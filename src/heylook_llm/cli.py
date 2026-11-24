# src/heylook_llm/cli.py
"""Heylook command-line utility (`uv run heylook ...`).

Lightweight CLI (zero MLX/torch imports) for managing heylook resources such as
presets. Talks to a running heylookllm server over HTTP by default and falls
back to direct local DuckDB access when the local server is not running (or
when `--db` is passed explicitly).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable, Protocol, Sequence, TextIO
import urllib.error
import urllib.parse
import urllib.request

from heylook_llm.preset_io import (
    ImportPlan,
    PresetFormatError,
    PresetRecord,
    format_combined_diffs,
    format_item_diff,
    load_presets_from_paths,
    plan_import,
    serialize_preset_json,
    serialize_preset_markdown,
    serialize_presets_json,
    slugify_preset_name,
    unique_markdown_filenames,
)

DEFAULT_SERVER_URL = "http://127.0.0.1:8000"
DEFAULT_API_KEY_PATH = Path.home() / ".heylook" / "api_key"


class PresetStore(Protocol):
    def list_presets(self) -> list[dict[str, Any]]: ...
    def create_preset(self, rec: PresetRecord) -> dict[str, Any]: ...
    def update_preset(self, preset_id: str, rec: PresetRecord) -> dict[str, Any]: ...


class StoreError(RuntimeError):
    """Raised on HTTP or DuckDB failures when reading/writing presets."""


def _resolve_api_key(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    env_key = os.environ.get("HEYLOOK_API_KEY", "").strip()
    if env_key:
        return env_key
    key_file_env = os.environ.get("HEYLOOK_API_KEY_FILE", "").strip()
    key_path = Path(key_file_env).expanduser() if key_file_env else DEFAULT_API_KEY_PATH
    try:
        if key_path.is_file():
            content = key_path.read_text(encoding="utf-8").strip()
            if content:
                return content
    except OSError:
        pass
    return None


class _HttpPresetStore:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        headers = {"Accept": "application/json"}
        data: bytes | None = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        if self.api_key:
            headers["x-api-key"] = self.api_key
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.reason
            try:
                body = json.loads(exc.read().decode("utf-8"))
                if isinstance(body, dict):
                    detail = body.get("detail") or body.get("error", {}).get("message") or detail
            except Exception:
                pass
            if exc.code == 401:
                raise StoreError(
                    f"Authentication required by {self.base_url} (401). "
                    "Set HEYLOOK_API_KEY, pass --api-key, or configure ~/.heylook/api_key."
                ) from None
            raise StoreError(f"HTTP {exc.code} on {method} {path}: {detail}") from None
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Cannot reach {self.base_url}: {exc.reason}") from exc

    def list_presets(self) -> list[dict[str, Any]]:
        data = self._request("GET", "/v1/presets")
        return list((data or {}).get("presets", []))

    def create_preset(self, rec: PresetRecord) -> dict[str, Any]:
        return self._request(
            "POST",
            "/v1/presets",
            {"name": rec.name, "system_prompt": rec.system_prompt, "params": rec.params},
        )

    def update_preset(self, preset_id: str, rec: PresetRecord) -> dict[str, Any]:
        return self._request(
            "PUT",
            f"/v1/presets/{urllib.parse.quote(preset_id, safe='')}",
            {"system_prompt": rec.system_prompt, "params": rec.params},
        )


class _DuckDBPresetStore:
    def __init__(self, db_path: str | Path | None = None):
        self.db_path = db_path

    def _run(self, fn):
        from heylook_llm import db

        async def _inner():
            conn = await db.get_connection(path=self.db_path)
            try:
                return await fn(db, conn)
            finally:
                await conn.close()

        try:
            return asyncio.run(_inner())
        except Exception as exc:
            raise StoreError(f"DuckDB error: {exc}") from None

    def list_presets(self) -> list[dict[str, Any]]:
        return self._run(lambda db, conn: db.list_presets(conn))

    def create_preset(self, rec: PresetRecord) -> dict[str, Any]:
        return self._run(
            lambda db, conn: db.create_preset(
                conn, name=rec.name, system_prompt=rec.system_prompt, params=rec.params
            )
        )

    def update_preset(self, preset_id: str, rec: PresetRecord) -> dict[str, Any]:
        return self._run(
            lambda db, conn: db.update_preset(
                conn, preset_id, system_prompt=rec.system_prompt, params=rec.params
            )
        )


def _open_store(
    *,
    server: str | None,
    api_key: str | None,
    db_path: str | None,
    err: TextIO,
) -> tuple[PresetStore, list[dict[str, Any]]]:
    """Connect to either the HTTP server or local DuckDB and load current presets."""
    if db_path:
        store: PresetStore = _DuckDBPresetStore(db_path)
        return store, store.list_presets()

    env_server = os.environ.get("HEYLOOK_SERVER", "").strip()
    url = (server or env_server or DEFAULT_SERVER_URL).rstrip("/")
    resolved_key = _resolve_api_key(api_key)
    http_store = _HttpPresetStore(url, api_key=resolved_key)
    try:
        return http_store, http_store.list_presets()
    except ConnectionError as exc:
        if server or env_server:
            raise StoreError(str(exc)) from None
        # Default local server is not running: fall back cleanly to local DuckDB.
        print(f"Note: server not reachable at {url}; using local DuckDB store.", file=err)
        local_store = _DuckDBPresetStore(None)
        return local_store, local_store.list_presets()


def _resolve_export_format(fmt: str | None, out_path: Path | None, *, bulk: bool) -> str:
    if fmt:
        return "md" if fmt in ("md", "markdown") else "json"
    if out_path is not None:
        suffix = out_path.suffix.lower()
        if suffix in (".md", ".markdown"):
            return "md"
        if suffix == ".json":
            return "json"
        if bulk and (out_path.is_dir() or str(out_path).endswith(os.sep)):
            return "md"
    return "json"


def _cmd_presets_list(
    presets: list[dict[str, Any]],
    *,
    out: TextIO,
) -> int:
    if not presets:
        print("No presets saved.", file=out)
        return 0
    for p in presets:
        prompt = p.get("system_prompt")
        prompt_desc = f"{len(prompt)} chars" if prompt else "settings only"
        params = p.get("params") or {}
        params_desc = ", ".join(f"{k}={v}" for k, v in sorted(params.items())) or "no pinned params"
        print(f"{p['name']}  ({prompt_desc}; {params_desc})", file=out)
    return 0


def _cmd_presets_export(
    args: argparse.Namespace,
    presets: list[dict[str, Any]],
    *,
    out: TextIO,
    err: TextIO,
) -> int:
    if args.all and args.names:
        print("Error: specify either preset names or --all, not both.", file=err)
        return 2
    if not args.all and not args.names:
        print("Error: provide one or more preset names, or pass --all.", file=err)
        return 2

    by_name = {p["name"]: p for p in presets}
    if args.all:
        selected = list(presets)
        if not selected:
            print("No presets to export.", file=err)
            return 1
    else:
        missing = [n for n in args.names if n not in by_name]
        if missing:
            print(f"Error: preset not found: {', '.join(missing)}", file=err)
            return 1
        selected = [by_name[n] for n in args.names]

    bulk = bool(args.all or len(selected) > 1)
    out_path = Path(args.output) if args.output else None
    fmt = _resolve_export_format(args.format, out_path, bulk=bulk)

    if not bulk:
        preset = selected[0]
        content = serialize_preset_markdown(preset) if fmt == "md" else serialize_preset_json(preset)
        if out_path is None:
            out.write(content)
            return 0
        if out_path.is_dir() or str(args.output).endswith(os.sep):
            out_path.mkdir(parents=True, exist_ok=True)
            ext = "md" if fmt == "md" else "json"
            target = out_path / f"{slugify_preset_name(preset['name'])}.{ext}"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            target = out_path
        target.write_text(content, encoding="utf-8")
        print(f"Exported {preset['name']!r} -> {target}", file=out)
        return 0

    # Bulk export
    if fmt == "json":
        content = serialize_presets_json(selected)
        if out_path is None:
            out.write(content)
            return 0
        if out_path.is_dir() or str(args.output).endswith(os.sep):
            out_path.mkdir(parents=True, exist_ok=True)
            target = out_path / "heylook-presets.json"
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            target = out_path
        target.write_text(content, encoding="utf-8")
        print(f"Exported {len(selected)} presets -> {target}", file=out)
        return 0

    # Bulk Markdown: one .md file per preset in output directory
    if out_path is None:
        print("Error: bulk Markdown export requires -o <directory>.", file=err)
        return 2
    if out_path.exists() and not out_path.is_dir():
        print(f"Error: bulk Markdown export target must be a directory: {out_path}", file=err)
        return 2
    out_path.mkdir(parents=True, exist_ok=True)
    filenames = unique_markdown_filenames(selected)
    for preset, filename in zip(selected, filenames):
        (out_path / filename).write_text(serialize_preset_markdown(preset), encoding="utf-8")
    print(f"Exported {len(selected)} Markdown preset files -> {out_path}", file=out)
    return 0


def _print_bulk_summary(plan: ImportPlan, out: TextIO) -> None:
    print(f"Bulk import summary ({len(plan.items)} presets):", file=out)
    new_items = plan.new_items
    existing_items = plan.existing_items
    if new_items:
        print(f"  New ({len(new_items)}):", file=out)
        for it in new_items:
            print(f"    + {it.incoming.name}", file=out)
    if existing_items:
        print(f"  Already exist ({len(existing_items)}):", file=out)
        for it in existing_items:
            tag = "changed" if it.changed else "unchanged"
            print(f"    ~ {it.incoming.name} ({tag})", file=out)


def _apply_plan(
    store: PresetStore,
    plan: ImportPlan,
    *,
    include_existing: bool,
    out: TextIO,
) -> int:
    created = 0
    updated = 0
    unchanged = 0
    skipped = 0

    for item in plan.items:
        if item.existing is None:
            store.create_preset(item.incoming)
            created += 1
        elif not include_existing:
            skipped += 1
        elif not item.changed:
            unchanged += 1
        else:
            store.update_preset(item.existing["id"], item.incoming)
            updated += 1

    parts = []
    if created:
        parts.append(f"{created} created")
    if updated:
        parts.append(f"{updated} updated")
    if unchanged:
        parts.append(f"{unchanged} unchanged")
    if skipped:
        parts.append(f"{skipped} skipped")
    summary = ", ".join(parts) or "nothing changed"
    print(f"Import complete: {summary}.", file=out)
    return 0


def _prompt_choice(
    prompt: str,
    *,
    input_fn: Callable[[str], str] | None,
    err: TextIO,
) -> str | None:
    if input_fn is None:
        if not sys.stdin.isatty():
            print(
                "Error: interactive confirmation required on a non-interactive terminal. "
                "Re-run with --yes, --new-only, or --diff.",
                file=err,
            )
            return None
        input_fn = input
    try:
        return input_fn(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.", file=err)
        return "c"


def _cmd_presets_import(
    args: argparse.Namespace,
    store: PresetStore,
    existing_presets: list[dict[str, Any]],
    *,
    out: TextIO,
    err: TextIO,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    try:
        incoming = load_presets_from_paths([Path(p) for p in args.paths], validate_params=True)
    except PresetFormatError as exc:
        print(f"Error: {exc}", file=err)
        return 1

    plan = plan_import(incoming, existing_presets)

    # Preview-only diff mode
    if args.diff:
        if len(plan.items) == 1:
            it = plan.items[0]
            if it.existing is None:
                print(f"New preset: {it.incoming.name}", file=out)
            else:
                print(f"Preset {it.incoming.name!r} already exists.", file=out)
                print(format_item_diff(it), file=out)
        else:
            _print_bulk_summary(plan, out)
            if plan.existing_items:
                print("", file=out)
                print(format_combined_diffs(plan), file=out)
        return 0

    # Non-interactive apply
    if args.yes:
        if len(plan.items) > 1:
            _print_bulk_summary(plan, out)
        return _apply_plan(store, plan, include_existing=not args.new_only, out=out)

    # Single-preset import flow (Requirement 5)
    if len(plan.items) == 1:
        item = plan.items[0]
        if item.existing is None:
            print(f"New preset: {item.incoming.name}", file=out)
            ans = _prompt_choice(f"Import preset {item.incoming.name!r}? [Y/n]: ", input_fn=input_fn, err=err)
            if ans is None:
                return 2
            if ans in ("", "y", "yes"):
                return _apply_plan(store, plan, include_existing=True, out=out)
            print("Cancelled.", file=out)
            return 0

        print(f"Preset {item.incoming.name!r} already exists.", file=out)
        print(format_item_diff(item), file=out)
        if not item.changed:
            print("No changes to apply.", file=out)
            return 0
        if args.new_only:
            print(f"Skipped existing preset {item.incoming.name!r} (--new-only).", file=out)
            return 0
        ans = _prompt_choice(
            f"Overwrite existing preset {item.incoming.name!r}? [y/N]: ",
            input_fn=input_fn,
            err=err,
        )
        if ans is None:
            return 2
        if ans in ("y", "yes"):
            return _apply_plan(store, plan, include_existing=True, out=out)
        print("Cancelled.", file=out)
        return 0

    # Bulk import flow (Requirements 5 & 6)
    _print_bulk_summary(plan, out)

    if args.new_only:
        if not plan.new_items:
            print("No new presets to import.", file=out)
            return 0
        ans = _prompt_choice(
            f"Import {len(plan.new_items)} new preset(s) (skipping existing)? [Y/n]: ",
            input_fn=input_fn,
            err=err,
        )
        if ans is None:
            return 2
        if ans in ("", "y", "yes"):
            return _apply_plan(store, plan, include_existing=False, out=out)
        print("Cancelled.", file=out)
        return 0

    if not plan.existing_items:
        ans = _prompt_choice(
            f"Import {len(plan.new_items)} new preset(s)? [Y/n]: ",
            input_fn=input_fn,
            err=err,
        )
        if ans is None:
            return 2
        if ans in ("", "y", "yes"):
            return _apply_plan(store, plan, include_existing=True, out=out)
        print("Cancelled.", file=out)
        return 0

    # Existing presets present in bulk import: offer combined diff view OR direct import & apply.
    while True:
        ans = _prompt_choice(
            "Choose action: [d] view combined diffs, [a] import and apply, [n] import new only, [c] cancel: ",
            input_fn=input_fn,
            err=err,
        )
        if ans is None:
            return 2
        if ans in ("d", "diff", "diffs"):
            print("", file=out)
            print(format_combined_diffs(plan), file=out)
            print("", file=out)
            follow = _prompt_choice(
                "Apply now? [a] import and apply, [n] import new only, [c] cancel: ",
                input_fn=input_fn,
                err=err,
            )
            if follow is None:
                return 2
            if follow in ("a", "apply", "y", "yes"):
                return _apply_plan(store, plan, include_existing=True, out=out)
            if follow in ("n", "new"):
                return _apply_plan(store, plan, include_existing=False, out=out)
            print("Cancelled.", file=out)
            return 0
        if ans in ("a", "apply", "y", "yes"):
            return _apply_plan(store, plan, include_existing=True, out=out)
        if ans in ("n", "new"):
            return _apply_plan(store, plan, include_existing=False, out=out)
        if ans in ("c", "cancel", "q", "quit"):
            print("Cancelled.", file=out)
            return 0
        print("Please enter d, a, n, or c.", file=out)


def _connection_parent_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--server",
        default=argparse.SUPPRESS,
        help=f"Running heylookllm server URL (default: $HEYLOOK_SERVER or {DEFAULT_SERVER_URL})",
    )
    p.add_argument(
        "--api-key",
        default=argparse.SUPPRESS,
        help="API key for authenticated servers (default: $HEYLOOK_API_KEY or ~/.heylook/api_key)",
    )
    p.add_argument(
        "--db",
        default=argparse.SUPPRESS,
        help="Read/write a local DuckDB file directly instead of calling the HTTP API",
    )
    return p


def build_parser() -> argparse.ArgumentParser:
    conn = _connection_parent_parser()
    parser = argparse.ArgumentParser(
        prog="heylook",
        description="Heylook CLI for managing presets and server resources.",
        parents=[conn],
    )

    subparsers = parser.add_subparsers(dest="group", required=True)

    presets_parser = subparsers.add_parser(
        "presets",
        help="List, export, and import presets",
        parents=[conn],
    )
    presets_sub = presets_parser.add_subparsers(dest="action", required=True)

    presets_sub.add_parser("list", help="List saved presets", parents=[conn])

    exp = presets_sub.add_parser(
        "export",
        help="Export one preset (JSON/Markdown) or bulk export (--all) to JSON or a folder of Markdown files",
        parents=[conn],
    )
    exp.add_argument("names", nargs="*", help="Preset name(s) to export")
    exp.add_argument("--all", action="store_true", help="Export all presets")
    exp.add_argument("-o", "--output", default=None, help="Output file path (or directory for bulk Markdown)")
    exp.add_argument(
        "--format",
        choices=["json", "md", "markdown"],
        default=None,
        help="Output format (default: inferred from -o extension, else json)",
    )

    imp = presets_sub.add_parser(
        "import",
        help="Import one or more presets from JSON file(s), Markdown file(s), or a directory",
        parents=[conn],
    )
    imp.add_argument("paths", nargs="+", help="Preset .json/.md file(s) or directory")
    imp.add_argument("-y", "--yes", action="store_true", help="Apply without interactive confirmation")
    imp.add_argument("--new-only", action="store_true", help="Import only new presets; skip existing names")
    imp.add_argument("--diff", action="store_true", help="Show summary and before/after diffs without writing")

    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    store: PresetStore | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
    input_fn: Callable[[str], str] | None = None,
) -> int:
    out_stream = out or sys.stdout
    err_stream = err or sys.stderr
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.group != "presets":
        parser.print_help(err_stream)
        return 2

    try:
        if store is None:
            active_store, current_presets = _open_store(
                server=getattr(args, "server", None),
                api_key=getattr(args, "api_key", None),
                db_path=getattr(args, "db", None),
                err=err_stream,
            )
        else:
            active_store = store
            current_presets = active_store.list_presets()
    except StoreError as exc:
        print(f"Error: {exc}", file=err_stream)
        return 1

    if args.action == "list":
        return _cmd_presets_list(current_presets, out=out_stream)
    if args.action == "export":
        return _cmd_presets_export(args, current_presets, out=out_stream, err=err_stream)
    if args.action == "import":
        return _cmd_presets_import(
            args,
            active_store,
            current_presets,
            out=out_stream,
            err=err_stream,
            input_fn=input_fn,
        )
    return 2


def entrypoint() -> None:
    sys.exit(main())


if __name__ == "__main__":
    entrypoint()
