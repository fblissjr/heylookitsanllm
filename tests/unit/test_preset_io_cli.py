# tests/unit/test_preset_io_cli.py
"""Unit tests for portable preset JSON/Markdown I/O, diffing, and `heylook` CLI."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from heylook_llm import cli, db
from heylook_llm.preset_io import (
    PresetFormatError,
    PresetRecord,
    format_combined_diffs,
    format_item_diff,
    load_presets_from_paths,
    parse_preset_markdown,
    parse_presets_json,
    plan_import,
    serialize_preset_json,
    serialize_preset_markdown,
    serialize_presets_json,
    unique_markdown_filenames,
)


class _MemoryPresetStore:
    """In-memory PresetStore implementing the CLI's store protocol."""

    def __init__(self, initial: list[dict] | None = None):
        self._items: list[dict] = []
        self._seq = 0
        for raw in initial or []:
            self._seq += 1
            self._items.append({
                "id": raw.get("id", f"p-{self._seq}"),
                "name": raw["name"],
                "system_prompt": raw.get("system_prompt"),
                "params": dict(raw.get("params") or {}),
            })

    def list_presets(self) -> list[dict]:
        return sorted(self._items, key=lambda p: p["name"])

    def create_preset(self, rec: PresetRecord) -> dict:
        if any(p["name"] == rec.name for p in self._items):
            raise RuntimeError(f"Duplicate name: {rec.name}")
        self._seq += 1
        row = {
            "id": f"p-{self._seq}",
            "name": rec.name,
            "system_prompt": rec.system_prompt,
            "params": dict(rec.params),
        }
        self._items.append(row)
        return row

    def update_preset(self, preset_id: str, rec: PresetRecord) -> dict:
        for row in self._items:
            if row["id"] == preset_id:
                row["system_prompt"] = rec.system_prompt
                row["params"] = dict(rec.params)
                return row
        raise RuntimeError(f"Unknown preset id: {preset_id}")

    def by_name(self) -> dict[str, dict]:
        return {p["name"]: p for p in self._items}


@pytest.mark.unit
class TestPresetSerialization:
    def test_single_json_roundtrip(self):
        rec = PresetRecord(
            name="Pirate",
            system_prompt="Talk like a pirate.\nAhoy!",
            params={"temperature": 1.2, "top_k": 40, "enable_thinking": True},
        )
        raw = serialize_preset_json(rec)
        parsed = parse_presets_json(raw)
        assert parsed == [rec]

    def test_bulk_json_roundtrip_and_bare_list(self):
        recs = [
            PresetRecord(name="Alpha", system_prompt="Be concise.", params={"temperature": 0.2}),
            PresetRecord(name="Beta", system_prompt=None, params={"seed": 7}),
        ]
        raw = serialize_presets_json(recs)
        assert "presets" in json.loads(raw)
        assert parse_presets_json(raw) == recs
        # Bare JSON list also accepted
        bare = json.dumps([r.to_export_dict() for r in recs])
        assert parse_presets_json(bare) == recs

    def test_markdown_roundtrip_normal_and_edge_cases(self):
        cases = [
            PresetRecord(
                name="Code Reviewer: Strict #1",
                system_prompt="Line 1\n\n---\n\nLine after rule.",
                params={
                    "temperature": 0.3,
                    "top_k": 20,
                    "enable_thinking": True,
                    "reasoning_effort": "medium",
                },
            ),
            PresetRecord(
                name="Settings Only",
                system_prompt=None,
                params={"temperature": 0.0, "seed": 42},
            ),
            PresetRecord(
                name="Prompt Only",
                system_prompt="Answer in haiku.",
                params={},
            ),
        ]
        for rec in cases:
            md = serialize_preset_markdown(rec)
            parsed = parse_preset_markdown(md)
            assert parsed == rec

    def test_markdown_plain_file_uses_filename_stem(self):
        parsed = parse_preset_markdown("Just a plain prompt.\n", fallback_name="PlainHelper")
        assert parsed == PresetRecord(name="PlainHelper", system_prompt="Just a plain prompt.", params={})

    def test_invalid_sampler_param_rejected(self):
        bad = json.dumps({"name": "Bad", "params": {"unknown_knob": 99}})
        with pytest.raises(PresetFormatError, match="Invalid preset params"):
            parse_presets_json(bad)

    def test_unique_markdown_filenames_avoid_collisions(self):
        names = unique_markdown_filenames([
            PresetRecord(name="Foo/Bar"),
            PresetRecord(name="Foo:Bar"),
            PresetRecord(name="Foo Bar"),
        ])
        assert names == ["Foo-Bar.md", "Foo-Bar-2.md", "Foo-Bar-3.md"]

    def test_duplicate_name_within_import_batch_rejected(self, tmp_path: Path):
        (tmp_path / "a.md").write_text("---\nname: Same\n---\n\nOne\n", encoding="utf-8")
        (tmp_path / "b.md").write_text("---\nname: Same\n---\n\nTwo\n", encoding="utf-8")
        with pytest.raises(PresetFormatError, match="Duplicate preset name"):
            load_presets_from_paths([tmp_path / "a.md", tmp_path / "b.md"])


@pytest.mark.unit
class TestPresetDiffAndPlan:
    def test_diff_detects_param_and_prompt_changes(self):
        existing = [
            {
                "id": "p-1",
                "name": "Pirate",
                "system_prompt": "Old line 1\nShared line",
                "params": {"temperature": 0.7, "top_k": 40},
            },
            {
                "id": "p-2",
                "name": "Same",
                "system_prompt": "Unchanged",
                "params": {"temperature": 0.5},
            },
        ]
        incoming = [
            PresetRecord(
                name="Pirate",
                system_prompt="New line 1\nShared line",
                params={"temperature": 1.2, "enable_thinking": True},
            ),
            PresetRecord(name="Same", system_prompt="Unchanged", params={"temperature": 0.5}),
            PresetRecord(name="Fresh", system_prompt="Brand new", params={}),
        ]
        plan = plan_import(incoming, existing)
        assert [i.incoming.name for i in plan.new_items] == ["Fresh"]
        assert [i.incoming.name for i in plan.changed_existing_items] == ["Pirate"]
        assert [i.incoming.name for i in plan.unchanged_existing_items] == ["Same"]

        pirate_diff = format_item_diff(plan.changed_existing_items[0])
        assert "- temperature: 0.7" in pirate_diff
        assert "+ temperature: 1.2" in pirate_diff
        assert "- top_k: 40" in pirate_diff
        assert "+ enable_thinking: true" in pirate_diff
        assert "-Old line 1" in pirate_diff
        assert "+New line 1" in pirate_diff

        combined = format_combined_diffs(plan)
        assert "=== Pirate (existing -> incoming) ===" in combined
        assert "=== Same (existing -> incoming) ===" in combined
        assert "(no differences)" in combined


@pytest.mark.unit
class TestHeylookCliPresets:
    def _run(self, argv: list[str], store: _MemoryPresetStore, inputs: list[str] | None = None):
        out = io.StringIO()
        err = io.StringIO()
        it = iter(inputs or [])
        prompts: list[str] = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return next(it)

        rc = cli.main(argv, store=store, out=out, err=err, input_fn=fake_input if inputs is not None else None)
        return rc, out.getvalue(), err.getvalue(), prompts

    def test_single_export_json_and_markdown(self, tmp_path: Path):
        store = _MemoryPresetStore([
            {"name": "Pirate", "system_prompt": "Talk pirate.", "params": {"temperature": 1.1}},
        ])
        json_out = tmp_path / "pirate.json"
        md_out = tmp_path / "pirate.md"

        rc, stdout, stderr, _ = self._run(["presets", "export", "Pirate", "-o", str(json_out)], store)
        assert rc == 0, stderr
        assert "Exported" in stdout
        assert parse_presets_json(json_out.read_text())[0].name == "Pirate"

        rc, stdout, stderr, _ = self._run(["presets", "export", "Pirate", "-o", str(md_out)], store)
        assert rc == 0, stderr
        parsed_md = parse_preset_markdown(md_out.read_text())
        assert parsed_md.name == "Pirate"
        assert parsed_md.system_prompt == "Talk pirate."
        assert parsed_md.params == {"temperature": 1.1}

    def test_bulk_export_json_and_markdown_dir(self, tmp_path: Path):
        store = _MemoryPresetStore([
            {"name": "Alpha", "system_prompt": "A", "params": {"temperature": 0.2}},
            {"name": "Beta", "system_prompt": "B", "params": {"top_p": 0.9}},
        ])
        json_bundle = tmp_path / "all.json"
        rc, _, stderr, _ = self._run(["presets", "export", "--all", "-o", str(json_bundle)], store)
        assert rc == 0, stderr
        assert len(parse_presets_json(json_bundle.read_text())) == 2

        md_dir = tmp_path / "md_bundle"
        rc, _, stderr, _ = self._run(
            ["presets", "export", "--all", "--format", "md", "-o", str(md_dir)], store
        )
        assert rc == 0, stderr
        md_files = sorted(p.name for p in md_dir.glob("*.md"))
        assert md_files == ["Alpha.md", "Beta.md"]

    def test_single_import_existing_shows_diff_and_respects_cancel_or_confirm(self, tmp_path: Path):
        store = _MemoryPresetStore([
            {"name": "Pirate", "system_prompt": "Old prompt", "params": {"temperature": 0.7}},
        ])
        incoming = tmp_path / "pirate.md"
        incoming.write_text(
            "---\nname: Pirate\nparams:\n  temperature: 1.3\n---\n\nNew prompt\n",
            encoding="utf-8",
        )

        # Cancel on conflict -> no write, diff shown before prompt
        rc, stdout, stderr, prompts = self._run(["presets", "import", str(incoming)], store, inputs=["n"])
        assert rc == 0, stderr
        assert "already exists" in stdout
        assert "- temperature: 0.7" in stdout
        assert "+ temperature: 1.3" in stdout
        assert "-Old prompt" in stdout
        assert "+New prompt" in stdout
        assert len(prompts) == 1
        assert store.by_name()["Pirate"]["system_prompt"] == "Old prompt"

        # Confirm overwrite -> applied
        rc, stdout, stderr, _ = self._run(["presets", "import", str(incoming)], store, inputs=["y"])
        assert rc == 0, stderr
        assert "1 updated" in stdout
        assert store.by_name()["Pirate"]["system_prompt"] == "New prompt"
        assert store.by_name()["Pirate"]["params"] == {"temperature": 1.3}

    def test_bulk_import_lists_new_and_existing_then_combined_diff_then_apply(self, tmp_path: Path):
        store = _MemoryPresetStore([
            {"name": "Alpha", "system_prompt": "Old A", "params": {"temperature": 0.5}},
            {"name": "Same", "system_prompt": "Keep", "params": {}},
        ])
        # Multiple markdown files (one per preset)
        f1 = tmp_path / "Alpha.md"
        f2 = tmp_path / "Same.md"
        f3 = tmp_path / "Gamma.md"
        f1.write_text("---\nname: Alpha\nparams:\n  temperature: 0.9\n---\n\nNew A\n", encoding="utf-8")
        f2.write_text("---\nname: Same\n---\n\nKeep\n", encoding="utf-8")
        f3.write_text("---\nname: Gamma\n---\n\nBrand new\n", encoding="utf-8")

        # User picks 'd' to see combined diffs, then 'a' to import and apply
        rc, stdout, stderr, prompts = self._run(
            ["presets", "import", str(f1), str(f2), str(f3)],
            store,
            inputs=["d", "a"],
        )
        assert rc == 0, stderr
        assert "New (1):" in stdout
        assert "+ Gamma" in stdout
        assert "Already exist (2):" in stdout
        assert "~ Alpha (changed)" in stdout
        assert "~ Same (unchanged)" in stdout
        assert "=== Alpha (existing -> incoming) ===" in stdout
        assert "-Old A" in stdout
        assert "+New A" in stdout
        assert "Import complete: 1 created, 1 updated, 1 unchanged." in stdout
        assert len(prompts) == 2

        after = store.by_name()
        assert after["Alpha"]["system_prompt"] == "New A"
        assert after["Alpha"]["params"] == {"temperature": 0.9}
        assert after["Gamma"]["system_prompt"] == "Brand new"

    def test_bulk_import_direct_apply_without_viewing_diffs(self, tmp_path: Path):
        store = _MemoryPresetStore([
            {"name": "Alpha", "system_prompt": "Old A", "params": {}},
        ])
        bundle = tmp_path / "bulk.json"
        bundle.write_text(
            serialize_presets_json([
                PresetRecord(name="Alpha", system_prompt="New A", params={"temperature": 0.8}),
                PresetRecord(name="Beta", system_prompt="New B", params={}),
            ]),
            encoding="utf-8",
        )
        rc, stdout, stderr, prompts = self._run(["presets", "import", str(bundle)], store, inputs=["a"])
        assert rc == 0, stderr
        assert "New (1):" in stdout
        assert "Already exist (1):" in stdout
        assert "=== Alpha" not in stdout  # Skipped combined diff view as requested
        assert len(prompts) == 1
        assert store.by_name()["Alpha"]["system_prompt"] == "New A"
        assert store.by_name()["Beta"]["system_prompt"] == "New B"

    def test_bulk_import_new_only_choice_preserves_existing(self, tmp_path: Path):
        store = _MemoryPresetStore([
            {"name": "Alpha", "system_prompt": "Keep old", "params": {}},
        ])
        bundle = tmp_path / "bulk.json"
        bundle.write_text(
            serialize_presets_json([
                PresetRecord(name="Alpha", system_prompt="Overwrite attempt", params={}),
                PresetRecord(name="Beta", system_prompt="New B", params={}),
            ]),
            encoding="utf-8",
        )
        rc, stdout, stderr, _ = self._run(["presets", "import", str(bundle)], store, inputs=["n"])
        assert rc == 0, stderr
        assert "1 created, 1 skipped" in stdout
        assert store.by_name()["Alpha"]["system_prompt"] == "Keep old"
        assert store.by_name()["Beta"]["system_prompt"] == "New B"

    def test_direct_duckdb_flag_roundtrip(self, tmp_path: Path):
        db_file = tmp_path / "presets_test.duckdb"
        md_file = tmp_path / "helper.md"
        md_file.write_text(
            "---\nname: Helper\nparams:\n  temperature: 0.4\n---\n\nBe helpful.\n",
            encoding="utf-8",
        )
        out = io.StringIO()
        err = io.StringIO()
        rc = cli.main(
            ["presets", "import", str(md_file), "--yes", "--db", str(db_file)],
            out=out,
            err=err,
        )
        assert rc == 0, err.getvalue()

        export_json = tmp_path / "exported.json"
        out2 = io.StringIO()
        err2 = io.StringIO()
        rc2 = cli.main(
            ["presets", "export", "Helper", "-o", str(export_json), "--db", str(db_file)],
            out=out2,
            err=err2,
        )
        assert rc2 == 0, err2.getvalue()
        got = parse_presets_json(export_json.read_text())[0]
        assert got == PresetRecord(name="Helper", system_prompt="Be helpful.", params={"temperature": 0.4})

    def test_js_and_python_markdown_and_zip_parity(self, tmp_path: Path):
        import shutil
        import subprocess
        import zipfile

        node = shutil.which("node")
        if not node:
            pytest.skip("node not installed")

        preset = PresetRecord(
            name="Code Reviewer: Strict #1",
            system_prompt="Line 1\n\n---\n\nLine 2",
            params={"temperature": 0.3, "top_k": 20, "enable_thinking": True, "reasoning_effort": "medium"},
        )
        py_md = serialize_preset_markdown(preset)
        zip_path = tmp_path / "out.zip"
        script = (
            "import fs from 'node:fs';"
            "import { buildZip, parsePresetMarkdown, serializePresetMarkdown } from './frontend/js/preset-io.js';"
            f"const pyMd = {json.dumps(py_md)};"
            "const parsed = parsePresetMarkdown(pyMd);"
            "const jsMd = serializePresetMarkdown(parsed);"
            "if (jsMd !== pyMd) { console.error({ jsMd, pyMd }); process.exit(1); }"
            f"fs.writeFileSync({json.dumps(str(zip_path))}, buildZip([{{ name: 'review.md', text: jsMd }}]));"
        )
        repo_root = Path(__file__).resolve().parents[2]
        proc = subprocess.run([node, "--input-type=module", "-e", script], cwd=repo_root, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

        with zipfile.ZipFile(zip_path, "r") as zf:
            assert zf.testzip() is None
            assert zf.namelist() == ["review.md"]
            assert parse_preset_markdown(zf.read("review.md").decode("utf-8")) == preset

    def test_http_store_roundtrip_with_api_key(self, monkeypatch):
        import urllib.error
        import urllib.request

        state: list[dict] = []
        seen: list[tuple[str, str, str | None]] = []

        class _FakeResp:
            def __init__(self, payload: dict):
                self._bytes = json.dumps(payload).encode("utf-8")

            def read(self):
                return self._bytes

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(req: urllib.request.Request, timeout: float = 15.0):
            key = req.get_header("X-api-key")
            seen.append((req.get_method(), req.full_url, key))
            if key != "secret-token":
                raise urllib.error.HTTPError(
                    req.full_url, 401, "Unauthorized", {}, io.BytesIO(b'{"detail":"auth required"}')
                )
            if req.get_method() == "GET":
                return _FakeResp({"presets": state, "total": len(state)})
            if req.get_method() == "POST":
                body = json.loads(req.data.decode("utf-8"))
                row = {"id": "p-1", **body}
                state.append(row)
                return _FakeResp(row)
            if req.get_method() == "PUT":
                body = json.loads(req.data.decode("utf-8"))
                state[0].update(body)
                return _FakeResp(state[0])
            raise AssertionError(f"Unexpected method: {req.get_method()}")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

        bad_store = cli._HttpPresetStore("http://127.0.0.1:8000", api_key="wrong")
        with pytest.raises(cli.StoreError, match="Authentication required"):
            bad_store.list_presets()

        http_store = cli._HttpPresetStore("http://127.0.0.1:8000", api_key="secret-token")
        assert http_store.list_presets() == []
        created = http_store.create_preset(PresetRecord(name="One", system_prompt="Hi", params={"temperature": 0.5}))
        assert created["id"] == "p-1"
        updated = http_store.update_preset("p-1", PresetRecord(name="One", system_prompt="Bye", params={"temperature": 0.8}))
        assert updated["system_prompt"] == "Bye"
        assert seen[1:] == [
            ("GET", "http://127.0.0.1:8000/v1/presets", "secret-token"),
            ("POST", "http://127.0.0.1:8000/v1/presets", "secret-token"),
            ("PUT", "http://127.0.0.1:8000/v1/presets/p-1", "secret-token"),
        ]
