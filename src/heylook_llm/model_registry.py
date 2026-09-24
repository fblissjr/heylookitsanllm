"""Discovery-as-registry: the model store decides what exists, models.toml only overrides.

Phase 6 item 1 of the roadmap ("registry-over-scan: structured, non-clobbering")
specifies merging by RESOLVED ``model_path`` rather than by id. This module is
that merge, applied at LOAD time instead of at write time -- so nothing is
generated into models.toml at all.

The rule, in one sentence: every ``[[models]]`` entry is served exactly as
written, and any model found under ``[scan].folders`` that no entry already
describes is served with derived defaults.

Consequences worth stating, because they are the whole point:

- A new download in a scan folder is servable with no import, no symlink, and
  no edit. models.toml is not touched -- there is nothing to clobber.
- models.toml shrinks to what cannot be derived: a hand-chosen id, a
  ``chat_template_path``, ``spec_type``, ``enabled = false``, a comment
  explaining a trap. Write an entry when you want to CHANGE something.
- Explicit always wins. That is what keeps models outside the scan folders
  working, and what makes ``enabled = false`` a real off switch rather than
  something a rescan undoes.

Matching is on the resolved path (``.resolve()`` follows symlinks) because id
matching is what broke: an id is derived from the directory name, so a
hand-renamed entry stops matching itself, and vendor symlinks in a model
folder make one file reachable by two spellings that share no prefix. Both failures
produced a real duplicate entry on 2026-08-17.

Discovery is best-effort by construction: a scan that raises is logged and
dropped, and the server comes up on models.toml alone. Serving fewer models
than expected is recoverable; refusing to start is not.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple


# Bootstrap-only, like HEYLOOK_DB_PATH: set by every launcher that is NOT the
# owner's daily server (scripts/dev_server.sh, the E2E harness, loop runs).
# Model folders, and the models.toml a dev server reads, are shared by every
# instance, so an automated run must not be able to change the owner's real
# model settings; it still serves them exactly (owner decision 2026-09-24).
READONLY_ENV = "HEYLOOK_READONLY_MODEL_CONFIG"


class ModelConfigReadOnly(RuntimeError):
    """A model-config write on an instance started read-only. The app maps
    it to 409 wherever it escapes a route (api.py)."""


def refuse_if_readonly(what: str) -> None:
    """Raise :class:`ModelConfigReadOnly` when this instance is read-only.
    Called inside every writer of model settings, so a new route inherits it."""
    import os

    if os.environ.get(READONLY_ENV, "") not in ("", "0"):
        raise ModelConfigReadOnly(
            f"this server was started with {READONLY_ENV} set, so it does not write "
            f"model settings ({what}); make the change on the daily server")


def path_identity(path: str) -> str:
    """Resolved, symlink-followed spelling of a model path -- THE identity rule.

    Import it; do not re-inline it. This is the single definition of "are these
    two spellings the same file", and the importer's dedup, the registry merge,
    and ModelService._configured_identity must not be able to disagree about it
    -- a future refinement (case folding on APFS, strict= semantics) has to
    land in exactly one place.

    Falls back to the literal string when the path cannot be resolved (a dead
    symlink, a permission wall) so an unreadable entry still deduplicates
    against itself instead of raising or colliding with everything else.
    """
    try:
        return str(Path(path).expanduser().resolve())
    except (OSError, RuntimeError, ValueError):
        return path


# Back-compat alias for the private spelling used before the rule was shared.
_identity = path_identity


def _entry_path(entry: dict) -> str:
    return str((entry.get("config") or {}).get("model_path") or "")


def merge_discovered(config_data: dict, discovered: list[dict]) -> dict:
    """Return ``config_data`` with unrepresented discovered models appended.

    ``config_data`` is the parsed models.toml. ``discovered`` is a list of
    entry dicts in the same shape (``{id, provider, enabled, config}``) as the
    importer builds. Neither input is mutated.
    """
    explicit: list[dict] = list(config_data.get("models") or [])

    # ALWAYS materialize `models`, even on the early returns. AppConfig.models
    # is a REQUIRED field, so handing back a dict without the key raises
    # ValidationError and the server does not start -- and the config shape
    # that hits it is the one this design promotes: a models.toml carrying
    # only [scan]. Empty folder, unmounted volume, or a failed scan all reach
    # here with discovered=[]. "Serve fewer models" must never become "refuse
    # to boot".
    merged = dict(config_data)
    merged["models"] = explicit
    if not discovered:
        return merged

    configured_paths = {
        path_identity(p) for e in explicit if (p := _entry_path(e))
    }
    configured_ids = {str(e["id"]) for e in explicit if e.get("id")}

    added: list[dict] = []
    for entry in discovered:
        path = _entry_path(entry)
        if not path:
            continue
        if path_identity(path) in configured_paths:
            continue  # models.toml already describes this file; it wins
        model_id = str(entry.get("id") or "")
        if not model_id:
            continue
        if model_id in configured_ids:
            # Same derived name, DIFFERENT file. Serving both would make the
            # id ambiguous and get_model_config() would silently pick one, so
            # decline and say which file went unserved -- a rename in
            # models.toml or on disk is the fix.
            logging.warning(
                "[registry] discovered model at %s not served: its derived id "
                "%r is already used by a different models.toml entry",
                path, model_id)
            continue
        configured_ids.add(model_id)
        # Extend BOTH sets with what we just accepted, or discovery only
        # dedupes against models.toml and not against itself. Two scanners
        # legitimately produce two ids for one file: scan_directory follows
        # symlinks, so two links to one store dir yield two names. Without this the same GGUF is servable twice and,
        # above max_loaded_models=1, loads into two llama-server processes.
        configured_paths.add(path_identity(path))
        added.append(entry)

    if not added:
        return merged

    logging.info(
        "[registry] serving %d discovered model(s) not in models.toml: %s",
        len(added), ", ".join(str(e["id"]) for e in added))
    merged["models"] = explicit + added
    return merged


def derived_for_explicit(config_data: dict, discovered: list[dict]) -> dict[str, dict]:
    """For each models.toml entry, the config discovery derives for the SAME
    file (matched by resolved path, the merge's identity rule), keyed by the
    entry's id. merge_discovered drops these, since the entry wins; the engine
    contract needs them to tell a stored value that differs from derivation
    (configured) from one that merely repeats it (a materialized copy).
    """
    by_path = {path_identity(p): dict(e.get("config") or {})
               for e in discovered if (p := _entry_path(e))}
    out: dict[str, dict] = {}
    for e in config_data.get("models") or []:
        p = _entry_path(e)
        if p and e.get("id") and path_identity(p) in by_path:
            out[str(e["id"])] = by_path[path_identity(p)]
    return out


class Discovery(NamedTuple):
    """What a scan found, and which sources it could not read.

    ``failed`` is what makes a comparison over two scans honest: a source that
    failed contributes no models, which is indistinguishable by count from
    those models being gone.
    """
    entries: list[dict]
    failed: list[str]


def discover(config_data: dict) -> list[dict]:
    """The models found under ``[scan].folders``; never raises. See :func:`scan`."""
    return scan(config_data).entries


def scan(config_data: dict) -> Discovery:
    """Scan the folders named by ``[scan].folders``; never raise.

    ``entries`` are entry dicts ready for :func:`merge_discovered`. They are
    empty for "no [scan] section", "scanning is off" and "the scan failed"
    alike -- all three mean models.toml stands alone -- and ``failed`` is what
    tells the last apart: each folder that is missing or raised, each model
    the importer rejected, and the importer itself if it would not construct.
    """
    scan_cfg = config_data.get("scan") or {}
    folders = [str(f) for f in (scan_cfg.get("folders") or [])]
    if not folders:
        return Discovery([], [])
    # scan_interval_seconds = 0 is the documented off switch (ScanConfig):
    # setting it to 0 to STOP scanning must not keep serving everything under
    # the folders. Its other values schedule nothing since the periodic rescan
    # was retired (v2.0.118).
    if int(scan_cfg.get("scan_interval_seconds", 900) or 0) <= 0:
        return Discovery([], [])

    try:
        # A FRESH importer per call is deliberate: its existing_ids
        # bookkeeping makes the scanners skip already-configured models, and
        # discovery wants everything. Deduplication is merge_discovered's job,
        # by resolved path, and it cannot do it for entries it never sees.
        from heylook_llm.model_importer import ModelImporter

        importer = ModelImporter()
    except Exception:
        logging.warning(
            "[registry] importer unavailable; serving models.toml alone",
            exc_info=True)
        return Discovery([], ["importer"])

    # PER-SOURCE isolation: one unmounted volume must not discard every other
    # folder's models. A single try around the whole loop turned one bad
    # directory into "discovery returned nothing".
    entries: list[dict] = []
    failed: list[str] = []
    for folder in folders:
        # The importer answers [] for a missing folder, which reads as "no
        # models here". An unmounted volume is a failure, so say so.
        if not Path(folder).expanduser().is_dir():
            logging.warning("[registry] scan folder %s is not a directory; skipping it", folder)
            failed.append(folder)
            continue
        try:
            entries.extend(importer.scan_directory(folder))
        except Exception:
            logging.warning(
                "[registry] scan of %s failed; skipping that folder only",
                folder, exc_info=True)
            failed.append(folder)
    failed.extend(path for path, _ in importer.rejected)

    return Discovery([e for e in entries if isinstance(e, dict) and e.get("config")], failed)


def served(config_data: dict, discovered: list[dict]):
    """The served config: the merge, validated. The router builds its
    ``AppConfig`` here and nowhere else, so anything that asks "what would be
    served" through this function cannot disagree with the server.

    Deep-copied first: ``AppConfig`` validation replaces each entry's nested
    ``config`` dict with a model instance IN PLACE, and ``merge_discovered``
    hands back the caller's own entry dicts.
    """
    from heylook_llm.config import AppConfig

    return AppConfig(**copy.deepcopy(merge_discovered(config_data, discovered)))


@dataclass
class ServedDiff:
    """What an edit does to the served set, by id.

    ``changed`` maps an id served on both sides to ``{field: [before, after]}``
    over the whole validated entry (config fields by name, the rest as
    ``entry.<name>``). ``renamed`` maps an old id to the new id serving the
    same file (resolved ``model_path``); a renamed model's field changes are in
    ``changed`` under its new id, and it is in neither ``lost`` nor
    ``gained``. ``unreliable`` names the sources either scan failed to read: a
    model under one of them can show as lost when it is only unread.
    """
    gained: list[str] = field(default_factory=list)
    lost: list[str] = field(default_factory=list)
    renamed: dict[str, str] = field(default_factory=dict)
    changed: dict[str, dict[str, list]] = field(default_factory=dict)
    unreliable: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.gained or self.lost or self.renamed or self.changed)


def _served_rows(app) -> dict[str, dict]:
    rows = {}
    for m in app.models:
        dumped = m.model_dump(mode="json")
        row = {k: v for k, v in (dumped.pop("config") or {}).items()}
        row.update({f"entry.{k}": v for k, v in dumped.items() if k != "id"})
        rows[m.id] = row
    return rows


def served_diff(before: tuple[dict, Discovery], after: tuple[dict, Discovery]) -> ServedDiff:
    """Compare what two configs serve, each side a ``(config_data, Discovery)``.

    Pure over its arguments: discovery is passed in, because it is the only
    part that touches the filesystem (the caller scans once, or twice for an
    edit that moves a folder). Each side goes through :func:`served`, the
    router's own merge and validation, never a second copy of the matching
    rule. A side the server would refuse raises, which is the answer.
    """
    (cfg_a, disc_a), (cfg_b, disc_b) = before, after
    rows_a = _served_rows(served(cfg_a, disc_a.entries))
    rows_b = _served_rows(served(cfg_b, disc_b.entries))
    lost = rows_a.keys() - rows_b.keys()
    gained = rows_b.keys() - rows_a.keys()
    # A lost id and a gained id serving one file (the merge's identity rule)
    # are a rename, not a loss: deleting a hand-named entry hands the file
    # back to discovery under its derived id.
    gained_by_path: dict[str, list[str]] = {}
    for mid in gained:
        gained_by_path.setdefault(path_identity(rows_b[mid]["model_path"]), []).append(mid)
    renamed = {}
    for mid in sorted(lost):
        heirs = gained_by_path.get(path_identity(rows_a[mid]["model_path"])) or []
        if len(heirs) == 1:
            renamed[mid] = heirs.pop()
    pairs = [(m, m) for m in rows_a.keys() & rows_b.keys()] + list(renamed.items())
    changed = {}
    for old, new in pairs:
        a, b = rows_a[old], rows_b[new]
        delta = {k: [a.get(k), b.get(k)] for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)}
        if delta:
            changed[new] = delta
    return ServedDiff(
        gained=sorted(gained - set(renamed.values())),
        lost=sorted(lost - renamed.keys()),
        renamed=renamed,
        changed=dict(sorted(changed.items())),
        unreliable=sorted(set(disc_a.failed) | set(disc_b.failed)),
    )
