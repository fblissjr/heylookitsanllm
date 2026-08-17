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
hand-renamed entry stops matching itself, and ``modelzoo/<vendor>`` symlinks
make one file reachable by two spellings that share no prefix. Both failures
produced a real duplicate entry on 2026-08-17.

Discovery is best-effort by construction: a scan that raises is logged and
dropped, and the server comes up on models.toml alone. Serving fewer models
than expected is recoverable; refusing to start is not.
"""

from __future__ import annotations

import logging
from pathlib import Path


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
        # legitimately produce two ids for one file: scan_hf_cache rewrites
        # the id to `org/name` after the directory-name id was registered,
        # and scan_directory follows symlinks, so two links to one store dir
        # yield two names. Without this the same GGUF is servable twice and,
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


def discover(config_data: dict) -> list[dict]:
    """Scan the folders named by ``[scan].folders``; never raise.

    Returns entry dicts ready for :func:`merge_discovered`. An empty list is
    the correct answer for "no [scan] section", "scanning is off", and "the
    scan failed" alike -- all three mean models.toml stands alone.
    """
    scan_cfg = config_data.get("scan") or {}
    folders = [str(f) for f in (scan_cfg.get("folders") or [])]
    watch_hf = bool(scan_cfg.get("watch_hf_cache", False))
    if not folders and not watch_hf:
        return []
    # scan_interval_seconds = 0 is the documented off switch ("0 disables
    # periodic rescans (no initial scan either)", ScanConfig) and
    # MemoryManager._maybe_rescan_models honors it. Load-time discovery has to
    # honor it too, or setting it to 0 to STOP scanning silently starts
    # serving everything under the folders instead -- the opposite of what the
    # operator asked for.
    if int(scan_cfg.get("scan_interval_seconds", 900) or 0) <= 0:
        return []

    entries: list[dict] = []
    try:
        # ModelImporter, NOT ModelService.scan_paths: the latter projects each
        # hit into a ScannedModel for the admin UI, and that projection drops
        # mmproj_path -- every vision GGUF would come back text-only. The
        # importer's raw dicts are already the models.toml entry shape, which
        # is exactly what the merge wants.
        #
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
        return []

    # PER-SOURCE isolation: one unmounted volume, or one model whose entry
    # fails ModelImporter._validate (which RAISES rather than skipping), must
    # not discard every other folder's models. A single try around the whole
    # loop turned one bad directory into "discovery returned nothing".
    for folder in folders:
        try:
            entries.extend(importer.scan_directory(folder))
        except Exception:
            logging.warning(
                "[registry] scan of %s failed; skipping that folder only",
                folder, exc_info=True)
    if watch_hf:
        try:
            entries.extend(importer.scan_hf_cache())
        except Exception:
            logging.warning(
                "[registry] HF cache scan failed; skipping it", exc_info=True)

    return [e for e in entries if isinstance(e, dict) and e.get("config")]
