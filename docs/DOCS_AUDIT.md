# docs/ Audit

Last updated: 2026-04-18

## Executive Summary

9 files audited (excluding `archive/` and `frontend-mocks/`). No removals warranted outright, but three issues need action: `openapi.json` is severely stale (v1.0.1, missing ~15 endpoints added since v1.14.0, contains endpoints for removed providers including STT/audio), `frontend_api_reference.md` has no "last updated" stamp and is missing the conversation, notebook, RLM, and data-clear API families added in v1.27.0 and prior, and `mlx_optimization_plan.md` has no inbound links from CLAUDE.md or README.md and its speculative-decoding Phase 4 still reads "planned improvements" despite being shipped in v1.15.0. The observability guide (new today) contains two forward-references to a local-only plan file (`plans/lets-look-at-our-prancy-gosling.md`) that does not exist on disk; those links will 404 for anyone reading the committed file.

---

## Per-File Status Table

| File | Last Updated | Inbound Links | Size (KB) | Status |
|------|-------------|--------------|-----------|--------|
| `rlm_guide.md` | none (last modified 2026-03-16) | README.md, CLAUDE.md, rlm_advanced.md | 16.2 | current |
| `rlm_advanced.md` | none (last modified 2026-03-16) | README.md, rlm_guide.md | 15.2 | current |
| `optloop_guide.md` | 2026-07-06 | README.md, CLAUDE.md | 13.7 | rewritten lib-only (app-level optloop retired 2026-07-06) |
| `optloop_advanced.md` | -- | -- | -- | DELETED 2026-07-06 (content merged into optloop_guide.md; .pth/activation-gap sections obsolete with app-level retirement) |
| `optimization_log.md` | none (last modified 2026-03-16) | CLAUDE.md, optloop_guide.md (×6), optloop_advanced.md (×2) | 2.3 | current |
| `observability_guide.md` | 2026-04-18 | README.md (×2), CLAUDE.md (×2) | 15.3 | stale cross-links |
| `mlx_optimization_plan.md` | 2026-02-23 | CHANGELOG.md entry only (no nav links) | 18.4 | orphan + partially stale |
| `frontend_api_reference.md` | none (last modified 2026-03-13) | CLAUDE.md | 36.1 | stale (missing API families) |
| `openapi.json` | none (last modified 2026-02-27) | frontend_api_reference.md (runtime URL only) | 133.8 | stale (v1.0.1, wrong endpoint set) |

---

## Proposed Actions

- [ ] **1. Fix observability_guide.md: replace local-only plan links** (`observability_guide.md`, lines 303, 417, 430, 434, 475)

  Two inline references and one Related-section entry point to `plans/lets-look-at-our-prancy-gosling.md`. That file is gitignored and does not exist on disk. Any reader of the committed file hits a dead link. The references are for S1.2b (preset taxonomy methodology) and S2.4 (idle unload daemon). Replace with descriptive prose: "the S1.2b methodology is documented locally; if the plan file isn't present, the jq recipes above are self-sufficient." Remove the Related-section entry or replace with the actual internal doc if one exists.

- [x] ~~**2. Fix optloop_guide.md and optloop_advanced.md: harden the .pth path**~~ MOOT 2026-07-06: the .pth activation mechanism belonged to the retired app-level optloop; the sections were removed in the lib-only guide rewrite.

- [ ] **3. Update mlx_optimization_plan.md Phase 4 "planned improvements" to reflect shipped state** (`mlx_optimization_plan.md`, lines 242--283)

  The Phase 4 section has two subsections titled "planned improvements -- speculative decoding" and "planned improvements -- kv quantization." Both DraftTuner acceptance tracking and the current KV quantization implementation are live (`generation_core.py` and `prompt_cache.py`). Retitle these to "implemented" and update the status marker in the phase overview table (line 21) from silent to `DONE (v1.15.0)` -- it currently shows no status. Also add a link to this file from CLAUDE.md or README.md so it is discoverable (currently referenced only in a CHANGELOG entry, not in any navigation surface).

- [ ] **4. Update frontend_api_reference.md: add missing API families** (`frontend_api_reference.md`, section 2 endpoint table and "Endpoint Quick Reference" at lines 1456--1480)

  The following API families added since the doc was last updated are entirely absent:
  - `/v1/conversations` (CRUD + message append/edit/truncate) -- added v1.27.0
  - `/v1/notebooks` -- added around same period
  - `/v1/rlm/completions` -- documented in `rlm_guide.md`, not in handoff
  - `POST /v1/data/clear` -- added 2026-03-16 per CLAUDE.md
  - `/v1/performance` and `/v1/performance/profile/{time_range}` -- exist in source
  - `/v1/admin/models/*` family (load, unload, status, toggle, bulk-profile, scan, validate) -- exist in source

  The handoff is the API reference for frontend developers. Missing these families means a frontend dev using this doc as the source of truth will not know these endpoints exist. Add at minimum the endpoint rows to section 2 and the quick reference; full implementation examples are optional but preferred for the conversation and RLM APIs since they are the most frontend-facing.

- [ ] **5. Regenerate openapi.json** (`openapi.json`)

  The file was generated at v1.0.1 (Feb 2026) and is now at v1.27.0. It contains endpoints that no longer exist (`/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/stt/models` -- the MLXSTTProvider was removed in Phase 2, 2026-03-13). It is missing every endpoint added since v1.14.0. Regenerate from a running server: `curl http://localhost:8080/openapi.json > docs/openapi.json`. The file is used by frontend_api_reference.md as a runtime URL reference, not imported directly, so it is safe to overwrite. Note: the `generated-api.ts` file in the frontend may also need regeneration -- check before committing the new schema.

- [ ] **6. Add "Last updated" date stamps to undated files** (all files except `observability_guide.md` and `mlx_optimization_plan.md`)

  Six files have no date stamp at all: `rlm_guide.md`, `rlm_advanced.md`, `optloop_guide.md`, `optloop_advanced.md`, `optimization_log.md`, `frontend_api_reference.md`. Add `Last updated: YYYY-MM-DD` as the second line (after the `#` title) using the last-modified date as a starting point. This is a low-effort signal for readers and agents about document freshness.

---

## Not Recommended

- **Merging rlm_guide.md + rlm_advanced.md**: Separation is clean. Guide covers reference (request fields, response, streaming, sandbox) and advanced covers patterns (pipeline, fan-out, retry). Cross-links are present and correct. Both are well-maintained and at similar lengths. No merge needed.
- **Merging optloop_guide.md + optloop_advanced.md**: ~~Keep split.~~ SUPERSEDED 2026-07-06: the app-level retirement removed most of the advanced guide's content (activation gap, monkey patching), so the remainder was merged into a single lib-only guide after all.
- **Removing mlx_optimization_plan.md**: It is a useful development history document showing why the current architecture is shaped the way it is, with the design rationale for radix caching, the speculative decoding pattern, and the pre-filled cache VLM approach. It should be preserved but fixed (action 3) and linked (action 3).
- **Removing optimization_log.md**: It is thin now (49 lines) but is the designated cross-session accumulator for the optloop workflow. Its thinness reflects that only one session has run, not document neglect.
