---
name: models-toml-validate
enabled: true
event: file
conditions:
  - field: file_path
    operator: regex_match
    pattern: models\.toml$
---

You just edited models.toml. Validate it NOW. The config classes are
`extra="forbid"`, so a typo'd key otherwise fails at SERVER START, not at edit
time:

```bash
uv run python -c "import tomllib; from heylook_llm.config import AppConfig; cfg = AppConfig(**tomllib.load(open('models.toml','rb'))); print(f'OK: {len(cfg.models)} entries')"
```

Then check that the edit did not remove a capability without saying so. An
explicit `[[models]]` entry receives NONE of discovery's derived fields, so
adding one field to a model that had no entry drops everything discovery was
giving it (mmproj, drafter, modalities). Compare against what discovery
derives:

```bash
uv run python -c "import tomllib; from heylook_llm.model_registry import discover, merge_discovered as m; d=tomllib.load(open('models.toml','rb')); print([x['id'] for x in m(d, discover(d))['models']])"
```

Common traps:

- mlx-only keys on a gguf entry, and vice versa. GGUFModelConfig and
  MLXModelConfig have disjoint knob sets.
- Fields that no longer exist, which `forbid` rejects:
  - `default_sampler`, `preset` and `profile` went with the named-sampler
    system in v2.0.30;
  - `vision_tokens` was removed in v2.0.64.
- Absolute-path typos in model_path, mmproj_path or draft_model_path.
- A hand-written value that contradicts what the model's own files say, e.g.
  `supports_thinking = true` on a template that reads no thinking switch.

Direction of travel: per-model entries are moving out of this file into
sidecars in each model's own directory (docs/project/plan_registry_sidecars.md,
W0 of plan_runtime_visibility.md). Do not add an entry where no setting needs
one.
