---
name: models-toml-validate
enabled: true
event: file
conditions:
  - field: file_path
    operator: regex_match
    pattern: models\.toml$
---

You just edited models.toml. Validate it NOW (the config classes are
`extra="forbid"`, so a typo'd key otherwise fails at SERVER START, not
edit time):

```bash
uv run python -c "import tomllib; from heylook_llm.config import AppConfig; cfg = AppConfig(**tomllib.load(open('models.toml','rb'))); print(f'OK: {len(cfg.models)} entries')"
```

If validation fails, fix the entry before moving on. Common traps:
mlx-only keys on a gguf entry (and vice versa -- GGUFModelConfig and
MLXModelConfig have disjoint knob sets), `preset`/`profile` instead of
`default_sampler`, and absolute-path typos in model_path/mmproj_path/
draft_model_path.
