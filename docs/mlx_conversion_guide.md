# Converting models to MLX (mlx_lm / mlx_vlm)

Last updated: 2026-07-20

The flags are self-documenting (`mlx_lm.convert --help`, `python -m mlx_vlm
convert --help`); this guide carries only the decisions the flags can't make
for you. Verified against mlx-vlm 0.6.6 / the pinned mlx-lm.

## Which tool

- VLM checkpoint (has `vision_config` etc.): `python -m mlx_vlm convert`.
  Using mlx_lm.convert on a VLM drops the vision tower.
- Text-only: `mlx_lm.convert`. Same quant flags, plus arbitrary
  `--quant-predicate` python paths.

## Quantization recipes, ranked (quality/perf bang-for-buck)

1. **4-bit affine, group 32** -- `-q --q-bits 4 --q-group-size 32`.
   The default recommendation, and the ONLY sane choice for QAT-q4_0
   checkpoints: QAT trains weights against q4_0's 32-element blocks, so
   affine g32 is the closest MLX lattice to what the weights were shaped
   for. ~0.5 bpw larger than g64 for strictly better fidelity.
2. **4-bit affine, group 64** (the `-q` default) -- ~10% smaller/faster
   weights (decode is bandwidth-bound), slightly coarser scales. Right
   when disk/bandwidth matters more than the last quality fraction.
3. **`--quant-predicate mixed_4_6`** -- Q4_K_M-style: 6-bit for
   v_proj/down_proj in the outer layer-eighths + embed/lm_head, 4-bit
   elsewhere (group 64 fixed). The hedge for NON-QAT checkpoints where
   plain 4-bit visibly degrades. For QAT checkpoints it deviates from the
   trained lattice on the boosted layers -- use 1 or 2 instead.

Anti-recommendations, with reasons:
- Do NOT give QAT checkpoints extra bits "for safety" (6/8-bit): the
  weights already sit on a 4-bit-friendly lattice; more bits buy almost
  no quality and cost decode bandwidth -- the entire point of QAT.
- `--q-mode mxfp4` is a different number format than what q4_0-QAT
  trained for; keep `affine` (the default) for QAT conversions.

## What you do NOT need to worry about (verified in source)

- **Vision/audio towers are never quantized** by plain `-q`: mlx-vlm's
  base predicate skips `vision_tower`/`multi_modal_projector`/audio
  modules, so they stay at the save dtype (bf16). No flag needed; this is
  why OCR/detail doesn't degrade with a quantized language side.
- Model classes can define their own `quant_predicate` (e.g. the gemma-4
  assistant protects its centroid table); convert honors it automatically.
- Groups must divide the row size; layers with `size % 64 != 0` are
  skipped automatically.

## After converting

- Import: `heylookllm import --folder <mlx-path>` (or the v3 models page
  scan). Template/eos come from the converted dir's own files -- for
  gemma-4, confirm the canonical `chat_template.jinja` came along.
- Verify behavior, not vibes: register the model and run the eval bank
  against old + new side by side --
  `uv run python tests/eval/run.py --server http://localhost:8080 --models <new>,<old>`.

Worked example (gemma-4 QAT downloaded to a local checkout dir):

```bash
python -m mlx_vlm convert \
  --hf-path <download-dir>/gemma-4-26B-A4B-it-qat-q4_0-unquantized \
  --mlx-path modelzoo/google/gemma-4-26B-A4B-it-qat-4bit-g32-mlx \
  -q --q-bits 4 --q-group-size 32
```
