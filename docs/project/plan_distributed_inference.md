# Plan: Distributed Inference Across M2 Ultra (192 GB) + M4 Pro (48 GB)

Last updated: 2026-10-25
Status: IMPLEMENTED (Phases 1 and 2 + October 2026 ecosystem optimizations)

## 1. Hardware & Interconnect Constraints

| Machine | Chip | Unified RAM | Memory Bandwidth | Thunderbolt Ports | Usable GPU Wired Budget (~90%) |
|---|---|---|---|---|---|
| **Primary Host** (Mac Studio) | Apple M2 Ultra | **192 GB** | ~800 GB/s | **Thunderbolt 4** (40 Gb/s) | ~176 GB |
| **Secondary Peer** (MacBook Pro) | Apple M4 Pro | **48 GB** | ~273 GB/s | **Thunderbolt 5** (80/120 Gb/s) | ~38–40 GB |
| **Combined Pool** | Asymmetric (4:1 RAM, 3:1 BW) | **240 GB** | — | **Negotiates at TB4 (40 Gb/s)** | **~215 GB** |

### Why Asymmetric Pipeline Parallelism (~80/20 Layer Split) Beats Tensor Parallelism (`shard`)

1. **Thunderbolt 4 negotiation**: Both MLX's RDMA backend (`jaccl.cpp`) and `llama.cpp`'s `GGML_RPC_RDMA_APPLE` (`coderef/llama.cpp/ggml/src/ggml-rpc/transport.cpp`) require Thunderbolt 5 on **both** peers (`rdma_ctl enable`). Because the M2 Ultra is Thunderbolt 4, a cable between the two Macs negotiates at **Thunderbolt 4 (40 Gb/s TCP over Thunderbolt Bridge)** and uses MLX's TCP ring backend (`ring.cpp`) or `ggml-rpc` TCP sockets.
2. **Multi-cable behavior over Thunderbolt Bridge**:
   - macOS's `Thunderbolt Bridge` (`bridge0`) runs Rapid Spanning Tree Protocol (RSTP) to prevent L2 broadcast loops and puts a second parallel cable between the same two hosts into `blocking` state.
   - To stripe TCP across two physical Thunderbolt cables on MLX (`ring.cpp` opens `N` parallel TCP sockets per peer when a hostfile entry lists `N` IPs), remove the second port from `bridge0` and assign each cable its own `/30` subnet. Single-token decode (8–14 KB per hop) is latency-bound, whereas weight staging (SMB3 Multichannel) and long-prompt prefill benefit from dual-cable striping.
3. **Tensor Parallelism (`model.shard()`) loses on both memory and latency**:
   - **Even-split memory cap**: `shard()` in `mlx-lm` and `mlx-vlm` (`muse_spark`, `qwen3_5`, `deepseek_v3`) splits weights `1 / N` evenly, capping each node at the M4 Pro's ~38 GB usable budget (~76 GB total across 2 nodes—less than half of what the M2 Ultra holds alone).
   - **Per-layer network lockstep**: 2 collective `all_sum`s per transformer layer (96–128 network round-trips per generated token) stall the higher-bandwidth M2 Ultra behind TCP round-trip latency and the lower-bandwidth M4 Pro on every layer.
4. **Asymmetric Pipeline Parallelism (`~80/20` layer split) pools all 240 GB with minimal wire traffic**:
   - **Rank 0 (M2 Ultra, 192 GB)** holds ~80% of decoder layers, the Vision Tower, `embed_tokens`, `norm`, `lm_head`, and APC cache.
   - **Rank 1 (M4 Pro, 48 GB)** holds ~20% of decoder layers (~30–36 GB of weights + 20% of the KV cache).
   - **Wire traffic**: Only **one hidden-state tensor `(B, L, hidden_size)`** crosses the cable **twice per forward pass** (`Rank 0 -> Rank 1` at layer `K`, `Rank 1 -> Rank 0` after layer `L-1`). On decode (`B=1, L=1, hidden=4096..7168, bf16`), that is **8–14 KB twice per token**, and each GPU runs its contiguous slice of layers at its native memory bandwidth without per-layer lockstep.

### Target Models That Benefit

| Model | Size | Current Bottleneck on 192 GB M2 Ultra Alone | What 192 GB + 48 GB (80/20 Split) Unlocks |
|---|---|---|---|
| `Qwen/Qwen3-VL-235B-A22B-Instruct-4bit-mlx` | 155 GB | Hits Metal GPU watchdog around ~53–56k ctx (`docs/project/CURRENT.md`) | Moves ~20 layers (~33 GB weights + 21% KV cache) to M4 Pro; frees **~35+ GB Metal working set** on M2 Ultra for 128k+ ctx + APC |
| `deepseek/unsloth_DeepSeek-V4-Flash-0731-GGUF` | 250 GB | Cannot fit in 192 GB RAM | Fits across combined 240 GB pool via `llama.cpp` `--rpc` + `--tensor-split` |
| `deepseek/unsloth_DeepSeek-V4-Flash-Vision-Exp-GGUF-UD-Q8_K_XL` | 152 GB | Tight at 384k ctx + speculative drafter (`docs/project/CURRENT.md`) | Offloads ~32 GB of target layers to M4 Pro; keeps drafter + full 384k context resident on M2 Ultra |
| `deepseek/unsloth_DeepSeek-V4-Flash-Vision-Exp-GGUF-UD-Q4_K_XL` | 145 GB | Drops drafter when free RAM is short (`src/heylook_llm/ram_fit.py`) | Retains drafter with ~30 GB extra local headroom on M2 Ultra |

---

## 2. October 2026 Ecosystem & Local-Source Audit

| Area | Mechanism Verified in Source | How `heylookitsanllm` Uses It |
|---|---|---|
| **MLX C++ Fast Stream Sync (`MLX_METAL_FAST_SYNCH=1`)** | In `mlx/backend/metal/fence.cpp`, all `mx.distributed` ops execute on a CPU communication stream while layers run on the GPU stream. By default (`MLX_METAL_FAST_SYNCH=0`), crossing streams uses `MTLSharedEvent`, forcing `CommandEncoder::end_encoding()` and splitting the Metal command buffer. Setting `MLX_METAL_FAST_SYNCH=1` (used by Meta's `muse-spark` cookbook) switches `Fence` to a 4-byte shared-memory atomic (`uint32_t`) + inline Metal kernels (`input_coherent`, `fence_update`, `fence_wait`) with CPU spin-wait, avoiding command-buffer splits on `send`/`recv`/`all_gather`. | Set `MLX_METAL_FAST_SYNCH=1` automatically when launching distributed MLX workers (`scripts/mlx_distributed_worker.py` and `mlx_pipeline.py`). |
| **MLX Metal Residency Refresh (`MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS`)** | In `mlx/backend/metal/resident.cpp`, macOS can drop wired residency after GPU idle despite a standing request. Setting `MLX_METAL_RESIDENCY_REFRESH_INTERVAL_MS=5000` runs a background condition-variable loop that periodically re-requests residency on active `MTLResidencySet`s. | Document and support for large resident models on Mac Studio. |
| **MLX Pipeline Graph Dependency (`mx.depends` + `all_gather`)** | In `mlx_lm/models/muse_spark.py` and `qwen3_5.py`, attaching `mx.depends` between the pipeline `send(h)` result and the last active layer's cache state prevents MLX's lazy graph evaluator from dead-code-eliminating `send(h)` during chunked prefill (`mx.eval([c.state for c in cache])`). Following the final layer with `all_gather(h)` broadcasts the final hidden state so both ranks can sample in SPMD lockstep without per-token control headers. | Adopted in `src/heylook_llm/providers/common/mlx_pipeline.py`. |
| **Lazy `mx.load` + Pre-Sanitize Selective Safetensors Loading** | Because `mlx-vlm`'s `sanitize_weights` slices and concatenates tensors eagerly inside `load_model()` (before returning the model), pruning layers *after* `vlm_load()` would still materialize off-rank shards during `sanitize_weights`. Wrapping `vlm_load()` in `selective_safetensors_load_scope` filters `mx.load` keys by `should_keep_weight_key` and skips off-rank shard files from `model.safetensors.index.json` *before* `sanitize_weights` runs, so Rank 1 (48 GB M4 Pro) never materializes Rank 0's weights. | Adopted in `src/heylook_llm/providers/common/mlx_pipeline.py` (`selective_pipeline_load`). |
| **Packed Gated DeltaNet Metal Kernel (`_make_gated_delta_packed_kernel`)** | Upstream `mlx_lm/models/gated_delta.py` packs 8 value rows (`rows_per_simdgroup = 8`, 4 lanes per row, 32 state elements per lane) into one 32-lane SIMD group for `Dk=128, Dv % 8 == 0` scalar-gated recurrence (used by Qwen3.5 / Qwen3.8), cutting `q`/`k` global loads and grid dispatches 8× while remaining bitwise-identical via an explicit ascending butterfly reduction. `mlx-vlm` still used the 1-row-per-SIMD-group kernel. | Ported into `forks/mlx-vlm/mlx_vlm/models/gated_delta.py` and carried at runtime via `PATCH_GDN_PACKED` in `src/heylook_llm/providers/common/mlx_vlm_patches.py` (guarded by `tests/contract/test_mlxvlm_surface.py`). |
| **`llama.cpp` `ggml-rpc` Subgraph & Weight Caching** | `coderef/llama.cpp/ggml/src/ggml-rpc/ggml-rpc.cpp` (`b11869`) implements `RPC_CMD_GRAPH_RECOMPUTE` (`GRAPH_CACHE_MAX = 1024`, sending a 12-byte UID instead of re-serializing the compute graph on every token), `RPC_CMD_SET_TENSOR_HASH` (`-c` 64-bit FNV-1a SSD tensor cache for tensors $>10\text{ MB}$), and `busy_spin_acquire` (`RPC_BUSY_SPIN_IDLE_TIME`) to avoid condition-variable wakeup latency during decode. | Enabled via `-DGGML_RPC=ON` in `scripts/build_llama.py` and `--cache` in `scripts/rpc_worker.sh`. |

---

## 2. Shared Prerequisite: Thunderbolt Point-to-Point Subnet

Connect a Thunderbolt 4 or 5 cable directly between the Mac Studio (M2 Ultra) and MacBook Pro (M4 Pro). Configure a deterministic point-to-point subnet on `Thunderbolt Bridge` (`bridge0`) so traffic never routes over Wi-Fi/LAN:

```sh
# On Mac Studio (M2 Ultra - Rank 0 / Primary):
sudo networksetup -setmanual "Thunderbolt Bridge" <primary-ip> 255.255.255.0

# On MacBook Pro (M4 Pro - Rank 1 / Worker):
sudo networksetup -setmanual "Thunderbolt Bridge" <worker-ip> 255.255.255.0
```

---

## 3. Phase 1 Specification: `llama.cpp` Distributed Mode (`ggml-rpc`)

### File-by-File Changes in `heylookitsanllm`

1. **`scripts/build_llama.py`**:
   - Add `"-DGGML_RPC=ON"` to `CMAKE_ARGS`.
   - Add `"ggml-rpc-server"` to `TARGETS` (upstream renamed the CMake target from `rpc-server` to `ggml-rpc-server` in `tools/rpc/CMakeLists.txt`).
2. **`src/heylook_llm/config.py`**:
   - Add `rpc: Optional[str]` (`--rpc`) and `tensor_split: Optional[str]` (`--tensor-split`) to `GGUFModelConfig`.
   - Add `pipeline_ratios: Optional[List[float]]` to `MLXModelConfig`.
   - Mark `rpc`, `tensor_split`, and `pipeline_ratios` as `EFFECT_REQUIRES_RELOAD` (`rpc` and `pipeline_ratios` are also `file_only=True`).
3. **`src/heylook_llm/providers/llama_server_provider.py` & `src/heylook_llm/providers/gguf_describe.py`**:
   - Append `["--rpc", c.rpc]` and `["--tensor-split", c.tensor_split]` in `_build_args()` and `describe_gguf_spawn()`.
   - Device order note: `llama.cpp` registers RPC devices before the local Metal device in `ggml_backend_reg` (verify with `llama-server --list-devices`), so `--tensor-split 20,80` assigns 20% to the RPC peer and 80% to local Metal.
4. **`src/heylook_llm/ram_fit.py`**:
   - Add `local_weight_fraction(config: dict)` scaling primary target weight sizing for both GGUF (`rpc` + `tensor_split`) and MLX (`pipeline_ratios`) so `heylook` does not drop speculative drafters when 20% of the target model lives on the worker Mac.
5. **`scripts/rpc_worker.sh`**:
   - Starts `coderef/llama.cpp/build/bin/ggml-rpc-server -H <worker-ip> -p 50052 -c` so the M4 Pro caches its assigned GGUF tensors on local SSD across reloads.

---

## 4. Phase 2 Specification: Model-Agnostic Asymmetric Pipeline Parallelism for MLX / `mlx-vlm`

### Why Upstream `PipelineMixin` Needs a General Pattern for VLMs on 192 GB + 48 GB

- **Model coverage**: In `mlx-vlm`, `mlx_vlm/models/pipeline.py` only implements an even `num_layers // N` split and is only wired into `deepseek_v3` and `deepseek_v4`; models like `qwen3_vl`, `qwen3_5`, `gemma4`, and `mistral3` do not inherit from `PipelineMixin`.
- **Weight loading**: `sharded_load()` in `mlx_vlm/utils.py` opens all `.safetensors` shards on every rank before slicing layers, which exceeds the 48 GB M4 Pro's memory on a 155 GB model.
- **Vision placement**: Putting early layers `0..K-1` + Vision Tower + `embed_tokens` + `norm` + `lm_head` on Rank 0 (M2 Ultra) and late layers `K..L-1` on Rank 1 (M4 Pro) keeps vision encoding, `inputs_embeds`, and `deepstack_visual_embeds` entirely on Rank 0.

### Core Design (`src/heylook_llm/providers/common/mlx_pipeline.py`)

Instead of editing individual model classes in `mlx-vlm`, hook the inner decoder (`inner = model.language_model.model` or `model.model`) and its underlying `inner.layers` list in-place while preserving `len(inner.layers) == L`:

1. **Selective Safetensors Loader (`selective_load_mlx_model`)**:
   - Filter `model.safetensors.index.json` before opening `.safetensors` files:
     - **Rank 0 (M2 Ultra, 192 GB)** loads `vision_tower`, `embed_tokens`, `layers.0 .. layers.{K-1}`, `norm`, and `lm_head`.
     - **Rank 1 (M4 Pro, 48 GB)** loads **only** `layers.{K} .. layers.{L-1}` (replacing `vision_tower` with `None`, `embed_tokens` with `_ZeroEmbed(hidden_size)`, and `lm_head` with `None`).
2. **In-Place `inner.layers` Wrappers**:
   - Replace off-rank layers (`K..L-1` on Rank 0; `0..K-1` on Rank 1) with `_PassThroughLayer` (0 parameters, advances `_OffsetStubCache` by `seq_len`, returns `x`).
   - Wrap layer `K-1` on Rank 0 with `_Rank0BoundarySendWrapper` (`mx.distributed.send(h, dst=1)`) and layer `L-1` on Rank 0 with `_Rank0BoundaryRecvWrapper` (`mx.distributed.recv_like(x, src=1)`).
   - Wrap layer `K` on Rank 1 with `_Rank1BoundaryRecvWrapper` (`mx.distributed.recv_like(x, src=0)`) and layer `L-1` on Rank 1 with `_Rank1BoundarySendWrapper` (`mx.distributed.send(h, dst=0)`).
   - Because `len(inner.layers) == L` is preserved and `_OffsetStubCache` tracks `.offset` identically on both ranks, every model's native mask construction (`fa_mask`, `swa_mask`, `linear_mask`), hybrid cache indexing (`fa_idx`, `swa_idx`), M-RoPE `position_embeddings`, and `deepstack_visual_embeds` work without modifying a single model file.
3. **SPMD Step Header (`mlx_pipeline_worker.py`)**:
   - Rank 0 broadcasts a 4-int32 control header `(op_code, batch_size, seq_len, has_3d_pos)` via `mx.distributed.all_sum` before each prefill chunk or decode step so Rank 1 steps its decoder in lockstep.
4. **Verification Gate**:
   - Single-machine 2-rank parity check (`uv run mlx.launch -n 2 --backend ring`) on `Qwen3.5-0.8B-MLX-8bit` asserting exact token-for-token greedy parity (`scripts/chain_probe.py`) before running across Thunderbolt on `Qwen3-VL-235B-A22B-Instruct-4bit-mlx`.
