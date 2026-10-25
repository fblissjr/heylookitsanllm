# Distributed Multi-Mac Inference (GGUF RPC & Asymmetric MLX Pipeline)

**heylookitsanllm** supports splitting large models across multiple Apple Silicon Macs connected over Thunderbolt Bridge using two engine-specific mechanisms:

1. **GGUF (`llama.cpp` `ggml-rpc`)**: Offloads a proportion of decoder layers from the primary `llama-server` process to one or more remote `ggml-rpc-server` workers over TCP (`--rpc` and `--tensor-split`).
2. **MLX (`mx.distributed` Asymmetric Pipeline)**: Partitions any `mlx-vlm` decoder's `layers` list asymmetrically across ranks (`pipeline_ratios`), filtering `.safetensors` shards before `sanitize_weights` so worker ranks materialize only their assigned layer slice and execute in lockstep with Rank 0.

---

## 1. Network Setup (Thunderbolt Bridge)

Connect a Thunderbolt 4 or Thunderbolt 5 cable between the primary Mac (Rank 0 / `llama-server` host) and the secondary Mac (Rank 1 / `ggml-rpc-server` worker).

### Single-Cable Static Subnet (`bridge0`)

Assign static `/24` IPv4 addresses on the `Thunderbolt Bridge` interface (`bridge0`) so discovery does not depend on mDNS or link-local addressing:

```bash
# On the primary Mac (e.g., Mac Studio):
sudo networksetup -setmanual "Thunderbolt Bridge" <primary-ip> 255.255.255.0

# On the secondary Mac (e.g., MacBook Pro):
sudo networksetup -setmanual "Thunderbolt Bridge" <worker-ip> 255.255.255.0
```

Verify connectivity and local binaries from the primary Mac using [`scripts/distributed_probe.py`](../../scripts/distributed_probe.py):

```bash
uv run python scripts/distributed_probe.py preflight --peer <worker-ip>:50052
```

### Multi-Cable Note (Why RSTP Blocks a Second Cable in `bridge0`)

By default, macOS places all Thunderbolt Ethernet ports (`en2`, `en3`, …) into a single virtual switch (`bridge0`) with Rapid Spanning Tree Protocol (RSTP) enabled. Connecting a second Thunderbolt cable between the same two Macs creates a Layer-2 loop, so macOS places the second link into `blocking` state.

To use two physical Thunderbolt cables simultaneously:
1. Remove the two Thunderbolt ports from `Thunderbolt Bridge` in macOS Network Settings on both machines.
2. Assign two separate point-to-point `/30` subnets (one per cable).
3. For `mlx.launch` (`ring` backend), list both IPs in the `"ips"` array of `hosts.json`: `mlx.core.distributed` opens parallel TCP sockets per IP and stripes ring sends across both cables.

---

## 2. Distributed GGUF via `ggml-rpc` (`rpc` & `tensor_split`)

### How It Works

- **Build**: [`scripts/build_llama.py`](../../scripts/build_llama.py) configures `llama.cpp` with `-DGGML_RPC=ON` and builds both `llama-server` and `ggml-rpc-server` into `~/.heylook/llama.cpp/build/bin/`.
- **Worker Daemon**: [`scripts/rpc_worker.sh`](../../scripts/rpc_worker.sh) launches `ggml-rpc-server` on the secondary Mac with `-c` (local SSD tensor caching keyed by FNV-1a hash) enabled by default, so subsequent model reloads read cached tensors from the worker's local SSD instead of re-streaming weights over the wire.
- **Device Ordering (`src/llama.cpp`)**: `llama.cpp` enumerates `ggml-rpc` endpoints **before** the local Metal GPU. Consequently, in a 2-device setup (`rpc` + local Metal), the **first** number in `tensor_split` is the remote RPC worker's share and the **last** number is the local Metal GPU's share (e.g., `tensor_split = "20,80"` puts 20% of offloaded layers on the remote RPC worker and 80% on local Metal).
- **Pre-Flight & Memory Fit**:
  - [`LlamaServerProvider._load_once`](../../src/heylook_llm/providers/llama_server_provider.py) verifies TCP reachability to every `host:port` in `rpc` before spawning `llama-server`, raising a descriptive `RuntimeError` if a worker is unreachable.
  - [`ram_fit.local_weight_fraction`](../../src/heylook_llm/ram_fit.py) scales the primary model weights by `weights[-1] / sum(weights)` when `rpc` and `tensor_split` are set, while keeping `mmproj_path` and `draft_model_path` sidecars at 100% local weight.

### Step-by-Step Guide (GGUF RPC)

1. **On the secondary Mac**, build (once) and start the RPC worker:
   ```bash
   uv run python scripts/build_llama.py
   ./scripts/rpc_worker.sh --host <worker-ip> --port 50052
   ```
2. **On the primary Mac**, probe reachability and local memory fit before editing config:
   ```bash
   uv run python scripts/distributed_probe.py gguf-rpc \
     --peer <worker-ip>:50052 \
     --model <model-id-or-gguf-path> \
     --tensor-split 20,80
   ```
3. **Configure the model's `model.heylook.toml`** (inside the model's folder; plain top-level keys, no `[table]`):
   ```toml
   rpc = "<worker-ip>:50052"
   tensor_split = "20,80"
   ```
   *(Note: `rpc` is `file_only=True` and `ui="hidden"` because it directs `llama-server` to open outbound TCP connections; `tensor_split` is `ui="advanced"` and can be adjusted live in the web UI's **Models $\to$ Configure** panel with real-time **Memory fit** feedback.)*

---

## 3. Asymmetric MLX Pipeline Parallelism (`pipeline_ratios`)

### How It Works

Upstream `mlx-lm` and `mlx-vlm` assume homogeneous tensor parallelism (`all_sum` on every attention and MLP block) or equal 50/50 pipeline splits. On an asymmetric pair (such as a high-memory primary Mac and a smaller secondary Mac), tensor parallelism throttles the faster GPU to the slower GPU and fires two collectives per layer.

Instead, [`src/heylook_llm/providers/common/mlx_pipeline.py`](../../src/heylook_llm/providers/common/mlx_pipeline.py) implements **model-agnostic asymmetric pipeline parallelism**:

- **Fast Metal Synchronization**: [`enable_fast_metal_synch`](../../src/heylook_llm/providers/common/mlx_pipeline.py) sets `MLX_METAL_FAST_SYNCH=1` before Metal device initialization so cross-stream synchronization uses 4-byte shared-memory atomic fences (`Fence` in `mlx/backend/metal/fence.cpp`) rather than `MTLSharedEvent` command-buffer splits.
- **Pre-Sanitize Selective Shard Loading**: [`selective_pipeline_load`](../../src/heylook_llm/providers/common/mlx_pipeline.py) wraps `mx.load` inside [`selective_safetensors_load_scope`](../../src/heylook_llm/providers/common/mlx_pipeline.py) during `mlx_vlm.utils.load(..., lazy=True, strict=False)`. Shards and tensor keys belonging to off-rank decoder layers (`layers.{i}`) or Rank-0-only modules (`vision_tower`, `embed_tokens`, `lm_head`, `norm`) are dropped *before* `sanitize_weights` runs, preventing worker ranks from materializing or evaluating Rank 0's weights.
- **Zero-Parameter Off-Rank Layers & Batch/APC Stub Caches**:
  - [`apply_asymmetric_pipeline`](../../src/heylook_llm/providers/common/mlx_pipeline.py) replaces off-rank entries in `language_model.model.layers` with [`_PassThroughLayer`](../../src/heylook_llm/providers/common/mlx_pipeline.py) (preserving `self_attn.rotary_emb` stubs for models that inspect rotary embeddings on layer 0), replaces worker `embed_tokens`/`norm`/`lm_head` with [`_IdentityEmbed`](../../src/heylook_llm/providers/common/mlx_pipeline.py), and repoints architecture probe indices (`fa_idx`, `ssm_idx`, `swa_idx`) into the rank's owned layer slice.
  - Off-rank cache slots use [`_OffsetStubCache`](../../src/heylook_llm/providers/common/mlx_pipeline.py), which implements `mlx-vlm`'s batching (`to_batch`, `extract`, `filter`, `extend`, `merge`) and Automatic Prefix Cache (`prefix_cache_snapshot`, `prefix_cache_restore`, `nbytes = 0`) contracts with zero GPU memory allocation.
- **DAG-Safe Point-to-Point Hand-Offs (`mx.depends`)**:
  - Neither `mx.distributed.send` nor `mx.distributed.recv_like` adds an edge to its reference input in MLX's C++ computation graph (`mlx/distributed/ops.cpp`). [`attach_send_dependency`](../../src/heylook_llm/providers/common/mlx_pipeline.py) and [`_FinalStageRecvWrapper`](../../src/heylook_llm/providers/common/mlx_pipeline.py) bind the communication tokens into the returned hidden state via `mx.depends` so lazy evaluation schedules sends and receives in strict pipeline order without deadlocking.
- **Lockstep Rank 0 $\to$ Rank 1 Coordinator**:
  - [`install_rank0_step_coordinator`](../../src/heylook_llm/providers/common/mlx_pipeline.py) broadcasts a compact `int32` control header (carrying batch size, sequence length, cache offset, and optional M-RoPE `position_ids` / `rope_deltas`) from Rank 0 before each `language_model` forward pass so [`run_pipeline_worker_loop`](../../src/heylook_llm/providers/common/mlx_pipeline.py) on Rank 1 executes its layer slice (`layers[K..L-1]`) in lockstep without running tokenization, the vision tower, or sampling.

### Step-by-Step Guide (MLX Pipeline)

1. **Run a 2-rank local parity check first (on a single Mac)**:
   ```bash
   uv run python scripts/distributed_probe.py mlx-local \
     --model Qwen3.5-0.8B-MLX-8bit \
     --ratios 0.8,0.2
   ```
2. **Run across two Macs over Thunderbolt Bridge (SSH-free `--peers` mode)**:
   - On the secondary Mac (`<worker-ip>`, Rank 1):
     ```bash
     uv run python scripts/mlx_pipeline_worker.py \
       --rank 1 --peers <primary-ip>,<worker-ip> \
       --model-path <mlx-model-path> \
       --ratios 0.8,0.2
     ```
   - On the primary Mac (`<primary-ip>`, Rank 0):
     ```bash
     uv run python scripts/mlx_pipeline_worker.py \
       --rank 0 --peers <primary-ip>,<worker-ip> \
       --model-path <mlx-model-path> \
       --ratios 0.8,0.2 \
       --smoke-prompt "The capital of France is"
     ```
3. **Alternative: Launch both ranks from Rank 0 via `mlx.launch` (`hosts.json`, requires passwordless SSH)**:
   ```json
   [
     {"ssh": "localhost", "ips": ["<primary-ip>"]},
     {"ssh": "<worker-ip>", "ips": ["<worker-ip>"]}
   ]
   ```
   ```bash
   uv run python scripts/distributed_probe.py mlx-cluster \
     --hostfile hosts.json \
     --model <mlx-model-id-or-path> \
     --ratios 0.8,0.2
   ```
4. **Configure `pipeline_ratios` and `pipeline_peers` in the model's `model.heylook.toml`** (inside the model's folder; plain top-level keys, no `[table]`):
   ```toml
   pipeline_ratios = [0.8, 0.2]
   pipeline_peers = ["<primary-ip>", "<worker-ip>"]
   ```
   With `pipeline_peers` set, `heylookllm` on the primary Mac initializes its TCP ring directly on model load (no `mlx.launch` or SSH required) and connects to the running `scripts/mlx_pipeline_worker.py --rank 1 --peers <primary-ip>,<worker-ip>` process on the worker Mac. When `heylookllm` unloads the model, it automatically sends `CMD_SHUTDOWN` to the worker rank.

---

## 4. Runtime Gated DeltaNet 8-Row Packed Metal Kernel

For Qwen3.5 and Qwen3.8 hybrid-recurrent models on MLX, [`src/heylook_llm/providers/common/mlx_vlm_patches.py`](../../src/heylook_llm/providers/common/mlx_vlm_patches.py) (`PATCH_GDN_PACKED`) carries the 8-row packed Gated DeltaNet Metal kernel (`_make_gated_delta_packed_kernel`) into the active `mlx-vlm` runtime whenever scalar-gated `Dk = 128, Dv % 8 == 0` recurrence is invoked, guarded by a retirement check in [`tests/contract/test_mlxvlm_surface.py`](../../tests/contract/test_mlxvlm_surface.py).
