"""Non-causal image decode guard (plan W8).

Some projectors' image tokens are decoded non-causally, and llama.cpp aborts
when such an image has more tokens than the micro-batch. The provider caps
them with --image-max-tokens from a small hand-copied table
(`LlamaServerProvider.NON_CAUSAL_IMAGE_PROJECTORS`). The first test pins that
table against the source of the build the provider actually spawns.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

from heylook_llm.config import GGUFModelConfig
from heylook_llm.providers.llama_server_provider import LlamaServerProvider

P = LlamaServerProvider


def _build_tree() -> Path:
    """The llama.cpp checkout behind the canonical binary, located through
    its build manifest. Skips, saying why, when that source cannot be trusted
    to be the binary's."""
    manifest_path = P.DEFAULT_BUILD.parents[1] / "heylook-build.json"
    if not manifest_path.is_file():
        pytest.skip(f"no llama.cpp build manifest at {manifest_path}; "
                    f"run scripts/build_llama.py")
    sha = json.loads(manifest_path.read_text()).get("sha")
    if not sha:
        pytest.skip("build manifest records no sha; rebuild with scripts/build_llama.py")
    tree = manifest_path.parents[1]
    head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    if head != sha:
        pytest.skip(f"llama.cpp tree is at {head[:10] or '?'}, the build at "
                    f"{sha[:10]}; the source is not the binary's")
    return tree


def _non_causal_names(tree: Path) -> set[str]:
    src = (tree / "tools/mtmd/mtmd.cpp").read_text()
    m = re.search(r"bool mtmd_decode_use_non_causal\(.*?\n}\n", src, re.S)
    assert m, "mtmd_decode_use_non_causal not found in mtmd.cpp"
    enums, pending = set(), []
    for line in m.group(0).splitlines():
        case = re.search(r"case (PROJECTOR_TYPE_\w+):", line)
        if case:
            pending.append(case.group(1))
        elif "return" in line and pending:
            # Anything but a flat `return false` counts as non-causal, so a
            # conditional case (gemma4v's text-width test) is listed whole.
            if not re.search(r"return\s+false\s*;", line):
                enums.update(pending)
            pending = []
    names = dict(re.findall(r'\{\s*(PROJECTOR_TYPE_\w+),\s*"([^"]+)"\s*\}',
                            (tree / "tools/mtmd/clip-impl.h").read_text()))
    return {names[e] for e in enums}


def _default_max(clip_src: str, enum: str) -> int | None:
    """The per-image token maximum a projector's case block sets, or None."""
    found = None
    for m in re.finditer(rf"case {enum}:", clip_src):
        block = clip_src[m.end():clip_src.find("break;", m.end())]
        hit = (re.search(r"set_limit_image_tokens\(\d+,\s*(\d+)\)", block)
               or re.search(r"dsv4_max_n_token\s*=\s*(\d+);", block))
        if hit:
            found = int(hit.group(1))
    return found


@pytest.mark.unit
def test_non_causal_table_matches_the_build():
    tree = _build_tree()
    assert _non_causal_names(tree) == set(P.NON_CAUSAL_IMAGE_PROJECTORS), (
        "mtmd_decode_use_non_causal changed; update NON_CAUSAL_IMAGE_PROJECTORS")

    clip_src = (tree / "tools/mtmd/clip.cpp").read_text()
    enums = {v: k for k, v in re.findall(
        r'\{\s*(PROJECTOR_TYPE_\w+),\s*"([^"]+)"\s*\}',
        (tree / "tools/mtmd/clip-impl.h").read_text())}
    for name, (default_max, cappable) in P.NON_CAUSAL_IMAGE_PROJECTORS.items():
        found = _default_max(clip_src, enums[name])
        if cappable:
            assert found == default_max, f"{name}: clip.cpp sets {found}, table says {default_max}"
        else:
            assert found is None, f"{name}: clip.cpp now sets a token limit ({found})"

    common = (tree / "common/common.h").read_text()
    assert re.search(rf"\bn_ubatch\s*=\s*{P.LLAMA_DEFAULT_N_UBATCH};", common)
    assert re.search(rf"\bn_batch\s*=\s*{GGUFModelConfig.LLAMA_DEFAULT_N_BATCH};", common)
    # The cache profile (engine.cache) reports these as what a spawn gets.
    for field, value in (("cache_ram_mib", P.LLAMA_DEFAULT_CACHE_RAM_MIB),
                         ("n_ctx_checkpoints", P.LLAMA_DEFAULT_CTX_CHECKPOINTS),
                         ("checkpoint_min_step", P.LLAMA_DEFAULT_CHECKPOINT_MIN_STEP)):
        assert re.search(rf"\b{field}\s*=\s*{value};", common), field


@pytest.mark.unit
@pytest.mark.parametrize("proj,config,auto,expected", [
    ("gemma4v", {}, None, 512),                        # default -ub 512 < 1120: cap
    ("gemma4v", {}, 2048, None),                       # auto 2048 fits: no flag
    ("gemma4v", {"n_ubatch": 1024}, 2048, 1024),       # stored value wins over auto
    ("gemma4v", {"n_batch": 1024}, 2048, 1024),        # -ub clamped to -b
    ("gemma4v", {"ctx_size": 768}, 2048, 768),         # -b clamped to the context first
    ("deepseek4v", {}, None, None),                    # 384 fits 512
    ("deepseek4v", {"n_ubatch": 256}, None, 256),
    ("gemma3", {"n_ubatch": 128}, None, None),         # fixed count: flag cannot help
    ("qwen3vl_merger", {}, None, None),                # causal: never capped
    ("gemma4v", {"extra_args": ["--image-max-tokens=300"]}, None, None),
])
def test_image_token_cap(monkeypatch, proj, config, auto, expected):
    from heylook_llm import gguf_metadata
    monkeypatch.setattr(gguf_metadata, "vision_projector_type", lambda _p: proj)
    provider = P.__new__(P)
    provider.model_id = "m"
    provider.config = {"model_path": "/x.gguf", "mmproj_path": "/mm.gguf", **config}
    cap = provider._image_token_cap(auto)
    assert cap == expected
    argv = provider._build_args(Path("/bin/llama-server"), 1, auto_ubatch=auto,
                                image_max_tokens=cap)
    assert ("--image-max-tokens" in argv) == (expected is not None)
