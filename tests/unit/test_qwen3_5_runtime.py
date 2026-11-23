from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import pytest

from heylook_llm.providers.common import gemma4_runtime, qwen3_5_runtime


def _clone_module_weights(src: nn.Module, dst: nn.Module) -> None:
    dst.load_weights(tree_flatten(src.parameters()), strict=True)
    mx.eval(src.parameters(), dst.parameters())


def _tiny_qwen3_5_configs(*, num_layers: int = 4, hidden_size: int = 512, vocab_size: int = 512):
    from mlx_vlm.models.qwen3_5.config import ModelConfig, TextConfig, VisionConfig

    text_cfg = TextConfig(
        model_type="qwen3_5_text",
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
        num_hidden_layers=num_layers,
        num_attention_heads=max(4, hidden_size // 128),
        rms_norm_eps=1e-6,
        vocab_size=vocab_size,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        head_dim=128,
        full_attention_interval=num_layers,
        tie_word_embeddings=False,
    )
    vision_cfg = VisionConfig(
        model_type="qwen3_5",
        depth=2,
        hidden_size=128,
        intermediate_size=256,
        out_hidden_size=hidden_size,
        num_heads=4,
        patch_size=16,
        spatial_patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    model_cfg = ModelConfig(
        text_config=text_cfg,
        vision_config=vision_cfg,
        model_type="qwen3_5",
        vocab_size=vocab_size,
    )
    return text_cfg, model_cfg


@pytest.mark.unit
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal device required")
def test_packed_gdn_with_states_bit_exact_and_fallback():
    from mlx_vlm.models.qwen3_5 import gated_delta as gd

    assert qwen3_5_runtime.install_qwen3_5_packed_gdn_with_states() is True
    wrapped = gd.gated_delta_update_with_states
    assert getattr(wrapped, "_heylook_patch", None) == qwen3_5_runtime.PATCH_GDN_PACKED_STATES
    upstream = getattr(wrapped, "_heylook_upstream", None)
    assert callable(upstream)

    mx.random.seed(0)
    B, Hk, Hv, Dk, Dv = 1, 4, 8, 128, 128
    A_log = mx.random.normal((Hv,), dtype=mx.float32) * 0.1
    dt_bias = mx.random.normal((Hv,), dtype=mx.bfloat16) * 0.1

    for T in (2, 3, 4, 8):
        q = mx.random.normal((B, T, Hk, Dk), dtype=mx.bfloat16) * 0.1
        k = mx.random.normal((B, T, Hk, Dk), dtype=mx.bfloat16) * 0.1
        v = mx.random.normal((B, T, Hv, Dv), dtype=mx.bfloat16) * 0.1
        a = mx.random.normal((B, T, Hv), dtype=mx.bfloat16) * 0.1
        b = mx.random.normal((B, T, Hv), dtype=mx.bfloat16) * 0.1
        state0 = mx.random.normal((B, Hv, Dv, Dk), dtype=mx.float32) * 0.1
        mx.eval(q, k, v, a, b, A_log, dt_bias, state0)

        for state_steps in (1, T - 1, T):
            for init_state in (None, state0):
                y_fast, sout_fast, states_fast = wrapped(
                    q, k, v, a, b, A_log, dt_bias, state=init_state, state_steps=state_steps
                )
                y_ref, sout_ref, states_ref = upstream(
                    q, k, v, a, b, A_log, dt_bias, state=init_state, state_steps=state_steps
                )
                mx.eval(y_fast, sout_fast, states_fast, y_ref, sout_ref, states_ref)
                assert mx.array_equal(y_fast, y_ref), f"y mismatch at T={T}, state_steps={state_steps}"
                assert mx.array_equal(sout_fast, sout_ref), f"state_out mismatch at T={T}, state_steps={state_steps}"
                assert mx.array_equal(states_fast, states_ref), f"states mismatch at T={T}, state_steps={state_steps}"

    # Fallback path 1: non-None mask -> delegates cleanly to upstream.
    q4 = mx.random.normal((B, 4, Hk, Dk), dtype=mx.bfloat16) * 0.1
    k4 = mx.random.normal((B, 4, Hk, Dk), dtype=mx.bfloat16) * 0.1
    v4 = mx.random.normal((B, 4, Hv, Dv), dtype=mx.bfloat16) * 0.1
    a4 = mx.random.normal((B, 4, Hv), dtype=mx.bfloat16) * 0.1
    b4 = mx.random.normal((B, 4, Hv), dtype=mx.bfloat16) * 0.1
    mask = mx.array([[True, True, False, True]])
    y_m, sout_m, st_m = wrapped(q4, k4, v4, a4, b4, A_log, dt_bias, mask=mask, state_steps=4)
    y_mu, sout_mu, st_mu = upstream(q4, k4, v4, a4, b4, A_log, dt_bias, mask=mask, state_steps=4)
    mx.eval(y_m, sout_m, st_m, y_mu, sout_mu, st_mu)
    assert mx.array_equal(y_m, y_mu)
    assert mx.array_equal(sout_m, sout_mu)
    assert mx.array_equal(st_m, st_mu)

    # Fallback path 2: Dk != 128 -> delegates cleanly to upstream.
    Dk_alt = 64
    q_alt = mx.random.normal((B, 3, Hk, Dk_alt), dtype=mx.bfloat16) * 0.1
    k_alt = mx.random.normal((B, 3, Hk, Dk_alt), dtype=mx.bfloat16) * 0.1
    v_alt = mx.random.normal((B, 3, Hv, Dv), dtype=mx.bfloat16) * 0.1
    a_alt = mx.random.normal((B, 3, Hv), dtype=mx.bfloat16) * 0.1
    b_alt = mx.random.normal((B, 3, Hv), dtype=mx.bfloat16) * 0.1
    y_d, sout_d, st_d = wrapped(q_alt, k_alt, v_alt, a_alt, b_alt, A_log, dt_bias, state_steps=3)
    y_du, sout_du, st_du = upstream(q_alt, k_alt, v_alt, a_alt, b_alt, A_log, dt_bias, state_steps=3)
    mx.eval(y_d, sout_d, st_d, y_du, sout_du, st_du)
    assert mx.array_equal(y_d, y_du)
    assert mx.array_equal(sout_d, sout_du)
    assert mx.array_equal(st_d, st_du)


@pytest.mark.unit
@pytest.mark.skipif(not mx.metal.is_available(), reason="Metal device required")
def test_bind_qwen3_5_verify_parity_both_hoisted_and_native(monkeypatch):
    from mlx_vlm.models.qwen3_5.language import LanguageModel

    mx.random.seed(7)
    text_cfg, model_cfg = _tiny_qwen3_5_configs()

    lm_up = LanguageModel(text_cfg, model_cfg)
    lm_up.set_dtype(mx.bfloat16)
    nn.quantize(lm_up, group_size=64, bits=8)

    lm_bd = LanguageModel(text_cfg, model_cfg)
    lm_bd.set_dtype(mx.bfloat16)
    nn.quantize(lm_bd, group_size=64, bits=8)
    _clone_module_weights(lm_up, lm_bd)

    assert qwen3_5_runtime.bind_qwen3_5_runtime(SimpleNamespace(language_model=lm_bd)) is True
    assert getattr(lm_bd, "_heylook_qwen3_5_runtime", False) is True
    assert getattr(lm_bd, "_heylook_qwen3_5_suffix_embed", False) is True
    assert (
        getattr(lm_bd.speculative_verify_hidden.__func__, "_heylook_patch", None)
        == qwen3_5_runtime._PATCH_VERIFY
    )

    prefill_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=mx.int32)

    def _assert_argmax_parity(arg_bd, arg_up, logits_up, logits_bd):
        assert arg_bd.dtype == mx.int32
        assert mx.array_equal(arg_bd, mx.argmax(logits_bd, axis=-1).astype(mx.int32))
        for t in range(int(arg_bd.shape[1])):
            bd_idx = int(arg_bd[0, t])
            up_idx = int(arg_up[0, t])
            if bd_idx != up_idx:
                margin = abs(
                    float(logits_up[0, t, up_idx]) - float(logits_up[0, t, bd_idx])
                )
                assert margin <= 0.05, f"t={t} argmax mismatch: {bd_idx} vs {up_idx} (margin={margin})"

    def _fresh_prefilled_caches():
        c_up = lm_up.make_cache()
        c_bd = lm_bd.make_cache()
        o_up = lm_up(prefill_ids, cache=c_up)
        o_bd = lm_bd(prefill_ids, cache=c_bd)
        mx.eval(o_up.logits, o_bd.logits)
        return c_up, c_bd

    # Case 1: hoisted 8-bit QMV branch (native_wins=False) at T=3 + commit(2) + post-commit decode.
    monkeypatch.setattr(qwen3_5_runtime, "native_8bit_qmv_beats_hoisted_for_t", lambda *_a, **_kw: False)
    cache_up, cache_bd = _fresh_prefilled_caches()
    draft3 = mx.array([[11, 12, 13]], dtype=mx.int32)
    h_up, _, tx_up = lm_up.speculative_verify_hidden(draft3, cache_up)
    h_bd, _, tx_bd = lm_bd.speculative_verify_hidden(draft3, cache_bd)
    logits_up = lm_up.speculative_logits_from_hidden(h_up)
    logits_bd = lm_bd.speculative_logits_from_hidden(h_bd)
    logits_bd_same_h = lm_bd.speculative_logits_from_hidden(h_up)
    arg_up = lm_up.speculative_argmax_from_hidden(h_up)
    arg_bd = lm_bd.speculative_argmax_from_hidden(h_bd)
    mx.eval(h_up, h_bd, logits_up, logits_bd, logits_bd_same_h, arg_up, arg_bd)

    assert mx.allclose(h_bd.astype(mx.float32), h_up.astype(mx.float32), atol=6e-2, rtol=5e-2)
    assert mx.allclose(logits_bd_same_h.astype(mx.float32), logits_up.astype(mx.float32), atol=3e-2, rtol=3e-2)
    assert mx.allclose(logits_bd.astype(mx.float32), logits_up.astype(mx.float32), atol=8e-2, rtol=5e-2)
    _assert_argmax_parity(arg_bd, arg_up, logits_up, logits_bd)

    tx_up.commit([2])
    tx_bd.commit([2])
    next_tok = mx.array([[21]], dtype=mx.int32)
    dec_up = lm_up(next_tok, cache=cache_up)
    dec_bd = lm_bd(next_tok, cache=cache_bd)
    mx.eval(dec_up.logits, dec_bd.logits)
    assert mx.allclose(dec_bd.logits.astype(mx.float32), dec_up.logits.astype(mx.float32), atol=8e-2, rtol=5e-2)

    # Case 2: native qmv_mma branch (native_wins=True) at T=4 + abort rollback + re-verify parity.
    monkeypatch.setattr(qwen3_5_runtime, "native_8bit_qmv_beats_hoisted_for_t", lambda *_a, **_kw: True)
    cache_up2, cache_bd2 = _fresh_prefilled_caches()
    draft4 = mx.array([[14, 15, 16, 17]], dtype=mx.int32)
    h2_up, _, tx2_up = lm_up.speculative_verify_hidden(draft4, cache_up2)
    h2_bd, _, tx2_bd = lm_bd.speculative_verify_hidden(draft4, cache_bd2)
    l2_up = lm_up.speculative_logits_from_hidden(h2_up)
    l2_bd = lm_bd.speculative_logits_from_hidden(h2_bd)
    l2_bd_same_h = lm_bd.speculative_logits_from_hidden(h2_up)
    a2_up = lm_up.speculative_argmax_from_hidden(h2_up)
    a2_bd = lm_bd.speculative_argmax_from_hidden(h2_bd)
    mx.eval(h2_up, h2_bd, l2_up, l2_bd, l2_bd_same_h, a2_up, a2_bd)

    assert mx.allclose(h2_bd.astype(mx.float32), h2_up.astype(mx.float32), atol=6e-2, rtol=5e-2)
    assert mx.allclose(l2_bd_same_h.astype(mx.float32), l2_up.astype(mx.float32), atol=3e-2, rtol=3e-2)
    assert mx.allclose(l2_bd.astype(mx.float32), l2_up.astype(mx.float32), atol=8e-2, rtol=5e-2)
    _assert_argmax_parity(a2_bd, a2_up, l2_up, l2_bd)

    tx2_up.abort()
    tx2_bd.abort()
    assert int(cache_bd2[text_cfg.full_attention_interval - 1].offset) == 10

    re_up, _, tx_re_up = lm_up.speculative_verify_hidden(draft4, cache_up2)
    re_bd, _, tx_re_bd = lm_bd.speculative_verify_hidden(draft4, cache_bd2)
    mx.eval(re_up, re_bd)
    assert mx.allclose(re_bd.astype(mx.float32), re_up.astype(mx.float32), atol=6e-2, rtol=5e-2)
    tx_re_up.abort()
    tx_re_bd.abort()

    # Fallback paths: T=1 and B=2 delegate cleanly to upstream without touching tx.
    single_tok = mx.array([[22]], dtype=mx.int32)
    hs1_up, _, txs1_up = lm_up.speculative_verify_hidden(single_tok, cache_up2)
    hs1_bd, _, txs1_bd = lm_bd.speculative_verify_hidden(single_tok, cache_bd2)
    mx.eval(hs1_up, hs1_bd)
    assert mx.allclose(hs1_bd.astype(mx.float32), hs1_up.astype(mx.float32), atol=6e-2, rtol=5e-2)
    txs1_up.abort()
    txs1_bd.abort()

    c_b2_up = lm_up.make_cache()
    c_b2_bd = lm_bd.make_cache()
    b2_ids = mx.array([[1, 2, 3], [4, 5, 6]], dtype=mx.int32)
    hb2_up, _, txb2_up = lm_up.speculative_verify_hidden(b2_ids, c_b2_up)
    hb2_bd, _, txb2_bd = lm_bd.speculative_verify_hidden(b2_ids, c_b2_bd)
    mx.eval(hb2_up, hb2_bd)
    assert mx.allclose(hb2_bd.astype(mx.float32), hb2_up.astype(mx.float32), atol=6e-2, rtol=5e-2)
    txb2_up.abort()
    txb2_bd.abort()


@pytest.mark.unit
def test_bind_qwen3_5_vision_parity_single_and_multi_image(monkeypatch):
    from mlx_vlm.models.qwen3_5.config import VisionConfig as Qwen35VisionConfig
    from mlx_vlm.models.qwen3_5.vision import VisionModel as Qwen35VisionModel
    from mlx_vlm.models.qwen3_vl.config import VisionConfig as Qwen3VLVisionConfig
    from mlx_vlm.models.qwen3_vl.vision import VisionModel as Qwen3VLVisionModel

    mx.random.seed(11)
    vcfg = Qwen35VisionConfig(
        model_type="qwen3_5",
        depth=2,
        hidden_size=128,
        intermediate_size=256,
        out_hidden_size=128,
        num_heads=4,
        patch_size=16,
        spatial_patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[],
    )
    vt_up = Qwen35VisionModel(vcfg)
    vt_bd = Qwen35VisionModel(vcfg)
    _clone_module_weights(vt_up, vt_bd)

    assert qwen3_5_runtime.bind_qwen3_5_vision(SimpleNamespace(vision_tower=vt_bd)) is True
    assert getattr(vt_bd, "_heylook_qwen3_5_vision", False) is True
    # Idempotent re-bind.
    assert qwen3_5_runtime.bind_qwen3_5_vision(SimpleNamespace(vision_tower=vt_bd)) is True

    patch_dim = vcfg.in_channels * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size

    # Single-image parity + verify rotary cos/sin hoisted outside all blocks (1 cos call vs depth=2).
    cos_calls = 0
    orig_cos = mx.cos

    def _spy_cos(arr):
        nonlocal cos_calls
        cos_calls += 1
        return orig_cos(arr)

    monkeypatch.setattr(mx, "cos", _spy_cos)

    px1 = mx.random.normal((16, patch_dim), dtype=mx.float32)
    grid1 = mx.array([[1, 4, 4]], dtype=mx.int32)
    cos_calls = 0
    out1_up, ds1_up = vt_up(px1, grid1)
    up_cos_calls = cos_calls
    cos_calls = 0
    out1_bd, ds1_bd = vt_bd(px1, grid1)
    bd_cos_calls = cos_calls
    mx.eval(out1_up, out1_bd)

    assert mx.allclose(out1_bd, out1_up, atol=1e-4, rtol=1e-4)
    assert ds1_bd == ds1_up == []
    assert bd_cos_calls == 1
    assert up_cos_calls >= 2

    # Multi-image parity.
    px2 = mx.random.normal((24, patch_dim), dtype=mx.float32)
    grid2 = mx.array([[1, 4, 4], [1, 2, 4]], dtype=mx.int32)
    out2_up, ds2_up = vt_up(px2, grid2)
    out2_bd, ds2_bd = vt_bd(px2, grid2)
    mx.eval(out2_up, out2_bd)
    assert mx.allclose(out2_bd, out2_up, atol=1e-4, rtol=1e-4)
    assert ds2_bd == ds2_up == []

    # qwen3_vl deepstack branch parity.
    vl_cfg = Qwen3VLVisionConfig(
        model_type="qwen3_vl",
        depth=2,
        hidden_size=128,
        intermediate_size=256,
        out_hidden_size=128,
        num_heads=4,
        patch_size=16,
        spatial_patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0],
    )
    vl_up = Qwen3VLVisionModel(vl_cfg)
    vl_bd = Qwen3VLVisionModel(vl_cfg)
    _clone_module_weights(vl_up, vl_bd)
    assert qwen3_5_runtime.bind_qwen3_5_vision(SimpleNamespace(vision_tower=vl_bd)) is True
    vout_up, vds_up = vl_up(px2, grid2)
    vout_bd, vds_bd = vl_bd(px2, grid2)
    mx.eval(vout_up, vout_bd, *vds_up, *vds_bd)
    assert mx.allclose(vout_bd, vout_up, atol=1e-4, rtol=1e-4)
    assert len(vds_bd) == len(vds_up) == 1
    assert mx.allclose(vds_bd[0], vds_up[0], atol=1e-4, rtol=1e-4)


@pytest.mark.unit
def test_native_8bit_qmv_for_t_env_overrides(monkeypatch):
    lin = nn.Linear(512, 512, bias=False)
    nn.quantize(lin, group_size=64, bits=8)
    assert gemma4_runtime.native_8bit_qmv_beats_hoisted_for_t(lin, 1) is False

    gemma4_runtime._NATIVE_8BIT_QMV_FOR_T_CACHE.clear()
    monkeypatch.setenv("HEYLOOK_8BIT_QMV", "0")
    assert gemma4_runtime.native_8bit_qmv_beats_hoisted_for_t(lin, 4) is True
    gemma4_runtime._NATIVE_8BIT_QMV_FOR_T_CACHE.clear()
    monkeypatch.setenv("HEYLOOK_8BIT_QMV", "1")
    assert gemma4_runtime.native_8bit_qmv_beats_hoisted_for_t(lin, 4) is False
    monkeypatch.delenv("HEYLOOK_8BIT_QMV")
    gemma4_runtime._NATIVE_8BIT_QMV_FOR_T_CACHE.clear()
