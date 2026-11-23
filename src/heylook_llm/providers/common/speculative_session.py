"""Owned single-row (B=1) speculative decoding session and stateful drafter prefiller.

Replaces ``mlx_vlm.speculative.mtp.MTPBatch`` inside ``b1_runner`` with a
direct single-row round driver that:
- clamps acceptance at ``stop_tokens`` BEFORE committing target/drafter caches
  (preventing post-EOS pollution of ``ArraysCache`` and ``KVCache``),
- keeps sampler state (pending draft distributions, penalty history) session-local
  instead of mutating attributes on the caller's ``sampler``,
- skips ``logsumexp`` reductions on greedy target verification when the target
  did not pre-sample ``target_tokens``, and
- feeds stateful MTP drafters (such as Qwen 3.8's ``Qwen3_5MTPDraftModel``)
  across chunked prefills, ``N-1``+``1`` recurrent splits, and warm cache hits.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx

from .mlx_vlm_patches import _apply_mtp_penalties


class _SessionSamplerProxy:
    """Session-local callable wrapper insulating the request sampler."""

    def __init__(self, sampler: Any, *, greedy: bool) -> None:
        self._base_sampler = sampler
        self._heylook_greedy = bool(greedy)
        self._heylook_spec_probs = (
            None if greedy else getattr(sampler, "_heylook_spec_probs", None)
        )
        self._heylook_pending_draft: Any = None
        self._heylook_captured_q: Optional[list[mx.array]] = None
        sample_target = getattr(sampler, "sample_target", None)
        if callable(sample_target):
            self.sample_target = sample_target

    def __call__(self, x: mx.array) -> mx.array:
        captured = self._heylook_captured_q
        if isinstance(captured, list) and callable(self._heylook_spec_probs):
            is_col3 = getattr(x, "ndim", 0) == 3 and x.shape[1] == 1
            flat = x[:, 0, :] if is_col3 else x
            q_p, q_scaled = self._heylook_spec_probs(flat, already_logprobs=False)
            tok = mx.random.categorical(q_scaled)
            captured.append(q_p)
            return tok[:, None] if is_col3 else tok
        if self._base_sampler is None:
            return mx.argmax(x, axis=-1)
        return self._base_sampler(x)

    def pop_pending_draft(self, draft_tokens: mx.array) -> Optional[mx.array]:
        pending = self._heylook_pending_draft
        self._heylook_pending_draft = None
        if (
            isinstance(pending, tuple)
            and len(pending) == 2
            and getattr(pending[0], "shape", None) == getattr(draft_tokens, "shape", None)
            and getattr(pending[1], "ndim", 0) == 2
            and int(pending[1].shape[0]) == int(draft_tokens.shape[1])
        ):
            return pending[1]
        return None


def is_stateful_drafter(draft: Any) -> bool:
    """True when ``draft`` maintains an autoregressive KV cache (e.g. Qwen 3.8 MTP)."""
    return bool(
        draft is not None
        and callable(getattr(draft, "prefill_from_target_hidden", None))
        and callable(getattr(draft, "_forward_tokens", None))
        and callable(getattr(draft, "_set_seed_from_hidden", None))
        and hasattr(draft, "_cache")
    )


def _apply_thinking_budget_single(
    accepted: int,
    round_tokens: list[int],
    criteria: Any,
) -> tuple[int, list[int]]:
    if criteria is None or not round_tokens:
        return accepted, round_tokens
    constrained: list[int] = []
    for index, token in enumerate(round_tokens):
        forced = criteria.pop_forced_token_id()
        chosen = int(token) if forced is None else int(forced)
        criteria(chosen)
        constrained.append(chosen)
        if chosen != int(token):
            accepted = min(accepted, index)
            break
    return accepted, constrained


class StatefulDrafterPrefiller:
    """Feeds hidden states from prefill chunks or warm restores into stateful drafters."""

    def __init__(
        self,
        model: Any,
        draft: Any,
        *,
        token_dtype: mx.Dtype,
        greedy: bool,
        proxy_sampler: _SessionSamplerProxy,
        cached_len: int,
        restored_drafter_layers: Optional[list[Any]] = None,
        restored_last_hidden: Optional[mx.array] = None,
    ) -> None:
        self.token_dtype = token_dtype
        self.greedy = bool(greedy)
        self.proxy_sampler = proxy_sampler
        self.draft = copy(draft)
        self.draft.reset(model)
        self.pending_prev_hidden: Optional[mx.array] = None
        self.enabled = False

        if cached_len <= 0:
            self.draft._next_position = 0
            self.enabled = True
        elif (
            restored_drafter_layers is not None
            and restored_last_hidden is not None
            and len(restored_drafter_layers) == len(self.draft._cache)
            and len(restored_drafter_layers) > 0
            and int(getattr(restored_drafter_layers[0], "offset", -1)) == cached_len - 1
        ):
            self.draft._cache = restored_drafter_layers
            self.draft._next_position = cached_len - 1
            self.pending_prev_hidden = restored_last_hidden
            self.enabled = True

    def feed_chunk(self, chunk_ids: mx.array, chunk_hidden: Optional[mx.array]) -> None:
        if not self.enabled or chunk_hidden is None:
            return
        length = int(chunk_ids.shape[1])
        if length <= 0:
            return
        if self.pending_prev_hidden is None:
            if length > 1:
                self.draft._forward_tokens(
                    chunk_ids[:, 1:],
                    chunk_hidden[:, : length - 1, :],
                    self.token_dtype,
                )
            self.pending_prev_hidden = chunk_hidden[:, -1:, :]
        else:
            paired_h = (
                self.pending_prev_hidden
                if length == 1
                else mx.concatenate(
                    [self.pending_prev_hidden, chunk_hidden[:, : length - 1, :]],
                    axis=1,
                )
            )
            self.draft._forward_tokens(chunk_ids, paired_h, self.token_dtype)
            self.pending_prev_hidden = chunk_hidden[:, -1:, :]

    def cache_snapshot_args(self) -> dict[str, Any]:
        """Return kwargs for ``PrefixCacheStore.store`` capturing pre-bonus drafter state."""
        if (
            not self.enabled
            or self.pending_prev_hidden is None
            or not getattr(self.draft, "_cache", None)
        ):
            return {}
        offsets = [getattr(c, "offset", None) for c in self.draft._cache]
        if any(not isinstance(o, int) or o <= 0 for o in offsets) or len(set(offsets)) != 1:
            return {}
        return {
            "drafter_cache": self.draft._cache,
            "drafter_target_len": int(offsets[0]),
            "drafter_last_hidden": self.pending_prev_hidden,
        }

    def finish_with_first_token(self, first_tok: int, rng: Any) -> None:
        if not self.enabled or self.pending_prev_hidden is None:
            return
        first_arr = mx.array([[int(first_tok)]], dtype=self.token_dtype)

        def _advance_and_seed() -> None:
            out = self.draft._forward_tokens(
                first_arr, self.pending_prev_hidden, self.token_dtype
            )
            if isinstance(out, tuple):
                logits_hidden, pre_hc_hidden = out[0], out[1]
            else:
                logits_hidden = pre_hc_hidden = out
            capture_q = not self.greedy and callable(
                self.proxy_sampler._heylook_spec_probs
            )
            if capture_q:
                buf: list[mx.array] = []
                self.proxy_sampler._heylook_captured_q = buf
                try:
                    self.draft._set_seed_from_hidden(
                        logits_hidden[:, -1:, :], self.proxy_sampler, self.greedy
                    )
                finally:
                    self.proxy_sampler._heylook_captured_q = None
                self.draft._heylook_seed_q = buf[-1] if len(buf) == 1 else None
            else:
                if hasattr(self.draft, "_heylook_seed_q"):
                    self.draft._heylook_seed_q = None
                self.draft._set_seed_from_hidden(
                    logits_hidden[:, -1:, :], self.proxy_sampler, self.greedy
                )
            if isinstance(out, tuple):
                self.draft._seed_hidden = pre_hc_hidden[:, -1:, ...]

        rng.draft_call(_advance_and_seed)


@dataclass
class SpeculativeStepResult:
    """Committed outcome of a single B=1 speculative verification round."""

    emitted_tokens: list[int]
    committed_tokens: list[int]
    finish_reason: Optional[str]
    accepted_count: int
    stopped_on_eos: bool = False


class SpeculativeSession:
    """Owned single-row speculative decoder for B=1 generation."""

    def __init__(
        self,
        model: Any,
        draft: Any,
        prompt_cache: list[Any],
        hidden: Optional[mx.array],
        first_token: int,
        block_size: int,
        sampler: Any,
        token_dtype: mx.Dtype,
        greedy: bool,
        *,
        shared_kv_states: Optional[dict] = None,
        prompt_tokens: Optional[mx.array] = None,
        rope_deltas: Optional[mx.array] = None,
        penalty_processors: Optional[list[Any]] = None,
        gen_history: Optional[list[int]] = None,
        prefiller: Optional[StatefulDrafterPrefiller] = None,
    ) -> None:
        from mlx_vlm.speculative.common import (
            _SpeculativeSamplerRNG,
            _dflash_block_total,
        )
        from mlx_vlm.speculative.mtp import (
            _mtp_cache_positions,
            _mtp_draft_hidden,
            _mtp_draft_kwargs,
        )

        self.model = getattr(model, "language_model", model)
        self.base_draft = draft
        self.cache = prompt_cache
        self.sampler = sampler
        self.dtype = token_dtype
        self.greedy = bool(greedy)
        self.penalty_processors = list(penalty_processors or [])
        self.gen_history = gen_history
        self.proxy_sampler = (
            prefiller.proxy_sampler
            if prefiller is not None
            else _SessionSamplerProxy(sampler, greedy=self.greedy)
        )
        self.positioned = callable(getattr(sampler, "sample_target", None))
        self.block_size = _dflash_block_total(draft, block_size)
        self.configured_block_size = int(
            getattr(getattr(draft, "config", None), "block_size", self.block_size)
            or self.block_size
        )
        self.rope_deltas = rope_deltas
        _, positions = _mtp_cache_positions(prompt_cache, 1)
        self.position = int(positions[0])
        self.bonus = int(first_token)

        if prefiller is not None:
            self.draft = prefiller.draft
            self.rng = _SpeculativeSamplerRNG(
                self.draft, enabled=not self.greedy and not self.positioned
            )
            if prefiller.enabled:
                prefiller.finish_with_first_token(self.bonus, self.rng)
            self.last_committed_hidden = prefiller.pending_prev_hidden
        else:
            self.draft = copy(draft)
            self.draft.reset(model)
            self.rng = _SpeculativeSamplerRNG(
                self.draft, enabled=not self.greedy and not self.positioned
            )
            prefill = getattr(self.draft, "prefill_from_target_hidden", None)
            if (
                callable(prefill)
                and hidden is not None
                and prompt_tokens is not None
                and int(hidden.shape[1]) >= self.position
                and int(prompt_tokens.shape[1]) >= self.position
            ):
                self.rng.draft_call(
                    prefill,
                    prompt_tokens[:, -self.position :],
                    hidden[:, -self.position :],
                    self.bonus,
                    self.proxy_sampler,
                    self.dtype,
                    **_mtp_draft_kwargs(self.draft, self.greedy, self.proxy_sampler),
                )
            self.last_committed_hidden = (
                hidden[:, -1:] if hidden is not None else None
            )

        self.hidden = (
            _mtp_draft_hidden(self.model, hidden[:, -1:])
            if hidden is not None
            else None
        )
        shared_kv_fn = getattr(self.model, "shared_kv_from_cache", None)
        init_shared = (
            shared_kv_fn(prompt_cache)
            if callable(shared_kv_fn)
            else (shared_kv_states or {})
        )
        self._bind_shared(init_shared or {})

    def _bind_shared(self, shared: dict, *, rejected: int = 0) -> None:
        from mlx_vlm.speculative.mtp import (
            _mtp_draft_position,
            _slice_shared_kv_after_reject,
        )
        from .gemma4_runtime import slice_gemma4_maskless_swa

        sliced = _slice_shared_kv_after_reject(shared, rejected) if shared else {}
        if sliced and "sliding_attention" in sliced:
            text_cfg = getattr(getattr(self.draft, "config", None), "text_config", None)
            window = getattr(text_cfg, "sliding_window", None)
            if window:
                sliced = slice_gemma4_maskless_swa(
                    sliced, position=self.position, window=int(window)
                )
        self.draft.set_shared_kv(
            sliced,
            kv_offset=self.position,
            position=_mtp_draft_position(self.position),
            kv_valid_len=self.position,
            left_padding=None,
        )

    def _apply_penalties(self, logits_2d: mx.array, draft_list: list[int]) -> mx.array:
        if not self.penalty_processors or self.gen_history is None:
            return logits_2d
        return _apply_mtp_penalties(
            mx, logits_2d, draft_list, (self.penalty_processors, self.gen_history)
        )

    def _walk(
        self,
        verify: Any,
        proposals: mx.array,
        q_probs: Optional[mx.array],
        budget: int,
        base_position: int,
    ) -> tuple[int, list[int]]:
        from mlx_vlm.generate.common import generation_stream
        from mlx_vlm.speculative.common import _speculative_walk
        from mlx_vlm.speculative.mtp import _mtp_logits_from_hidden

        if verify.target_tokens is not None:
            mx.async_eval(verify.target_tokens, verify.hidden)
            return _speculative_walk(proposals, verify.target_tokens, budget)

        n_draft = int(proposals.shape[1])
        total_t = n_draft + 1
        draft_list = [int(tok) for tok in proposals[0].tolist()]

        if self.greedy and not self.penalty_processors and not self.positioned:
            with mx.stream(generation_stream):
                logits = _mtp_logits_from_hidden(
                    self.model, verify.hidden[:, :total_t, :]
                )
                if logits.ndim == 3 and logits.shape[0] == 1:
                    logits = logits[0]
                target_tokens = mx.argmax(logits, axis=-1)
            mx.eval(target_tokens)
            return _speculative_walk(proposals, target_tokens[None, :], budget)

        spec_probs = self.proxy_sampler._heylook_spec_probs
        if (
            not self.positioned
            and callable(spec_probs)
            and q_probs is not None
        ):
            with mx.stream(generation_stream):
                logits = _mtp_logits_from_hidden(
                    self.model, verify.hidden[:, :total_t, :]
                )
                if logits.ndim == 3 and logits.shape[0] == 1:
                    logits = logits[0]
                logits = self._apply_penalties(logits, draft_list)
                logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
                p_probs, _ = spec_probs(logprobs, already_logprobs=True)
                if int(q_probs.shape[1]) == int(p_probs.shape[1]):
                    draft_1d = proposals[0].astype(mx.int32)
                    p_sel = mx.take_along_axis(
                        p_probs[:n_draft], draft_1d[:, None], axis=-1
                    ).squeeze(-1)
                    q_sel = mx.take_along_axis(
                        q_probs, draft_1d[:, None], axis=-1
                    ).squeeze(-1)
                    u = mx.random.uniform(shape=(n_draft,))
                    rej = (u * q_sel) >= p_sel
                    rej_padded = mx.concatenate([rej, mx.array([True])])
                    acc_idx = mx.argmax(rej_padded.astype(mx.int32))
                    q_padded = mx.concatenate(
                        [
                            q_probs,
                            mx.zeros((1, p_probs.shape[1]), dtype=p_probs.dtype),
                        ],
                        axis=0,
                    )
                    p_row = p_probs[acc_idx]
                    q_row = q_padded[acc_idx]
                    res = mx.maximum(p_row - q_row, 0.0)
                    res_sum = mx.sum(res)
                    chosen = mx.where(res_sum > 1e-12, res, p_row)
                    log_dist = mx.where(chosen > 0, mx.log(chosen), -mx.inf)
                    bonus_tok = mx.random.categorical(log_dist)
                    run_leviathan = True
                else:
                    run_leviathan = False
            if run_leviathan:
                mx.eval(acc_idx, bonus_tok)
                accepted = int(acc_idx.item())
                bonus = int(bonus_tok.item())
                new_tokens = draft_list[:accepted]
                if len(new_tokens) < budget:
                    new_tokens.append(bonus)
                return accepted, new_tokens[:budget]

            # Vocab width mismatch fallback reuses the already computed logprobs.
            mx.eval(logprobs)
            accepted = 0
            new_tokens = []
            for pos in range(total_t):
                with mx.stream(generation_stream):
                    target_tok = self.sampler(logprobs[pos : pos + 1])
                mx.eval(target_tok)
                token = int(target_tok.reshape(-1).item())
                if pos < n_draft and token == draft_list[pos]:
                    accepted += 1
                    if len(new_tokens) < budget:
                        new_tokens.append(token)
                    continue
                if len(new_tokens) < budget:
                    new_tokens.append(token)
                break
            return accepted, new_tokens

        with mx.stream(generation_stream):
            logits = _mtp_logits_from_hidden(
                self.model, verify.hidden[:, :total_t, :]
            )
            if logits.ndim == 3 and logits.shape[0] == 1:
                logits = logits[0]
            logits = self._apply_penalties(logits, draft_list)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            if self.positioned:
                positions = [int(base_position) + pos for pos in range(total_t)]
                target_tokens = self.sampler.sample_target(
                    logprobs,
                    row_ids=[0] * total_t,
                    positions=positions,
                )
            elif self.greedy:
                target_tokens = mx.argmax(logprobs, axis=-1)
            else:
                target_tokens = None

        if target_tokens is not None:
            mx.eval(target_tokens)
            return _speculative_walk(proposals, target_tokens[None, :], budget)

        mx.eval(logprobs)
        accepted = 0
        new_tokens: list[int] = []
        for pos in range(total_t):
            with mx.stream(generation_stream):
                target_tok = self.sampler(logprobs[pos : pos + 1])
            mx.eval(target_tok)
            token = int(target_tok.reshape(-1).item())
            if pos < n_draft and token == draft_list[pos]:
                accepted += 1
                if len(new_tokens) < budget:
                    new_tokens.append(token)
                continue
            if len(new_tokens) < budget:
                new_tokens.append(token)
            break
        return accepted, new_tokens

    def step(
        self,
        budget: int,
        position: int,
        *,
        stop_tokens: set[int],
        thinking_budget_criteria: Any = None,
    ) -> SpeculativeStepResult:
        from mlx_vlm.speculative.common import _record_speculative_round
        from mlx_vlm.speculative.mtp import (
            _mtp_draft_hidden,
            _mtp_draft_kwargs,
            _mtp_next_block_size,
            _mtp_verify_target,
            _release_shared_kv,
        )

        if budget <= 0:
            raise ValueError("SpeculativeSession.step requires positive token budget")

        block_size = _mtp_next_block_size(
            self.draft, self.block_size, self.configured_block_size, budget + 1
        )
        if self.rope_deltas is not None:
            self.model._rope_deltas = self.rope_deltas
            self.model._position_ids = None

        verify = None
        draft_kw = _mtp_draft_kwargs(self.draft, self.greedy, self.proxy_sampler)
        try:
            proposals = self.rng.draft_tokens(
                self.draft.draft_block,
                self.bonus,
                self.hidden,
                None,
                block_size,
                self.proxy_sampler,
                self.dtype,
                **draft_kw,
            )
            q_probs = self.proxy_sampler.pop_pending_draft(proposals)
            _release_shared_kv(self.draft)

            prev_bonus = int(self.bonus)
            verify_input = mx.concatenate(
                [mx.array([[prev_bonus]], dtype=self.dtype), proposals], axis=1
            )
            verify = _mtp_verify_target(
                self.model,
                verify_input,
                self.cache,
                self.sampler,
                sample_target_tokens=self.greedy and not self.penalty_processors,
            )
            self.rng.target_eval(verify.hidden, verify.target_tokens)

            accepted, round_tokens = self._walk(
                verify, proposals, q_probs, budget, position
            )
            if round_tokens:
                accepted = min(accepted, len(round_tokens) - 1)
            else:
                accepted = 0

            accepted, round_tokens = _apply_thinking_budget_single(
                accepted, round_tokens, thinking_budget_criteria
            )

            stop_idx: Optional[int] = None
            for idx, tok in enumerate(round_tokens):
                if int(tok) in stop_tokens:
                    stop_idx = idx
                    break

            if stop_idx is not None:
                emitted = [int(t) for t in round_tokens[:stop_idx]]
                commit_accepted = int(stop_idx)
                committed_tokens = [prev_bonus, *emitted]
                finish_reason: Optional[str] = "stop"
                draft_new_tokens: list[int] = []
                stopped_on_eos = True
            else:
                emitted = [int(t) for t in round_tokens]
                commit_accepted = int(accepted)
                committed_tokens = [prev_bonus, *emitted[:-1]] if emitted else [prev_bonus]
                finish_reason = "length" if len(emitted) >= budget else None
                draft_new_tokens = emitted
                stopped_on_eos = False

            self.rng.target_sampled(sync_draft=not self.positioned)
            accept_fn = getattr(self.draft, "accept_verified_tokens", None)
            if callable(accept_fn):
                self.rng.draft_call(
                    accept_fn,
                    verify.hidden,
                    proposals,
                    commit_accepted,
                    draft_new_tokens,
                    self.proxy_sampler,
                    self.dtype,
                    **draft_kw,
                )
            verify.commit(self.model, self.cache, [commit_accepted], block_size)
        except BaseException:
            if verify is not None:
                verify.abort()
            abort_draft = getattr(self.draft, "abort_draft_round", None)
            if callable(abort_draft):
                abort_draft()
            raise

        self.last_committed_hidden = verify.hidden[:, commit_accepted : commit_accepted + 1]
        self.hidden = _mtp_draft_hidden(self.model, self.last_committed_hidden)
        self.bonus = int(round_tokens[-1]) if round_tokens else prev_bonus
        self.position += commit_accepted + 1
        if finish_reason is None:
            if verify.committed_shared_kv is not None:
                self._bind_shared(verify.committed_shared_kv, rejected=0)
            else:
                self._bind_shared(
                    verify.shared_kv_states or {},
                    rejected=max(0, block_size - commit_accepted - 1),
                )
        else:
            _release_shared_kv(self.draft)

        _record_speculative_round(self.draft, commit_accepted, block_size - 1)
        _record_speculative_round(self.base_draft, commit_accepted, block_size - 1)
        return SpeculativeStepResult(
            emitted_tokens=emitted,
            committed_tokens=committed_tokens,
            finish_reason=finish_reason,
            accepted_count=commit_accepted,
            stopped_on_eos=stopped_on_eos,
        )

    def drafter_cache_for_store(self, target_token_len: int) -> dict[str, Any]:
        """Return ``PrefixCacheStore.store`` kwargs for stateful drafter persistence."""
        if (
            not is_stateful_drafter(self.draft)
            or not getattr(self.draft, "_cache", None)
            or self.last_committed_hidden is None
            or target_token_len < 2
        ):
            return {}
        want = int(target_token_len) - 1
        offsets = [getattr(c, "offset", None) for c in self.draft._cache]
        if any(not isinstance(o, int) or o < want for o in offsets) or len(set(offsets)) != 1:
            return {}
        return {
            "drafter_cache": self.draft._cache,
            "drafter_trim_tail": int(offsets[0]) - want,
            "drafter_target_len": want,
            "drafter_last_hidden": self.last_committed_hidden,
        }


__all__ = [
    "SpeculativeSession",
    "SpeculativeStepResult",
    "StatefulDrafterPrefiller",
    "is_stateful_drafter",
]
