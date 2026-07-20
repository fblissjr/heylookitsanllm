# Pure, network-free judges shared by tasks.py. Each takes plain values (never
# a live HTTP response) and returns a Verdict -- unit-testable in isolation,
# and reusable across multiple tasks that happen to want the same property
# check (e.g. marker_leak is used by every thinking task).
#
# Generalized from two ad-hoc debugging scripts (full_matrix.py,
# repro_multiimage.py) that hardcoded model-specific strings ("The image",
# "Image 1"). These check PROPERTIES instead (colors mentioned, no leak
# markers, terminated cleanly, one word produced) so they don't rot the next
# time a model's phrasing changes.
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class Verdict:
    passed: bool
    evidence: str


def combine_verdicts(*verdicts: Verdict, sep: str = "; ") -> Verdict:
    """AND together several sub-checks into one Verdict, keeping each piece
    of evidence -- tasks that need more than one property (e.g. "colors
    mentioned" AND "no leak") compose this instead of writing one-off logic.
    """
    return Verdict(
        passed=all(v.passed for v in verdicts),
        evidence=sep.join(v.evidence for v in verdicts),
    )


def color_mention(text: str, expected_colors: list[str]) -> Verdict:
    lower = (text or "").lower()
    hits = [c for c in expected_colors if c.lower() in lower]
    passed = len(hits) == len(expected_colors)
    return Verdict(
        passed=passed,
        evidence=f"colors {len(hits)}/{len(expected_colors)}: {', '.join(expected_colors)}",
    )


_LEAK_MARKERS = ("<think>", "</think>", "<|channel>")


def marker_leak(content: str) -> Verdict:
    """passed=True means NO leak found (mirrors the original scripts' `leak`
    boolean, inverted so passed=True is always "good" like every other judge
    here)."""
    text = content or ""
    marker_hit = next((m for m in _LEAK_MARKERS if m in text), None)
    starts_with_thought = text.strip().lower().startswith("thought")
    leaked = marker_hit is not None or starts_with_thought
    if not leaked:
        return Verdict(passed=True, evidence="no leak markers")
    reason = f"marker {marker_hit!r}" if marker_hit else "starts with 'thought'"
    return Verdict(passed=False, evidence=f"LEAK: {reason}")


def repetition(content: str, min_len: int = 12, max_repeats: int = 2) -> Verdict:
    """Generalizes the seed scripts' `content.count("The image") > 3` style
    check: split on sentence punctuation, count repeats of any chunk long
    enough to be a real sentence (short chunks repeat legitimately, e.g.
    "Yes." or "4."), fail if one repeats too often -- a proxy for the model
    getting stuck in a loop rather than terminating cleanly.
    """
    chunks = [c.strip() for c in re.split(r"[.!?]+", content or "")]
    chunks = [c for c in chunks if len(c) >= min_len]
    counts = Counter(chunks)
    worst = counts.most_common(1)
    if not worst or worst[0][1] <= max_repeats:
        return Verdict(passed=True, evidence=f"no chunk repeats > {max_repeats}x")
    chunk, n = worst[0]
    return Verdict(passed=False, evidence=f"chunk repeated {n}x: {chunk[:60]!r}")


def token_budget_exhausted(completion_tokens: int, max_tokens: int) -> Verdict:
    """passed=True means generation stopped BEFORE the cap (clean stop);
    passed=False means it hit max_tokens exactly, a proxy for "never found a
    stopping point" / runaway generation that got truncated rather than
    terminated.
    """
    exhausted = completion_tokens == max_tokens
    evidence = f"completion_tokens={completion_tokens} max_tokens={max_tokens}"
    if exhausted:
        evidence += " (exhausted budget, likely truncated)"
    return Verdict(passed=not exhausted, evidence=evidence)


def exact_word_count(content: str, expected_words: int = 1) -> Verdict:
    tokens = (content or "").strip().split()
    passed = len(tokens) == expected_words
    return Verdict(passed=passed, evidence=f"word count {len(tokens)} (want {expected_words}): {tokens!r}")


def non_empty_non_gibberish(content: str) -> Verdict:
    """Generalizes repro_multiimage.py's
    `content.count(":") > 15 or len(set(content.strip())) < 8` heuristic --
    a sanity floor for "did the model produce prose at all", not a
    content-accuracy check (used for the random-noise heatmap task, where
    there is no ground truth to check against).
    """
    text = content or ""
    if not text.strip():
        return Verdict(passed=False, evidence="empty content")
    colon_heavy = text.count(":") > 15
    low_variety = len(set(text.strip())) < 8
    if colon_heavy or low_variety:
        reason = "colon-heavy" if colon_heavy else "low character variety"
        return Verdict(passed=False, evidence=f"gibberish ({reason}): {text[:80]!r}")
    return Verdict(passed=True, evidence=f"looks like prose: {text[:60]!r}")


def substring_present(content: str, needle: str) -> Verdict:
    """The one judge allowed to check an exact-ish string: it's checking
    factual-QA ground truth (e.g. "paris"), not a model's phrasing."""
    passed = needle.lower() in (content or "").lower()
    return Verdict(passed=passed, evidence=f"{'found' if passed else 'missing'} {needle!r}")
