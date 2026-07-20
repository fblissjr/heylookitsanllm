# The task bank as data. Generalizes two ad-hoc debugging scripts
# (full_matrix.py: model x think x image-count x vision-budget matrix;
# repro_multiimage.py: multi-image + thinking gibberish repro) into a
# reusable, filterable bank. Fixture images are generated at import time with
# PIL/numpy (no files on disk) since the originals' pre-existing PNGs no
# longer exist.
#
# Each EvalTask.build_request() returns a /v1/chat/completions body WITHOUT
# "model"/"stream" -- run.py injects those. judge() takes a small context
# dict (content, thinking, completion_tokens, max_tokens) and returns a
# judges.Verdict. Expected values (colors, needles) are closed over by the
# judge lambda rather than stored as separate EvalTask fields, so adding a
# task never requires touching the EvalTask shape itself.
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from judges import Verdict, color_mention, combine_verdicts, exact_word_count, marker_leak, non_empty_non_gibberish, repetition, substring_present, token_budget_exhausted


@dataclass
class EvalTask:
    name: str
    category: str  # "vision" | "thinking" | "stop" | "text" -- matches --tasks filter
    required_capabilities: tuple[str, ...]  # subset of a model's /v1/models capabilities; () = runs for every model
    description: str
    build_request: Callable[[], dict]
    judge: Callable[[dict], Verdict]
    timeout: float = 300.0  # generous, mirrors the seed scripts (vision/thinking decode is slow)


# ---------------------------------------------------------------------------
# Fixture generation (runtime, no files on disk)
# ---------------------------------------------------------------------------

def _data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode()


def _color_letter_png(rgb: tuple[int, int, int], letter: str, size: int = 256) -> bytes:
    img = Image.new("RGB", (size, size), rgb)
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # no bundled font files -- default is legible enough
    brightness = sum(rgb) / 3
    fg = (0, 0, 0) if brightness > 140 else (255, 255, 255)  # crude contrast pick
    draw.text((size // 2 - 6, size // 2 - 6), letter, fill=fg, font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _heatmap_png(width: int = 1400, height: int = 900) -> bytes:
    # Random noise, not a real heatmap -- big + content-free so the "sanity"
    # task has no ground-truth colors to hallucinate about. Seeded so the
    # fixture (and therefore the vision-feature-cache key) is stable run to run.
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Distinct color/letter pairs, following the seed scripts' img_<color>_<letter> naming.
_RED = _data_url(_color_letter_png((230, 30, 30), "A"))
_BLUE = _data_url(_color_letter_png((30, 60, 220), "B"))
_GREEN = _data_url(_color_letter_png((30, 150, 60), "C"))
_YELLOW = _data_url(_color_letter_png((225, 210, 20), "D"))
_HEATMAP = _data_url(_heatmap_png())


def _vision_content(urls: list[str], prompt: str) -> list[dict]:
    content = [{"type": "image_url", "image_url": {"url": u}} for u in urls]
    content.append({"type": "text", "text": prompt})
    return content


def _text_body(prompt: str, **extra) -> dict:
    return {"messages": [{"role": "user", "content": prompt}], **extra}


def _vision_body(urls: list[str], prompt: str, **extra) -> dict:
    return {"messages": [{"role": "user", "content": _vision_content(urls, prompt)}], **extra}


# ---------------------------------------------------------------------------
# Vision tasks
# ---------------------------------------------------------------------------

def _judge_single_color(ctx: dict) -> Verdict:
    return combine_verdicts(
        color_mention(ctx["content"], ["red"]),
        marker_leak(ctx["content"]),
        repetition(ctx["content"]),
    )


TASK_VISION_SINGLE_COLOR_LETTER = EvalTask(
    name="vision_single_color_letter",
    category="vision",
    required_capabilities=("vision",),
    description="One generated solid-color image with a letter overlay; checks color mention, no leak markers, no runaway repetition. (Letter-mention is not separately judged -- color_mention alone is the property check; see README.)",
    build_request=lambda: _vision_body([_RED], "What color is this image? Also name any letter you see on it.", max_tokens=60),
    judge=_judge_single_color,
    timeout=300,
)


def _judge_two_image(ctx: dict) -> Verdict:
    base = combine_verdicts(color_mention(ctx["content"], ["green", "yellow"]), marker_leak(ctx["content"]))
    lower = (ctx["content"] or "").lower()
    ig, iy = lower.find("green"), lower.find("yellow")
    order = "green-before-yellow" if 0 <= ig < iy else ("yellow-before-green" if 0 <= iy < ig else "order-unclear")
    return Verdict(passed=base.passed, evidence=f"{base.evidence}; order: {order} (soft, not hard-failed)")


TASK_VISION_TWO_IMAGE_DISCRIMINATION = EvalTask(
    name="vision_two_image_discrimination",
    category="vision",
    required_capabilities=("vision",),
    description="Two distinct-color images, one prompt; requires BOTH colors mentioned. Ordering is recorded as a soft signal, never hard-failed.",
    build_request=lambda: _vision_body([_GREEN, _YELLOW], "Describe each image in one short sentence.", max_tokens=100),
    judge=_judge_two_image,
    timeout=300,
)


def _judge_heatmap(ctx: dict) -> Verdict:
    return combine_verdicts(non_empty_non_gibberish(ctx["content"]), marker_leak(ctx["content"]))


TASK_VISION_LARGE_HEATMAP_SANITY = EvalTask(
    name="vision_large_heatmap_sanity",
    category="vision",
    required_capabilities=("vision",),
    description="~1400x900 random-noise image; sanity only (non-empty, non-gibberish, no leak) -- no ground truth to judge color-accuracy against.",
    build_request=lambda: _vision_body([_HEATMAP], "Describe this image.", max_tokens=150),
    judge=_judge_heatmap,
    timeout=600,
)


def _judge_budget(color: str, vt: int) -> Callable[[dict], Verdict]:
    def judge(ctx: dict) -> Verdict:
        v = color_mention(ctx["content"], [color])
        return Verdict(passed=v.passed, evidence=f"vision_tokens={vt}: {v.evidence}")
    return judge


TASK_VISION_BUDGET_LOW_TOKENS = EvalTask(
    name="vision_budget_low_tokens",
    category="vision",
    required_capabilities=("vision",),
    description="Same fixture as the single-color task but with vision_tokens=70 (low budget); paired with vision_budget_high_tokens -- see README for why these are two tasks, not one.",
    build_request=lambda: _vision_body([_RED], "What color is this image?", max_tokens=60, vision_tokens=70),
    judge=_judge_budget("red", 70),
    timeout=300,
)

TASK_VISION_BUDGET_HIGH_TOKENS = EvalTask(
    name="vision_budget_high_tokens",
    category="vision",
    required_capabilities=("vision",),
    description="Same fixture as vision_budget_low_tokens but with vision_tokens=1120 (high budget); both must pass independently, differences are recorded not hard-failed.",
    build_request=lambda: _vision_body([_RED], "What color is this image?", max_tokens=60, vision_tokens=1120),
    judge=_judge_budget("red", 1120),
    timeout=300,
)


def _judge_vision_thinking_off(ctx: dict) -> Verdict:
    return combine_verdicts(color_mention(ctx["content"], ["blue"]), marker_leak(ctx["content"]))


TASK_VISION_THINKING_OFF_PURITY = EvalTask(
    name="vision_thinking_off_purity",
    category="vision",
    required_capabilities=("vision",),
    description="One image with enable_thinking=False; checks the vision answer stays clean (color mentioned, no leak markers) with thinking off.",
    build_request=lambda: _vision_body([_BLUE], "Describe this image in one short sentence.", max_tokens=60, enable_thinking=False),
    judge=_judge_vision_thinking_off,
    timeout=300,
)


# ---------------------------------------------------------------------------
# Thinking tasks
# ---------------------------------------------------------------------------

def _judge_thinking_split(ctx: dict) -> Verdict:
    has_thinking = bool((ctx["thinking"] or "").strip())
    thinking_v = Verdict(passed=has_thinking, evidence=f"thinking field {'present' if has_thinking else 'MISSING'}")
    return combine_verdicts(thinking_v, marker_leak(ctx["content"]))


TASK_THINKING_REQUESTED_SPLIT = EvalTask(
    name="thinking_requested_split",
    category="thinking",
    required_capabilities=("thinking",),
    description="enable_thinking=True on a plain prompt; requires a non-empty `thinking` field AND content free of leak markers (<think>, <|channel>, 'thought'-prefix).",
    build_request=lambda: _text_body("Explain briefly why the sky appears blue.", max_tokens=200, enable_thinking=True),
    judge=_judge_thinking_split,
    timeout=600,
)


def _judge_thinking_off(ctx: dict) -> Verdict:
    no_thinking = Verdict(passed=ctx["thinking"] is None, evidence=f"thinking field {'absent (good)' if ctx['thinking'] is None else 'PRESENT'}")
    return combine_verdicts(no_thinking, marker_leak(ctx["content"]))


TASK_THINKING_OFF_PURITY = EvalTask(
    name="thinking_off_purity",
    category="thinking",
    required_capabilities=("thinking",),
    description="enable_thinking=False; requires NO `thinking` field in the response and no explicit leak markers in content (does not attempt to detect 'reasoning-sounding' prose -- that's model-version-brittle).",
    build_request=lambda: _text_body("Explain briefly why the sky appears blue.", max_tokens=100, enable_thinking=False),
    judge=_judge_thinking_off,
    timeout=300,
)


def _judge_thinking_multi_image(ctx: dict) -> Verdict:
    has_thinking = bool((ctx["thinking"] or "").strip())
    thinking_v = Verdict(passed=has_thinking, evidence=f"thinking field {'present' if has_thinking else 'MISSING'}")
    leak_v = marker_leak(ctx["content"])
    combined_text = (ctx["content"] or "") + " " + (ctx["thinking"] or "")
    colors_v = color_mention(combined_text, ["red", "blue"])
    return combine_verdicts(thinking_v, leak_v, colors_v)


TASK_THINKING_MULTI_IMAGE_COMBINED = EvalTask(
    name="thinking_multi_image_combined",
    category="thinking",  # judged category is thinking-split behavior under vision load, not color accuracy -- see README
    required_capabilities=("vision", "thinking"),
    description="enable_thinking=True with two distinct-color images; requires thinking present, no leak, and both colors mentioned across content+thinking combined (mirrors full_matrix.py's judge).",
    build_request=lambda: _vision_body([_RED, _BLUE], "Describe each image in one short sentence.", max_tokens=300, enable_thinking=True),
    judge=_judge_thinking_multi_image,
    timeout=900,
)


# ---------------------------------------------------------------------------
# Stop discipline
# ---------------------------------------------------------------------------

def _judge_stop(ctx: dict) -> Verdict:
    return combine_verdicts(
        repetition(ctx["content"]),
        token_budget_exhausted(ctx["completion_tokens"], ctx["max_tokens"]),
    )


TASK_STOP_DISCIPLINE_SHORT_ANSWER = EvalTask(
    name="stop_discipline_short_answer",
    category="stop",
    required_capabilities=(),
    description="Short-answer prompt with a generous max_tokens; fails on runaway sentence repetition or on hitting the token cap exactly (proxy for never finding a stopping point).",
    build_request=lambda: _text_body("What is 2+2? Answer in one short sentence.", max_tokens=200, temperature=0.1),
    judge=_judge_stop,
    timeout=300,
)

TASK_STOP_DISCIPLINE_LONG_FORM = EvalTask(
    name="stop_discipline_long_form",
    category="stop",
    required_capabilities=(),
    description="Zero-image, thinking-off creative prompt with more room to ramble; same repetition + token-budget-exhaustion checks as stop_discipline_short_answer on a longer generation.",
    build_request=lambda: _text_body("Write a short paragraph describing the ocean at sunset.", max_tokens=150, enable_thinking=False),
    judge=_judge_stop,
    timeout=300,
)


# ---------------------------------------------------------------------------
# Text sanity
# ---------------------------------------------------------------------------

TASK_TEXT_FACTUAL_QA_CAPITAL = EvalTask(
    name="text_factual_qa_capital",
    category="text",
    required_capabilities=(),
    description="'What is the capital of France?' -- checks 'paris' appears (case-insensitive). The one task allowed an exact-ish string check: it's ground truth, not phrasing.",
    build_request=lambda: _text_body("What is the capital of France?", max_tokens=30),
    judge=lambda ctx: substring_present(ctx["content"], "paris"),
    timeout=120,
)


def _judge_single_word(ctx: dict) -> Verdict:
    v = exact_word_count(ctx["content"], 1)
    word = (ctx["content"] or "").strip().strip(".,!?\"'").lower()
    soft_note = f"; soft check word={word!r} (not hard-required to be 'blue')"
    return Verdict(passed=v.passed, evidence=v.evidence + soft_note)


TASK_TEXT_SINGLE_WORD_INSTRUCTION = EvalTask(
    name="text_single_word_instruction",
    category="text",
    required_capabilities=(),
    description="'Respond with exactly one word: the color of the sky.' -- checks content is exactly one whitespace-separated token; does not hard-require the word be 'blue' (content-brittleness).",
    build_request=lambda: _text_body("Respond with exactly one word: the color of the sky.", max_tokens=10),
    judge=_judge_single_word,
    timeout=120,
)


TASKS: list[EvalTask] = [
    TASK_VISION_SINGLE_COLOR_LETTER,
    TASK_VISION_TWO_IMAGE_DISCRIMINATION,
    TASK_VISION_LARGE_HEATMAP_SANITY,
    TASK_VISION_BUDGET_LOW_TOKENS,
    TASK_VISION_BUDGET_HIGH_TOKENS,
    TASK_VISION_THINKING_OFF_PURITY,
    TASK_THINKING_REQUESTED_SPLIT,
    TASK_THINKING_OFF_PURITY,
    TASK_THINKING_MULTI_IMAGE_COMBINED,
    TASK_STOP_DISCIPLINE_SHORT_ANSWER,
    TASK_STOP_DISCIPLINE_LONG_FORM,
    TASK_TEXT_FACTUAL_QA_CAPITAL,
    TASK_TEXT_SINGLE_WORD_INSTRUCTION,
]
