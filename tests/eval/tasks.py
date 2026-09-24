# The task bank as data. Generalizes two ad-hoc debugging scripts
# (full_matrix.py: model x think x image-count x vision-budget matrix;
# repro_multiimage.py: multi-image + thinking gibberish repro) into a
# reusable, filterable bank. Fixture images are generated at import time with
# PIL/numpy (no files on disk) since the originals' pre-existing PNGs no
# longer exist.
#
# Each EvalTask.build_request() returns an OpenAI-shaped chat body WITHOUT
# "model"/"stream"; run.py injects those and adapts the body to /v1/messages
# (run.to_messages_request). judge() takes a small context dict (content,
# thinking, completion_tokens, max_tokens, stop_reason) and returns a
# judges.Verdict. Expected values (colors, needles) are closed over by the
# judge lambda rather than stored as separate EvalTask fields, so adding a
# task never requires touching the EvalTask shape itself.
from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageFont

from judges import Verdict, color_mention, combine_verdicts, exact_word_count, marker_leak, non_empty_non_gibberish, not_refusal, repetition, substring_present, token_budget_exhausted


@dataclass
class EvalTask:
    name: str
    category: str  # the --tasks filter; run.py lists the categories in use
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
    # Four flat color quadrants: still big (exercises large-image token
    # budgets) but DESCRIBABLE, so the judge can demand actual perception.
    # The earlier pure-noise version let a refusal ("I cannot describe this
    # image") score as a prose-pass -- there was nothing to check against.
    # Deterministic drawing keeps the fixture (and the vision-feature-cache
    # key) stable run to run.
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for box, rgb in (
        ((0, 0, width // 2, height // 2), (230, 30, 30)),        # red
        ((width // 2, 0, width, height // 2), (30, 60, 220)),    # blue
        ((0, height // 2, width // 2, height), (30, 150, 60)),   # green
        ((width // 2, height // 2, width, height), (240, 150, 20)),  # orange
    ):
        draw.rectangle(box, fill=rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Distinct color/letter pairs, following the seed scripts' img_<color>_<letter> naming.
_RED = _data_url(_color_letter_png((230, 30, 30), "A"))
_BLUE = _data_url(_color_letter_png((30, 60, 220), "B"))
_GREEN = _data_url(_color_letter_png((30, 150, 60), "C"))
_YELLOW = _data_url(_color_letter_png((225, 210, 20), "D"))
_HEATMAP = _data_url(_heatmap_png())


# The least any task gives a model to answer in. An always-reasoning model
# (gpt-oss's harmony analysis channel) reasons even with thinking off, and a
# budget below this ran out mid-analysis: empty content, judged as a wrong
# answer when nothing was answered at all. Tasks that need more say so.
ANSWER_BUDGET = 256


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
    build_request=lambda: _vision_body([_RED], "What color is this image? Also name any letter you see on it.", max_tokens=ANSWER_BUDGET, enable_thinking=False),
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
    build_request=lambda: _vision_body([_GREEN, _YELLOW], "Describe each image in one short sentence.", max_tokens=ANSWER_BUDGET, enable_thinking=False),
    judge=_judge_two_image,
    timeout=300,
)


def _judge_heatmap(ctx: dict) -> Verdict:
    content = (ctx["content"] or "").lower()
    hits = [c for c in ("red", "blue", "green", "orange") if c in content]
    colors_v = Verdict(
        passed=len(hits) >= 2,
        evidence=f"quadrant colors {len(hits)}/4: {', '.join(hits) or 'none'}",
    )
    return combine_verdicts(
        colors_v,
        not_refusal(ctx["content"]),
        non_empty_non_gibberish(ctx["content"]),
        marker_leak(ctx["content"]),
    )


TASK_VISION_LARGE_HEATMAP_SANITY = EvalTask(
    name="vision_large_heatmap_sanity",
    category="vision",
    required_capabilities=("vision",),
    description="~1400x900 four-quadrant color-block image; requires >=2 quadrant colors named, non-refusal, non-gibberish, no leak markers -- large-image budget sanity WITH ground truth.",
    build_request=lambda: _vision_body([_HEATMAP], "Describe this image.", max_tokens=ANSWER_BUDGET, enable_thinking=False),
    judge=_judge_heatmap,
    timeout=600,
)



def _judge_vision_thinking_off(ctx: dict) -> Verdict:
    return combine_verdicts(color_mention(ctx["content"], ["blue"]), marker_leak(ctx["content"]))


TASK_VISION_THINKING_OFF_PURITY = EvalTask(
    name="vision_thinking_off_purity",
    category="vision",
    required_capabilities=("vision",),
    description="One image with enable_thinking=False; checks the vision answer stays clean (color mentioned, no leak markers) with thinking off.",
    build_request=lambda: _vision_body([_BLUE], "Describe this image in one short sentence.", max_tokens=ANSWER_BUDGET, enable_thinking=False),
    judge=_judge_vision_thinking_off,
    timeout=300,
)


# ---------------------------------------------------------------------------
# Thinking tasks
# ---------------------------------------------------------------------------

# The hard thinking cap the split task runs under (plan W7). Without one the
# task measured verbosity, not the split: a small model that thinks past the
# token budget failed it with a clean split (Qwen3.5-0.8B, v2.0.71 port).
THINKING_BUDGET = 256
# Streamed thinking is counted per chunk and the engine forces a newline and
# the close marker after the budget is passed, so allow a few over.
THINKING_BUDGET_SLACK = 8


def _judge_thinking_split(ctx: dict) -> Verdict:
    has_thinking = bool((ctx["thinking"] or "").strip())
    thinking_v = Verdict(passed=has_thinking, evidence=f"thinking field {'present' if has_thinking else 'MISSING'}")
    has_answer = bool((ctx["content"] or "").strip())
    answer_v = Verdict(passed=has_answer,
                       evidence=f"answer after the thinking {'present' if has_answer else 'MISSING'}")
    spent = ctx.get("thinking_tokens")
    cap = THINKING_BUDGET + THINKING_BUDGET_SLACK
    budget_v = Verdict(passed=spent is not None and spent <= cap,
                       evidence=f"thinking tokens {spent} (budget {THINKING_BUDGET}, allowed {cap})")
    return combine_verdicts(
        thinking_v, answer_v, budget_v,
        marker_leak(ctx["content"]),
        token_budget_exhausted(ctx["completion_tokens"], ctx["max_tokens"],
                               ctx.get("stop_reason")),
    )


TASK_THINKING_REQUESTED_SPLIT = EvalTask(
    name="thinking_requested_split",
    category="thinking",
    required_capabilities=("thinking", "thinking_budget"),
    description=f"thinking on with a {THINKING_BUDGET}-token hard budget on a plain prompt; requires non-empty `thinking`, an answer after it, thinking within the budget, and content free of leak markers (<think>, <|channel>, 'thought'-prefix).",
    build_request=lambda: _text_body("Explain briefly why the sky appears blue.", max_tokens=1536,
                                     thinking={"type": "enabled", "budget_tokens": THINKING_BUDGET}),
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
    build_request=lambda: _text_body("Explain briefly why the sky appears blue.", max_tokens=ANSWER_BUDGET, enable_thinking=False),
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
    build_request=lambda: _vision_body([_RED, _BLUE], "Describe each image in one short sentence.", max_tokens=768, enable_thinking=True),
    judge=_judge_thinking_multi_image,
    timeout=900,
)


# ---------------------------------------------------------------------------
# Stop discipline
# ---------------------------------------------------------------------------

def _judge_stop(ctx: dict) -> Verdict:
    return combine_verdicts(
        repetition(ctx["content"]),
        token_budget_exhausted(ctx["completion_tokens"], ctx["max_tokens"],
                               ctx.get("stop_reason")),
    )


TASK_STOP_DISCIPLINE_SHORT_ANSWER = EvalTask(
    name="stop_discipline_short_answer",
    category="stop",
    required_capabilities=(),
    description="Short-answer prompt with a generous max_tokens; fails on runaway sentence repetition or on hitting the token cap exactly (proxy for never finding a stopping point).",
    build_request=lambda: _text_body("What is 2+2? Answer in one short sentence.", max_tokens=ANSWER_BUDGET, enable_thinking=False),
    judge=_judge_stop,
    timeout=300,
)

TASK_STOP_DISCIPLINE_LONG_FORM = EvalTask(
    name="stop_discipline_long_form",
    category="stop",
    required_capabilities=(),
    description="Zero-image, thinking-off creative prompt with more room to ramble; same repetition + token-budget-exhaustion checks as stop_discipline_short_answer on a longer generation.",
    build_request=lambda: _text_body("Write a short paragraph describing the ocean at sunset.", max_tokens=400, enable_thinking=False),
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
    build_request=lambda: _text_body("What is the capital of France?", max_tokens=ANSWER_BUDGET, enable_thinking=False),
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
    build_request=lambda: _text_body("Respond with exactly one word: the color of the sky.", max_tokens=ANSWER_BUDGET, enable_thinking=False),
    judge=_judge_single_word,
    timeout=120,
)


# ---------------------------------------------------------------------------
# Audio tasks (plan Phase 7d; gguf/llama-server models only -- MLX 400s audio)
# ---------------------------------------------------------------------------

def _wav_base64_from_file() -> str:
    # The one committed audio fixture: real speech (a "six seven" chant),
    # survivor of the deleted STT provider. Real speech is deliberate --
    # gemma-4's audio stack is speech-trained, and the keyword property
    # below fails when the audio embedding never reaches the model
    # (a broken pipeline hallucinates a description without the keywords).
    wav_path = Path(__file__).parent.parent / "input" / "test_15sec.wav"
    return base64.b64encode(wav_path.read_bytes()).decode()


def _sine_wav_base64(freq_hz: int = 440, seconds: float = 2.0, rate: int = 16000) -> str:
    # Deterministic pure tone, synthesized at import like the PIL vision
    # fixtures -- no numpy, stdlib wave + math only.
    import math
    import struct
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        n = int(rate * seconds)
        frames = b"".join(
            struct.pack("<h", int(20000 * math.sin(2 * math.pi * freq_hz * i / rate)))
            for i in range(n)
        )
        w.writeframes(frames)
    return base64.b64encode(buf.getvalue()).decode()


_SPEECH_B64 = _wav_base64_from_file()
_TONE_B64 = _sine_wav_base64()


def _audio_body(audio_b64: str, prompt: str, **extra) -> dict:
    return {"messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
    ]}], **extra}


def _mentions_any(text: str, words: list[str], label: str) -> Verdict:
    lower = (text or "").lower()
    hits = [w for w in words if w in lower]
    return Verdict(passed=bool(hits),
                   evidence=f"{label}: {'mentions ' + ', '.join(hits) if hits else 'none of ' + '/'.join(words)}")


def _judge_speech_keywords(ctx: dict) -> Verdict:
    return combine_verdicts(
        _mentions_any(ctx["content"], ["six", "seven"], "speech keywords"),
        not_refusal(ctx["content"]),
        marker_leak(ctx["content"]),
    )


TASK_AUDIO_SPEECH_KEYWORDS = EvalTask(
    name="audio_speech_keywords",
    category="audio",
    required_capabilities=("audio",),
    description="Real speech clip (committed fixture); the transcription/description must surface a keyword actually said ('six'/'seven'). Property, not exact-transcript: proves the audio embedding reaches the model.",
    build_request=lambda: _audio_body(_SPEECH_B64, "Briefly, what do you hear in this audio? Quote any words or numbers you can make out.", max_tokens=ANSWER_BUDGET, enable_thinking=False),
    judge=_judge_speech_keywords,
    timeout=300,
)


def _judge_tone(ctx: dict) -> Verdict:
    return combine_verdicts(
        _mentions_any(ctx["content"], ["tone", "beep", "note", "sine", "hum", "buzz", "synth", "pitch"], "tone words"),
        marker_leak(ctx["content"]),
    )


TASK_AUDIO_TONE_VS_SPEECH = EvalTask(
    name="audio_tone_vs_speech",
    category="audio",
    required_capabilities=("audio",),
    description="Synthesized 440Hz sine (deterministic, stdlib): asked speech-or-tone, the answer must use tone vocabulary -- discriminates listening from confabulating speech.",
    build_request=lambda: _audio_body(_TONE_B64, "Is this sound human speech or a simple synthesized tone? Describe it in one sentence.", max_tokens=ANSWER_BUDGET, enable_thinking=False),
    judge=_judge_tone,
    timeout=300,
)


TASKS: list[EvalTask] = [
    TASK_AUDIO_SPEECH_KEYWORDS,
    TASK_AUDIO_TONE_VS_SPEECH,
    TASK_VISION_SINGLE_COLOR_LETTER,
    TASK_VISION_TWO_IMAGE_DISCRIMINATION,
    TASK_VISION_LARGE_HEATMAP_SANITY,
    TASK_VISION_THINKING_OFF_PURITY,
    TASK_THINKING_REQUESTED_SPLIT,
    TASK_THINKING_OFF_PURITY,
    TASK_THINKING_MULTI_IMAGE_COMBINED,
    TASK_STOP_DISCIPLINE_SHORT_ANSWER,
    TASK_STOP_DISCIPLINE_LONG_FORM,
    TASK_TEXT_FACTUAL_QA_CAPITAL,
    TASK_TEXT_SINGLE_WORD_INSTRUCTION,
]
