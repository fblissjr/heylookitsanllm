"""Starter contrastive corpora for Qwen 3.8 thinking + vision steering experiments.

All text pairs are balanced across identical topics so setting/noun bias does
not leak into extracted subspaces (avoiding the H3 `de_llm_slop` office-cabinet
correlation trap). Vision pairs synthesize paired Pillow images on the fly so
the corpus is 100% self-contained with zero external file paths.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ContrastivePair:
    id: str
    user_prompt: str
    system_a: str  # Pole A (source / unwanted / baseline)
    system_b: str  # Pole B (target / clean)
    image_variant: Optional[int] = None  # optional synthetic image for VLM capture


@dataclass(frozen=True)
class CorpusSpec:
    id: str
    label: str
    operator: str  # "regime_a" | "regime_b" | "conditional"
    phase: str
    token_mask: str
    default_span: str
    layers: tuple[int, ...]
    pairs: tuple[ContrastivePair, ...]
    notes: str = ""
    kind: str = "contrast"  # "contrast" | "hold"


def make_synthetic_image_data_uri(variant: int) -> str:
    """Create a deterministic synthetic RGB PNG data URI for VLM steering probes."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (384, 384), (24, 34, 48))
    draw = ImageDraw.Draw(im)
    if variant % 3 == 0:
        draw.rectangle((0, 260, 384, 384), fill=(52, 60, 68))
        draw.ellipse((112, 72, 272, 232), fill=(228, 92, 54))
        draw.rectangle((48, 176, 104, 310), fill=(196, 206, 214))
    elif variant % 3 == 1:
        draw.rectangle((0, 0, 384, 192), fill=(238, 198, 110))
        draw.rectangle((0, 192, 384, 384), fill=(36, 52, 74))
        draw.polygon([(192, 60), (312, 290), (72, 290)], fill=(26, 28, 34))
    else:
        draw.rectangle((0, 0, 384, 384), fill=(22, 46, 38))
        for x in range(24, 360, 56):
            draw.rectangle((x, 48, x + 28, 336), fill=(190, 210, 185))
        draw.ellipse((142, 142, 242, 242), fill=(235, 195, 65))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


_SHARED_TOPICS = (
    ("cache", "Explain why a sliding-window attention layer needs different prompt caching than full attention."),
    ("bridge", "Describe how a suspension bridge carries traffic load down to its foundations."),
    ("coffee", "Explain what happens during espresso extraction when the grind is too fine."),
    ("storm", "Describe how a supercell thunderstorm forms and sustains rotation."),
    ("clock", "Explain how a mechanical escapement regulates a watch."),
    ("memory", "Explain unified memory bandwidth tradeoffs on Apple Silicon during autoregressive decoding."),
    ("piano", "Describe how pressing a piano key transfers energy to the string and soundboard."),
    ("optics", "Explain why stopping a camera lens down from f/1.4 to f/5.6 deepens focus."),
)

_CLINICAL_SYS_A = (
    "Write in lush, metaphorical, evocative literary prose rich with sensory imagery and poetic cadence."
)
_CLINICAL_SYS_B = (
    "Write in crisp, clinical engineering prose: concrete mechanisms, exact terminology, zero metaphor."
)

_BLUNT_SYS_A = (
    "Be diplomatically hedged, tentative, and reassuring. Qualify statements with softeners like "
    "'perhaps', 'arguably', 'it could be said', and validate the reader's curiosity."
)
_BLUNT_SYS_B = (
    "Be blunt, direct, and unhedged. State conclusions plainly on line one with zero preamble or flattery."
)

_SOCRATIC_SYS_A = (
    "Answer declaratively as a complete, self-contained explanation with no questions back to the reader."
)
_SOCRATIC_SYS_B = (
    "Respond in Socratic style: guide the reader via crisp, probing diagnostic questions that expose the core mechanism."
)

_SLOP_SYS_A = (
    "Use upbeat corporate-AI blog style with phrases like 'In today's rapidly evolving landscape', "
    "'delve into', 'rich tapestry', 'multifaceted', 'crucial role', and 'it's important to note'."
)
_SLOP_SYS_B = (
    "Write plain, concrete, unadorned prose with zero clichés, zero throat-clearing, and zero meta-commentary."
)

_VISION_PROMPTS = (
    ("vis_0", "Describe what you see in this image in three sentences.", 0),
    ("vis_1", "Analyze the composition and focal elements of this image.", 1),
    ("vis_2", "Describe the light, contrast, and geometry in this image.", 2),
    ("vis_3", "Summarize the primary shapes, color palette, and balance shown here.", 0),
    ("vis_4", "Explain how a viewer's eye moves across this frame.", 1),
    ("vis_5", "Describe the foreground and background separation in this scene.", 2),
)

_VISION_SYS_A = (
    "Describe the image through abstract emotional mood, dreamy aesthetic vibes, and poetic interpretation."
)
_VISION_SYS_B = (
    "Describe the image strictly through concrete geometry, coordinates, RGB color contrast, and optical structure."
)

_HOLD_PROMPTS = (
    ("hold_math_1", "Compute 7^100 mod 13. Give the final integer clearly."),
    ("hold_math_2", "Find the smallest positive integer x with x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)."),
    ("hold_code_1", "Write a Python function merge_intervals(intervals) that does not mutate the input list."),
    ("hold_code_2", "Write a Python function has_cycle(graph) for a directed graph using 3-color DFS."),
    ("hold_json_1", "Return ONLY valid JSON with keys 'primes' (first 5 primes) and 'sum' (their integer sum)."),
    ("hold_json_2", "Return ONLY valid JSON factoring 2520 into [{'prime': p, 'exp': e}]."),
)


def default_corpora() -> dict[str, CorpusSpec]:
    """Return all starter contrastive corpora + capability holds keyed by concept ID."""
    return {
        "clinical_vs_poetic": CorpusSpec(
            id="clinical_vs_poetic",
            label="Clinical Technical (vs. Poetic)",
            operator="regime_a",
            phase="content",
            token_mask="decode",
            default_span="answer",
            layers=(39, 43, 47, 51),
            pairs=tuple(
                ContrastivePair(f"cp_{tid}", prompt, _CLINICAL_SYS_A, _CLINICAL_SYS_B)
                for tid, prompt in _SHARED_TOPICS
            ),
            notes="Regime-A register axis on the ~60-80% Full-Attention band (idx % 4 == 3: L39, L43, L47, L51).",
        ),
        "blunt_vs_hedged": CorpusSpec(
            id="blunt_vs_hedged",
            label="Blunt Directness (vs. Hedged)",
            operator="regime_a",
            phase="content",
            token_mask="decode",
            default_span="answer",
            layers=(39, 43, 47, 51),
            pairs=tuple(
                ContrastivePair(f"bh_{tid}", prompt, _BLUNT_SYS_A, _BLUNT_SYS_B)
                for tid, prompt in _SHARED_TOPICS
            ),
            notes="Regime-A directness axis on the FA band paired with clinical_vs_poetic to test peer_ros.",
        ),
        "socratic_vs_declarative": CorpusSpec(
            id="socratic_vs_declarative",
            label="Socratic Questioning (vs. Declarative)",
            operator="regime_a",
            phase="content",
            token_mask="decode",
            default_span="answer",
            layers=(39, 43, 47, 51),
            pairs=tuple(
                ContrastivePair(f"sd_{tid}", prompt, _SOCRATIC_SYS_A, _SOCRATIC_SYS_B)
                for tid, prompt in _SHARED_TOPICS
            ),
            notes="Third Regime-A FA-band axis to stress-test 3-slider composition and clamped union SVD.",
        ),
        "anti_slop": CorpusSpec(
            id="anti_slop",
            label="Anti-AI-Slop Eraser (Regime B)",
            operator="regime_b",
            phase="content",
            token_mask="decode",
            default_span="answer",
            layers=(63,),
            pairs=tuple(
                ContrastivePair(f"slop_{tid}", prompt, _SLOP_SYS_A, _SLOP_SYS_B)
                for tid, prompt in _SHARED_TOPICS
            ),
            notes="Regime-B active noise cancellation at terminal L63 using paired-shift vr_diff.",
        ),
        "vision_optical_vs_mood": CorpusSpec(
            id="vision_optical_vs_mood",
            label="Optical/Physical Vision (vs. Abstract Mood)",
            operator="regime_a",
            phase="all",
            token_mask="image_pad",
            default_span="image_pad",
            layers=(27, 47),
            pairs=tuple(
                ContrastivePair(vid, prompt, _VISION_SYS_A, _VISION_SYS_B, image_variant=var)
                for vid, prompt, var in _VISION_PROMPTS
            ),
            notes="VLM Dual-Lock probe comparing <|image_pad|>-only vs content-only steering.",
        ),
        "capability_holds": CorpusSpec(
            id="capability_holds",
            label="Math / Code / JSON Capability Holds",
            operator="regime_a",
            phase="all",
            token_mask="all",
            default_span="answer",
            layers=(15, 27, 39, 43, 47, 49, 51, 55, 62, 63),
            pairs=tuple(
                ContrastivePair(hid, prompt, "", "")
                for hid, prompt in _HOLD_PROMPTS
            ),
            notes="Clean structured-reasoning spans projected out of Regime-A sliders.",
            kind="hold",
        ),
    }


def list_corpora_summary() -> list[dict[str, Any]]:
    return [
        {
            "id": c.id,
            "label": c.label,
            "operator": c.operator,
            "kind": c.kind,
            "phase": c.phase,
            "token_mask": c.token_mask,
            "default_span": c.default_span,
            "layers": list(c.layers),
            "pair_count": len(c.pairs),
            "notes": c.notes,
        }
        for c in default_corpora().values()
    ]
