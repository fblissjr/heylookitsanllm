"""Task templates: built-in labeling tasks and custom task TOML loading.

A task bundles a system prompt, a user prompt, output expectations
(freeform vs JSON + required keys), and generation defaults (server-side
sampler preset, max_tokens). CLI flags override task defaults; task
defaults override server/model defaults via the server's cascade.
"""

import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

DEFAULT_USER_PROMPT = "Analyze this image."


@dataclass(frozen=True)
class Task:
    name: str
    description: str
    system_prompt: str
    user_prompt: str = DEFAULT_USER_PROMPT
    expects_json: bool = True
    required_keys: tuple[str, ...] = ()
    # Generation defaults (overridable from the CLI). sampler refers to the
    # server's named-sampler registry (discoverable via /v1/capabilities).
    sampler: str | None = None
    max_tokens: int | None = None


_LABEL_SYSTEM_PROMPT = """\
You are a meticulous image annotation engine producing training-quality labels.

Respond with EXACTLY one JSON object and nothing else: no markdown fences, no
commentary, no trailing text. Use null for fields you cannot determine.

Schema (all fields required unless marked nullable):
{
  "category": string,        // primary category from the taxonomy below
  "subcategory": string,     // from the taxonomy; "other" if none fits
  "description": string,     // 1-3 dense, concrete sentences; describe what IS visible, no speculation
  "objects": [string],       // up to 8 distinct visible objects, most prominent first
  "text_content": string|null,  // verbatim visible text (signs, labels, UI), null if none
  "dominant_colors": [string],  // 1-4 simple color names ("blue", "dark green", "off-white")
  "style": string,           // photo | illustration | render_3d | screenshot | scan | painting | pixel_art | diagram | other
  "setting": string|null,    // indoor | outdoor | studio | aerial | underwater | space | abstract | null
  "lighting": string|null,   // natural | artificial | low_light | overexposed | mixed | flat | dramatic | null
  "people_count": integer,   // 0 if none; count visible people
  "mood": string,            // neutral | cheerful | somber | dramatic | calm | chaotic | clinical | nostalgic | tense
  "quality_issues": [string],  // any of: blur, noise, compression_artifacts, overexposure, underexposure, occlusion, low_resolution; [] if clean
  "confidence": number       // 0.0-1.0; below 0.5 if blurry, ambiguous, or a poor taxonomy fit
}

Taxonomy (category: subcategories):
- landscape: natural, urban, aerial, underwater, night
- portrait: human, animal, group, selfie
- object: food, vehicle, electronics, furniture, clothing, tool, product, plant
- scene: interior, street, event, sports, workplace
- text: document, sign, screenshot, handwritten, receipt
- diagram: flowchart, graph, schematic, map, ui_mockup
- art: painting, illustration, sculpture, digital, photo_manipulation
- other: abstract, unclear

Rules:
- "description" must mention text content briefly if any text is visible.
- Prefer specific object names ("espresso machine") over generic ones ("appliance").
- Never invent objects, people, or text that are not clearly visible.\
"""

_CAPTION_SYSTEM_PROMPT = """\
You write dense, factual image captions suitable for training image-generation
and captioning models.

Respond with a single paragraph of 2-5 sentences and nothing else. No preamble
("This image shows..."), no markdown, no bullet points.

Style rules:
- Lead with the main subject and action, then setting, then notable details.
- Be concrete: name specific objects, materials, colors, and spatial relations.
- Note the medium and style when apparent (photograph, watercolor, 3D render,
  screenshot), plus camera angle or framing if distinctive (close-up, aerial,
  low-angle).
- Transcribe short prominent text verbatim in quotes; summarize long text.
- Describe only what is visible. No speculation about intent, emotion of the
  photographer, or anything outside the frame.\
"""

_TAGS_SYSTEM_PROMPT = """\
You are an image tagging engine. Produce flat keyword tags for search and
dataset filtering.

Respond with EXACTLY one JSON object and nothing else:
{"tags": [string]}

Rules:
- 5-20 tags, lowercase, singular nouns or short noun phrases ("golden retriever",
  "sunset", "brick wall"). Use underscores never; spaces are fine.
- Order from most to least salient.
- Cover: main subjects, notable objects, setting, style/medium, dominant colors,
  and distinctive attributes (e.g. "black and white", "macro", "night").
- Tag only what is clearly visible. No abstract concepts, no duplicates.\
"""

_OCR_SYSTEM_PROMPT = """\
You are a text extraction (OCR) engine.

Respond with EXACTLY one JSON object and nothing else:
{
  "text": string,           // all legible text, reading order, "\\n" between blocks; "" if none
  "language": string|null,  // dominant language as ISO 639-1 ("en", "de"), null if no text
  "text_types": [string],   // any of: printed, handwritten, sign, screen, receipt, document; [] if none
  "legibility": string      // full | partial | none -- "partial" if some text is unreadable
}

Rules:
- Transcribe verbatim: preserve casing, punctuation, and numbers exactly.
- Do not translate, correct spelling, or expand abbreviations.
- For tables or columns, read left-to-right, top-to-bottom, one line per row.
- Mark unreadable spans as [illegible] rather than guessing.\
"""

BUILTIN_TASKS: dict[str, Task] = {
    "label": Task(
        name="label",
        description="Structured taxonomy labels: category, objects, colors, style, mood, quality, confidence.",
        system_prompt=_LABEL_SYSTEM_PROMPT,
        user_prompt="Label this image following the schema.",
        expects_json=True,
        required_keys=("category", "subcategory", "description", "objects", "confidence"),
        sampler="vlm-extract",
        max_tokens=1024,
    ),
    "caption": Task(
        name="caption",
        description="Dense natural-language caption (training-data style), single paragraph.",
        system_prompt=_CAPTION_SYSTEM_PROMPT,
        user_prompt="Caption this image.",
        expects_json=False,
        sampler="vlm-describe",
        max_tokens=512,
    ),
    "tags": Task(
        name="tags",
        description="Flat keyword tags for search/filtering: {\"tags\": [...]}.",
        system_prompt=_TAGS_SYSTEM_PROMPT,
        user_prompt="Tag this image.",
        expects_json=True,
        required_keys=("tags",),
        sampler="vlm-extract",
        max_tokens=512,
    ),
    "ocr": Task(
        name="ocr",
        description="Verbatim text extraction with language and legibility metadata.",
        system_prompt=_OCR_SYSTEM_PROMPT,
        user_prompt="Extract all text from this image.",
        expects_json=True,
        required_keys=("text", "legibility"),
        sampler="vlm-extract",
        max_tokens=2048,
    ),
}


def get_task(name: str) -> Task:
    """Look up a built-in task; KeyError lists valid names."""
    if name not in BUILTIN_TASKS:
        raise KeyError(
            f"unknown task {name!r}; built-ins: {', '.join(sorted(BUILTIN_TASKS))}"
        )
    return BUILTIN_TASKS[name]


_TASK_FIELDS = {f.name for f in fields(Task)}
_REQUIRED_FIELDS = ("name", "description", "system_prompt")


def load_task_file(path: Path | str) -> Task:
    """Load a custom task from a TOML file with a [task] table.

    Unknown keys are an error (catches typos silently changing behavior).
    """
    path = Path(path)
    with path.open("rb") as f:
        data = tomllib.load(f)

    table = data.get("task")
    if not isinstance(table, dict):
        raise ValueError(f"{path}: expected a [task] table")

    unknown = set(table) - _TASK_FIELDS
    if unknown:
        raise ValueError(
            f"{path}: unknown task keys: {', '.join(sorted(unknown))}; "
            f"valid: {', '.join(sorted(_TASK_FIELDS))}"
        )
    missing = [k for k in _REQUIRED_FIELDS if not table.get(k)]
    if missing:
        raise ValueError(f"{path}: missing required task keys: {', '.join(missing)}")

    if "required_keys" in table:
        table["required_keys"] = tuple(table["required_keys"])
    return Task(**table)


def missing_required_keys(parsed: object, task: Task) -> list[str]:
    """Which of the task's required keys are absent from a parsed label.

    A non-dict parse (or None) is missing everything.
    """
    if not task.required_keys:
        return []
    if not isinstance(parsed, dict):
        return list(task.required_keys)
    return [k for k in task.required_keys if k not in parsed]
