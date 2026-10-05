"""The frontend's colour tokens are light-dark() pairs that meet WCAG 2.

`frontend/css/app.css` `:root` is the one place colour lives: every colour
token is a `light-dark(light, dark)` pair (so the two themes cannot drift
apart) and no colour literal appears outside that block. This test parses
the block, converts each OKLCH value to sRGB luminance, and recomputes the
contrast table from docs/project/plan_dark_mode.md (Phase 1) for BOTH
themes. Why a test and not a note: a hand-copied contrast table is a defect
with a delay (AGENTS.md "Derive, never hand-copy").

Known light-theme misses are listed by name, each tied to the plan's open
question that would close it; dark has none.

`frontend/index.html` cannot read a CSS variable, so its `theme-color` metas
are a hex copy of `--surface`. The last test derives the hex from the token,
so the copy cannot drift from it.
"""
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CSS = ROOT / "frontend" / "css" / "app.css"
INDEX = ROOT / "frontend" / "index.html"

TEXT_FLOOR = 4.5      # WCAG 2 AA, body text
NON_TEXT_FLOOR = 3.0  # WCAG 2 AA, UI component borders and placeholders

# (foreground, background, floor). The plan's Phase 1 table, plus the light
# pairs DESIGN.md section 1 has always claimed.
PAIRS = [
    ("ink", "bg", TEXT_FLOOR),
    ("ink", "surface", TEXT_FLOOR),
    ("ink", "surface-2", TEXT_FLOOR),
    ("ink", "brand-tint", TEXT_FLOOR),
    ("ink-muted", "bg", TEXT_FLOOR),
    ("ink-muted", "surface", TEXT_FLOOR),
    ("ink-muted", "surface-2", TEXT_FLOOR),
    ("accent", "bg", TEXT_FLOOR),
    ("accent", "surface", TEXT_FLOOR),
    ("accent", "brand-tint", TEXT_FLOOR),
    ("on-accent", "accent", TEXT_FLOOR),
    ("on-accent", "accent-hover", TEXT_FLOOR),
    ("on-accent", "danger", TEXT_FLOOR),
    ("danger", "bg", TEXT_FLOOR),
    ("danger", "danger-tint", TEXT_FLOOR),
    ("warn", "warn-tint", TEXT_FLOOR),
    ("line-strong", "bg", NON_TEXT_FLOOR),
    ("line-strong", "surface", NON_TEXT_FLOOR),
    ("ink-faint", "bg", NON_TEXT_FLOOR),
]

# Light pairs under their floor, kept by decision (plan_dark_mode.md open
# question 2); the dark theme was designed to the floors and gets no exemptions.
KNOWN_LIGHT_MISSES = {
    ("warn", "warn-tint"): "plan_dark_mode.md open question 2 (light warn hue)",
}


def _strip_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


def _root_block(css: str) -> str:
    m = re.search(r":root\s*\{(.*?)\}", css, flags=re.S)
    assert m, "app.css has no :root block"
    return m.group(1)


def _oklch_to_linear_srgb(value: str) -> tuple[float, float, float]:
    """CSS oklch(L C H [/ a]) -> linear sRGB, clipped to gamut."""
    m = re.fullmatch(r"oklch\(\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(?:/.*)?\)", value.strip())
    assert m, f"not a bare oklch() value: {value!r}"
    L, C, h = (float(x) for x in m.groups())
    a, b = C * math.cos(math.radians(h)), C * math.sin(math.radians(h))
    l_ = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m_ = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s_ = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    r = 4.0767416621 * l_ - 3.3077115913 * m_ + 0.2309699292 * s_
    g = -1.2684380046 * l_ + 2.6097574011 * m_ - 0.3413193965 * s_
    bl = -0.0041960863 * l_ - 0.7034186147 * m_ + 1.7076147010 * s_
    clip = lambda x: min(1.0, max(0.0, x))  # noqa: E731
    return clip(r), clip(g), clip(bl)


def _oklch_to_luminance(value: str) -> float:
    """CSS oklch(L C H [/ a]) -> WCAG relative luminance."""
    r, g, bl = _oklch_to_linear_srgb(value)
    return 0.2126 * r + 0.7152 * g + 0.0722 * bl


def _oklch_to_hex(value: str) -> str:
    """CSS oklch(L C H) -> the #rrggbb a browser paints for it."""
    encode = lambda x: 12.92 * x if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055  # noqa: E731
    return "#" + "".join(f"{round(encode(x) * 255):02x}" for x in _oklch_to_linear_srgb(value))


def _contrast(fg: float, bg: float) -> float:
    hi, lo = max(fg, bg), min(fg, bg)
    return (hi + 0.05) / (lo + 0.05)


def _tokens() -> dict[str, dict[str, str]]:
    """name -> {"light": oklch(...), "dark": oklch(...)} for every colour token."""
    block = _root_block(_strip_comments(CSS.read_text()))
    out: dict[str, dict[str, str]] = {}
    for name, raw in re.findall(r"--([\w-]+)\s*:\s*([^;]+);", block):
        if "oklch(" not in raw:
            continue  # fonts, sizes, timings
        values = re.findall(r"oklch\([^)]*\)", raw)
        if raw.strip().startswith("light-dark("):
            assert len(values) == 2, f"--{name}: light-dark() needs two colours: {raw!r}"
            out[name] = {"light": values[0], "dark": values[1]}
        else:
            assert len(values) == 1, f"--{name}: {raw!r}"
            out[name] = {"light": values[0], "dark": values[0]}
    return out


TOKENS = _tokens()
THEMES = ("light", "dark")


def _more_contrast_tokens() -> dict[str, dict[str, str]]:
    """TOKENS with the `@media (prefers-contrast: more)` :root block laid over it."""
    css = _strip_comments(CSS.read_text())
    m = re.search(r"@media \(prefers-contrast: more\)\s*\{\s*:root\s*\{(.*?)\}", css, flags=re.S)
    assert m, "app.css has no prefers-contrast: more block (plan_dark_mode.md Phase 3)"
    out = {k: dict(v) for k, v in TOKENS.items()}
    for name, raw in re.findall(r"--([\w-]+)\s*:\s*([^;]+);", m.group(1)):
        values = re.findall(r"oklch\([^)]*\)", raw)
        assert raw.strip().startswith("light-dark(") and len(values) == 2, f"--{name} in the contrast block: {raw!r}"
        out[name] = {"light": values[0], "dark": values[1]}
    return out


MORE_CONTRAST = _more_contrast_tokens()


def test_every_colour_token_is_a_light_dark_pair():
    # --brand is the seed and is the same in both themes by design.
    block = _root_block(_strip_comments(CSS.read_text()))
    bare = [n for n, raw in re.findall(r"--([\w-]+)\s*:\s*([^;]+);", block)
            if "oklch(" in raw and not raw.strip().startswith("light-dark(") and n != "brand"]
    assert not bare, f"colour tokens with one value only (dark would inherit light): {bare}"
    assert "color-scheme: light dark" in block, "the theme must follow the system"


def test_no_colour_literal_outside_root():
    # Every `:root { ... }` block is a token block (the base one and the
    # prefers-contrast overlay); colour may live in those and nowhere else.
    css = _strip_comments(CSS.read_text())
    rest = re.sub(r":root\s*\{.*?\}", "", css, flags=re.S)
    literals = re.findall(r"oklch\([^)]*\)|#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)|hsla?\([^)]*\)", rest)
    assert not literals, f"colour literals outside :root (make them tokens): {literals}"


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fg,bg,floor", PAIRS, ids=lambda x: str(x))
def test_pair_meets_floor(theme, fg, bg, floor):
    for name in (fg, bg):
        assert name in TOKENS, f"--{name} is not a colour token in :root"
    ratio = _contrast(_oklch_to_luminance(TOKENS[fg][theme]),
                      _oklch_to_luminance(TOKENS[bg][theme]))
    if theme == "light" and (fg, bg) in KNOWN_LIGHT_MISSES:
        assert ratio < floor, (
            f"light {fg} on {bg} now passes at {ratio:.2f}:1; drop it from KNOWN_LIGHT_MISSES "
            f"({KNOWN_LIGHT_MISSES[(fg, bg)]})")
        return
    assert ratio >= floor, f"{theme}: {fg} on {bg} is {ratio:.2f}:1, floor {floor}"


@pytest.mark.parametrize("theme", THEMES)
def test_theme_color_meta_is_the_surface_token(theme):
    # --surface is the bottom nav, the element that touches the browser's bar.
    html = INDEX.read_text()
    m = re.search(
        rf'<meta name="theme-color" media="\(prefers-color-scheme: {theme}\)" content="([^"]+)">', html)
    assert m, f"index.html has no theme-color meta for {theme}"
    want = _oklch_to_hex(TOKENS["surface"][theme])
    assert m.group(1).lower() == want, f"{theme} theme-color is {m.group(1)}, --surface is {want}"
    scheme = re.search(r'<meta name="color-scheme" content="([^"]+)">', html)
    assert scheme and scheme.group(1) == "light dark", "index.html must declare the scheme :root follows"
    assert html.index(scheme.group(0)) < html.index('rel="stylesheet"'), (
        "color-scheme must come before the stylesheet, or the first paint is light")


@pytest.mark.parametrize("theme", THEMES)
@pytest.mark.parametrize("fg,bg,floor", PAIRS, ids=lambda x: str(x))
def test_pair_meets_floor_under_increase_contrast(theme, fg, bg, floor):
    """Under prefers-contrast: more every pair holds at least its normal ratio
    and still meets its floor, so the overlay can only raise contrast. The one
    light exemption (warn on warn-tint) is untouched by the overlay and keeps
    its known value."""
    base = _contrast(_oklch_to_luminance(TOKENS[fg][theme]), _oklch_to_luminance(TOKENS[bg][theme]))
    more = _contrast(_oklch_to_luminance(MORE_CONTRAST[fg][theme]), _oklch_to_luminance(MORE_CONTRAST[bg][theme]))
    assert more >= base - 1e-9, f"{theme}: {fg} on {bg} fell from {base:.2f} to {more:.2f} under more contrast"
    if theme == "light" and (fg, bg) in KNOWN_LIGHT_MISSES:
        return
    assert more >= floor, f"{theme} (more contrast): {fg} on {bg} is {more:.2f}:1, floor {floor}"
