"""Contract: the frontend's ``depthOffered`` (frontend/js/settings.js) agrees
with the server's ``thinking_controls.check_depth`` on every fixture template.

The frontend keeps its own copy of the rule in another language, to drop a
depth the selected model does not offer before it reaches the wire. The two
drifting means the panel either sends a value the server 400s, or hides one
the model takes. So the JS function itself runs under node, over ``detect()``
of every ``tests/fixtures/chat_templates/*.jinja``: each offered value, each
alias, a value another template offers but this one does not, and a made-up
one, plus a template with no depth control and one that could not be judged.

Not compared: a null value. The server accepts it (no depth requested), and
the frontend never asks about one.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from heylook_llm.thinking_controls import check_depth, detect

REPO = Path(__file__).resolve().parents[2]
SETTINGS_JS = REPO / "frontend" / "js" / "settings.js"
FIXTURES = sorted((REPO / "tests" / "fixtures" / "chat_templates").glob("*.jinja"))
MADE_UP = "zz-not-a-level"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _cases() -> list[tuple[str, str, dict | None]]:
    controls = {p.stem: detect(p.read_text()) for p in FIXTURES}
    every_value = {v for c in controls.values() if c and c.get("depth")
                   for v in c["depth"]["values"]}
    cases = []
    for name, c in controls.items():
        depth = (c or {}).get("depth")
        if not depth:
            cases.append((name, "high", c))
            continue
        offered = set(depth["values"]) | set(depth["aliases"])
        cases += [(name, v, c) for v in depth["values"]]
        cases += [(name, a, c) for a in depth["aliases"]]
        foreign = sorted(every_value - offered)
        if foreign:
            cases.append((name, foreign[0], c))
        cases.append((name, MADE_UP, c))
    cases.append(("<not judged>", "high", None))
    return cases


def test_depth_offered_matches_check_depth():
    cases = _cases()
    assert len(FIXTURES) >= 5 and any(c and c.get("depth") for _, _, c in cases), \
        "no fixture with a depth control: this check would compare nothing"
    script = (
        f"import {{ depthOffered }} from {json.dumps(SETTINGS_JS.as_uri())};\n"
        "let raw = ''; for await (const c of process.stdin) raw += c;\n"
        "const out = JSON.parse(raw).map(([v, t]) => depthOffered(v, t));\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    run = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        input=json.dumps([[v, c] for _, v, c in cases]),
        capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    js = json.loads(run.stdout)
    disagree = [(name, value, got, check_depth(value, c))
                for (name, value, c), got in zip(cases, js)
                if got != (check_depth(value, c) is None)]
    assert not disagree, (
        "frontend depthOffered disagrees with the server's check_depth "
        "(template, value, js says offered, server's refusal): " + repr(disagree))
