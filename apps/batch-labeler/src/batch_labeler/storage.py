"""JSONL append-only storage with resume support for batch labeling results."""

from pathlib import Path

import orjson


def extract_json(text: str) -> str | None:
    """Try to extract valid JSON from model output, handling markdown fences.

    Returns re-serialized JSON string, or None if no valid JSON found.
    """
    text = text.strip()

    # Strip markdown code fences
    if text.startswith('```'):
        lines = text.split('\n')
        if lines[-1].strip() == '```':
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = '\n'.join(lines).strip()

    # Try parsing as-is
    try:
        parsed = orjson.loads(text)
        return orjson.dumps(parsed).decode()
    except (orjson.JSONDecodeError, ValueError, TypeError):
        pass

    # Try finding JSON object boundaries
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end > start:
        try:
            parsed = orjson.loads(text[start:end + 1])
            return orjson.dumps(parsed).decode()
        except (orjson.JSONDecodeError, ValueError, TypeError):
            pass

    return None


def load_processed(output_path: str) -> set[str]:
    """Read existing JSONL file, return set of file hashes for resume."""
    path = Path(output_path)
    if not path.exists():
        return set()

    hashes = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = orjson.loads(line)
            if h := record.get('file_hash'):
                hashes.add(h)
        except (orjson.JSONDecodeError, ValueError, TypeError):
            continue
    return hashes


def append_result(output_path: str, result: dict) -> None:
    """Append one JSON line to the output file."""
    with open(output_path, 'ab') as f:
        f.write(orjson.dumps(result))
        f.write(b'\n')
