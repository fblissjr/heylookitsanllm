"""Tests for storage module: JSON extraction, JSONL append, resume."""

import orjson

from batch_labeler.storage import extract_json, load_processed, append_result


class TestExtractJson:
    def test_valid_json(self):
        result = extract_json('{"key": "value"}')
        assert result is not None
        parsed = orjson.loads(result)
        assert parsed["key"] == "value"

    def test_markdown_fenced(self):
        text = '```json\n{"key": "value"}\n```'
        result = extract_json(text)
        assert result is not None
        assert orjson.loads(result)["key"] == "value"

    def test_json_with_surrounding_text(self):
        text = 'Here is the result:\n{"key": "value"}\nDone.'
        result = extract_json(text)
        assert result is not None
        assert orjson.loads(result)["key"] == "value"

    def test_malformed_returns_none(self):
        assert extract_json("not json at all") is None

    def test_empty_string(self):
        assert extract_json("") is None

    def test_nested_json(self):
        text = '{"outer": {"inner": 42}}'
        result = extract_json(text)
        assert result is not None
        parsed = orjson.loads(result)
        assert parsed["outer"]["inner"] == 42

    def test_markdown_fence_no_closing(self):
        text = '```json\n{"key": "value"}'
        result = extract_json(text)
        assert result is not None
        assert orjson.loads(result)["key"] == "value"


class TestLoadProcessed:
    def test_nonexistent_file(self, tmp_path):
        result = load_processed(str(tmp_path / "nope.jsonl"))
        assert result == set()

    def test_reads_hashes(self, tmp_path):
        f = tmp_path / "results.jsonl"
        lines = [
            orjson.dumps({"file_hash": "abc123", "file_name": "a.jpg"}).decode(),
            orjson.dumps({"file_hash": "def456", "file_name": "b.jpg"}).decode(),
        ]
        f.write_text('\n'.join(lines) + '\n')
        result = load_processed(str(f))
        assert result == {"abc123", "def456"}

    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text('{"file_hash": "abc"}\n\n{"file_hash": "def"}\n')
        result = load_processed(str(f))
        assert len(result) == 2

    def test_skips_malformed_lines(self, tmp_path):
        f = tmp_path / "results.jsonl"
        f.write_text('{"file_hash": "abc"}\nnot json\n{"file_hash": "def"}\n')
        result = load_processed(str(f))
        assert result == {"abc", "def"}


class TestAppendResult:
    def test_appends_line(self, tmp_path):
        f = tmp_path / "out.jsonl"
        append_result(str(f), {"file_hash": "abc", "label": {"tag": "cat"}})
        append_result(str(f), {"file_hash": "def", "label": {"tag": "dog"}})

        lines = f.read_text().strip().split('\n')
        assert len(lines) == 2
        assert orjson.loads(lines[0])["file_hash"] == "abc"
        assert orjson.loads(lines[1])["file_hash"] == "def"

    def test_resume_round_trip(self, tmp_path):
        f = tmp_path / "out.jsonl"
        append_result(str(f), {"file_hash": "hash1", "label": {}})
        append_result(str(f), {"file_hash": "hash2", "label": {}})

        hashes = load_processed(str(f))
        assert hashes == {"hash1", "hash2"}
