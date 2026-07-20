"""Tests for task templates: built-ins, TOML loading, validation."""

import pytest

from batch_labeler.tasks import (
    BUILTIN_TASKS,
    Task,
    get_task,
    load_task_file,
    missing_required_keys,
)


class TestBuiltins:
    def test_expected_builtins_present(self):
        for name in ("label", "caption", "tags", "ocr"):
            assert name in BUILTIN_TASKS

    def test_all_builtins_have_prompts(self):
        for task in BUILTIN_TASKS.values():
            assert task.system_prompt.strip()
            assert task.user_prompt.strip()
            assert task.description.strip()

    def test_label_is_structured(self):
        task = BUILTIN_TASKS["label"]
        assert task.expects_json
        assert "category" in task.required_keys
        assert "description" in task.required_keys

    def test_caption_is_freeform(self):
        task = BUILTIN_TASKS["caption"]
        assert not task.expects_json
        assert task.required_keys == ()

    def test_json_tasks_state_json_only_in_prompt(self):
        for task in BUILTIN_TASKS.values():
            if task.expects_json:
                assert "JSON" in task.system_prompt

    def test_get_task_by_name(self):
        assert get_task("label").name == "label"

    def test_get_task_unknown_raises(self):
        with pytest.raises(KeyError) as exc:
            get_task("nope")
        # Error should list valid names to be helpful
        assert "label" in str(exc.value)


class TestLoadTaskFile:
    def test_load_minimal(self, tmp_path):
        f = tmp_path / "mytask.toml"
        f.write_text(
            '[task]\nname = "my-task"\ndescription = "d"\n'
            'system_prompt = "Respond with JSON."\n'
        )
        task = load_task_file(f)
        assert task.name == "my-task"
        assert task.system_prompt == "Respond with JSON."
        assert task.expects_json  # default
        assert task.user_prompt  # falls back to a default

    def test_load_full(self, tmp_path):
        f = tmp_path / "full.toml"
        f.write_text(
            "[task]\n"
            'name = "full"\n'
            'description = "d"\n'
            'system_prompt = "sys"\n'
            'user_prompt = "usr"\n'
            "expects_json = false\n"
            'required_keys = ["a", "b"]\n'
            'preset = "vlm-extract"\n'
            "max_tokens = 512\n"
        )
        task = load_task_file(f)
        assert task.user_prompt == "usr"
        assert not task.expects_json
        assert task.required_keys == ("a", "b")
        assert task.preset == "vlm-extract"
        assert task.max_tokens == 512

    def test_missing_required_field_raises(self, tmp_path):
        f = tmp_path / "bad.toml"
        f.write_text('[task]\nname = "x"\n')
        with pytest.raises(ValueError):
            load_task_file(f)

    def test_unknown_key_raises(self, tmp_path):
        f = tmp_path / "bad2.toml"
        f.write_text(
            '[task]\nname = "x"\ndescription = "d"\nsystem_prompt = "s"\ntypo_key = 1\n'
        )
        with pytest.raises(ValueError) as exc:
            load_task_file(f)
        assert "typo_key" in str(exc.value)


class TestValidation:
    def test_all_present(self):
        task = Task(
            name="t", description="d", system_prompt="s",
            required_keys=("a", "b"),
        )
        assert missing_required_keys({"a": 1, "b": 2, "c": 3}, task) == []

    def test_missing_reported(self):
        task = Task(
            name="t", description="d", system_prompt="s",
            required_keys=("a", "b"),
        )
        assert missing_required_keys({"a": 1}, task) == ["b"]

    def test_non_dict_reports_all(self):
        task = Task(
            name="t", description="d", system_prompt="s",
            required_keys=("a",),
        )
        assert missing_required_keys(["not", "a", "dict"], task) == ["a"]

    def test_no_requirements(self):
        task = Task(name="t", description="d", system_prompt="s")
        assert missing_required_keys(None, task) == []
