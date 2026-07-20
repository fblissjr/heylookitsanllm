"""Tests for CLI: arg parsing, task/options resolution, settings echo."""

from rich.console import Console

from batch_labeler.cli import (
    _build_parser,
    _resolve_options,
    _resolve_task,
    _settings_echo,
)
from batch_labeler.tasks import get_task


def parse(argv):
    return _build_parser().parse_args(argv)


def quiet_console():
    return Console(quiet=True)


class TestParser:
    def test_run_defaults(self):
        args = parse(["run", "imgs"])
        assert args.image_dir == "imgs"
        assert args.task == "label"
        assert args.output == "results.jsonl"
        assert args.recursive is True
        assert args.enable_thinking is None
        assert args.retries == 2

    def test_think_flags(self):
        assert parse(["run", "imgs", "--think"]).enable_thinking is True
        assert parse(["run", "imgs", "--no-think"]).enable_thinking is False

    def test_try_takes_single_image(self):
        args = parse(["try", "photo.jpg", "--model", "m"])
        assert args.image == "photo.jpg"
        assert args.model == "m"

    def test_vision_knobs(self):
        args = parse([
            "run", "imgs", "--vision-tokens", "1024",
            "--resize-max", "768", "--image-quality", "90",
        ])
        assert args.vision_tokens == 1024
        assert args.resize_max == 768
        assert args.image_quality == 90


class TestResolveTask:
    def test_default_label_task(self):
        args = parse(["run", "imgs"])
        task = _resolve_task(args, quiet_console())
        assert task is not None and task.name == "label"

    def test_system_prompt_override(self):
        args = parse(["run", "imgs", "--system-prompt", "custom sys"])
        task = _resolve_task(args, quiet_console())
        assert task is not None
        assert task.system_prompt == "custom sys"
        # Other task fields survive the override
        assert task.name == "label"
        assert task.required_keys

    def test_user_prompt_override(self):
        args = parse(["run", "imgs", "--user-prompt", "look closely"])
        task = _resolve_task(args, quiet_console())
        assert task is not None and task.user_prompt == "look closely"

    def test_unknown_task_returns_none(self):
        args = parse(["run", "imgs", "--task", "nope"])
        assert _resolve_task(args, quiet_console()) is None

    def test_task_file_wins_over_task_name(self, tmp_path):
        f = tmp_path / "t.toml"
        f.write_text(
            '[task]\nname = "custom"\ndescription = "d"\nsystem_prompt = "s"\n'
        )
        args = parse(["run", "imgs", "--task", "ocr", "--task-file", str(f)])
        task = _resolve_task(args, quiet_console())
        assert task is not None and task.name == "custom"


class TestResolveOptions:
    def test_task_defaults_apply(self):
        args = parse(["run", "imgs"])
        task = get_task("label")
        opts = _resolve_options(args, task)
        assert opts.preset == task.preset
        assert opts.max_tokens == task.max_tokens
        assert opts.temperature is None

    def test_cli_overrides_task(self):
        args = parse([
            "run", "imgs", "--preset", "thinking", "--max-tokens", "99",
            "--temperature", "0.5",
        ])
        opts = _resolve_options(args, get_task("label"))
        assert opts.preset == "thinking"
        assert opts.max_tokens == 99
        assert opts.temperature == 0.5

    def test_settings_echo_drops_none(self):
        args = parse(["run", "imgs", "--seed", "7"])
        task = get_task("caption")
        opts = _resolve_options(args, task)
        settings = _settings_echo("m", task, opts)
        assert settings["model"] == "m"
        assert settings["task"] == "caption"
        assert settings["seed"] == 7
        assert "temperature" not in settings
        assert "vision_tokens" not in settings
