# tests/unit/test_toml_comment_preservation.py
"""models.toml comment carry-forward (toml_comments.merge_comments).

Every admin write regenerates the file through tomli_w, which emits no
comments. merge_comments copies the old file's comments onto the fresh
render -- but ONLY while their anchor is unchanged, so a note can never
outlive what it describes. These tests pin:

- values/layout stay tomli_w's (old formatting is never spliced in);
- untouched models keep every comment shape (block above the header,
  inline on a key, standalone inside the body, above/inside sub-tables,
  trailing block above the next header);
- a changed model drops ALL its comments; a changed root key drops its
  comment; the block above the next model's header drops when THAT model
  changes;
- the merge is best-effort: any failure or value drift degrades to the
  comment-less render, never a refused or corrupted write.
"""

import textwrap
import tomllib

import tomli_w
import pytest

from heylook_llm.model_service import ModelService
from heylook_llm.toml_comments import merge_comments


def _model(mid: str, path: str = "/w", **config) -> dict:
    return {
        "id": mid,
        "provider": "mlx",
        "enabled": True,
        "config": {"model_path": path, **config},
    }


def _render(data: dict) -> str:
    return tomli_w.dumps(data)


OLD = textwrap.dedent("""\
    # banner: what this file is
    default_model = "a"  # routes here
    max_loaded_models = 1

    # --- section divider above first model ---
    [[models]]
    id = "a"  # the workhorse
    provider = "mlx"
    # standalone: why enabled
    enabled = true

    # about the config sub-table
    [models.config]
    model_path = "/w"  # local weights
    temperature = 0.7

    # trailing block: describes model b below
    [[models]]
    id = "b"
    provider = "mlx"
    enabled = true

    [models.config]
    model_path = "/w2"
    """)

OLD_DATA = tomllib.loads(OLD)


class TestUnchangedEverythingCarries:
    def test_identical_data_keeps_every_comment(self):
        merged = merge_comments(OLD, _render(OLD_DATA))
        for line in (
            "# banner: what this file is",
            "# routes here",
            "# --- section divider above first model ---",
            "# the workhorse",
            "# standalone: why enabled",
            "# about the config sub-table",
            "# local weights",
            "# trailing block: describes model b below",
        ):
            assert line in merged, f"lost: {line}"

    def test_merged_values_identical_to_fresh_render(self):
        fresh = _render(OLD_DATA)
        merged = merge_comments(OLD, fresh)
        assert tomllib.loads(merged) == tomllib.loads(fresh)

    def test_layout_is_tomli_w_not_old_file(self):
        # The old file could be hand-formatted (indented headers, inline
        # config tables); values must still come out in tomli_w's layout.
        hand = textwrap.dedent("""\
            default_model = "a"

            [[models]]
            id = "a"  # keep me
            provider = "mlx"
            enabled = true
            config = { model_path = "/w", temperature = 0.7 }
            """)
        data = tomllib.loads(hand)
        merged = merge_comments(hand, _render(data))
        assert "[models.config]" in merged, "tomli_w layout is authoritative"
        assert "config = {" not in merged
        assert "# keep me" in merged

    def test_comment_position_is_preserved(self):
        merged = merge_comments(OLD, _render(OLD_DATA)).splitlines()
        divider = merged.index("# --- section divider above first model ---")
        assert merged[divider + 1] == "[[models]]"
        trailing = merged.index("# trailing block: describes model b below")
        assert merged[trailing + 1] == "[[models]]"
        standalone = merged.index("# standalone: why enabled")
        assert merged[standalone + 1] == "enabled = true"


class TestChangedAnchorsDrop:
    def test_patched_model_drops_its_comments_others_survive(self):
        data = tomllib.loads(OLD)
        data["models"][0]["config"]["temperature"] = 0.2  # patch model a
        merged = merge_comments(OLD, _render(data))
        for gone in ("# the workhorse", "# standalone: why enabled",
                     "# about the config sub-table", "# local weights"):
            assert gone not in merged, f"comment outlived its model: {gone}"
        # Root keys unchanged -> their comments stay.
        assert "# banner: what this file is" in merged
        assert "# routes here" in merged

    def test_divider_above_first_model_drops_when_it_changes(self):
        data = tomllib.loads(OLD)
        data["models"][0]["enabled"] = False
        merged = merge_comments(OLD, _render(data))
        assert "# --- section divider above first model ---" not in merged

    def test_trailing_block_drops_when_the_next_model_changes(self):
        # The block sits above model b's header; it describes b, even though
        # TOML-structurally it lives at the end of model a's section.
        data = tomllib.loads(OLD)
        data["models"][1]["config"]["model_path"] = "/moved"
        merged = merge_comments(OLD, _render(data))
        assert "# trailing block: describes model b below" not in merged
        # Model a untouched -> its own comments survive.
        assert "# the workhorse" in merged

    def test_changed_root_key_drops_only_its_comment(self):
        data = tomllib.loads(OLD)
        data["default_model"] = "b"
        merged = merge_comments(OLD, _render(data))
        assert "# routes here" not in merged
        # The banner is anchored to default_model too -- it drops with it.
        assert "# banner: what this file is" not in merged
        assert "# the workhorse" in merged

    def test_removed_model_takes_its_comments_along(self):
        data = tomllib.loads(OLD)
        del data["models"][0]
        merged = merge_comments(OLD, _render(data))
        assert "# the workhorse" not in merged
        assert "# local weights" not in merged
        # b changed neighbors, and the trailing block's anchor pair broke.
        assert "# trailing block: describes model b below" not in merged

    def test_added_model_carries_nothing_and_breaks_nothing(self):
        data = tomllib.loads(OLD)
        data["models"].append(_model("c", "/w3"))
        merged = merge_comments(OLD, _render(data))
        assert "# the workhorse" in merged
        assert tomllib.loads(merged) == data


class TestBestEffortNeverBlocks:
    def test_unparseable_old_text_returns_fresh_render(self):
        fresh = _render(OLD_DATA)
        assert merge_comments("not [ valid { toml", fresh) == fresh

    def test_commentless_old_text_is_a_noop(self):
        fresh = _render(OLD_DATA)
        assert merge_comments(fresh, fresh) == fresh

    def test_merge_never_changes_parsed_values(self, monkeypatch):
        # Even if injection misplaced a line, the value-equality gate must
        # refuse the merged text rather than write drifted values.
        import heylook_llm.toml_comments as tc

        def bad_merge(*args):
            return args[1].replace('id = "a"', 'id = "mangled"')

        monkeypatch.setattr(tc, "_merge", bad_merge)
        fresh = _render(OLD_DATA)
        assert tc.merge_comments(OLD, fresh) == fresh


class TestThroughModelService:
    @pytest.fixture
    def config_path(self, tmp_path):
        weights_a = tmp_path / "weights" / "a"
        weights_b = tmp_path / "weights" / "b"
        weights_a.mkdir(parents=True)
        weights_b.mkdir(parents=True)
        text = OLD.replace('"/w2"', f'"{weights_b}"').replace('"/w"', f'"{weights_a}"')
        p = tmp_path / "models.toml"
        p.write_text(text)
        return p

    def test_admin_patch_keeps_other_models_comments(self, config_path):
        service = ModelService(str(config_path))
        service.update_config("b", {"config": {"temperature": 0.5}})
        text = config_path.read_text()
        assert "# banner: what this file is" in text
        assert "# the workhorse" in text
        assert "# local weights" in text
        # The patched model's neighborhood note drops with it.
        assert "# trailing block: describes model b below" not in text
        assert 'temperature = 0.5' in text

    def test_second_patch_still_carries(self, config_path):
        # Comments must survive REPEATED rewrites, not just the first.
        service = ModelService(str(config_path))
        service.update_config("b", {"config": {"temperature": 0.5}})
        service.update_config("b", {"config": {"temperature": 0.6}})
        text = config_path.read_text()
        assert "# the workhorse" in text
        assert "# standalone: why enabled" in text

    def test_written_file_is_valid_and_parses_to_patched_values(self, config_path):
        service = ModelService(str(config_path))
        service.update_config("a", {"config": {"temperature": 0.1}})
        data = tomllib.loads(config_path.read_text())
        assert data["models"][0]["config"]["temperature"] == 0.1
