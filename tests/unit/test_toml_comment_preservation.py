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
    # Identical data keeps every comment shape, each directly above its anchor.
    # Rows map a comment to the line that must follow it (None = presence only;
    # inline comments such as "# routes here" are checked as substrings).
    @pytest.mark.parametrize("expected", [
        pytest.param({
            "# banner: what this file is": None,
            "# routes here": None,
            "# --- section divider above first model ---": None,
            "# the workhorse": None,
            "# standalone: why enabled": None,
            "# about the config sub-table": None,
            "# local weights": None,
            "# trailing block: describes model b below": None,
        }, id="identical_data_keeps_every_comment"),
        pytest.param({
            "# --- section divider above first model ---": "[[models]]",
            "# trailing block: describes model b below": "[[models]]",
            "# standalone: why enabled": "enabled = true",
        }, id="comment_position_is_preserved"),
    ])
    def test_unchanged_data_keeps_comments_in_place(self, expected):
        merged = merge_comments(OLD, _render(OLD_DATA))
        lines = merged.splitlines()
        for comment, next_line in expected.items():
            assert comment in merged, f"lost: {comment}"
            if next_line is not None:
                assert lines[lines.index(comment) + 1] == next_line

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


def _patch_model_a(data):
    data["models"][0]["config"]["temperature"] = 0.2


def _disable_model_a(data):
    data["models"][0]["enabled"] = False


def _move_model_b(data):
    data["models"][1]["config"]["model_path"] = "/moved"


def _change_root_key(data):
    data["default_model"] = "b"


def _remove_model_a(data):
    del data["models"][0]


def _add_model_c(data):
    data["models"].append(_model("c", "/w3"))


# Edit -> comments that must drop / must survive. Every row's merged text also
# parses to exactly the edited data.
# - patched: a changed model drops ALL its comments; unchanged root keys keep
#   theirs.
# - trailing: the block sits above model b's header; it describes b, even
#   though TOML-structurally it lives at the end of model a's section. Model a
#   is untouched, so its own comments survive.
# - root key: the banner is anchored to default_model too -- it drops with it.
# - removed: b changed neighbours, so the trailing block's anchor pair broke.
_DROP_ROWS = [
    pytest.param(
        _patch_model_a,
        ("# the workhorse", "# standalone: why enabled",
         "# about the config sub-table", "# local weights"),
        ("# banner: what this file is", "# routes here"),
        id="patched_model_drops_its_comments_others_survive",
    ),
    pytest.param(_disable_model_a, ("# --- section divider above first model ---",), (),
                 id="divider_above_first_model_drops_when_it_changes"),
    pytest.param(_move_model_b, ("# trailing block: describes model b below",),
                 ("# the workhorse",),
                 id="trailing_block_drops_when_the_next_model_changes"),
    pytest.param(_change_root_key, ("# routes here", "# banner: what this file is"),
                 ("# the workhorse",),
                 id="changed_root_key_drops_only_its_comment"),
    pytest.param(_remove_model_a,
                 ("# the workhorse", "# local weights",
                  "# trailing block: describes model b below"), (),
                 id="removed_model_takes_its_comments_along"),
    pytest.param(_add_model_c, (), ("# the workhorse",),
                 id="added_model_carries_nothing_and_breaks_nothing"),
]


class TestChangedAnchorsDrop:
    @pytest.mark.parametrize("edit, must_drop, must_keep", _DROP_ROWS)
    def test_changed_anchor_drops_its_comments(self, edit, must_drop, must_keep):
        data = tomllib.loads(OLD)
        edit(data)
        merged = merge_comments(OLD, _render(data))
        for gone in must_drop:
            assert gone not in merged, f"comment outlived its anchor: {gone}"
        for kept in must_keep:
            assert kept in merged, f"lost: {kept}"
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

    # Admin patches through ModelService. Rows: (patches applied in order,
    # comments that must survive, comments that must drop, (model index,
    # temperature) the written file must parse to and render).
    # - admin_patch: the patched model's neighbourhood note drops with it.
    # - second_patch: comments must survive REPEATED rewrites, not just the
    #   first.
    @pytest.mark.parametrize("patches, must_keep, must_drop, expected", [
        pytest.param(
            [("b", 0.5)],
            ("# banner: what this file is", "# the workhorse", "# local weights"),
            ("# trailing block: describes model b below",),
            (1, 0.5),
            id="admin_patch_keeps_other_models_comments",
        ),
        pytest.param(
            [("b", 0.5), ("b", 0.6)],
            ("# the workhorse", "# standalone: why enabled"),
            (),
            (1, 0.6),
            id="second_patch_still_carries",
        ),
        pytest.param([("a", 0.1)], (), (), (0, 0.1),
                     id="written_file_is_valid_and_parses_to_patched_values"),
    ])
    def test_admin_patch_through_the_service(
        self, config_path, patches, must_keep, must_drop, expected
    ):
        service = ModelService(str(config_path))
        for model_id, temperature in patches:
            service.update_config(model_id, {"config": {"temperature": temperature}})
        text = config_path.read_text()
        for kept in must_keep:
            assert kept in text, f"lost: {kept}"
        for gone in must_drop:
            assert gone not in text, f"comment outlived its anchor: {gone}"
        index, temperature = expected
        assert f"temperature = {temperature}" in text
        data = tomllib.loads(text)
        assert data["models"][index]["config"]["temperature"] == temperature
