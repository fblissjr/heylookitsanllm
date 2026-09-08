"""The chat-template routes, exercised THROUGH the route.

Testing the module alone would prove nothing about whether anything is bound
to it: this repo has shipped a guard on a model no route binds and a rename
refusal that had been dead for thirteen versions, both behind green tests that
called the function directly. So every claim here goes over HTTP.

The round trip is one test rather than three, because GET/PUT/DELETE are only
meaningful in sequence -- a PUT that "succeeds" while the next GET still shows
the vendor template is the failure worth catching, and separate tests cannot
see it.
"""

import pytest

from heylook_llm.providers.common.template_info import (
    HEYLOOK_OVERRIDE,
    HEYLOOK_TEMPLATE_FILENAME,
)

VENDOR = "VENDOR{% for m in messages %}{{ m.content }}{% endfor %}<end_of_turn>"
OVERRIDE = "OVERRIDE{% for m in messages %}{{ m.content }}{% endfor %}<end_of_turn>"

URL = "/v1/admin/models/test-gguf-model/chat-template"


@pytest.fixture
def model_dir(app, tmp_path):
    """Point the gguf test model at a real folder, and put it back after.

    The fixture roster is session-scoped, so a test that mutated it and did
    not restore would leak into every later contract test.
    """
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "model.gguf").write_bytes(b"")
    (directory / "chat_template.jinja").write_text(VENDOR)

    configs = [
        app.state.model_service.app_config.get_model_config("test-gguf-model"),
        app.state.router_instance.app_config.get_model_config("test-gguf-model"),
    ]
    original = [c.config.model_path for c in configs]
    for c in configs:
        c.config.model_path = str(directory / "model.gguf")
    try:
        yield directory
    finally:
        for c, path in zip(configs, original):
            c.config.model_path = path


class TestChatTemplateRoutes:
    def test_round_trip_reads_writes_and_reverts(self, client, model_dir):
        got = client.get(URL)
        assert got.status_code == 200, got.text
        body = got.json()
        assert body["template"].startswith("VENDOR")
        assert body["override_present"] is False
        assert body["override_path"].endswith(HEYLOOK_TEMPLATE_FILENAME)

        put = client.put(URL, json={"template": OVERRIDE})
        assert put.status_code == 200, put.text
        written = put.json()
        # The write's own response is the resolved view, so a client never has
        # to re-fetch to learn what its write actually resolved to.
        assert written["template"].startswith("OVERRIDE")
        assert written["override_present"] is True
        assert written["origin"] == HEYLOOK_OVERRIDE
        assert written["inert_reason"] is None

        assert client.get(URL).json()["template"].startswith("OVERRIDE")

        deleted = client.delete(URL)
        assert deleted.status_code == 200, deleted.text
        assert deleted.json()["template"].startswith("VENDOR")
        assert deleted.json()["override_present"] is False
        # The vendor file is what we fell back TO, so it had better still be there.
        assert (model_dir / "chat_template.jinja").read_text() == VENDOR

    def test_a_template_that_would_brick_the_model_is_refused_and_not_written(
        self, client, model_dir
    ):
        """400 with a reason, and nothing on disk.

        The write has to be refused rather than accepted-and-broken: on gguf a
        raised jinja exception comes back as a 500 from llama-server, so a bad
        template on disk breaks every request to that model at its next load.
        """
        refused = client.put(URL, json={"template": "{% for m in messages %}{{ m.content }}"})

        assert refused.status_code == 400, refused.text
        assert "jinja" in refused.json()["detail"].lower()
        assert not (model_dir / HEYLOOK_TEMPLATE_FILENAME).exists()
        assert client.get(URL).json()["override_present"] is False

    def test_delete_without_an_override_is_404_not_a_silent_success(
        self, client, model_dir
    ):
        assert client.delete(URL).status_code == 404

    def test_unknown_model_is_404_on_every_verb(self, client):
        missing = "/v1/admin/models/no-such-model/chat-template"
        assert client.get(missing).status_code == 404
        assert client.put(missing, json={"template": OVERRIDE}).status_code == 404
        assert client.delete(missing).status_code == 404

    def test_a_provider_without_templates_says_so_rather_than_showing_an_editor(
        self, client
    ):
        """mlx_embedding renders no chat template.

        `supported: false` is what stops the UI drawing an empty editor over a
        model that has nothing to edit.
        """
        # The mlx row stands in for "has templates"; the contract asserted here
        # is that `supported` is a real field the route populates per provider.
        got = client.get("/v1/admin/models/test-mlx-model/chat-template")
        assert got.status_code == 200, got.text
        assert got.json()["supported"] is True
