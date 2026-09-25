"""Contract: apps/batch-labeler speaks the server's Messages wire.

The app has its own venv and cannot import heylook, so its own tests can only
check the shape it believes in. This one checks that belief against the
server's classes: a payload the app builds (from a real image, resized)
validates as a ``MessageCreateRequest``, and a response built from the
server's own ``MessageResponse`` parses back to the reply's text and thinking.
The app sat on the removed /v1/chat/completions route for months with its own
suite green; that is the gap this closes.
"""

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "apps" / "batch-labeler" / "src"))
from batch_labeler.client import (  # noqa: E402
    GenerationOptions, build_payload, image_block, parse_message_response)

from heylook_llm.schema.content_blocks import TextBlock, ThinkingBlock  # noqa: E402
from heylook_llm.schema.messages import MessageCreateRequest  # noqa: E402
from heylook_llm.schema.responses import MessageResponse, Usage  # noqa: E402


@pytest.mark.parametrize("thinking", [None, False, True])
def test_the_app_payload_is_a_valid_messages_request(tmp_path, thinking):
    path = tmp_path / "photo.jpg"
    Image.new("RGB", (3000, 2000), (90, 120, 40)).save(path, format="JPEG")
    payload = build_payload(
        model_id="some-vlm", system_prompt="label it", user_prompt="Label this image.",
        image=image_block(path, max_edge=1024),
        options=GenerationOptions(max_tokens=64, temperature=0.2, seed=1, thinking=thinking))

    req = MessageCreateRequest.model_validate(payload)
    assert req.system == "label it"
    image, text = req.messages[0].content
    assert (image.type, image.source_type, image.media_type) == ("image", "base64", "image/jpeg")
    assert image.data and text.text == "Label this image."
    assert req.thinking == thinking


def test_the_app_reads_the_server_response():
    response = MessageResponse(
        id="msg_1", model="some-vlm",
        content=[ThinkingBlock(thinking="looking"), TextBlock(text='{"category": "scene"}')],
        usage=Usage(input_tokens=900, output_tokens=12),
    ).model_dump(mode="json")
    parsed = parse_message_response(response)
    assert (parsed.content, parsed.thinking) == ('{"category": "scene"}', "looking")
    assert parsed.usage["output_tokens"] == 12
