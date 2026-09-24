"""utils.load_image: every source decodes or raises; nothing becomes a
placeholder image (the red squares removed 2026-09-24)."""
import base64
import io

import pytest
from PIL import Image

from heylook_llm.utils import load_image


def _png_b64(wrap: bool) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 0, 255)).save(buf, format="PNG")
    raw = buf.getvalue()
    enc = base64.encodebytes(raw).decode() if wrap else base64.b64encode(raw).decode()
    return "data:image/png;base64," + enc


@pytest.mark.unit
@pytest.mark.parametrize("wrap", [False, True])
def test_base64_images_decode_wrapped_or_not(wrap):
    # encodebytes wraps at 76 columns; those requests decoded before and must still
    img = load_image(_png_b64(wrap))
    assert img.size == (8, 8) and img.getpixel((0, 0)) == (0, 0, 255)


@pytest.mark.unit
def test_an_unreadable_image_raises():
    with pytest.raises(Exception):
        load_image("data:image/png;base64,bm90IGFuIGltYWdl")
