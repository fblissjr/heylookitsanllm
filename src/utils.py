# src/utils.py
# src/utils.py
import base64, io, requests
from PIL import Image, ImageOps
from typing import List, Tuple

# Why: We now import apply_chat_template from our local vendored copy.
from .vendored.mlx_vlm.prompt_utils import apply_chat_template

def load_image(source_str: str) -> Image.Image:
    if source_str.startswith("data:image"):
        try: header, encoded = source_str.split(",", 1); return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
        except: raise ValueError("Invalid Base64 image data.")
    elif source_str.startswith("http"):
        return Image.open(requests.get(source_str, stream=True).raw).convert("RGB")
    else: return ImageOps.exif_transpose(Image.open(source_str)).convert("RGB")

def process_vlm_messages(processor, model_config, messages: List) -> Tuple[List[Image.Image], list]:
    images, text_messages = [], []
    for msg in messages:
        if isinstance(msg.get('content'), list):
            text_parts = []
            for part in msg['content']:
                if part.get('type') == 'text': text_parts.append(part['text'])
                elif part.get('type') == 'image_url': images.append(load_image(part['image_url']['url']))
            text_messages.append({"role": msg['role'], "content": "".join(text_parts)})
        else:
            text_messages.append(msg)

    # Use the apply_chat_template function we imported from our local copy
    return images, apply_chat_template(processor, model_config, text_messages, num_images=len(images))
