"""One embedding item is one vector, including when it contains mixed media."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_ITEMS = 16
MAX_MEDIA_BYTES = 32 * 1024 * 1024            # image and audio blocks
MAX_VIDEO_BYTES = 256 * 1024 * 1024           # video blocks: windowing decodes one window at a time (v2.0.235)
MAX_REQUEST_MEDIA_BYTES = 256 * 1024 * 1024
MEDIA_BYTES_BY_TYPE = {"image": MAX_MEDIA_BYTES, "audio": MAX_MEDIA_BYTES, "video": MAX_VIDEO_BYTES}


MAX_TEXT_CHARS = 131072
MAX_BLOCKS_PER_ITEM = 32


def base64_chars(n_bytes):
    return (n_bytes + 2) // 3 * 4


def request_limits():
    """The request limits, as engine.embedding.limits reports them to clients.
    Built from the constants the validators use, so a client never copies them."""
    return {"items": MAX_ITEMS, "blocks_per_item": MAX_BLOCKS_PER_ITEM, "text_chars": MAX_TEXT_CHARS,
            "media_bytes": dict(MEDIA_BYTES_BY_TYPE), "request_media_bytes": MAX_REQUEST_MEDIA_BYTES}


class EmbeddingSource(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["base64", "url"]
    media_type: str
    data: str | None = Field(default=None, max_length=base64_chars(max(MEDIA_BYTES_BY_TYPE.values())))
    url: str | None = Field(default=None, max_length=8192)

    @model_validator(mode="after")
    def payload(self):
        from urllib.parse import urlsplit
        if self.type == "base64":
            if not self.data or self.url is not None:
                raise ValueError("base64 source requires data and no url")
        elif not self.url or self.data is not None or urlsplit(self.url).scheme not in ("http", "https"):
            raise ValueError("url source requires an HTTP(S) url and no data")
        return self


class EmbeddingText(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["text"] = "text"
    text: str = Field(max_length=MAX_TEXT_CHARS)


class VideoWindows(BaseModel):
    """Opt-in: also embed a video as consecutive windows, one vector each.

    ``seconds`` omitted means the longest window the model trained on (the
    engine reports it as ``engine.embedding.video_windows.max_seconds``).
    ``sampling`` picks frames inside a window: ``fps`` (the default) takes
    frames at the model's own rate; ``content`` keeps evenly spaced anchors
    plus the frames that differ most from them within the window's frame budget.
    """
    model_config = ConfigDict(extra="forbid")
    seconds: float | None = Field(default=None, gt=0)
    sampling: Literal["content", "fps"] = "fps"


class EmbeddingMedia(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["image", "audio", "video"]
    source: EmbeddingSource
    windows: VideoWindows | None = None

    @model_validator(mode="after")
    def mime_matches(self):
        if not self.source.media_type.startswith(self.type + "/"):
            raise ValueError("media_type must match the content block type")
        if self.windows is not None and self.type != "video":
            raise ValueError("windows applies to video blocks only")
        cap = MEDIA_BYTES_BY_TYPE[self.type]
        if self.source.data is not None and len(self.source.data) > base64_chars(cap):
            raise ValueError(f"{self.type} data exceeds {cap // (1024 * 1024)} MiB")
        return self


EmbeddingBlock = Annotated[EmbeddingText | EmbeddingMedia, Field(discriminator="type")]


class EmbeddingItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    content: list[EmbeddingBlock] = Field(min_length=1, max_length=MAX_BLOCKS_PER_ITEM)
    prompt_name: str | None = Field(default=None, max_length=128)

    @model_validator(mode="after")
    def windowed_video_stands_alone(self):
        # Window times belong to one video; another block would make them ambiguous.
        if any(getattr(b, "windows", None) is not None for b in self.content) and len(self.content) != 1:
            raise ValueError("an item with video windows must contain only that video block")
        return self


class EmbeddingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1)
    input: list[EmbeddingItem] = Field(min_length=1, max_length=MAX_ITEMS)
    dimensions: int | None = Field(default=None, gt=0, strict=True)

    @model_validator(mode="before")
    @classmethod
    def text_shorthand(cls, value):
        if isinstance(value, dict):
            value = dict(value)
            items = value.get("input")
            if isinstance(items, str):
                items = [items]
            if isinstance(items, list):
                value["input"] = [
                    {"content": [{"type": "text", "text": item}]} if isinstance(item, str) else item
                    for item in items
                ]
        return value


class EmbeddingWindow(BaseModel):
    start: float
    end: float
    frame_times: list[float]        # seconds of the frames this window's vector saw
    embedding: list[float]


class EmbeddingVector(BaseModel):
    index: int
    embedding: list[float]
    prompt_name: str | None = None
    windows: list[EmbeddingWindow] | None = None   # only when the item's video asked for windows


class EmbeddingResult(BaseModel):
    model: str
    dimensions: int
    data: list[EmbeddingVector]
    input_tokens: int | None = None
    request_id: str | None = None
    elapsed_ms: float | None = None
