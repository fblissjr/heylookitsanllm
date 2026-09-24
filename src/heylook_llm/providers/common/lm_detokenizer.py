# src/heylook_llm/providers/common/lm_detokenizer.py
#
# Vendored from mlx-lm (https://github.com/ml-explore/mlx-lm) at commit
# c69d1288440a0dc4e6401fc417098b07598dccd5 (0.32.0), file
# mlx_lm/tokenizer_utils.py: its streaming detokenizers and the
# tokenizer.json decoder predicates that choose one, unchanged. heylook's
# edits, all of them: everything else in that file is removed
# (TokenizerWrapper, the thinking/tool-parser/chat-template inference,
# NewlineTokenizer, load, no_bos_or_eos), unused imports with it, and
# ``detokenizer_class_for`` at the bottom is heylook's, lifted from load's
# class-selection block.
#
# Why it is here (plan W10 stage 3): mlx-vlm drives every MLX model, and its
# own BPE detokenizer holds space-free text until the end of the stream
# (internal/claude/w10/mlx_vlm_bpe_detokenizer.md); these stream per token.
# It can go when mlx-vlm's does the same.
#
# MIT License
#
# Copyright © 2023 Apple Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# --- vendored ---
# Copyright © 2024 Apple Inc.

import abc
import functools
import json
from functools import partial
from pathlib import Path
from typing import List


class StreamingDetokenizer(abc.ABC):
    """The streaming detokenizer interface so that we can detokenize one token at a time.

    Example usage is as follows:

        detokenizer = ...

        # Reset the tokenizer state
        detokenizer.reset()

        for token in generate(...):
            detokenizer.add_token(token.item())

            # Contains the whole text so far. Some tokens may not be included
            # since it contains whole words usually.
            detokenizer.text

            # Contains the printable segment (usually a word) since the last
            # time it was accessed
            detokenizer.last_segment

            # Contains all the tokens added so far
            detokenizer.tokens

        # Make sure that we detokenize any remaining tokens
        detokenizer.finalize()

        # Now detokenizer.text should match tokenizer.decode(detokenizer.tokens)
    """

    # Set by reset(); text is a property on some subclasses.
    text: str
    tokens: List[int]
    offset: int

    @abc.abstractmethod
    def reset(self):
        """Drop all streaming state, keeping data derived from the tokenizer."""

    @abc.abstractmethod
    def add_token(self, token):
        """Consume one token id."""

    @abc.abstractmethod
    def finalize(self):
        """Flush any text held back waiting for more tokens."""

    @property
    def last_segment(self):
        """Return the last segment of readable text since last time this property was accessed."""
        text = self.text
        segment = text[self.offset :]
        self.offset = len(text)
        return segment


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    """NaiveStreamingDetokenizer relies on the underlying tokenizer
    implementation and should work with every tokenizer.

    Its complexity is O(T^2) where T is the longest line since it will
    repeatedly detokenize the same tokens until a new line is generated.
    """

    def __init__(self, tokenizer):
        super().__init__()
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])
        probe = tokenizer.encode("a ,b", add_special_tokens=False)
        self._clean_spaces = " ," not in tokenizer.decode(probe)
        self.reset()

    def reset(self):
        self.offset = 0
        self.tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token):
        self._current_tokens.append(token)
        self.tokens.append(token)

    def finalize(self):
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
            if self._current_text.endswith("\ufffd"):
                # An incomplete character can decode to several replacements.
                self._current_text = self._current_text.rstrip("\ufffd")
            elif (
                self._clean_spaces
                and len(self._current_text) > 0
                and self._current_text[-1] == " "
            ):
                self._current_text = self._current_text[:-1]
        if self._current_text and self._current_text[-1] == "\n":
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for SPM models.

    It adds tokens to the text if the next token starts with the special SPM
    underscore which results in linear complexity.
    """

    _sep = "\u2581".encode("utf-8")

    def __init__(self, tokenizer, trim_space=True):
        super().__init__()
        self.trim_space = trim_space

        ids = list(range(len(tokenizer)))
        tokens = tokenizer.convert_ids_to_tokens(ids)
        self.tokenmap = [
            # Byte tokens carry their value in hex.
            bytes([int(t[3:5], 16)]) if t.startswith("<0x") else t.encode("utf-8")
            for t in tokens
        ]

        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = b""
        self.text = ""
        self.tokens = []

    def _try_flush(self, force=False):
        text = self._unflushed.replace(self._sep, b" ").decode("utf-8", "replace")
        if not force and text.endswith("\ufffd"):
            return
        if not self.text and self.trim_space and text and text[0] == " ":
            text = text[1:]
        self.text += text
        self._unflushed = b""

    def add_token(self, token):
        self.tokens.append(token)
        v = self.tokenmap[token]
        self._unflushed += v
        self._try_flush()

    def finalize(self):
        self._try_flush(force=True)
        self._unflushed = b""


@functools.lru_cache(maxsize=1)
def _byte_decoder():
    """See https://github.com/openai/gpt-2/blob/master/src/encoder.py for the rationale."""
    char_to_bytes = {}
    limits = [
        0,
        ord("!"),
        ord("~") + 1,
        ord("¡"),
        ord("¬") + 1,
        ord("®"),
        ord("ÿ") + 1,
    ]
    n = 0
    for i, (start, stop) in enumerate(zip(limits, limits[1:])):
        if i % 2 == 0:
            for b in range(start, stop):
                char_to_bytes[chr(2**8 + n)] = b
                n += 1
        else:
            for b in range(start, stop):
                char_to_bytes[chr(b)] = b
    return char_to_bytes


class BPEStreamingDetokenizer(StreamingDetokenizer):
    """A streaming detokenizer for OpenAI style BPE models.

    It adds tokens to the text if the next token starts with a space similar to
    the SPM detokenizer.
    """

    def __init__(self, tokenizer):
        super().__init__()
        ids = list(range(len(tokenizer)))
        self.tokenmap = tokenizer.convert_ids_to_tokens(ids)

        self.reset()

    def reset(self):
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def _decode_bytes(self, seq):
        byte_decoder = _byte_decoder()
        barr = bytearray()
        for c in seq:
            res = byte_decoder.get(c, False)
            if res:
                barr.append(res)
            else:
                barr.extend(bytes(c, "utf-8"))
        return barr.decode("utf-8", "replace")

    def _maybe_trim_space(self, current_text):
        if len(current_text) == 0:
            return current_text
        elif current_text[0] != " ":
            return current_text
        elif not self.text:
            return current_text[1:]
        return current_text

    def add_token(self, token):
        self.tokens.append(token)
        # Undocumented fallback from #418, likely for a padded model vocab.
        # TODO(michalk8): check whether this is still needed.
        v = self.tokenmap[token] if token < len(self.tokenmap) else "!"
        self._unflushed += v
        text = self._decode_bytes(self._unflushed)

        # For multi-byte utf-8 wait until they are complete
        # For single spaces wait until the next token to clean it if needed
        if not text.endswith("\ufffd") and not (
            len(v) == 1 and _byte_decoder().get(v[0]) == 32
        ):
            self.text += self._maybe_trim_space(text)
            self._unflushed = ""

    def finalize(self):
        byte_decoder = _byte_decoder()
        current_text = bytearray(byte_decoder[c] for c in self._unflushed).decode(
            "utf-8",
            "replace",
        )
        self.text += self._maybe_trim_space(current_text)
        self._unflushed = ""


def _match(a, b):
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))

    return a == b


def _is_spm_decoder(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder):
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder):
    return isinstance(decoder, dict) and decoder.get("type", None) == "ByteLevel"


# --- heylook ---


def detokenizer_class_for(model_path):
    """The streaming detokenizer class mlx-lm's ``load`` would choose for the
    tokenizer at ``model_path``, from tokenizer.json's decoder; the naive one
    when there is no tokenizer.json or its decoder matches none. Build it on
    the tokenizer already loaded from the same files."""
    tokenizer_file = Path(model_path) / "tokenizer.json"
    if tokenizer_file.exists():
        with open(tokenizer_file, "r", encoding="utf-8") as fid:
            decoder = json.load(fid).get("decoder")
        if decoder is not None:
            if _is_spm_decoder(decoder):
                return SPMStreamingDetokenizer
            if _is_spm_decoder_no_space(decoder):
                return partial(SPMStreamingDetokenizer, trim_space=False)
            if _is_bpe_decoder(decoder):
                return BPEStreamingDetokenizer
    return NaiveStreamingDetokenizer
