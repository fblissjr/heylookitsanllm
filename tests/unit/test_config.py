# tests/unit/test_config.py
"""Unit tests for Pydantic config models."""
import pytest

from heylook_llm.config import (
    AppConfig,
    ChatMessage,
    ChatRequest,
    ImageContentPart,
    ImageUrl,
    LlamaCppModelConfig,
    MLXModelConfig,
    MLXSTTModelConfig,
    ModelConfig,
    TextContentPart,
)


@pytest.mark.unit
class TestChatMessage:
    def test_simple_text_message(self):
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.thinking is None

    def test_assistant_message(self):
        msg = ChatMessage(role="assistant", content="hi there")
        assert msg.role == "assistant"

    def test_system_message(self):
        msg = ChatMessage(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_multimodal_content(self):
        parts = [
            TextContentPart(type="text", text="What is this?"),
            ImageContentPart(type="image_url", image_url=ImageUrl(url="https://example.com/img.png")),
        ]
        msg = ChatMessage(role="user", content=parts)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_thinking_optional(self):
        msg = ChatMessage(role="assistant", content="hi")
        assert msg.thinking is None

    def test_thinking_roundtrip(self):
        msg = ChatMessage(role="assistant", content="answer", thinking="reasoning")
        assert msg.thinking == "reasoning"
        d = msg.model_dump()
        assert d["thinking"] == "reasoning"

    def test_thinking_excluded_when_none(self):
        msg = ChatMessage(role="assistant", content="hi")
        d = msg.model_dump(exclude_none=True)
        assert "thinking" not in d

    def test_optional_fields_default_none(self):
        msg = ChatMessage(role="user", content="test")
        assert msg.name is None
        assert msg.tool_call_id is None
        assert msg.tool_calls is None


@pytest.mark.unit
class TestChatRequest:
    def test_minimal_request(self):
        req = ChatRequest(messages=[ChatMessage(role="user", content="hi")])
        assert req.model is None
        assert req.stream is False
        assert len(req.messages) == 1

    def test_empty_messages_rejected(self):
        with pytest.raises(ValueError, match="Messages list cannot be empty"):
            ChatRequest(messages=[])

    def test_all_sampler_params(self):
        req = ChatRequest(
            model="test",
            messages=[ChatMessage(role="user", content="hi")],
            temperature=0.5,
            top_p=0.9,
            top_k=40,
            min_p=0.1,
            repetition_penalty=1.1,
            max_tokens=256,
            seed=42,
        )
        assert req.temperature == 0.5
        assert req.top_k == 40
        assert req.seed == 42

    def test_stream_options(self):
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            stream=True,
            stream_options={"include_usage": True},
        )
        assert req.stream is True
        assert req.stream_options["include_usage"] is True

    def test_logprobs_params(self):
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            logprobs=True,
            top_logprobs=5,
        )
        assert req.logprobs is True
        assert req.top_logprobs == 5

    def test_enable_thinking(self):
        req = ChatRequest(
            messages=[ChatMessage(role="user", content="hi")],
            enable_thinking=True,
        )
        assert req.enable_thinking is True


@pytest.mark.unit
class TestModelConfig:
    def test_mlx_model_config(self):
        mc = ModelConfig(
            id="test-mlx",
            provider="mlx",
            config={"model_path": "/fake/path"},
        )
        assert mc.id == "test-mlx"
        assert mc.provider == "mlx"
        assert isinstance(mc.config, MLXModelConfig)
        assert mc.config.model_path == "/fake/path"

    def test_llama_cpp_model_config(self):
        mc = ModelConfig(
            id="test-gguf",
            provider="llama_cpp",
            config={"model_path": "/fake/model.gguf"},
        )
        assert isinstance(mc.config, LlamaCppModelConfig)

    def test_gguf_alias(self):
        mc = ModelConfig(
            id="test-gguf",
            provider="gguf",
            config={"model_path": "/fake/model.gguf"},
        )
        assert isinstance(mc.config, LlamaCppModelConfig)

    def test_mlx_stt_config(self):
        mc = ModelConfig(
            id="test-stt",
            provider="mlx_stt",
            config={"model_path": "mlx-community/parakeet"},
        )
        assert isinstance(mc.config, MLXSTTModelConfig)

    def test_invalid_config_for_provider_rejected(self):
        """Config missing required fields raises validation error."""
        with pytest.raises(ValueError):
            ModelConfig(
                id="bad",
                provider="mlx",
                config={"bad_field_only": True},
            )

    def test_mlx_config_defaults(self):
        mc = MLXModelConfig(model_path="/fake")
        assert mc.vision is False
        assert mc.enable_thinking is False
        assert mc.cache_type == "standard"
        assert mc.default_hidden_layer == -2
        assert mc.default_max_length == 512

    def test_capabilities_list(self):
        mc = ModelConfig(
            id="test",
            provider="mlx",
            config={"model_path": "/fake"},
            capabilities=["chat", "thinking", "vision"],
        )
        assert "thinking" in mc.capabilities

    def test_enabled_default_true(self):
        mc = ModelConfig(
            id="test",
            provider="mlx",
            config={"model_path": "/fake"},
        )
        assert mc.enabled is True


@pytest.mark.unit
class TestAppConfig:
    def test_get_model_config(self):
        cfg = AppConfig(
            models=[
                ModelConfig(id="m1", provider="mlx", config={"model_path": "/a"}, enabled=True),
                ModelConfig(id="m2", provider="mlx", config={"model_path": "/b"}, enabled=False),
            ]
        )
        assert cfg.get_model_config("m1") is not None
        assert cfg.get_model_config("m2") is None  # disabled
        assert cfg.get_model_config("m3") is None  # missing

    def test_get_enabled_models(self):
        cfg = AppConfig(
            models=[
                ModelConfig(id="m1", provider="mlx", config={"model_path": "/a"}, enabled=True),
                ModelConfig(id="m2", provider="mlx", config={"model_path": "/b"}, enabled=False),
            ]
        )
        enabled = cfg.get_enabled_models()
        assert len(enabled) == 1
        assert enabled[0].id == "m1"

    def test_max_loaded_models_default(self):
        cfg = AppConfig(models=[])
        assert cfg.max_loaded_models == 2
