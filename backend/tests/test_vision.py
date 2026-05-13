"""Tests for vision (image) inference in LocalLLM.

All tests use mocked model and processor — no GPU or real weights required.
The real images in tests/images/ are used only by the manual probe script
(backend/probe_vision.py), not by this suite.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image as PILImage

from src.model import LocalLLM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_image() -> PILImage.Image:
    return PILImage.new("RGB", (64, 64), color=(128, 64, 200))


@pytest.fixture
def vision_inputs() -> dict:
    """Fake processor output matching the shape returned by Gemma 4's processor."""
    return {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
        "pixel_values": torch.randn(1, 2520, 768),
        "mm_token_type_ids": torch.zeros(1, 10, dtype=torch.long),
        "image_position_ids": torch.zeros(1, 2520, 2, dtype=torch.long),
    }


@pytest.fixture
def vision_llm(mock_llm, vision_inputs) -> LocalLLM:
    """mock_llm with processor.apply_chat_template configured to return vision inputs."""
    mock_llm.processor.apply_chat_template.return_value = vision_inputs
    return mock_llm


# ---------------------------------------------------------------------------
# _inject_image
# ---------------------------------------------------------------------------

class TestInjectImage:
    def test_injects_image_into_last_user_message(self, vision_llm, sample_image):
        messages = [{"role": "user", "content": "what is this?"}]
        result = vision_llm._inject_image(messages, sample_image)
        content = result[-1]["content"]
        assert isinstance(content, list)
        assert any(c["type"] == "image" and c["image"] is sample_image for c in content)
        assert any(c["type"] == "text" and c["text"] == "what is this?" for c in content)

    def test_preserves_system_message(self, vision_llm, sample_image):
        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "describe this"},
        ]
        result = vision_llm._inject_image(messages, sample_image)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "Be helpful."

    def test_targets_last_user_turn_in_multi_turn_conversation(self, vision_llm, sample_image):
        messages = [
            {"role": "user", "content": "first message"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "now look at this"},
        ]
        result = vision_llm._inject_image(messages, sample_image)
        assert result[0]["content"] == "first message"
        assert isinstance(result[2]["content"], list)

    def test_does_not_mutate_original_list(self, vision_llm, sample_image):
        messages = [{"role": "user", "content": "hello"}]
        original_content = messages[0]["content"]
        vision_llm._inject_image(messages, sample_image)
        assert messages[0]["content"] == original_content


# ---------------------------------------------------------------------------
# generate_with_image
# ---------------------------------------------------------------------------

class TestGenerateWithImage:
    def test_returns_string(self, vision_llm, sample_image):
        result = vision_llm.generate_with_image(sample_image, "what do you see?")
        assert isinstance(result, str)

    def test_uses_single_step_apply_chat_template(self, vision_llm, sample_image):
        vision_llm.generate_with_image(sample_image, "describe")
        kwargs = vision_llm.processor.apply_chat_template.call_args.kwargs
        assert kwargs.get("tokenize") is True
        assert kwargs.get("return_dict") is True
        assert kwargs.get("return_tensors") == "pt"

    def test_pixel_values_passed_to_model_generate(self, vision_llm, sample_image):
        vision_llm.generate_with_image(sample_image, "describe")
        gen_kwargs = vision_llm.model.generate.call_args.kwargs
        assert "pixel_values" in gen_kwargs

    def test_uses_greedy_decoding(self, vision_llm, sample_image):
        vision_llm.generate_with_image(sample_image, "describe")
        gen_kwargs = vision_llm.model.generate.call_args.kwargs
        assert gen_kwargs.get("do_sample") is False

    def test_respects_max_new_tokens(self, vision_llm, sample_image):
        vision_llm.generate_with_image(sample_image, "describe", max_new_tokens=42)
        gen_kwargs = vision_llm.model.generate.call_args.kwargs
        assert gen_kwargs["max_new_tokens"] == 42

    def test_messages_contain_image_object(self, vision_llm, sample_image):
        vision_llm.generate_with_image(sample_image, "look")
        passed_messages = vision_llm.processor.apply_chat_template.call_args.args[0]
        image_items = [
            item
            for msg in passed_messages
            for item in (msg["content"] if isinstance(msg.get("content"), list) else [])
            if isinstance(item, dict) and item.get("type") == "image"
        ]
        assert any(item["image"] is sample_image for item in image_items)

    def test_requires_auto_processor(self, cfg, sample_image):
        llm = LocalLLM(cfg)
        llm.model = MagicMock()
        llm.processor = MagicMock(spec=[])   # no .tokenizer attr
        with pytest.raises(AssertionError):
            llm.generate_with_image(sample_image, "describe")


# ---------------------------------------------------------------------------
# stream_with_image
# ---------------------------------------------------------------------------

class TestStreamWithImage:
    def test_yields_tokens(self, vision_llm, sample_image):
        messages = [{"role": "user", "content": "what is this?"}]
        with patch("src.model.TextIteratorStreamer") as mock_streamer_cls:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["A", " dog", "!"]))
            mock_streamer_cls.return_value = mock_streamer

            tokens = list(vision_llm.stream_with_image(sample_image, messages))
            assert tokens == ["A", " dog", "!"]

    def test_pixel_values_passed_to_model_generate(self, vision_llm, sample_image):
        messages = [{"role": "user", "content": "describe"}]
        with patch("src.model.TextIteratorStreamer") as mock_streamer_cls:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter([]))
            mock_streamer_cls.return_value = mock_streamer
            list(vision_llm.stream_with_image(sample_image, messages))

        gen_kwargs = vision_llm.model.generate.call_args.kwargs
        assert "pixel_values" in gen_kwargs

    def test_image_injected_before_template_call(self, vision_llm, sample_image):
        messages = [{"role": "user", "content": "look"}]
        with patch("src.model.TextIteratorStreamer") as mock_streamer_cls:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter([]))
            mock_streamer_cls.return_value = mock_streamer
            list(vision_llm.stream_with_image(sample_image, messages))

        passed_messages = vision_llm.processor.apply_chat_template.call_args.args[0]
        image_found = any(
            item.get("image") is sample_image
            for msg in passed_messages
            for item in (msg["content"] if isinstance(msg.get("content"), list) else [])
        )
        assert image_found

    def test_starts_generation_in_background_thread(self, vision_llm, sample_image):
        messages = [{"role": "user", "content": "hi"}]
        with patch("src.model.TextIteratorStreamer") as mock_streamer_cls, \
             patch("src.model.threading.Thread") as mock_thread_cls:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter([]))
            mock_streamer_cls.return_value = mock_streamer
            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            list(vision_llm.stream_with_image(sample_image, messages))
            mock_thread.start.assert_called_once()
            mock_thread.join.assert_called_once()


# ---------------------------------------------------------------------------
# BnB config — vision quantization skip
# ---------------------------------------------------------------------------

class TestVisionQuantizationConfig:
    def test_4bit_skip_modules_use_full_path_prefixes(self, cfg):
        """Short names like 'vision_tower' are silently ignored by
        should_convert_module() which uses re.match() anchored at pos 0.
        Full path prefixes are required."""
        llm = LocalLLM(cfg)
        bnb = llm._build_bnb_config("4bit")
        skip = bnb.llm_int8_skip_modules
        assert any(s.startswith("model.vision_tower") for s in skip)
        assert any("lm_head" in s for s in skip)

    def test_8bit_skip_modules_use_full_path_prefixes(self, cfg):
        llm = LocalLLM(cfg)
        bnb = llm._build_bnb_config("8bit")
        skip = bnb.llm_int8_skip_modules
        assert any(s.startswith("model.vision_tower") for s in skip)

    def test_no_short_names_in_skip_list(self, cfg):
        """Bare 'vision_tower' without the 'model.' prefix would silently fail."""
        llm = LocalLLM(cfg)
        for quant in ("4bit", "8bit"):
            bnb = llm._build_bnb_config(quant)
            for s in bnb.llm_int8_skip_modules:
                assert s != "vision_tower", (
                    f"Bare 'vision_tower' in skip list for {quant} — "
                    "should be 'model.vision_tower' (re.match anchors at pos 0)"
                )
