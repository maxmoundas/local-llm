"""Tests for the LocalLLM model wrapper."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from src.model import LocalLLM


class TestLocalLLMConfig:
    def test_uses_config_defaults(self, cfg):
        llm = LocalLLM(cfg)
        assert llm.cfg["repo_id"] == "google/gemma-4-E4B-it"
        assert llm.cfg["quantization"] == "4bit"

    def test_is_loaded_false_before_load(self, cfg):
        llm = LocalLLM(cfg)
        assert llm.is_loaded is False

    def test_is_loaded_true_after_mock_load(self, mock_llm):
        assert mock_llm.is_loaded is True

    def test_repo_id_property(self, mock_llm):
        assert mock_llm.repo_id == "google/gemma-4-E4B-it"


class TestGenerate:
    def test_generate_returns_string(self, mock_llm):
        result = mock_llm.generate([{"role": "user", "content": "hello"}])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_calls_model(self, mock_llm):
        mock_llm.generate([{"role": "user", "content": "hello"}])
        mock_llm.model.generate.assert_called_once()

    def test_generate_respects_max_new_tokens(self, mock_llm):
        mock_llm.generate([{"role": "user", "content": "hi"}], max_new_tokens=32)
        call_kwargs = mock_llm.model.generate.call_args.kwargs
        assert call_kwargs["max_new_tokens"] == 32

    def test_generate_with_temperature_zero_disables_sampling(self, mock_llm):
        mock_llm.generate([{"role": "user", "content": "hi"}], temperature=0.0)
        call_kwargs = mock_llm.model.generate.call_args.kwargs
        assert call_kwargs["do_sample"] is False
        assert call_kwargs.get("temperature") is None

    def test_generate_with_system_message(self, mock_llm):
        messages = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "hello"},
        ]
        result = mock_llm.generate(messages)
        assert isinstance(result, str)

    def test_generate_multi_turn(self, mock_llm):
        messages = [
            {"role": "user", "content": "what is 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "and 4+4?"},
        ]
        result = mock_llm.generate(messages)
        assert isinstance(result, str)


class TestStream:
    def test_stream_yields_strings(self, mock_llm):
        # Mock TextIteratorStreamer to yield tokens
        with patch("src.model.TextIteratorStreamer") as mock_streamer_cls:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter(["Hello", " world", "!"]))
            mock_streamer_cls.return_value = mock_streamer

            tokens = list(mock_llm.stream([{"role": "user", "content": "hi"}]))
            assert tokens == ["Hello", " world", "!"]

    def test_stream_starts_generation_thread(self, mock_llm):
        with patch("src.model.TextIteratorStreamer") as mock_streamer_cls, \
             patch("src.model.threading.Thread") as mock_thread_cls:
            mock_streamer = MagicMock()
            mock_streamer.__iter__ = MagicMock(return_value=iter([]))
            mock_streamer_cls.return_value = mock_streamer

            mock_thread = MagicMock()
            mock_thread_cls.return_value = mock_thread

            list(mock_llm.stream([{"role": "user", "content": "hi"}]))
            mock_thread.start.assert_called_once()
            mock_thread.join.assert_called_once()


class TestTokenCount:
    def test_count_tokens_returns_int(self, mock_llm):
        count = mock_llm.count_tokens("hello world")
        assert isinstance(count, int)
        assert count > 0

    def test_count_tokens_longer_text_more_tokens(self, mock_llm):
        short = mock_llm.count_tokens("hi")
        long = mock_llm.count_tokens("hello world this is a longer message")
        assert long > short


class TestBnbConfig:
    def test_4bit_config(self, cfg):
        llm = LocalLLM(cfg)
        bnb = llm._build_bnb_config("4bit")
        assert bnb is not None
        assert bnb.load_in_4bit is True

    def test_8bit_config(self, cfg):
        llm = LocalLLM(cfg)
        bnb = llm._build_bnb_config("8bit")
        assert bnb is not None
        assert bnb.load_in_8bit is True

    def test_none_config(self, cfg):
        llm = LocalLLM(cfg)
        bnb = llm._build_bnb_config("none")
        assert bnb is None
