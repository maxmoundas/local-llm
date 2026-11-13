"""Shared fixtures for tests. Uses a mock model to avoid requiring the real weights."""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def cfg() -> dict[str, Any]:
    return {
        "model": {
            "repo_id": "google/gemma-4-E4B-it",
            "cache_dir": "./models",
            "quantization": "4bit",
            "device_map": "auto",
            "max_new_tokens": 64,
            "temperature": 0.7,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "do_sample": True,
        },
        "server": {"host": "0.0.0.0", "port": 8000},
        "logging": {"level": "DEBUG", "log_dir": "logs", "console": False},
    }


@pytest.fixture
def mock_llm(cfg):
    """Returns a LocalLLM with mocked processor+model (no real weights needed).

    Skips on installs without torch (the MLX backend on macOS).
    """
    torch = pytest.importorskip("torch")
    from src.model import LocalLLM
    llm = LocalLLM(cfg)

    # Mock the underlying tokenizer (returned by processor.tokenizer)
    mock_tok = MagicMock()
    mock_tok.eos_token_id = 1
    mock_tok.apply_chat_template.return_value = (
        "<bos><start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"
    )
    mock_tok.decode.return_value = "Hello! How can I help you?"
    mock_tok.encode.side_effect = lambda text, **kw: list(range(len(text.split())))

    # Mock processor: has a .tokenizer attr and is callable (returns input_ids)
    mock_processor = MagicMock()
    mock_processor.tokenizer = mock_tok
    # processor(text=..., return_tensors="pt") returns dict with input_ids + attention_mask
    mock_processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }

    mock_model = MagicMock()
    mock_model.parameters.side_effect = lambda: iter([MagicMock(device="cpu")])
    mock_model.hf_device_map = {"layer.0": "cuda:0", "layer.1": "cpu"}
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])

    llm.processor = mock_processor
    llm.model = mock_model
    return llm


@pytest.fixture
def mock_registry(mock_llm):
    from src.providers.local import LocalProvider
    from src.providers.registry import ProviderRegistry
    registry = ProviderRegistry()
    registry.register(LocalProvider(mock_llm))
    return registry


@pytest.fixture
def test_client(mock_llm, mock_registry) -> TestClient:
    from src import server
    from src.llm_service import LLMService
    server._llm = mock_llm
    server._registry = mock_registry
    server._llm_service = LLMService(mock_registry)
    return TestClient(server.app)


@pytest.fixture(autouse=True)
def isolate_data_files(tmp_path, monkeypatch):
    """Keep every test away from the real memory.txt, settings.json, and chats/.

    Message handling starts background memory updates, so any test that sends a
    message could otherwise write a mocked reply into the user's real memory file.
    """
    import src.agent.sandbox as sb
    import src.chat_store as cs
    import src.tasks.store as ts
    import src.memory_store as ms
    import src.settings_store as ss
    monkeypatch.setattr(ms, "MEMORY_PATH", tmp_path / "memory.txt")
    monkeypatch.setattr(ss, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.setattr(cs, "CHATS_DIR", tmp_path / "chats")
    monkeypatch.setattr(ts, "DB_PATH", tmp_path / "tasks.db")
    monkeypatch.setattr(sb, "WORKSPACES_DIR", tmp_path / "workspaces")


def pytest_addoption(parser):
    parser.addoption("--model", action="append", default=[],
                     help="Model id(s) from models.yaml to run the contract suite against (tests/contract).")
