"""Model contract suite — does a model work with this app's agent loop?

    pytest tests/contract -m contract --model claude-haiku-5-5 --model llama3.2:3b -v

Uses the real models.yaml catalog and API keys from backend/.env, so it costs a few cents
on paid models. Tool checks run in both `native` and `prompted` modes; the summary at the
end recommends the `tools:` setting for models.yaml.
"""
from __future__ import annotations

from collections import defaultdict

import pytest

_results: dict[str, dict[str, bool]] = defaultdict(dict)


def pytest_generate_tests(metafunc):
    if "model_id" in metafunc.fixturenames:
        ids = metafunc.config.getoption("--model")
        metafunc.parametrize("model_id", ids or [pytest.param(None, marks=pytest.mark.skip(
            reason="pass --model <id> to run the contract suite"))], scope="session")
    if "tool_mode" in metafunc.fixturenames:
        metafunc.parametrize("tool_mode", ["native", "prompted"])


@pytest.fixture(scope="session")
def registry():
    from dotenv import load_dotenv
    load_dotenv()
    from src.providers.registry import build_registry
    return build_registry(None)  # cloud + OpenAI-compatible providers; the in-process model isn't loaded here


@pytest.fixture(scope="session")
def svc(registry):
    from src.llm_service import LLMService
    return LLMService(registry)


@pytest.fixture
def model(registry, model_id):
    try:
        return registry.get_model(model_id)
    except KeyError:
        pytest.skip(f"{model_id} is not available (check models.yaml, API keys, local servers)")


@pytest.fixture
def record(request, model_id):
    def _record(check: str, ok: bool) -> None:
        mode = request.node.callspec.params.get("tool_mode", "")
        _results[model_id][f"{check}[{mode}]" if mode else check] = ok
    return _record


@pytest.fixture
def tools_as(model, tool_mode):
    original = model.tools
    model.tools = tool_mode
    yield tool_mode
    model.tools = original


def pytest_terminal_summary(terminalreporter):
    if not _results:
        return
    tr = terminalreporter
    tr.section("model contract summary")
    for model_id, checks in _results.items():
        passed = sum(checks.values())
        tr.write_line(f"{model_id}: {passed}/{len(checks)} checks passed")
        for name, ok in sorted(checks.items()):
            tr.write_line(f"   {'✓' if ok else '✗'} {name}")
        native = [ok for k, ok in checks.items() if k.endswith("[native]")]
        prompted = [ok for k, ok in checks.items() if k.endswith("[prompted]")]
        if native and prompted:
            rec = "native" if sum(native) >= sum(prompted) and all(native) else (
                "prompted" if sum(prompted) > sum(native) else "auto")
            tr.write_line(f"   → recommended models.yaml setting: tools: {rec}")
