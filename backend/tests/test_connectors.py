"""ConnectorHub against a real MCP server over stdio (skipped when the `mcp` package is missing)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("mcp")

from src.agent.connectors import ConnectorHub, _expand, _tool_name  # noqa: E402

SERVER = Path(__file__).parent / "fixtures" / "mcp_test_server.py"


@pytest.fixture(scope="module")
def hub(tmp_path_factory):
    cfg = tmp_path_factory.mktemp("c") / "connectors.yaml"
    cfg.write_text(f"servers:\n  demo:\n    command: {sys.executable}\n    args: ['{SERVER}']\n"
                   "    confirm: ['send_*']\n")
    h = ConnectorHub(cfg)
    h.start(timeout=30)
    yield h
    h.stop()


def test_tools_registered_with_prefix_and_confirm(hub):
    tools = {t.name: t for t in hub.tools}
    assert set(tools) == {"demo__add", "demo__send_message"}
    assert tools["demo__send_message"].confirm and not tools["demo__add"].confirm
    assert tools["demo__add"].parameters["properties"]["a"]
    assert hub.status()["demo"].startswith("connected")


def test_call_tool(hub):
    add = next(t for t in hub.tools if t.name == "demo__add")
    result = add.run({"a": 2, "b": 3})
    assert result.text.strip() == "5" and result.summary == "done"


def test_bad_server_reports_error(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("servers:\n  broken:\n    command: /nonexistent/binary\n")
    h = ConnectorHub(cfg)
    h.start(timeout=10)
    assert h.tools == [] and h.status()["broken"].startswith("error")


def test_env_expansion_and_names(monkeypatch):
    monkeypatch.setenv("TOK", "abc")
    assert _expand({"h": ["Bearer ${TOK}"]}) == {"h": ["Bearer abc"]}
    assert _tool_name("my server", "do.thing") == "my_server__do_thing"
