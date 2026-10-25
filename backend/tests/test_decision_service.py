"""Tests for DecisionService (Jev client). HTTP is mocked with httpx.MockTransport."""
from __future__ import annotations

import json

import httpx
import pytest

from src import decision_service
from src.decision_service import Answer, DecisionError, DecisionService, choice, noul, score


def _service(handler, **kw) -> DecisionService:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return DecisionService("test-key", client=client, **kw)


def _ok(answers: dict) -> httpx.Response:
    return httpx.Response(200, json={"model": "jev-1.14.0", "answers": answers, "usage": {}})


class TestQuestionBuilders:
    def test_noul_without_criteria(self):
        assert noul("Is it urgent?") == {"type": "noul", "instructions": "Is it urgent?"}

    def test_noul_with_criteria(self):
        q = noul("Is it urgent?", true="yes", false="no")
        assert q["criteria"] == {"true": "yes", "false": "no"}

    def test_choice(self):
        q = choice("Team?", {"a": "A team"})
        assert q == {"type": "choice", "instructions": "Team?", "criteria": {"a": "A team"}}

    def test_score(self):
        assert score("How bad?", ["low", "high"])["criteria"] == ["low", "high"]


class TestAnswer:
    def test_noul_from_api(self):
        a = Answer.from_api({"type": "noul", "noul": 0.9})
        assert a.noul == 0.9 and a.confidence is None

    def test_choice_from_api(self):
        a = Answer.from_api({"type": "choice", "choice": "b", "confidence": 0.8, "probabilities": {"b": 0.9}})
        assert a.choice == "b" and a.confidence == 0.8 and a.probabilities == {"b": 0.9}

    def test_score_from_api(self):
        assert Answer.from_api({"type": "score", "score": 1.5, "confidence": 0.4}).score == 1.5


class TestAvailability:
    def test_unconfigured_without_key(self):
        svc = DecisionService(None)
        assert not svc.configured and not svc.available

    def test_disabled_is_unavailable(self):
        svc = DecisionService("k", enabled=False)
        assert svc.configured and not svc.available

    def test_ask_raises_when_unavailable(self):
        with pytest.raises(DecisionError):
            DecisionService(None).ask("s", {"q": noul("x")})


class TestAsk:
    def test_sends_bearer_model_state_questions(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers["authorization"]
            seen["body"] = json.loads(request.content)
            return _ok({"q": {"type": "noul", "noul": 0.7}})

        _service(handler, model="jev-preview").ask({"a": 1}, {"q": noul("x")})
        assert seen["auth"] == "Bearer test-key"
        assert seen["body"]["model"] == "jev-preview"
        assert seen["body"]["state"] == {"a": 1}
        assert seen["body"]["questions"]["q"]["type"] == "noul"

    def test_returns_typed_answers(self):
        svc = _service(lambda r: _ok({
            "n": {"type": "noul", "noul": 0.2},
            "c": {"type": "choice", "choice": "x", "confidence": 1.0, "probabilities": {"x": 1.0}},
        }))
        answers = svc.ask("s", {"n": noul("n"), "c": choice("c", {"x": "X"})})
        assert answers["n"].noul == 0.2
        assert answers["c"].choice == "x"

    def test_retries_once_on_503(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(503) if len(calls) == 1 else _ok({"q": {"type": "noul", "noul": 0.5}})

        assert _service(handler).ask("s", {"q": noul("x")})["q"].noul == 0.5
        assert len(calls) == 2

    def test_raises_on_400_without_retry(self):
        calls = []

        def handler(request):
            calls.append(1)
            return httpx.Response(422, json={"detail": "bad"})

        with pytest.raises(DecisionError):
            _service(handler).ask("s", {"q": noul("x")})
        assert len(calls) == 1

    def test_raises_on_network_error_after_retry(self):
        def handler(request):
            raise httpx.ConnectError("down")

        with pytest.raises(DecisionError):
            _service(handler).ask("s", {"q": noul("x")})

    def test_raises_when_answer_missing(self):
        with pytest.raises(DecisionError):
            _service(lambda r: _ok({})).ask("s", {"q": noul("x")})


class TestModuleInstance:
    def test_configure_and_get(self):
        original = decision_service.get()
        svc = DecisionService("k")
        try:
            decision_service.configure(svc)
            assert decision_service.get() is svc
        finally:
            decision_service.configure(original)
