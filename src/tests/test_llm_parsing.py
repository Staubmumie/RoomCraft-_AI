"""Reading structured answers: models sometimes add trailing content after a complete JSON object."""
import asyncio
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ValidationError

import llm


class Answer(BaseModel):
    action: str
    note: str


GOOD = '{"action":"finish","note":"Sofa for it in the lounge."}'


def test_plain_json_is_parsed():
    assert llm.parse_structured(GOOD, Answer).action == "finish"


@pytest.mark.parametrize("content", [GOOD + "\n\n" + GOOD, GOOD + "\n\nDone.", "  " + GOOD + "\n}"])
def test_first_complete_object_is_used_despite_trailing_characters(content):
    assert llm.parse_structured(content, Answer).note == "Sofa for it in the lounge."


@pytest.mark.parametrize("content", ['{"action":"finish"}', "not json", '{"action":"fin'])
def test_answers_that_do_not_match_still_fail(content):
    with pytest.raises(ValidationError):
        llm.parse_structured(content, Answer)


def _completion(content, finish_reason="stop"):
    message = SimpleNamespace(content=content, refusal=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], model="m",
                           model_extra={}, id="gen", usage=None)


def _client(monkeypatch, *contents):
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return _completion(contents[len(calls) - 1])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(llm, "get_client", lambda: client)
    return calls


def test_structured_completion_sends_the_strict_schema_and_tolerates_trailing_text(monkeypatch):
    calls = _client(monkeypatch, GOOD + "\n\n" + GOOD)
    result = asyncio.run(llm.structured_completion("m", [], Answer))
    assert result.parsed.action == "finish" and len(calls) == 1
    assert calls[0]["response_format"]["type"] == "json_schema"
    assert calls[0]["response_format"]["json_schema"]["strict"] is True


def test_unreadable_answer_is_asked_for_once_more(monkeypatch):
    calls = _client(monkeypatch, "garbage", GOOD)
    result = asyncio.run(llm.structured_completion("m", [], Answer))
    assert result.parsed.action == "finish" and len(calls) == 2
    assert [a["attempt"] for a in result.info.attempts] == [1, 2] and "error" in result.info.attempts[0]


def test_unreadable_answers_fail_after_the_retry(monkeypatch):
    _client(monkeypatch, "garbage", "garbage")
    with pytest.raises(llm.LLMError, match="expected layout format") as exc:
        asyncio.run(llm.structured_completion("m", [], Answer))
    assert len(exc.value.info.attempts) == 2
