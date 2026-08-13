import json

import pytest

from config import settings
from engine import fact_extractor


def _enable(monkeypatch):
    monkeypatch.setattr(settings, "fact_extraction_enabled", True)
    monkeypatch.setattr(fact_extractor.llm_client, "is_configured", lambda: True)


def test_long_session_is_split_without_truncating_turns(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(settings, "fact_extraction_max_chars_per_request", 90)
    monkeypatch.setattr(settings, "fact_extraction_max_requests_per_session", 8)
    seen = []

    def fake_call(messages, max_tokens, response_format=None):
        seen.append(messages[1]["content"])
        return json.dumps({"facts": []})

    monkeypatch.setattr(fact_extractor, "_call_llm", fake_call)
    turns = [
        {"speaker": "user", "text": f"unique-{index} " + "x" * 45, "date": ""}
        for index in range(4)
    ]

    assert fact_extractor.extract_facts_from_session(turns) == []
    assert len(seen) == 4
    joined = "\n".join(seen)
    assert all(f"unique-{index}" in joined for index in range(4))


def test_provider_failure_is_not_returned_as_factless(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(fact_extractor, "_call_llm", lambda *args, **kwargs: None)
    with pytest.raises(fact_extractor.FactExtractionError, match="no usable response"):
        fact_extractor.extract_facts_from_session(
            [{"speaker": "user", "text": "I live in Delhi", "date": ""}]
        )


def test_malformed_response_is_not_returned_as_factless():
    with pytest.raises(fact_extractor.FactExtractionError, match="malformed JSON"):
        fact_extractor._parse_facts_content("not-json and no array")


def test_request_ceiling_rejects_before_any_provider_call(monkeypatch):
    _enable(monkeypatch)
    monkeypatch.setattr(settings, "fact_extraction_max_chars_per_request", 80)
    monkeypatch.setattr(settings, "fact_extraction_max_requests_per_session", 2)
    calls = []
    monkeypatch.setattr(fact_extractor, "_call_llm", lambda *args, **kwargs: calls.append(1))
    turns = [
        {"speaker": "u", "text": f"turn-{index} " + "x" * 45, "date": ""}
        for index in range(3)
    ]

    with pytest.raises(fact_extractor.FactExtractionError, match="ceiling is 2"):
        fact_extractor.extract_facts_from_session(turns)
    assert calls == []


def test_invalid_structured_fields_fail_closed():
    content = json.dumps(
        {
            "facts": [
                {
                    "fact": "Akshat moved.",
                    "entities": "Akshat",
                    "date": "",
                    "memory_kind": "experience",
                    "confidence": 0.9,
                    "caused_by": [],
                }
            ]
        }
    )
    with pytest.raises(fact_extractor.FactExtractionError, match="invalid entities"):
        fact_extractor._parse_facts_content(content)


@pytest.mark.parametrize(
    "date_value",
    ["last Friday", "2026-13-40", "2026-01-01 trailing text"],
)
def test_extracted_dates_must_be_complete_iso_8601_values(date_value):
    content = json.dumps(
        {
            "facts": [
                {
                    "fact": "Akshat moved.",
                    "entities": ["Akshat"],
                    "date": date_value,
                    "memory_kind": "experience",
                    "confidence": 0.9,
                    "caused_by": [],
                }
            ]
        }
    )

    with pytest.raises(fact_extractor.FactExtractionError, match="invalid ISO-8601 date"):
        fact_extractor._parse_facts_content(content)


@pytest.mark.parametrize("date_value", ["2026-08-13", "2026-08-13T12:30:00Z", ""])
def test_extracted_dates_accept_iso_8601_or_empty(date_value):
    content = json.dumps(
        {
            "facts": [
                {
                    "fact": "Akshat moved.",
                    "entities": ["Akshat"],
                    "date": date_value,
                    "memory_kind": "experience",
                    "confidence": 0.9,
                    "caused_by": [],
                }
            ]
        }
    )

    assert fact_extractor._parse_facts_content(content)[0]["date"] == date_value
