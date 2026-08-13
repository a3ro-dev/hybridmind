import json

import pytest

from engine.llm import LLMEngine, LLMOutputError


def _engine_with_response(response: str) -> LLMEngine:
    engine = object.__new__(LLMEngine)
    engine.model = "offline-test"
    engine._complete = lambda *args, **kwargs: response
    return engine


def test_unstructured_extraction_rejects_silent_input_truncation():
    engine = _engine_with_response("{}")
    with pytest.raises(ValueError, match="maximum is 12000"):
        engine.process_unstructured("x" * 12_001)


def test_unstructured_extraction_rejects_malformed_output():
    engine = _engine_with_response("not json")
    with pytest.raises(LLMOutputError, match="malformed JSON"):
        engine.process_unstructured("source text")


def test_unstructured_extraction_rejects_dangling_edges():
    engine = _engine_with_response(
        json.dumps(
            {
                "nodes": [{"text": "one", "metadata": {}}],
                "edges": [{"source_index": 0, "target_index": 2, "type": "relates_to"}],
                "summary": "",
            }
        )
    )
    with pytest.raises(LLMOutputError, match="missing node"):
        engine.process_unstructured("source text")


def test_unstructured_extraction_accepts_valid_json_fence():
    payload = {
        "nodes": [{"text": "one", "metadata": {"type": "fact"}}],
        "edges": [],
        "summary": "one",
    }
    engine = _engine_with_response(f"```json\n{json.dumps(payload)}\n```")
    assert engine.process_unstructured("source text") == payload


def test_metadata_parse_failure_is_not_returned_as_fake_summary():
    engine = _engine_with_response("not-json")
    with pytest.raises(LLMOutputError):
        engine.extract_metadata("source text")
