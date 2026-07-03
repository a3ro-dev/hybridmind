"""
Shared LLM QA-answering helper for eval_*.py scripts.

The historical failure mode (docs/LOCOMO_BENCHMARK_REPORT.md): an answering LLM
would abstain ("Answer: None") even when the correct context was retrieved,
tanking single-hop accuracy to 0% despite 60%+ Hit@10. That bug lived in the
external `memorybench/` harness (not part of this repo) and has since been
fixed there via a rephrase-and-extract retry.

The eval_*.py scripts in this repo are retrieval-only (no answering LLM at
all), so they were immune to that bug but also couldn't report answer
accuracy. This module adds that capability directly, reusing the same
HackClub-proxy + structured-JSON + retry pattern as engine/fact_extractor.py,
with the same defensive rephrase-retry so the bug class can't recur here.
"""
import json
import logging
import os
import random
import re
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_HC_API_KEY = os.getenv("HC_API_KEY", "")
_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://ai.hackclub.com/proxy/v1").rstrip("/")
DEFAULT_ANSWER_MODEL = os.getenv("HYBRIDMIND_QA_MODEL", "openai/gpt-4o")

_MAX_ATTEMPTS = 3
_BACKOFF_BASE = 1.5
_BACKOFF_CAP = 20.0

_CLIENT: httpx.Client | None = None

_STOPWORDS = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "was", "it", "and", "or", "but"}

_ABSTENTION_RE = re.compile(r"^\s*(i don'?t know|none|not (specified|available|mentioned)|n/?a)\b", re.I)

_ANSWER_SCHEMA = {
    "name": "answer",
    "schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "found_in_context": {"type": "boolean"},
        },
        "required": ["answer", "found_in_context"],
    },
    "strict": True,
}


def _get_client() -> httpx.Client:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = httpx.Client(
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
            timeout=httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=30.0),
        )
    return _CLIENT


def _sleep_backoff(attempt: int, reason: str) -> None:
    delay = min(_BACKOFF_CAP, _BACKOFF_BASE * (2 ** attempt) * (0.5 + random.random()))
    logger.warning(f"eval_common: retry {attempt + 1}/{_MAX_ATTEMPTS} after {delay:.1f}s ({reason})")
    time.sleep(delay)


def is_abstention(text: str) -> bool:
    text = (text or "").strip()
    return (not text) or bool(_ABSTENTION_RE.match(text))


def _call(payload: dict) -> str | None:
    if not _HC_API_KEY:
        return None
    headers = {"Authorization": f"Bearer {_HC_API_KEY}", "Content-Type": "application/json"}
    client = _get_client()
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = client.post(f"{_BASE_URL}/chat/completions", headers=headers, json=payload)
            if resp.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError(
                    f"retryable {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as e:
            status = e.response.status_code if e.response is not None else None
            if status in (429, 500, 502, 503, 504) and attempt < _MAX_ATTEMPTS - 1:
                _sleep_backoff(attempt, f"HTTP {status}")
                continue
            logger.error(f"eval_common HTTP error: {status}")
            return None
        except (
            httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError,
            httpx.ReadError, httpx.WriteError, httpx.PoolTimeout,
        ) as e:
            if attempt < _MAX_ATTEMPTS - 1:
                _sleep_backoff(attempt, f"{type(e).__name__}")
                continue
            logger.error(f"eval_common transient error after {_MAX_ATTEMPTS} attempts: {e}")
            return None
        except Exception as e:
            logger.error(f"eval_common unexpected error: {e}")
            return None
    return None


def llm_answer(question: str, snippets: list[str], question_date: str = "", model: str | None = None) -> str:
    """
    Answer `question` from retrieved `snippets` via structured-JSON LLM call.
    Automatically retries with an explicit span-extraction prompt (no schema
    constraint) if the first pass abstains — this is the fix for the
    "Answer: None despite correct context" failure mode.

    Returns "" if no answer could be produced (including when HC_API_KEY unset).
    """
    model = model or DEFAULT_ANSWER_MODEL
    if not snippets or not _HC_API_KEY:
        return ""
    context = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(snippets[:10]))
    date_line = f"Question date: {question_date}\n" if question_date else ""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You are a precise question-answering system. Answer ONLY from the "
                "provided context snippets. If the answer isn't in the snippets, set "
                'found_in_context to false and answer to "".'
            )},
            {"role": "user", "content": f"{date_line}Context snippets:\n{context}\n\nQuestion: {question}"},
        ],
        "max_tokens": 256,
        "temperature": 0.0,
        "response_format": {"type": "json_schema", "json_schema": _ANSWER_SCHEMA},
    }
    content = _call(payload)
    answer = ""
    if content:
        try:
            parsed = json.loads(content)
            answer = str(parsed.get("answer", "")).strip()
        except json.JSONDecodeError:
            answer = content.strip()

    if not is_abstention(answer):
        return answer

    # Rephrase-and-retry: drop the structured-output constraint entirely in
    # case schema-following itself was what caused the abstention.
    retry_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "Extract the exact answer to the question from the snippets below. "
                "Quote the relevant fact verbatim, no explanation. If truly nothing "
                'is relevant, reply with exactly "I don\'t know".'
            )},
            {"role": "user", "content": f"{date_line}Snippets:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    retry_content = _call(retry_payload)
    if retry_content and not is_abstention(retry_content):
        return retry_content.strip()

    return answer  # still empty/abstained after both attempts


def judge_correct(hypothesis: str, gold_answer: str) -> bool:
    """Deterministic overlap-based judge — no extra LLM call, cheap and repeatable."""
    if not hypothesis or not gold_answer:
        return False
    hyp_l, gold_l = hypothesis.lower(), gold_answer.lower()
    if gold_l in hyp_l:
        return True
    gold_toks = set(re.findall(r"[A-Za-z0-9']+", gold_l)) - _STOPWORDS
    if not gold_toks:
        return False
    hyp_toks = set(re.findall(r"[A-Za-z0-9']+", hyp_l))
    if len(gold_toks) <= 3:
        return gold_toks.issubset(hyp_toks)
    return len(gold_toks & hyp_toks) / len(gold_toks) >= 0.7
