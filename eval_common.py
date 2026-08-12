"""
Shared LLM QA-answering helper for eval_*.py scripts.

The historical failure mode (docs/LOCOMO_BENCHMARK_REPORT.md): an answering LLM
would abstain ("Answer: None") even when the correct context was retrieved,
tanking single-hop accuracy to 0% despite 60%+ Hit@10. That bug lived in the
external `memorybench/` harness (not part of this repo) and has since been
fixed there via a rephrase-and-extract retry.

The eval_*.py scripts in this repo are retrieval-only (no answering LLM at
all), so they were immune to that bug but also couldn't report answer
accuracy. This module adds that capability directly, using Z.AI's
OpenAI-compatible API with the same defensive rephrase-retry so the bug class
can't recur here.
"""
import json
import logging
import os
import re
from pathlib import Path

from config import settings
from engine import llm_client

logger = logging.getLogger(__name__)

DEFAULT_ANSWER_MODEL = settings.qa_model

# Phase 6.1 (docs/PHASE_6_REALISTIC.md section 3): set to "true" to fall back to
# the pre-6.1 single-shot answering prompt (no citation, no multi-hop
# iteration, no answer normalization before judging), so the two behaviors
# can be A/B'd against each other via the ledger.
LEGACY_ANSWERING = os.getenv("HYBRIDMIND_EVAL_LEGACY_ANSWERING", "false").strip().lower() == "true"

_STOPWORDS = {"the", "a", "an", "in", "on", "at", "to", "for", "of", "is", "was", "it", "and", "or", "but"}

_ABSTENTION_RE = re.compile(r"^\s*(i don'?t know|none|not (specified|available|mentioned)|n/?a)\b", re.I)

# Versioned eval prompts (docs/PHASE_6_REALISTIC.md §6.0.2): a changed prompt
# invalidates prior ledger rows for A/B comparison, so the version travels
# with every judged answer. Bump this whenever the prompt text below changes.
QA_PROMPT_VERSION = "qa_v1"                # legacy single-shot prompt (llm_answer)
QA_CITATION_PROMPT_VERSION = "qa_citation_v1"   # 6.1(a): evidence-citation-then-answer
QA_MULTIHOP_PROMPT_VERSION = "qa_multihop_v1"   # 6.1(c): iterative evidence-then-conclude

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


def is_abstention(text: str) -> bool:
    text = (text or "").strip()
    return (not text) or bool(_ABSTENTION_RE.match(text))


def _is_llm_available(model: str | None = None) -> bool:
    # The explicit research opt-in selects the free proxy for evals so a failed
    # research call cannot unexpectedly spend the Z.AI budget. Production mode
    # remains pinned to canonical Z.AI.
    preferred = "research_proxy" if settings.allow_research_proxy else "zai"
    return llm_client.is_configured(preferred, allow_fallback=False)


def _call(payload: dict) -> str | None:
    """Call canonical Z.AI, or the explicitly selected research-only proxy."""
    preferred = "research_proxy" if settings.allow_research_proxy else "zai"
    return llm_client.chat_completion(
        payload.get("messages", []),
        max_tokens=payload.get("max_tokens", 512),
        temperature=payload.get("temperature", 0.0),
        model=settings.qa_model,
        response_format=payload.get("response_format"),
        preferred=preferred,
        allow_fallback=False,
    )

def llm_answer(question: str, snippets: list[str], question_date: str = "", model: str | None = None) -> str:
    """
    Answer `question` from retrieved `snippets` via structured-JSON LLM call.
    Automatically retries with an explicit span-extraction prompt (no schema
    constraint) if the first pass abstains — this is the fix for the
    "Answer: None despite correct context" failure mode.

    Returns "" if no answer could be produced (including when ZAI_API_KEY is unset).
    """
    model = model or DEFAULT_ANSWER_MODEL
    if not snippets or not _is_llm_available(model):
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


_CITATION_SCHEMA = {
    "name": "cited_answer",
    "schema": {
        "type": "object",
        "properties": {
            "citations": {"type": "array", "items": {"type": "integer"}},
            "answer": {"type": "string"},
            "found_in_context": {"type": "boolean"},
        },
        "required": ["citations", "answer", "found_in_context"],
    },
    "strict": True,
}


def llm_answer_citation(question: str, snippets: list[str], question_date: str = "", model: str | None = None) -> str:
    """
    6.1(a): evidence-citation-then-answer. Forces the model to name which
    snippet indices support its answer before writing the answer itself —
    the citation step is discarded here (the retriever, not this eval, is what
    Phase 6.2 improves), but requiring it first has been shown to reduce
    answers invented without grounding in the provided context.
    """
    model = model or DEFAULT_ANSWER_MODEL
    if not snippets or not _is_llm_available(model):
        return ""
    context = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(snippets[:10]))
    date_line = f"Question date: {question_date}\n" if question_date else ""

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "You are a precise question-answering system. First identify which "
                "numbered snippets (if any) contain the answer, listing their indices "
                "in `citations`. Then answer ONLY using the cited snippets. If no "
                'snippet supports an answer, set found_in_context to false, citations '
                'to [], and answer to "".'
            )},
            {"role": "user", "content": f"{date_line}Context snippets:\n{context}\n\nQuestion: {question}"},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
        "response_format": {"type": "json_schema", "json_schema": _CITATION_SCHEMA},
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

    # Same defensive rephrase-retry as llm_answer(), unconstrained by the schema.
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
    return answer


def llm_answer_multihop(question: str, snippets: list[str], question_date: str = "", model: str | None = None) -> str:
    """
    6.1(c): iterative evidence-then-conclude, for router-classified multi-hop
    questions. Pass 1 asks the model to state, per aspect of the question,
    which fact (if any) each snippet establishes; pass 2 asks it to compose
    the final answer strictly from those extracted facts. This gives multi-hop
    composition an explicit intermediate step instead of asking for the final
    answer directly from a flat snippet list.
    """
    model = model or DEFAULT_ANSWER_MODEL
    if not snippets or not _is_llm_available(model):
        return ""
    context = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(snippets[:10]))
    date_line = f"Question date: {question_date}\n" if question_date else ""

    extract_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "This question likely requires combining facts from more than one "
                "snippet. List each fact relevant to answering it as a short bullet "
                "point, citing the snippet index. If a snippet is irrelevant, ignore it. "
                "Output only the bullet points, no preamble."
            )},
            {"role": "user", "content": f"{date_line}Context snippets:\n{context}\n\nQuestion: {question}"},
        ],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    facts = _call(extract_payload)
    if not facts or is_abstention(facts):
        # No intermediate facts extracted — fall through to the citation prompt
        # rather than returning empty outright.
        return llm_answer_citation(question, snippets, question_date=question_date, model=model)

    conclude_payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": (
                "Given the extracted facts below, compose the final answer to the "
                "question by combining them. Answer with the fact itself, no "
                'explanation. If the facts do not answer the question, reply with '
                'exactly "I don\'t know".'
            )},
            {"role": "user", "content": f"{date_line}Extracted facts:\n{facts}\n\nQuestion: {question}\n\nAnswer:"},
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    conclusion = _call(conclude_payload)
    if conclusion and not is_abstention(conclusion):
        return conclusion.strip()
    return ""


def answer_question(
    question: str,
    snippets: list[str],
    question_type: str = "default",
    question_date: str = "",
    model: str | None = None,
) -> tuple[str, str]:
    """
    Single dispatch point used by all eval_*.py scripts. Returns
    (hypothesis, prompt_version) so the ledger always records which prompt
    strategy produced the answer.

    HYBRIDMIND_EVAL_LEGACY_ANSWERING=true reverts to the pre-6.1 single-shot
    prompt (llm_answer) for A/B comparison against the 6.1 interventions.
    """
    if LEGACY_ANSWERING:
        return llm_answer(question, snippets, question_date=question_date, model=model), QA_PROMPT_VERSION
    if question_type == "multihop":
        return (
            llm_answer_multihop(question, snippets, question_date=question_date, model=model),
            QA_MULTIHOP_PROMPT_VERSION,
        )
    return (
        llm_answer_citation(question, snippets, question_date=question_date, model=model),
        QA_CITATION_PROMPT_VERSION,
    )


# --------------------------------------------------------------------- #
# 6.1(b): deterministic answer normalization, applied BEFORE judging.
# Pure Python — never delegated to the LLM, so it can't itself hallucinate.
# --------------------------------------------------------------------- #

_MONTHS = {
    "january": "01", "february": "02", "march": "03", "april": "04", "may": "05", "june": "06",
    "july": "07", "august": "08", "september": "09", "october": "10", "november": "11", "december": "12",
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "jun": "06", "jul": "07",
    "aug": "08", "sep": "09", "sept": "09", "oct": "10", "nov": "11", "dec": "12",
}

_WORD_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
    "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
}

_DATE_TEXT_RE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b", re.I
)
_DATE_TEXT_RE2 = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")\.?,?\s+(\d{4})\b", re.I
)
_LEADING_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.I)


def normalize_answer(text: str) -> str:
    """
    Canonicalize dates, number words, and casing so semantically-identical
    answers ("5 January 2023" vs "January 5th, 2023" vs "5") compare equal to
    the judge instead of diverging on surface form alone.
    """
    if not text:
        return ""
    s = text.strip()

    def _month_day_year(m):
        month = _MONTHS[m.group(1).lower()]
        day = m.group(2).zfill(2)
        year = m.group(3)
        return f"{year}-{month}-{day}"

    def _day_month_year(m):
        day = m.group(1).zfill(2)
        month = _MONTHS[m.group(2).lower()]
        year = m.group(3)
        return f"{year}-{month}-{day}"

    s = _DATE_TEXT_RE.sub(_month_day_year, s)
    s = _DATE_TEXT_RE2.sub(_day_month_year, s)

    s = s.lower().strip()
    s = _LEADING_ARTICLE_RE.sub("", s)

    words = s.split()
    words = [_WORD_NUMBERS.get(w.strip(",."), w) for w in words]
    s = " ".join(words)

    s = re.sub(r"\s+", " ", s).strip(" .,")
    return s


def judge_correct_normalized(hypothesis: str, gold_answer: str) -> tuple[bool, str]:
    """Same as judge_correct_with_rationale(), but normalizes both sides first (6.1b)."""
    norm_hyp = normalize_answer(hypothesis)
    norm_gold = normalize_answer(gold_answer)
    ok, rationale = judge_correct_with_rationale(norm_hyp, norm_gold)
    return ok, f"[normalized] {rationale}"


def retrieve_with_decomposition(
    query_text: str,
    question_type: str,
    post_fn,
    decompose_enabled: bool = True,
    model: str | None = None,
) -> list:
    """
    6.2.2: for multihop-routed queries, decompose into sub-questions and
    retrieve per sub-question via `post_fn(query_string) -> list[dict]`,
    unioning candidates by node_id (keeping the highest combined_score seen
    across sub-question retrievals). Falls through to a single
    post_fn(query_text) call whenever decomposition doesn't apply — including
    the guards inside decompose_query() (RunPod not configured, degenerates
    to <=1 sub-question, or all sub-questions rejected for novel entities).
    """
    from engine.query_decomposition import decompose_query

    if question_type != "multihop":
        return post_fn(query_text)

    sub_questions = decompose_query(query_text, model=model, enabled=decompose_enabled)
    if not sub_questions:
        return post_fn(query_text)

    seen: dict = {}
    for sq in sub_questions:
        for r in post_fn(sq):
            nid = r.get("node_id") or r.get("id")
            if nid is None:
                continue
            if nid not in seen or r.get("combined_score", 0) > seen[nid].get("combined_score", 0):
                seen[nid] = r
    unioned = sorted(seen.values(), key=lambda r: -r.get("combined_score", 0))
    return unioned or post_fn(query_text)


def export_training_record(path: str, query_id: str, query_type: str, candidates: list) -> bool:
    """
    Append one training record to a shared fusion_train_data.jsonl.

    candidates: [{"node_id": str, "dense_score": float, "bm25_score": float,
                   "graph_score": float, "is_relevant": bool}, ...]
                 — pass the FULL rerank pool, not just the trimmed top-k, so
                   correct-but-buried candidates are still usable training pairs.

    Returns True if the record has at least one positive and one negative
    candidate (i.e. is actually usable for pairwise training).
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    record = {"query_id": query_id, "query_type": query_type, "candidates": candidates}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    has_pos = any(c.get("is_relevant") for c in candidates)
    has_neg = any(not c.get("is_relevant") for c in candidates)
    return has_pos and has_neg


def judge_correct(hypothesis: str, gold_answer: str) -> bool:
    """Deterministic overlap-based judge — no extra LLM call, cheap and repeatable."""
    return judge_correct_with_rationale(hypothesis, gold_answer)[0]


def judge_correct_with_rationale(hypothesis: str, gold_answer: str) -> tuple[bool, str]:
    """Same verdict as judge_correct(), plus the rule that fired (for the ledger)."""
    if not hypothesis or not gold_answer:
        return False, "empty hypothesis or gold answer"
    hyp_l, gold_l = hypothesis.lower(), gold_answer.lower()
    if gold_l in hyp_l:
        return True, "gold answer is a substring of the hypothesis"
    gold_toks = set(re.findall(r"[A-Za-z0-9']+", gold_l)) - _STOPWORDS
    if not gold_toks:
        return False, "gold answer has no non-stopword tokens"
    hyp_toks = set(re.findall(r"[A-Za-z0-9']+", hyp_l))
    if len(gold_toks) <= 3:
        ok = gold_toks.issubset(hyp_toks)
        return ok, f"gold token subset check ({'passed' if ok else 'failed'}, {len(gold_toks)} tokens)"
    overlap = len(gold_toks & hyp_toks) / len(gold_toks)
    ok = overlap >= 0.7
    return ok, f"token overlap {overlap:.2f} {'>=' if ok else '<'} 0.70 threshold"
