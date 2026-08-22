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
import math
import os
import re
import time
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
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
    # A CLI --answer-model override is a Z.AI model override.  Research-proxy
    # model selection remains config-owned so a GLM model name can never leak
    # across provider boundaries.
    requested_model = (
        settings.qa_model
        if preferred == "research_proxy"
        else payload.get("model") or settings.qa_model
    )
    messages = payload.get("messages", [])
    max_tokens = int(payload.get("max_tokens", 512))
    budget = active_budget()
    reserved_output_tokens = (
        budget.before_llm(messages, max_tokens) if budget is not None else 0
    )
    content = None
    try:
        content = llm_client.chat_completion(
            messages,
            max_tokens=max_tokens,
            temperature=payload.get("temperature", 0.0),
            model=requested_model,
            response_format=payload.get("response_format"),
            preferred=preferred,
            allow_fallback=False,
        )
    finally:
        if budget is not None:
            budget.after_llm(content, reserved_output_tokens)
    if content is None:
        raise AnswerProviderError(f"{preferred} returned no completion")
    return content


class AnswerProviderError(RuntimeError):
    """Configured answer provider failed or returned an unusable response."""


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    prompt_version: str
    status: str
    error: str | None = None


class EvaluationBudgetExceeded(RuntimeError):
    """A live evaluation crossed a predeclared resource or spend ceiling."""


class SearchExecutionAttestationError(RuntimeError):
    """The server response cannot prove the requested retrieval condition ran."""


_SEARCH_STAGES = {
    "vector_only": {"dense"},
    "sparse_only": {"sparse"},
    "vector_sparse": {"dense", "sparse"},
    "graph_only": {"graph"},
    "hybrid": {"dense", "sparse", "graph"},
}


def validate_search_execution(
    response_body: Mapping,
    *,
    expected_request: Mapping,
    require_reranker: bool,
) -> dict:
    """Validate and return a server-attested controlled-search trace.

    Evaluator request labels are not execution evidence. A completed condition
    therefore requires the API's versioned trace, exact requested mode,
    resolved control values, corpus generation, and every mode-required stage.
    Positive reranker conditions additionally require a successful, non-empty
    cross-encoder execution.
    """
    if not isinstance(response_body, Mapping):
        raise SearchExecutionAttestationError("search response must be an object")
    trace = response_body.get("execution_trace")
    if not isinstance(trace, Mapping):
        raise SearchExecutionAttestationError(
            "search response is missing execution_trace"
        )
    if trace.get("schema_version") != "hybridmind.search-execution/v1":
        raise SearchExecutionAttestationError(
            "search execution trace schema is missing or unsupported"
        )

    expected_mode = str(expected_request.get("search_mode") or "hybrid")
    if expected_mode not in _SEARCH_STAGES:
        raise SearchExecutionAttestationError(
            f"unsupported expected search mode: {expected_mode}"
        )
    if trace.get("search_mode") != expected_mode:
        raise SearchExecutionAttestationError(
            "server-attested search mode does not match the request"
        )

    corpus_generation = trace.get("corpus_generation")
    if (
        isinstance(corpus_generation, bool)
        or not isinstance(corpus_generation, int)
        or corpus_generation < 0
    ):
        raise SearchExecutionAttestationError(
            "execution trace lacks a valid corpus generation"
        )
    config_sha = trace.get("resolved_config_sha256")
    if not isinstance(config_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", config_sha):
        raise SearchExecutionAttestationError(
            "execution trace lacks a valid resolved configuration hash"
        )

    controls = trace.get("resolved_controls")
    if not isinstance(controls, Mapping):
        raise SearchExecutionAttestationError(
            "execution trace lacks resolved controls"
        )
    control_fields = {
        "search_mode": "search_mode",
        "top_k": "top_k",
        "rerank_pool": "rerank_pool",
        "route_weights": "route_weights",
        "fusion_mode": "fusion_mode",
        "vector_weight": "vector_weight",
        "graph_weight": "graph_weight",
        "bm25_boost_weight": "sparse_weight",
    }
    stage_bound_controls = {
        "vector_weight": "dense",
        "graph_weight": "graph",
        "bm25_boost_weight": "sparse",
    }
    for request_name, control_name in control_fields.items():
        if request_name not in expected_request:
            continue
        bound_stage = stage_bound_controls.get(request_name)
        if bound_stage is not None and bound_stage not in _SEARCH_STAGES[expected_mode]:
            # Controlled modes force unused signal weights to zero.
            if controls.get(control_name) != 0.0:
                raise SearchExecutionAttestationError(
                    f"disabled control {control_name} was not resolved to zero"
                )
            continue
        expected = expected_request.get(request_name)
        if expected is None:
            continue
        if controls.get(control_name) != expected:
            raise SearchExecutionAttestationError(
                f"resolved control {control_name} does not match the request"
            )

    stages = trace.get("stages")
    if not isinstance(stages, Mapping):
        raise SearchExecutionAttestationError("execution trace lacks stage evidence")
    required = _SEARCH_STAGES[expected_mode]
    for stage_name in ("dense", "sparse", "graph"):
        stage = stages.get(stage_name)
        if not isinstance(stage, Mapping):
            raise SearchExecutionAttestationError(
                f"execution trace lacks {stage_name} stage evidence"
            )
        should_run = stage_name in required
        if stage.get("requested") is not should_run:
            raise SearchExecutionAttestationError(
                f"server stage request flag is invalid for {stage_name}"
            )
        if stage.get("executed") is not should_run:
            raise SearchExecutionAttestationError(
                f"requested {stage_name} retrieval stage did not execute"
                if should_run
                else f"unrequested {stage_name} retrieval stage executed"
            )

    reranker = stages.get("cross_encoder")
    if not isinstance(reranker, Mapping):
        raise SearchExecutionAttestationError(
            "execution trace lacks cross-encoder stage evidence"
        )
    if require_reranker and not (
        reranker.get("attempted") is True
        and reranker.get("applied") is True
        and int(reranker.get("candidates") or 0) > 0
        and not reranker.get("failure_type")
    ):
        raise SearchExecutionAttestationError(
            "required cross-encoder reranker did not execute successfully"
        )
    return dict(trace)


def validate_rerank_pool(*, top_k: int, rerank_pool: int) -> None:
    """Validate the evaluator/API reranking window contract.

    A positive pool is a hard upper bound on cross-encoder work and must still
    contain every requested final result.  Zero is the only disabled value.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if rerank_pool < 0:
        raise ValueError("rerank_pool must be non-negative; use 0 to disable reranking")
    if 0 < rerank_pool < top_k:
        raise ValueError(
            "positive rerank_pool must be greater than or equal to top_k; "
            "use 0 to disable reranking"
        )


@dataclass
class EvaluationBudget:
    max_queries: int
    max_llm_calls: int
    max_embedding_texts: int
    max_input_tokens: int
    max_output_tokens: int
    max_wall_seconds: float
    max_estimated_spend_usd: float
    input_cost_per_million_tokens: float = 0.0
    output_cost_per_million_tokens: float = 0.0
    embedding_cost_per_text_usd: float | None = None
    allow_unpriced_embedding: bool = False
    queries: int = 0
    llm_calls: int = 0
    embedding_texts: int = 0
    input_tokens_estimated: int = 0
    output_tokens_estimated: int = 0
    started_monotonic: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "max_queries", "max_llm_calls", "max_embedding_texts",
            "max_input_tokens", "max_output_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.embedding_cost_per_text_usd is not None and self.embedding_cost_per_text_usd < 0:
            raise ValueError("embedding_cost_per_text_usd must be non-negative")
        for name in (
            "max_wall_seconds", "max_estimated_spend_usd",
            "input_cost_per_million_tokens", "output_cost_per_million_tokens",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        self.started_monotonic = time.monotonic()

    @staticmethod
    def estimate_tokens_from_chars(characters: int) -> int:
        return math.ceil(max(0, characters) / 4)

    def _wall_seconds(self) -> float:
        return time.monotonic() - self.started_monotonic

    def _estimated_spend(self, input_tokens: int | None = None, output_tokens: int | None = None) -> float:
        inputs = self.input_tokens_estimated if input_tokens is None else input_tokens
        outputs = self.output_tokens_estimated if output_tokens is None else output_tokens
        token_spend = (
            inputs * self.input_cost_per_million_tokens
            + outputs * self.output_cost_per_million_tokens
        ) / 1_000_000
        embedding_spend = self.embedding_texts * (self.embedding_cost_per_text_usd or 0.0)
        return token_spend + embedding_spend

    def _check_wall(self) -> None:
        if self._wall_seconds() > self.max_wall_seconds:
            raise EvaluationBudgetExceeded(
                f"wall-time budget exceeded ({self.max_wall_seconds}s)"
            )

    def record_query(self, *, embedding_texts: int = 1) -> None:
        self._check_wall()
        if (
            embedding_texts
            and self.embedding_cost_per_text_usd is None
            and not self.allow_unpriced_embedding
        ):
            raise EvaluationBudgetExceeded(
                "embedding spend is unpriced; set --embedding-cost-per-text-usd "
                "or explicitly --allow-unpriced-embedding"
            )
        if self.queries + 1 > self.max_queries:
            raise EvaluationBudgetExceeded("retrieval-query budget exceeded")
        if self.embedding_texts + embedding_texts > self.max_embedding_texts:
            raise EvaluationBudgetExceeded("embedding-text budget exceeded")
        self.queries += 1
        self.embedding_texts += embedding_texts
        if self._estimated_spend() > self.max_estimated_spend_usd:
            # Counts remain recorded because the query budget reservation has
            # occurred, but the external request has not yet been sent.
            raise EvaluationBudgetExceeded("estimated provider-spend budget exceeded")

    def before_llm(self, messages: list[dict], max_output_tokens: int) -> int:
        self._check_wall()
        input_chars = sum(len(str(message.get("content", ""))) for message in messages)
        input_tokens = self.estimate_tokens_from_chars(input_chars)
        prospective_inputs = self.input_tokens_estimated + input_tokens
        prospective_outputs = self.output_tokens_estimated + max_output_tokens
        if self.llm_calls + 1 > self.max_llm_calls:
            raise EvaluationBudgetExceeded("LLM-call budget exceeded")
        if prospective_inputs > self.max_input_tokens:
            raise EvaluationBudgetExceeded("estimated LLM input-token budget exceeded")
        if prospective_outputs > self.max_output_tokens:
            raise EvaluationBudgetExceeded("conservative LLM output-token budget exceeded")
        if self._estimated_spend(prospective_inputs, prospective_outputs) > self.max_estimated_spend_usd:
            raise EvaluationBudgetExceeded("estimated provider-spend budget exceeded")
        self.llm_calls += 1
        self.input_tokens_estimated = prospective_inputs
        return max_output_tokens

    def after_llm(self, content: str | None, reserved_output_tokens: int) -> None:
        actual = self.estimate_tokens_from_chars(len(content or ""))
        # The pre-call check reserved the provider's max_tokens. Usage records
        # the conservative reservation so actual spend cannot exceed the cap.
        self.output_tokens_estimated += reserved_output_tokens
        self._check_wall()

    def ceilings(self) -> dict:
        return {
            "max_queries": self.max_queries,
            "max_llm_calls": self.max_llm_calls,
            "max_embedding_texts": self.max_embedding_texts,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
            "max_wall_seconds": self.max_wall_seconds,
            "max_estimated_spend_usd": self.max_estimated_spend_usd,
            "input_cost_per_million_tokens": self.input_cost_per_million_tokens,
            "output_cost_per_million_tokens": self.output_cost_per_million_tokens,
            "embedding_cost_per_text_usd": self.embedding_cost_per_text_usd,
            "allow_unpriced_embedding": self.allow_unpriced_embedding,
        }

    def usage(self) -> dict:
        return {
            "queries": self.queries,
            "llm_calls": self.llm_calls,
            "embedding_texts": self.embedding_texts,
            "input_tokens_estimated": self.input_tokens_estimated,
            "output_tokens_conservative": self.output_tokens_estimated,
            "wall_seconds": self._wall_seconds(),
            "estimated_spend_usd_conservative": self._estimated_spend(),
        }

    @contextmanager
    def activate(self):
        token = _ACTIVE_EVAL_BUDGET.set(self)
        try:
            yield self
        finally:
            _ACTIVE_EVAL_BUDGET.reset(token)


_ACTIVE_EVAL_BUDGET: ContextVar[EvaluationBudget | None] = ContextVar(
    "active_eval_budget", default=None
)


def active_budget() -> EvaluationBudget | None:
    return _ACTIVE_EVAL_BUDGET.get()


def record_retrieval_query(*, embedding_texts: int = 1) -> None:
    budget = active_budget()
    if budget is not None:
        budget.record_query(embedding_texts=embedding_texts)


def live_request_timeout(default_seconds: float) -> float:
    budget = active_budget()
    if budget is None:
        return default_seconds
    budget._check_wall()
    remaining = budget.max_wall_seconds - budget._wall_seconds()
    if remaining <= 0:
        raise EvaluationBudgetExceeded("wall-time budget exceeded")
    return max(0.001, min(default_seconds, remaining))


def record_retrieval_response() -> None:
    budget = active_budget()
    if budget is not None:
        budget._check_wall()


def active_budget_provenance() -> dict | None:
    budget = active_budget()
    return None if budget is None else {"ceilings": budget.ceilings(), "usage": budget.usage()}


def api_headers() -> dict[str, str]:
    api_key = str(settings.api_key or "").strip()
    return {"X-HybridMind-API-Key": api_key} if api_key else {}


def sanitized_error(exc: BaseException) -> str:
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    return f"{type(exc).__name__} (HTTP {status_code})" if status_code else type(exc).__name__


def enforce_priced_llm_budget(
    args, *, answer_requested: bool = False, decomposition_requested: bool = False
) -> None:
    """Reject live paid-provider paths whose spend cannot be bounded.

    QA is pinned to Z.AI outside explicit research mode. Decomposition uses
    the automatic provider chain, so Z.AI pricing is required whenever that
    chain could fall through to Z.AI, even if a self-hosted RunPod is first.
    """
    paid_possible = (
        answer_requested and not settings.allow_research_proxy
    ) or (
        decomposition_requested
        and "zai" in llm_client.provider_chain()
    )
    if paid_possible and (
        args.input_cost_per_million_tokens <= 0
        or args.output_cost_per_million_tokens <= 0
    ):
        raise SystemExit(
            "Paid LLM evaluation requires explicit input/output token prices "
            "for spend enforcement"
        )


def add_budget_arguments(parser) -> None:
    """Add safe live-evaluation ceilings to an argparse parser."""
    parser.add_argument("--execute", action="store_true", help="Perform live API/provider calls (default: dry plan only)")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--max-llm-calls", type=int, default=0)
    parser.add_argument("--max-embedding-texts", type=int, default=100)
    parser.add_argument("--max-input-tokens", type=int, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=0)
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    parser.add_argument("--max-estimated-spend-usd", type=float, default=0.0)
    parser.add_argument("--input-cost-per-million-tokens", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million-tokens", type=float, default=0.0)
    parser.add_argument("--embedding-cost-per-text-usd", type=float)
    parser.add_argument("--allow-unpriced-embedding", action="store_true")


def budget_from_args(args) -> EvaluationBudget:
    return EvaluationBudget(
        max_queries=args.max_queries,
        max_llm_calls=args.max_llm_calls,
        max_embedding_texts=args.max_embedding_texts,
        max_input_tokens=args.max_input_tokens,
        max_output_tokens=args.max_output_tokens,
        max_wall_seconds=args.max_wall_seconds,
        max_estimated_spend_usd=args.max_estimated_spend_usd,
        input_cost_per_million_tokens=args.input_cost_per_million_tokens,
        output_cost_per_million_tokens=args.output_cost_per_million_tokens,
        embedding_cost_per_text_usd=args.embedding_cost_per_text_usd,
        allow_unpriced_embedding=args.allow_unpriced_embedding,
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


def answer_question_with_status(
    question: str,
    snippets: list[str],
    question_type: str = "default",
    question_date: str = "",
    model: str | None = None,
) -> AnswerResult:
    """Generate an answer without collapsing infrastructure failures into zero.

    ``abstained`` is a model outcome and may legitimately count as an incorrect
    answer. ``provider_unavailable`` and ``provider_error`` are run failures;
    evaluators must ledger them and fail closed instead of adding a zero to the
    accuracy denominator.
    """
    prompt_version = (
        QA_PROMPT_VERSION
        if LEGACY_ANSWERING
        else QA_MULTIHOP_PROMPT_VERSION
        if question_type == "multihop"
        else QA_CITATION_PROMPT_VERSION
    )
    if not snippets:
        return AnswerResult("", prompt_version, "no_context")
    if not _is_llm_available(model):
        return AnswerResult("", prompt_version, "provider_unavailable")
    try:
        answer, actual_prompt_version = answer_question(
            question,
            snippets,
            question_type=question_type,
            question_date=question_date,
            model=model,
        )
    except AnswerProviderError as exc:
        return AnswerResult("", prompt_version, "provider_error", str(exc))
    if is_abstention(answer):
        return AnswerResult(answer, actual_prompt_version, "abstained")
    return AnswerResult(answer, actual_prompt_version, "completed")


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
    from engine.query_decomposition import _DECOMPOSE_SYSTEM_PROMPT, decompose_query

    if question_type != "multihop":
        return post_fn(query_text)

    budget = active_budget()
    reserved_output_tokens = 0
    decomposition_will_call = bool(
        decompose_enabled
        and query_text.strip()
        and len(query_text.strip()) <= 2_000
        and llm_client.is_configured()
    )
    if budget is not None and decomposition_will_call:
        reserved_output_tokens = budget.before_llm(
            [
                {"role": "system", "content": _DECOMPOSE_SYSTEM_PROMPT},
                {"role": "user", "content": query_text},
            ],
            300,
        )
    try:
        sub_questions = decompose_query(query_text, model=model, enabled=decompose_enabled)
    finally:
        if budget is not None and decomposition_will_call:
            rendered = json.dumps(locals().get("sub_questions", []))
            budget.after_llm(rendered, reserved_output_tokens)
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
    """Deterministic answer-overlap heuristic; this is not an LLM judge."""
    return judge_correct_with_rationale(hypothesis, gold_answer)[0]


def judge_correct_with_rationale(hypothesis: str, gold_answer: str) -> tuple[bool, str]:
    """Return the deterministic answer-overlap verdict and rule that fired."""
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
