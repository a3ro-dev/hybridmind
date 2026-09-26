"""LoCoMo and LongMemEval loaders that produce scope-local ``ScopedCorpus`` views.

Zero provider calls. One ``Conversation`` is one retrieval scope: a LoCoMo
conversation (many questions) or a LongMemEval question haystack (one
question). Every channel ranks the same ``TurnRecord`` list, so gold evidence
IDs here are directly comparable with channel output.

Conventions follow the E-series offline harness
(``scripts/offline_budgeted_evidence.py``) unless noted in ``DEVIATIONS``:

- LoCoMo ``index_text`` is the E-series ``spk_cap`` representation
  (``index_text(t, "spk_cap")``, chosen in research/experiments/e1-budgeted-units
  and e2-aggregation-keys): ``"{speaker}: {text}"`` plus ``" {blip_caption}"``.
- LoCoMo category IDs use the official mapping (``CATEGORY`` in
  ``scripts/offline_locomo_sparse_baseline.py``; tests/test_locomo_category_map.py).
- LongMemEval gold turns are ``has_answer`` turns; ``gold_user_evidence`` keeps
  only user-side ones, which is the official retrieval convention
  (LongMemEval ``src/retrieval/run_retrieval.py``; see
  ``scripts/reproduce_longmemeval_retrieval.py``).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from engine.corpus import ScopedCorpus, TurnRecord
from scripts.offline_locomo_sparse_baseline import CATEGORY as LOCOMO_CATEGORY

# Known dataset files: {"bytes", "sha256"}.
LONGMEMEVAL_S_CLEANED = {
    # HF xiaowu0162/longmemeval-cleaned LFS oid; research/experiments/e4-lme-confirmation and
    # experiments/results/offline-lme-s-cleaned-official-bm25-session-20260925.json.
    "bytes": 277_383_467,
    "sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
}
# memorybench/data/benchmarks/longmemeval/longmemeval_s.json has the size of the *oracle*
# file (evidence sessions only). It is not a retrieval corpus and is refused.
LONGMEMEVAL_S_ORACLE_INVALID = {
    "bytes": 15_388_478,
    "sha256": "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
}
# memorybench/data/benchmarks/locomo/locomo10.json (used by every E-series and engine run).
LOCOMO10 = {
    "bytes": 2_805_274,
    "sha256": "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
}
# EMG run_config dataset (github.com/Sun668/em_graph_memory data/locomo10.json). A derivative:
# its 446 category-5 rows fold ``adversarial_answer`` into ``answer`` and drop the former key;
# everything else is identical, so both files load to the same Conversations.
EMG_LOCOMO10 = {
    "bytes": 2_737_755,
    "sha256": "047d8e2528126afd02ab4b6fbade825ad390447e7a91861d35960c85752b4d74",
}
KNOWN_DATASETS = {
    "longmemeval_s_cleaned": LONGMEMEVAL_S_CLEANED,
    "locomo10": LOCOMO10,
    "emg_locomo10": EMG_LOCOMO10,
}

DEVIATIONS = [
    "LoCoMo question_date: the dataset has none; questions are dated at the last session's "
    "datetime (conversation end), as in scripts/offline_budgeted_evidence.py.",
    "LoCoMo gold evidence: malformed strings are split into D<n>:<m> ids ('D8:6; D9:17', "
    "'D9:1 D4:4 D4:6'); the official scorer and eval_locomo_retrieval.py keep them whole "
    "(never matchable). Unparseable strings ('D', 'D:11:26') and ids with no turn in the "
    "conversation go to malformed_evidence, not gold_evidence.",
    "LoCoMo index_text strips the raw turn text before prefixing the speaker (E-series spk_cap "
    "parity; 209 turns carry edge whitespace). TurnRecord.text stays raw.",
    "LongMemEval: 13 haystacks repeat a session id (identical content, other date, never an "
    "answer session). The first occurrence keeps session_key=session_id; later ones become "
    "'{session_id}@{position}' so node ids stay unique. metadata['session_id'] keeps the raw "
    "id; longmemeval_official_turn_view maps repeats back to it, as upstream does.",
    "LongMemEval is streamed with ijson (as scripts/offline_budgeted_evidence.py and "
    "scripts/reproduce_longmemeval_retrieval.py already do; same data, bounded RAM) instead "
    "of json.load. Requested question_ids missing from the file raise instead of being skipped.",
    "LongMemEval empty-content turns (12 in S-cleaned) are kept; the E-series harness dropped "
    "them. Evidence ids are positional, so ids are unaffected.",
]

_SESSION = re.compile(r"^session_(\d+)$")
_DIA = re.compile(r"D\d+:\d+")
_LOCOMO_DATE = "%I:%M %p on %d %B, %Y"  # '1:56 pm on 8 May, 2023'
_LME_DATE = "%Y/%m/%d (%a) %H:%M"  # '2023/05/20 (Sat) 02:21'


@dataclass(frozen=True)
class Question:
    """One benchmark question. Gold is kept as annotated (abstention rows included);
    ``benchmarks.conversational_metrics.scoring_gold`` applies a denominator protocol."""

    qid: str
    scope_key: str
    question: str
    answer: str
    category: Union[int, str]
    category_name: str
    gold_evidence: Tuple[str, ...]
    gold_sessions: Tuple[str, ...]
    # LongMemEval only (user-side has_answer turns); empty for LoCoMo.
    gold_user_evidence: Tuple[str, ...]
    question_date: Optional[str]
    abstention: bool
    malformed_evidence: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Conversation:
    scope_key: str
    corpus: ScopedCorpus
    questions: Tuple[Question, ...]


def dataset_identity(path: Union[str, Path]) -> Dict[str, Any]:
    """``{path, bytes, sha256, known}``; raises on the invalid LongMemEval oracle file."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    identity = {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    if identity["sha256"] == LONGMEMEVAL_S_ORACLE_INVALID["sha256"]:
        raise ValueError(_oracle_message(path))
    identity["known"] = next(
        (name for name, ref in KNOWN_DATASETS.items()
         if ref["sha256"] == identity["sha256"] and ref["bytes"] == identity["bytes"]),
        None,
    )
    return identity


def _oracle_message(path: Path) -> str:
    return (
        f"{path} is the LongMemEval oracle file ({LONGMEMEVAL_S_ORACLE_INVALID['bytes']:,} bytes, "
        "evidence sessions only), not a retrieval haystack; use longmemeval_s_cleaned.json "
        f"({LONGMEMEVAL_S_CLEANED['bytes']:,} bytes, sha256 {LONGMEMEVAL_S_CLEANED['sha256']})"
    )


def _iso(value: str, fmt: str, what: str) -> str:
    try:
        return datetime.strptime(value.strip(), fmt).isoformat()
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"unparseable {what} date {value!r} (expected {fmt!r})") from exc


def parse_locomo_datetime(value: str) -> str:
    """'1:56 pm on 8 May, 2023' -> '2023-05-08T13:56:00'."""
    return _iso(value, _LOCOMO_DATE, "LoCoMo session")


def parse_longmemeval_datetime(value: str) -> str:
    """'2023/05/20 (Sat) 02:21' -> '2023-05-20T02:21:00'."""
    return _iso(value, _LME_DATE, "LongMemEval")


def split_locomo_evidence(raw: Sequence[Any], known_ids: Iterable[str]) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Return ``(gold_ids, malformed)``; gold ids keep annotation order, deduplicated."""
    known = set(known_ids)
    gold: List[str] = []
    malformed: List[str] = []
    for value in raw:
        text = str(value)
        ids = _DIA.findall(text)
        if not ids or _DIA.sub("", text).strip(" ;,\t\r\n"):
            malformed.append(text)
        for dia in ids:
            if dia not in known:
                malformed.append(dia)
            elif dia not in gold:
                gold.append(dia)
    return tuple(gold), tuple(malformed)


def load_locomo(path: Union[str, Path]) -> List[Conversation]:
    """Ten LoCoMo conversations, one scope each, with every QA row as a ``Question``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_locomo_conversation(item) for item in data]


def _locomo_conversation(item: Dict[str, Any]) -> Conversation:
    sid = str(item["sample_id"])
    scope = f"locomo:{sid}"
    conv = item["conversation"]
    numbers = sorted(int(m.group(1)) for m in map(_SESSION.match, conv) if m)
    turns: List[TurnRecord] = []
    session_of: Dict[str, str] = {}
    last_iso = None
    for n in numbers:
        raw_date = conv.get(f"session_{n}_date_time")
        if raw_date is None:
            raise ValueError(f"{sid}: session_{n} has no session_{n}_date_time")
        iso = parse_locomo_datetime(str(raw_date))
        last_iso = max(last_iso or iso, iso)
        for i, m in enumerate(conv[f"session_{n}"]):
            dia, speaker, text = str(m["dia_id"]), str(m.get("speaker") or ""), str(m.get("text") or "")
            caption = m.get("blip_caption")
            index_text = f"{speaker}: {text.strip()}" + (f" {str(caption).strip()}" if caption else "")
            session_of[dia] = f"{sid}:S{n}"
            turns.append(TurnRecord(
                node_id=f"locomo:{sid}:{dia}",
                evidence_id=dia,
                text=text,
                session_key=f"{sid}:S{n}",
                order=(iso, n, i),
                speaker=speaker,
                timestamp=iso,
                index_text=index_text,
                metadata={
                    "dia_id": dia, "session_number": n, "turn_index": i,
                    "blip_caption": caption, "img_url": m.get("img_url"),
                    "sample_id": sid, "session_date_time": str(raw_date),
                },
            ))
    corpus = ScopedCorpus(turns, scope_key=scope)
    questions = []
    for qa_index, qa in enumerate(item.get("qa") or []):
        gold, malformed = split_locomo_evidence(qa.get("evidence") or [], session_of)
        category = int(qa["category"])
        answer = qa.get("answer")
        if answer is None and category == 5:
            answer = qa.get("adversarial_answer")
        questions.append(Question(
            qid=f"locomo:{sid}:{qa_index}",
            scope_key=scope,
            question=str(qa["question"]),
            answer="" if answer is None else str(answer),
            category=category,
            category_name=LOCOMO_CATEGORY[category],
            gold_evidence=gold,
            gold_sessions=tuple(dict.fromkeys(session_of[g] for g in gold)),
            gold_user_evidence=(),
            question_date=last_iso,
            abstention=category == 5,
            malformed_evidence=malformed,
        ))
    return Conversation(scope_key=scope, corpus=corpus, questions=tuple(questions))


def load_longmemeval(
    path: Union[str, Path],
    limit: Optional[int] = None,
    question_ids: Optional[Sequence[str]] = None,
) -> Iterator[Conversation]:
    """One ``Conversation`` per question haystack, streamed in file order.

    ``question_ids`` filters (then ``limit`` caps the yielded count); ids absent from
    the file raise once the stream ends. The oracle file is refused at call time.
    """
    path = Path(path)
    if path.stat().st_size == LONGMEMEVAL_S_ORACLE_INVALID["bytes"]:
        raise ValueError(_oracle_message(path))
    return _stream_longmemeval(path, limit, None if question_ids is None else set(question_ids))


def _stream_longmemeval(path: Path, limit: Optional[int], wanted: Optional[set]) -> Iterator[Conversation]:
    import ijson

    if limit is not None and limit <= 0:
        return
    found = set()
    with path.open("rb") as handle:
        for item in ijson.items(handle, "item"):
            qid = str(item["question_id"])
            if wanted is not None and qid not in wanted:
                continue
            found.add(qid)
            yield _longmemeval_conversation(item)
            if limit is not None and len(found) >= limit:
                return
    if wanted is not None and wanted - found:
        raise ValueError(f"question ids not in {path.name}: {sorted(wanted - found)[:5]}")


def longmemeval_official_turn_view(corpus: ScopedCorpus) -> Tuple[List[str], Dict[str, str]]:
    """Official LongMemEval turn-granularity view: ``(user_turn_evidence_ids, session_of)``.

    ``process_item_flat_index(granularity='turn')`` indexes user turns only and relabels a
    user turn without ``has_answer`` in a session whose id contains 'answer' to
    ``session_id.replace('answer', 'noans')``; ``evaluate_retrieval_turn2session`` scores
    sessions by those labels. With ``gold = Question.gold_user_evidence`` and a ranking over
    these ids, ``turn_metrics_at_k`` / ``turn2session_metrics_at_k`` equal the official values.
    """
    ids: List[str] = []
    session_of: Dict[str, str] = {}
    for turn in corpus.turns:
        if turn.metadata["role"] != "user":
            continue
        label = str(turn.metadata["session_id"])
        if "answer" in label and not turn.metadata["has_answer"]:
            label = label.replace("answer", "noans")
        ids.append(turn.evidence_id)
        session_of[turn.evidence_id] = label
    return ids, session_of


def _longmemeval_conversation(item: Dict[str, Any]) -> Conversation:
    qid = str(item["question_id"])
    scope = f"lme:{qid}"
    ids, dates, sessions = item["haystack_session_ids"], item["haystack_dates"], item["haystack_sessions"]
    if not len(ids) == len(dates) == len(sessions):
        raise ValueError(f"{qid}: haystack ids/dates/sessions lengths differ")
    turns: List[TurnRecord] = []
    gold: List[str] = []
    gold_user: List[str] = []
    seen_sessions = set()
    for position, (session_id, raw_date, messages) in enumerate(zip(ids, dates, sessions)):
        session_id = str(session_id)
        key = session_id if session_id not in seen_sessions else f"{session_id}@{position}"
        seen_sessions.add(session_id)
        iso = parse_longmemeval_datetime(str(raw_date))
        for i, m in enumerate(messages):
            role = str(m.get("role") or "")
            has_answer = m.get("has_answer") is True
            evidence_id = f"{key}:{i}"
            if has_answer:
                gold.append(evidence_id)
                if role == "user":
                    gold_user.append(evidence_id)
            turns.append(TurnRecord(
                node_id=f"lme:{qid}:{key}:{i}",
                evidence_id=evidence_id,
                text=str(m.get("content") or ""),
                session_key=key,
                order=(iso, position, i),
                speaker=role,
                timestamp=iso,
                metadata={"has_answer": has_answer, "role": role, "session_id": session_id,
                          "session_position": position, "turn_index": i},
            ))
    question_type = str(item["question_type"])
    question = Question(
        qid=qid,
        scope_key=scope,
        question=str(item["question"]),
        answer=str(item["answer"]),
        category=question_type,
        category_name=question_type,
        gold_evidence=tuple(gold),
        gold_sessions=tuple(str(s) for s in item.get("answer_session_ids") or []),
        gold_user_evidence=tuple(gold_user),
        question_date=parse_longmemeval_datetime(str(item["question_date"])) if item.get("question_date") else None,
        abstention=qid.endswith("_abs"),
    )
    return Conversation(scope_key=scope, corpus=ScopedCorpus(turns, scope_key=scope), questions=(question,))
