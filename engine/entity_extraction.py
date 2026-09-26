"""Entity mention extraction for the entity-memory graph channel.

Two extractors emit the same ``EntityMention(key, value, type)`` records, so a
graph built from either is scored by the same code in ``engine.entity_graph``:

* ``LexicalEntityExtractor`` (version ``lexical-v1``) is HybridMind-original and
  fully offline. It is NOT the EMG paper configuration and must be reported as
  its own experimental condition.
* ``LLMEntityExtractor`` ports EMG's LLM extractor (prompt, response parsing and
  post-processing verbatim). It never builds a client: callers inject
  ``complete(messages) -> str``, which keeps provider admission (preflight,
  priced plans) outside this module. Tests inject a fake.

Port attribution (``normalize_entity_key``, ``ENTITY_TYPES``,
``ENTITY_EXTRACTION_PROMPT``, ``normalize_entity_value``,
``canonicalize_entity_type``, ``postprocess_entities``,
``parse_entity_response``, ``LLMEntityExtractor``):
EMG, https://github.com/Sun668/em_graph_memory, commit
f020e855be06ac9f33ec888945ff6b305d81cb07, files code/em_graph/build/entity_keys.py,
code/em_graph/build/config.py, code/em_graph/build/entity_extractor.py,
code/em_graph/recall/retrieval.py (``entity_keys_from_extracted``,
``extract_question_entity_keys``). SPDX-License-Identifier: MIT.
Copyright (c) 2026 Sun668.

``NLTK_ENGLISH_STOPWORDS`` is vendored data from the NLTK stopwords corpus
(see ``NLTK_STOPWORDS_PROVENANCE``). The RAKE candidate/scoring rule in the
lexical extractor follows Rose, Engel, Cramer & Cowley (2010), "Automatic
keyword extraction from individual documents", in Text Mining: Applications
and Theory; it is a re-implementation, not ported code.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

DEVIATIONS = [
    "LLMEntityExtractor takes an injected complete(messages)->str instead of "
    "EMG's common.llm.run_chat; EMG's request settings are recorded in "
    "LLM_REQUEST_PARAMS for the caller to apply.",
    "LLMEntityExtractor has no text-digest cache (EMG EntityCache); persistence "
    "is SQLiteStore.put_entity_extraction keyed by (node_id, extractor).",
    "LLMEntityExtractor retries a parse failure up to 3 times like EMG but does "
    "not sleep 1s between attempts; pacing belongs to the injected callable.",
    "Extractors return EntityMention(key, value, type) with key = "
    "normalize_entity_key(value); EMG returns (value, type) and computes the "
    "key in the graph builder. Mentions whose key normalises to '' are dropped "
    "here instead of in the builder.",
    "LexicalEntityExtractor is HybridMind-original (no upstream counterpart).",
]

# --------------------------------------------------------------------------- #
# Entity keys (EMG build/entity_keys.py)
# --------------------------------------------------------------------------- #


def normalize_entity_key(value: str) -> str:
    """Lowercase key used for Entity identity and recall matching (EMG verbatim).

    Note: the possessive strip runs before whitespace collapsing and the result
    is not re-stripped, exactly as upstream.
    """
    key = str(value or "").lower().strip()
    key = re.sub(r"'s$", "", key)
    return re.sub(r"\s+", " ", key)


# --------------------------------------------------------------------------- #
# Vendored NLTK English stopwords
# --------------------------------------------------------------------------- #

NLTK_STOPWORDS_PROVENANCE = {
    "source": "nltk_data 'stopwords' package, corpora/stopwords/english",
    "retrieved_with": "nltk 3.9.4 nltk.download('stopwords', "
    "download_dir='D:/hybridmind/tmp/nltk_data') on 2026-09-26",
    "english_file_sha256": "f6d005956f407dbc6ea32e5ff0c7e8e6f71488d3239b9023efdc7fc139d6375b",
    "stopwords_zip_sha256": "48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b",
    "size": 198,
    "note": "EMG lowercases stopwords.words('english'); every entry is already "
    "lowercase. Entries containing an apostrophe can never equal a "
    "[a-z0-9]+ token, so they are inert in tokenize_for_bm25 (older 179-word "
    "NLTK lists differ from this one only in such contractions).",
}

NLTK_ENGLISH_STOPWORDS = frozenset({
    "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an",
    "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been",
    "before", "being", "below", "between", "both", "but", "by", "can", "couldn",
    "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't",
    "doing", "don", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven",
    "haven't", "having", "he", "he'd", "he'll", "her", "here", "hers", "herself",
    "he's", "him", "himself", "his", "how", "i", "i'd", "if", "i'll", "i'm", "in",
    "into", "is", "isn", "isn't", "it", "it'd", "it'll", "it's", "its", "itself",
    "i've", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most",
    "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not",
    "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn", "shouldn't", "should've",
    "so", "some", "such", "t", "than", "that", "that'll", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "we'd",
    "we'll", "we're", "were", "weren", "weren't", "we've", "what", "when",
    "where", "which", "while", "who", "whom", "why", "will", "with", "won",
    "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "your",
    "you're", "yours", "yourself", "yourselves", "you've",
})

# --------------------------------------------------------------------------- #
# Mention record
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EntityMention:
    """One extracted entity: normalised identity key, surface value, EMG type."""

    key: str
    value: str
    type: str

    def to_dict(self) -> Dict[str, str]:
        return {"key": self.key, "value": self.value, "type": self.type}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EntityMention":
        return cls(key=str(data["key"]), value=str(data["value"]), type=str(data["type"]))


def _mention(value: str, entity_type: str) -> Optional[EntityMention]:
    key = normalize_entity_key(value)
    return EntityMention(key=key, value=value, type=entity_type) if key else None


# --------------------------------------------------------------------------- #
# EMG LLM extractor (build/config.py + build/entity_extractor.py)
# --------------------------------------------------------------------------- #

ENTITY_EXTRACT_VERSION = "v4"

ENTITY_TYPES: Set[str] = {
    "Who",
    "What",
    "When",
    "Where",
    "Why",
    "How",
    "How much",
}

# Verbatim from EMG build/config.py (a str.format template: doubled braces).
ENTITY_EXTRACTION_PROMPT = """Extract high-signal entities for memory-graph retrieval from the text below.

Use ONLY these types: Who, What, When, Where, Why, How, How much.

Rules (same for dialog text and questions):
1. Prefer people, places, times, objects, events, states, and attributes useful for retrieval.
2. Default to short noun phrases (about 1-3 words) so the same concept can reuse one entity key across sentences.
3. Use a longer multi-word value only when needed for a meaningful state or proper concept
   (e.g. a fixed status phrase or named group). Do not copy whole clauses or sentences.
4. Whenever possible, extract one subject-predicate-object structure that captures the core meaning of each sentence:
   - Usually label the subject as Who.
   - Label the core predicate or action as What.
   - Label the object with the entity type appropriate to its meaning.
   - Omit a component if it is absent, not explicitly stated, or would violate rule 9.
5. Do NOT extract low-information wrappers around another entity
   (prepositional shells or filler phrases whose head is already the real entity).
6. Deduplicate near-identical values; keep the clearest short form.
7. Only extract what is actually present; do not invent facts.
8. Do not extract as many as possible — skip weak or redundant mentions.
9. Do NOT extract:
   - Interrogative or function words
     (what, which, who, whom, whose, where, when, why, how, a, an, the, it, they, we, you, i, me, my, your, their).
   - Auxiliary verbs, copular verbs, or generic function verbs without independent retrieval value
     (do, does, did, be, am, is, are, was, were, have, has, had, can, could, will, would, shall, should, may, might, must).
   Keep core actions or content predicates with real retrieval value, such as research, adopt, travel, paint, study, or recommend.

Return a JSON array of objects with "value" (string) and "type" (string).

Example 1:

Text: "The technician will repair the machine in the workshop tomorrow."
Output:
[{{"value": "technician", "type": "Who"}}, {{"value": "repair", "type": "What"}}, {{"value": "machine", "type": "What"}}, {{"value": "workshop", "type": "Where"}}, {{"value": "tomorrow", "type": "When"}}]

Example 2:

Text: "What will the technician repair?"
Output:
[{{"value": "technician", "type": "Who"}}, {{"value": "repair", "type": "What"}}]

Text to extract from:
{text}

Return only the JSON array, no additional text or explanation:"""

# EMG EntityExtractor._call_llm -> common.llm.run_chat: one user message,
# temperature 0.3, max_tokens 2500 (4000 for gpt-5* models).
LLM_REQUEST_PARAMS = {"temperature": 0.3, "max_tokens": 2500, "max_tokens_gpt5": 4000}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_entity_value(value: str) -> str:
    text = _WHITESPACE_RE.sub(" ", str(value or "").strip())
    text = re.sub(r"'s$", "", text, flags=re.I)
    return text.strip(" .,;:!?\"'")


def canonicalize_entity_type(entity_type: str) -> Optional[str]:
    t = str(entity_type or "").strip()
    return t if t in ENTITY_TYPES else None


def postprocess_entities(entities: Iterable[Mapping[str, str]]) -> List[Dict[str, str]]:
    """Normalise values, canonicalise types, dedupe (EMG ``postprocess_entities``)."""
    best: Dict[str, Dict[str, str]] = {}
    for ent in entities:
        etype = canonicalize_entity_type(ent["type"])
        if etype is None:
            continue
        value = normalize_entity_value(ent["value"])
        if not value:
            continue
        key = value.lower()
        prev = best.get(key)
        if prev is None or len(value) > len(prev["value"]):
            best[key] = {"value": value, "type": etype}
    out = list(best.values())
    out.sort(key=lambda e: (0 if e["type"] == "Who" else 1, -len(e["value"]), e["value"].lower()))
    return out


def parse_entity_response(response: str) -> List[Dict[str, str]]:
    """Parse an extraction completion (EMG ``EntityExtractor._parse_response``)."""
    response = response.strip()
    # Strip one leading fence and any trailing fence.
    if response.startswith("```json"):
        response = response[7:].strip()
    elif response.startswith("```"):
        response = response[3:].strip()
    if response.endswith("```"):
        response = response[:-3].strip()
    # Models sometimes emit "[]```json\n[]" — drop leftover fences mid-text.
    response = re.sub(r"```(?:json)?", "", response).strip()

    cleaned = re.sub(r",\s*([}\]])", r"\1", response)
    cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)

    entities = None
    parse_errors: List[str] = []

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "entities" in parsed:
            entities = parsed["entities"]
        elif isinstance(parsed, list):
            entities = parsed
    except json.JSONDecodeError as exc:
        parse_errors.append(f"Direct parse failed: {exc}")

    # Prefer the first JSON value when trailing junk remains.
    if entities is None:
        try:
            parsed, _end = json.JSONDecoder().raw_decode(cleaned.lstrip())
            if isinstance(parsed, dict) and "entities" in parsed:
                entities = parsed["entities"]
            elif isinstance(parsed, list):
                entities = parsed
        except json.JSONDecodeError as exc:
            parse_errors.append(f"raw_decode failed: {exc}")

    if entities is None:
        try:
            start_idx = cleaned.find("[")
            end_idx = cleaned.rfind("]")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                arr_str = cleaned[start_idx : end_idx + 1]
                arr_str = re.sub(r",\s*([}\]])", r"\1", arr_str)
                entities = json.loads(arr_str)
        except (json.JSONDecodeError, ValueError) as exc:
            parse_errors.append(f"Array extraction failed: {exc}")

    if entities is None:
        try:
            entity_pattern = (
                r'\{\s*"value"\s*:\s*"([^"]*)"\s*,\s*"type"\s*:\s*"([^"]*)"\s*\}'
            )
            matches = re.findall(entity_pattern, response)
            if matches:
                entities = [{"value": m[0], "type": m[1]} for m in matches]
        except Exception as exc:
            parse_errors.append(f"Regex extraction failed: {exc}")

    if entities is None:
        raise ValueError(
            "Failed to parse entity JSON:\n"
            + "\n".join(parse_errors)
            + f"\n\nResponse was:\n{response[:500]}..."
        )
    if not isinstance(entities, list):
        raise ValueError(f"Expected list, got {type(entities)}")

    cleaned_entities: List[Dict[str, str]] = []
    for entity in entities:
        if not isinstance(entity, dict):
            raise ValueError(f"Expected dict, got {type(entity)}")
        if "value" not in entity or "type" not in entity:
            raise ValueError(f"Entity missing value/type: {entity}")
        etype = canonicalize_entity_type(str(entity["type"]))
        if etype is None:
            etype = "What"
        value = normalize_entity_value(str(entity["value"]))
        if value:
            cleaned_entities.append({"value": value, "type": etype})
    return cleaned_entities


CompleteFn = Callable[[List[Dict[str, str]]], str]


class LLMEntityExtractor:
    """EMG LLM entity extractor over an injected chat-completion callable."""

    version = f"emg-llm-{ENTITY_EXTRACT_VERSION}"

    def __init__(self, complete: CompleteFn, *, max_retries: int = 3):
        if not callable(complete):
            raise TypeError("complete must be a callable(messages) -> str")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        self._complete = complete
        self.max_retries = int(max_retries)

    @staticmethod
    def messages_for(text: str) -> List[Dict[str, str]]:
        return [{"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(text=text)}]

    def extract(self, text: str) -> List[EntityMention]:
        text = str(text or "").strip()
        if not text:
            return []
        last_error: Optional[ValueError] = None
        for _attempt in range(self.max_retries):
            try:
                raw = parse_entity_response(self._complete(self.messages_for(text)))
                break
            except ValueError as exc:
                last_error = exc
        else:
            assert last_error is not None
            raise last_error
        mentions = (_mention(item["value"], item["type"]) for item in postprocess_entities(raw))
        return [m for m in mentions if m is not None]

    def extract_query(self, question: str) -> Set[str]:
        """EMG ``extract_question_entity_keys``: the same prompt on the question."""
        return {m.key for m in self.extract(question)}


# --------------------------------------------------------------------------- #
# HybridMind-original lexical extractor
# --------------------------------------------------------------------------- #

LEXICAL_EXTRACTOR_VERSION = "lexical-v1"

# Conversational filler that NLTK's list lacks. Used only to delimit lexical
# candidates; tokenize_for_bm25 keeps the exact NLTK list.
CONVERSATIONAL_STOPWORDS = frozenset({
    "hey", "hi", "hello", "wow", "oh", "ooh", "aww", "aw", "yeah", "yes", "yep",
    "nope", "ok", "okay", "sure", "thanks", "thank", "lol", "haha", "omg", "um",
    "uh", "hmm", "well", "really", "also", "would", "could", "might", "must",
    "shall", "may", "get", "got", "gets", "getting", "go", "goes", "going",
    "gonna", "wanna", "gotta", "know", "think", "thought", "like", "likes",
    "liked", "lot", "lots", "much", "many", "something", "anything", "everything",
    "nothing", "thing", "things", "stuff", "one", "even", "still", "always",
    "never", "maybe", "definitely", "totally", "absolutely", "though", "way",
    "kind", "sort", "pretty", "quite", "bit", "let", "let's", "us", "that's",
    "what's", "there's", "here's", "who's", "where's", "how's", "can't", "cannot",
    "yet", "ever", "every", "since", "etc", "good", "great", "nice", "awesome",
    "amazing", "cool", "glad", "happy", "sorry", "hope", "want", "wanted",
    "make", "made", "take", "took", "say", "said", "tell", "told", "see", "saw",
    "look", "looks", "feel", "feels", "felt", "keep", "new", "lately",
    "recently", "whose", "else", "anyway", "actually", "especially",
})

_LEXICAL_STOPWORDS = NLTK_ENGLISH_STOPWORDS | CONVERSATIONAL_STOPWORDS

_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'"})
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['\-][A-Za-z0-9]+)*")

_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_WEEKDAY_NAMES = r"monday|tuesday|wednesday|thursday|friday|saturday|sunday"
_WEEKDAY = rf"(?:{_WEEKDAY_NAMES})"
_ORDINAL = r"\d{1,2}(?:st|nd|rd|th)?"
_COUNT = (
    r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
    r"|a\s+few|a\s+couple\s+of|a\s+couple|few|several)"
)
_PERIOD = (
    r"(?:week|weekend|month|year|night|morning|afternoon|evening|summer|winter"
    r"|spring|fall|autumn|" + _WEEKDAY_NAMES + r")"
)
# Priority order inside the alternation matters: longest forms first.
_TIME_RE = re.compile(
    r"\b(?:"
    rf"{_ORDINAL}\s+(?:of\s+)?{_MONTH}(?:,?\s+\d{{4}})?"
    rf"|{_MONTH}(?:\s+{_ORDINAL})?(?:,?\s+\d{{4}})?"
    rf"|(?i:{_COUNT}\s+(?:day|week|month|year|hour|minute|decade)s?\s+(?:ago|later|earlier))"
    rf"|(?i:(?:last|next|this|past|coming|previous)\s+{_PERIOD}s?)"
    r"|(?i:yesterday|today|tonight|tomorrow)"
    rf"|(?i:{_WEEKDAY}s?)"
    r"|(?:1[89]|20)\d{2}s?"
    r")\b"
)
_UNITS = (
    r"%|percent|km|kilometers?|kilometres?|miles?|meters?|metres?|cm|mm|feet|foot|ft"
    r"|inch(?:es)?|kg|kilograms?|grams?|lbs?|pounds?|oz|ounces?|liters?|litres?|ml"
    r"|gallons?|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?"
    r"|years?|yrs?|times|dollars?|bucks|euros?|cents?|mph|k"
)
_QUANTITY_RE = re.compile(
    r"(?:[$\u20ac\u00a3]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|million|billion|thousand)\b)?"
    rf"|\b\d[\d,]*(?:\.\d+)?\s?(?:{_UNITS})(?![A-Za-z0-9]))",
    re.IGNORECASE,
)

_TIER_TIME_QUANTITY = 3
_TIER_CAPITALIZED = 2
_TIER_PHRASE = 1


def _is_lexical_stopword(token: str) -> bool:
    low = token.lower()
    return low in _LEXICAL_STOPWORDS or low.split("'", 1)[0] in _LEXICAL_STOPWORDS


def _is_acronym(token: str) -> bool:
    return len(token) >= 2 and token.isalpha() and token.isupper()


def _sentence_initial(text: str, start: int) -> bool:
    prev = text[:start].rstrip(" \t\"'([{")
    return not prev or prev[-1] in ".!?\n"


def rake_phrase_scores(phrases: Sequence[Sequence[str]]) -> List[float]:
    """RAKE (Rose et al. 2010): word score deg(w)/freq(w); phrase = sum of words.

    ``deg(w)`` sums the lengths of every candidate occurrence containing ``w``
    (co-occurrence degree including itself); ``freq(w)`` counts occurrences.
    Words are compared lowercased.
    """
    freq: Dict[str, int] = {}
    deg: Dict[str, int] = {}
    for phrase in phrases:
        for word in phrase:
            w = word.lower()
            freq[w] = freq.get(w, 0) + 1
            deg[w] = deg.get(w, 0) + len(phrase)
    return [sum(deg[w.lower()] / freq[w.lower()] for w in phrase) for phrase in phrases]


class LexicalEntityExtractor:
    """Deterministic offline extractor emitting EMG-typed mentions.

    Candidates, in priority tier order:

    1. ``When``: month dates, years, weekdays, relative expressions
       ("yesterday", "last week", "4 years ago"); ``How much``: numbers with
       units or currency ("5 km", "$20", "3 hours").
    2. Capitalized spans and acronyms (``Who`` when the span is a known speaker,
       else ``What``). A single capitalized word at a sentence start is skipped
       unless it is an acronym.
    3. RAKE-style content phrases: maximal runs of non-stopword tokens between
       stopwords/punctuation/consumed spans, chunked to at most 3 tokens,
       scored by RAKE (``What``).

    Each turn keeps at most ``max_entities`` content mentions ranked by the
    deterministic score ``(tier, rake_score or span length, -position)``; the
    speaker (``Who``) is always emitted first and is not counted in the cap.
    """

    version = LEXICAL_EXTRACTOR_VERSION

    def __init__(self, max_entities: int = 6, speakers: Iterable[str] = ()):
        if max_entities < 1:
            raise ValueError("max_entities must be >= 1")
        self.max_entities = int(max_entities)
        self._speaker_keys = frozenset(k for k in (normalize_entity_key(s) for s in speakers) if k)

    def extract(self, text: str, speaker: str = "") -> List[EntityMention]:
        out: List[EntityMention] = []
        speaker_mention = _mention(str(speaker or "").strip(), "Who")
        if speaker_mention is not None:
            out.append(speaker_mention)
        seen = {m.key for m in out}
        ranked = sorted(self._candidates(text), key=lambda c: (-c[0], -c[1], c[2]))
        kept = 0
        for _tier, _score, _start, mention in ranked:
            if kept >= self.max_entities:
                break
            if mention.key in seen:
                continue
            seen.add(mention.key)
            out.append(mention)
            kept += 1
        return out

    def extract_query(self, question: str) -> Set[str]:
        """All candidate keys of a question (no speaker, no per-turn cap)."""
        return {mention.key for *_rest, mention in self._candidates(question)}

    # -- candidate generation ------------------------------------------------

    def _candidates(self, text: str) -> List[Tuple[int, float, int, EntityMention]]:
        text = str(text or "").translate(_APOSTROPHES)
        consumed: List[Tuple[int, int]] = []
        found: List[Tuple[int, float, int, EntityMention]] = []

        def overlaps(start: int, end: int) -> bool:
            return any(start < c_end and c_start < end for c_start, c_end in consumed)

        for pattern, entity_type in ((_TIME_RE, "When"), (_QUANTITY_RE, "How much")):
            for match in pattern.finditer(text):
                start, end = match.span()
                if overlaps(start, end):
                    continue
                consumed.append((start, end))
                mention = _mention(_WHITESPACE_RE.sub(" ", match.group(0).strip()), entity_type)
                if mention is not None:
                    found.append((_TIER_TIME_QUANTITY, 0.0, start, mention))

        tokens = [m for m in _WORD_RE.finditer(text) if not overlaps(*m.span())]
        found.extend(self._capitalized(text, tokens, consumed))
        found.extend(self._phrases(text, [t for t in tokens if not overlaps(*t.span())]))
        return found

    def _capitalized(self, text: str, tokens: List[re.Match], consumed: List[Tuple[int, int]]):
        runs: List[List[re.Match]] = []
        previous: Optional[re.Match] = None
        for tok in tokens:
            word = tok.group(0)
            if not word[0].isupper() or _is_lexical_stopword(word):
                previous = None
                continue
            if previous is not None and text[previous.end() : tok.start()].isspace():
                runs[-1].append(tok)
            else:
                runs.append([tok])
            previous = tok
        out = []
        for run in runs:
            if len(run) == 1 and _sentence_initial(text, run[0].start()) and not _is_acronym(run[0].group(0)):
                continue
            start, end = run[0].start(), run[-1].end()
            consumed.append((start, end))
            value = " ".join(t.group(0) for t in run)
            mention = _mention(value, "Who" if normalize_entity_key(value) in self._speaker_keys else "What")
            if mention is not None:
                out.append((_TIER_CAPITALIZED, float(len(run)), start, mention))
        return out

    @staticmethod
    def _phrases(text: str, tokens: List[re.Match]):
        runs: List[List[re.Match]] = []
        previous: Optional[re.Match] = None
        for tok in tokens:
            word = tok.group(0)
            content = not _is_lexical_stopword(word) and len(word) >= 2 and any(c.isalpha() for c in word)
            if not content:
                previous = None
                continue
            joined = previous is not None and text[previous.end() : tok.start()].isspace()
            if joined:
                runs[-1].append(tok)
            else:
                runs.append([tok])
            previous = tok
        chunks = [run[i : i + 3] for run in runs for i in range(0, len(run), 3)]
        scores = rake_phrase_scores([[t.group(0) for t in chunk] for chunk in chunks])
        out = []
        for chunk, score in zip(chunks, scores):
            mention = _mention(" ".join(t.group(0) for t in chunk), "What")
            if mention is not None:
                out.append((_TIER_PHRASE, score, chunk[0].start(), mention))
        return out
