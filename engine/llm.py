"""
LLM Engine for processing unstructured data.

Uses the centralized HybridMind LLM provider policy for intelligent extraction
of entities, relationships, and metadata from text.
Called ONLY at ingest time — never at query time.
"""

import json
from typing import Optional
from dataclasses import dataclass

from config import settings
from engine import llm_client


@dataclass
class ExtractedData:
    """Structured data extracted from unstructured text."""
    summary: str
    entities: list[dict]  # [{name, type, description}]
    topics: list[str]
    relationships: list[dict]  # [{source, target, relationship}]
    key_facts: list[str]
    sentiment: str  # positive, negative, neutral
    language: str


class LLMOutputError(RuntimeError):
    """The configured provider returned output that cannot be stored safely."""


def _bounded_text(text: str, *, operation: str, max_chars: int) -> str:
    clean = str(text).strip()
    if not clean:
        raise ValueError(f"{operation} requires non-empty text")
    if len(clean) > max_chars:
        raise ValueError(
            f"{operation} input has {len(clean)} characters; maximum is {max_chars}. "
            "Split the source explicitly so provenance and provider cost remain visible."
        )
    return clean


def _json_content(content: str):
    """Parse one JSON value without treating malformed model output as data."""
    clean = content.strip()
    if clean.startswith("```"):
        lines = clean.splitlines()
        if lines and lines[0].strip().lower() in {"```", "```json"}:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines).strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError as exc:
        raise LLMOutputError("provider returned malformed JSON") from exc


class LLMEngine:
    """
    LLM-powered engine for processing unstructured data.

    Uses Z.AI or self-hosted RunPod for extraction at ingest time. The research
    proxy is usable only through the repository-wide explicit opt-in.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize the LLM engine.

        Args:
            api_key: Deprecated and ignored compatibility argument. Provider
                credentials come from Settings.
            model: Deprecated compatibility argument. Provider models come
                from Settings.
            base_url: Deprecated and ignored compatibility argument. Provider
                URLs come from Settings.
        """
        if not llm_client.is_configured():
            raise ValueError("No policy-allowed LLM provider is configured")
        self.model = settings.fact_model

    def _complete(self, messages: list[dict], *, temperature: float, max_tokens: int) -> str:
        content = llm_client.chat_completion(
            messages,
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if content is None:
            raise RuntimeError("All policy-allowed LLM providers failed")
        return content.strip()

    def extract_metadata(self, text: str) -> ExtractedData:
        """
        Extract structured metadata from unstructured text.

        Args:
            text: Raw unstructured text to process

        Returns:
            ExtractedData with entities, topics, relationships, etc.
        """
        text = _bounded_text(text, operation="metadata extraction", max_chars=4_000)
        prompt = f"""Analyze the following text and extract structured information.

TEXT:
{text}

Return a JSON object with these fields:
{{
  "summary": "A concise 1-2 sentence summary",
  "entities": [
    {{"name": "entity name", "type": "PERSON|PLACE|ORG|CONCEPT|WORK|EVENT|DATE", "description": "brief description"}}
  ],
  "topics": ["topic1", "topic2"],
  "relationships": [
    {{"source": "entity1", "target": "entity2", "relationship": "relationship type"}}
  ],
  "key_facts": ["fact1", "fact2"],
  "sentiment": "positive|negative|neutral",
  "language": "detected language"
}}

Be thorough but concise. Extract ALL named entities and their relationships.
Return ONLY valid JSON, no markdown or explanation."""

        content = self._complete(
            [
                {
                    "role": "system",
                    "content": "You are a precise data extraction assistant. Extract structured information from text and return valid JSON only."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )

        data = _json_content(content)
        if not isinstance(data, dict):
            raise LLMOutputError("metadata extraction response must be a JSON object")
        for field in ("entities", "topics", "relationships", "key_facts"):
            if not isinstance(data.get(field, []), list):
                raise LLMOutputError(f"metadata extraction field {field!r} must be an array")
        return ExtractedData(
            summary=str(data.get("summary", "")),
            entities=data.get("entities", []),
            topics=data.get("topics", []),
            relationships=data.get("relationships", []),
            key_facts=data.get("key_facts", []),
            sentiment=str(data.get("sentiment", "neutral")),
            language=str(data.get("language", "en")),
        )

    def process_unstructured(self, text: str) -> dict:
        """
        Process unstructured text and return nodes and edges for the knowledge graph.

        Args:
            text: Raw unstructured text (can be very large)

        Returns:
            Dict with 'nodes' and 'edges' ready for import
        """
        text = _bounded_text(text, operation="knowledge graph extraction", max_chars=12_000)
        prompt = f"""You are a knowledge graph extraction system. Analyze this text and extract:
1. Knowledge nodes (distinct concepts, facts, entities)
2. Relationships between nodes

TEXT:
{text}

Return a JSON object with this exact structure:
{{
  "nodes": [
    {{
      "text": "The actual content/description of this knowledge unit",
      "metadata": {{
        "type": "fact|concept|entity|event|definition",
        "topic": "main topic",
        "entities": ["entity1", "entity2"],
        "importance": "high|medium|low"
      }}
    }}
  ],
  "edges": [
    {{
      "source_index": 0,
      "target_index": 1,
      "type": "relates_to|causes|is_part_of|describes|follows|contradicts",
      "weight": 0.8
    }}
  ],
  "summary": "Brief summary of the entire text"
}}

Guidelines:
- Create 5-20 nodes depending on text complexity
- Each node should be a self-contained piece of knowledge
- Node text should be 50-500 characters
- Only create edges for clear relationships
- source_index and target_index refer to positions in the nodes array

Return ONLY valid JSON."""

        content = self._complete(
            [
                {
                    "role": "system",
                    "content": "You are a knowledge extraction system. Convert unstructured text into structured knowledge graphs. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )

        data = _json_content(content)
        if not isinstance(data, dict):
            raise LLMOutputError("knowledge graph extraction response must be a JSON object")
        nodes = data.get("nodes")
        edges = data.get("edges")
        if not isinstance(nodes, list) or not isinstance(edges, list):
            raise LLMOutputError("knowledge graph extraction requires nodes and edges arrays")
        for index, node in enumerate(nodes):
            if not isinstance(node, dict) or not isinstance(node.get("text"), str) or not node["text"].strip():
                raise LLMOutputError(f"knowledge graph node {index} has invalid text")
            if not isinstance(node.get("metadata", {}), dict):
                raise LLMOutputError(f"knowledge graph node {index} has invalid metadata")
        for index, edge in enumerate(edges):
            if not isinstance(edge, dict):
                raise LLMOutputError(f"knowledge graph edge {index} is not an object")
            source = edge.get("source_index")
            target = edge.get("target_index")
            if not isinstance(source, int) or not isinstance(target, int):
                raise LLMOutputError(f"knowledge graph edge {index} has invalid endpoints")
            if not 0 <= source < len(nodes) or not 0 <= target < len(nodes):
                raise LLMOutputError(f"knowledge graph edge {index} references a missing node")
        return {"nodes": nodes, "edges": edges, "summary": str(data.get("summary", ""))}

    def smart_chunk(self, text: str, max_chunk_size: int = 1500) -> list[dict]:
        """
        Intelligently chunk text based on semantic boundaries.
        """
        text = _bounded_text(text, operation="semantic chunking", max_chars=8_000)
        prompt = f"""Divide the following text into semantically meaningful chunks.
Each chunk should:
- Be self-contained and focus on a single topic/concept
- Be between 200-{max_chunk_size} characters
- Preserve context and meaning

TEXT:
{text}

Return a JSON array:
[
  {{
    "text": "chunk content",
    "topic": "main topic",
    "entities": ["key entities"]
  }}
]

Return ONLY valid JSON."""

        content = self._complete(
            [
                {
                    "role": "system",
                    "content": "You are a text processing assistant. Divide text into meaningful semantic chunks."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.1,
            max_tokens=4000
        )

        chunks = _json_content(content)
        if not isinstance(chunks, list):
            raise LLMOutputError("semantic chunking response must be a JSON array")
        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
                raise LLMOutputError(f"semantic chunk {index} has invalid text")
        return chunks

    def chat(self, message: str, context: Optional[str] = None) -> str:
        """Simple chat interface for ad-hoc queries."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant for a knowledge database system."
            }
        ]

        if context:
            messages.append({
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {message}"
            })
        else:
            messages.append({
                "role": "user",
                "content": message
            })

        return self._complete(
            messages,
            temperature=0.7,
            max_tokens=1000
        )


# Singleton instance
_llm_engine: Optional[LLMEngine] = None


def get_llm_engine(api_key: Optional[str] = None) -> LLMEngine:
    """Get or create the LLM engine singleton."""
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine(api_key=api_key)
    return _llm_engine
