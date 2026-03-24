"""
LLM Engine for processing unstructured data.

Uses OSM API with Qwen3.5 397B A17B for intelligent
extraction of entities, relationships, and metadata from text.
"""

import json
import os
from typing import Optional
from dataclasses import dataclass

from openai import OpenAI


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


class LLMEngine:
    """
    LLM-powered engine for processing unstructured data.
    
    Uses OSM API to access various LLM providers
    through a unified OpenAI-compatible API.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen3.5-397b-a17b",
        base_url: str = "https://api.osmapi.com/v1"
    ):
        """
        Initialize the LLM engine.
        
        Args:
            api_key: OSM API key (defaults to env var)
            model: Model to use (default: qwen3.5-397b-a17b)
            base_url: OSM API URL
        """
        self.api_key = api_key or os.getenv("OSM_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key required. Set OSM_API_KEY env var or pass api_key"
            )
        
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=base_url
        )
    
    def extract_metadata(self, text: str) -> ExtractedData:
        """
        Extract structured metadata from unstructured text.
        
        Args:
            text: Raw unstructured text to process
            
        Returns:
            ExtractedData with entities, topics, relationships, etc.
        """
        prompt = f"""Analyze the following text and extract structured information.

TEXT:
{text[:4000]}

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
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
        
        content = response.choices[0].message.content.strip()
        
        # Clean up response (remove markdown if present)
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        try:
            data = json.loads(content)
            return ExtractedData(
                summary=data.get("summary", ""),
                entities=data.get("entities", []),
                topics=data.get("topics", []),
                relationships=data.get("relationships", []),
                key_facts=data.get("key_facts", []),
                sentiment=data.get("sentiment", "neutral"),
                language=data.get("language", "en")
            )
        except json.JSONDecodeError:
            # Fallback for malformed responses
            return ExtractedData(
                summary=text[:200],
                entities=[],
                topics=[],
                relationships=[],
                key_facts=[],
                sentiment="neutral",
                language="en"
            )
    
    def process_unstructured(self, text: str) -> dict:
        """
        Process unstructured text and return nodes and edges for the knowledge graph.
        
        Args:
            text: Raw unstructured text (can be very large)
            
        Returns:
            Dict with 'nodes' and 'edges' ready for import
        """
        prompt = f"""You are a knowledge graph extraction system. Analyze this text and extract:
1. Knowledge nodes (distinct concepts, facts, entities)
2. Relationships between nodes

TEXT:
{text[:12000]}

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

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
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
        
        content = response.choices[0].message.content.strip()
        
        # Clean up response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            return {
                "nodes": [{"text": text[:1000], "metadata": {"type": "raw", "error": str(e)}}],
                "edges": [],
                "summary": "Failed to parse - raw text stored"
            }
    
    def smart_chunk(self, text: str, max_chunk_size: int = 1500) -> list[dict]:
        """
        Intelligently chunk text based on semantic boundaries.
        """
        prompt = f"""Divide the following text into semantically meaningful chunks.
Each chunk should:
- Be self-contained and focus on a single topic/concept
- Be between 200-{max_chunk_size} characters
- Preserve context and meaning

TEXT:
{text[:8000]}

Return a JSON array:
[
  {{
    "text": "chunk content",
    "topic": "main topic",
    "entities": ["key entities"]
  }}
]

Return ONLY valid JSON."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
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
        
        content = response.choices[0].message.content.strip()
        
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            paragraphs = text.split("\n\n")
            return [{"text": p.strip(), "topic": "", "entities": []} 
                    for p in paragraphs if len(p.strip()) > 50]
    
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
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content


# Singleton instance
_llm_engine: Optional[LLMEngine] = None


def get_llm_engine(api_key: Optional[str] = None) -> LLMEngine:
    """Get or create the LLM engine singleton."""
    global _llm_engine
    if _llm_engine is None:
        _llm_engine = LLMEngine(api_key=api_key)
    return _llm_engine
