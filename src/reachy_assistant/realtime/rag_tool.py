"""
Realtime-tool som kobler modellen til lokal RAG-søk i Læringslab-dokumentasjonen.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reachy_assistant.rag_openai.embeddings import OpenAIEmbeddings
from reachy_assistant.rag_openai.search import OpenAIRagSearch
from reachy_assistant.rag_openai.store import QdrantVectorStore


DEFAULT_STORE_DIR = Path(os.getenv("RAG_OPENAI_STORE_DIR", "data/rag_openai_store"))
DEFAULT_EMBED_MODEL = os.getenv("RAG_OPENAI_EMBED_MODEL", "text-embedding-3-small")


@dataclass
class OpenAIRagRealtimeTool:
    searcher: OpenAIRagSearch

    @classmethod
    def from_env(cls, api_key: str) -> "OpenAIRagRealtimeTool | None":
        if not api_key:
            return None

        store_dir = DEFAULT_STORE_DIR
        if not store_dir.exists():
            return None

        try:
            vector_store = QdrantVectorStore(path=str(store_dir))
        except Exception:
            return None

        if vector_store.is_empty():
            return None

        embeddings = OpenAIEmbeddings(api_key=api_key, model=DEFAULT_EMBED_MODEL)
        return cls(searcher=OpenAIRagSearch(vector_store=vector_store, embeddings=embeddings))

    @staticmethod
    def definition() -> dict[str, Any]:
        return {
            "type": "function",
            "name": "openai_rag_search",
            "description": (
                "Search the local Læringslab documentation for grounded answers about equipment, "
                "rules, procedures, rooms, and project resources. "
                "Use this for factual questions about what exists in the lab, what can be borrowed, "
                "how something is used, and local lab policies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The user question or search query.",
                    },
                    "category": {
                        "type": "string",
                        "description": "Optional category filter, for example equipment, safety, or rooms.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                        "default": 3,
                        "description": "Maximum number of relevant chunks to retrieve.",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        }

    def execute(self, arguments_json: str) -> str:
        try:
            payload = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON arguments."}, ensure_ascii=False)

        query = str(payload.get("query", "")).strip()
        category = payload.get("category")
        limit = payload.get("limit", 3)

        try:
            limit = max(1, min(int(limit), 5))
        except (TypeError, ValueError):
            limit = 3

        result = self.searcher.search(query=query, category=category, limit=limit)
        return json.dumps(result, ensure_ascii=False)

    def debug_summary(self, arguments_json: str) -> dict[str, Any]:
        try:
            payload = json.loads(arguments_json or "{}")
        except json.JSONDecodeError:
            return {
                "query": "",
                "category": None,
                "limit": None,
                "error": "Invalid JSON arguments.",
            }

        query = str(payload.get("query", "")).strip()
        category = payload.get("category")
        limit = payload.get("limit", 3)

        try:
            limit = max(1, min(int(limit), 5))
        except (TypeError, ValueError):
            limit = 3

        result = self.searcher.search(query=query, category=category, limit=limit)
        return {
            "query": query,
            "category": category,
            "limit": limit,
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "results_count": len(result.get("results", [])),
            "error": result.get("error"),
        }
