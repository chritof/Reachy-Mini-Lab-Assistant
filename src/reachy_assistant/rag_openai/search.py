from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class OpenAIRagSearch:
    vector_store: object
    embeddings: object

    def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 3,
    ) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"error": "query is required"}

        query_vector = self.embeddings.embed_one(query)
        results = self.vector_store.search(query_vector, category=category, limit=limit)

        if not results:
            return {"answer": "Jeg finner ikke dette i dokumentasjonen."}

        context = "\n\n---\n\n".join(
            f"[{item['source']}]\n{item['text']}" for item in results
        )
        return {
            "results": results,
            "context": context,
        }
