from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

COLLECTION_NAME = "reachy_lab_content"
VECTOR_SIZE = 1536


class QdrantVectorStore:
    """Local Qdrant-backed vector store for a second RAG path."""

    def __init__(self, path: str, vector_size: int = VECTOR_SIZE) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:
            raise ImportError("Install 'qdrant-client' to use rag_openai.") from exc

        Path(path).mkdir(parents=True, exist_ok=True)
        self._client = QdrantClient(path=path)
        self._distance = Distance
        self._vector_params = VectorParams
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self._client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=self._vector_params(
                    size=self.vector_size,
                    distance=self._distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'", COLLECTION_NAME)

    def is_empty(self) -> bool:
        info = self._client.get_collection(COLLECTION_NAME)
        return info.points_count == 0

    def upsert(self, points: list[Any]) -> None:
        if points:
            self._client.upsert(collection_name=COLLECTION_NAME, points=points)

    def delete_by_file(self, file_path: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[FieldCondition(key="source_file", match=MatchValue(value=file_path))]
            ),
        )

    def search(
        self,
        query_vector: list[float],
        category: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        query_filter = None
        if category:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_filter = Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            )

        response = self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        return [
            {
                "text": point.payload.get("text", ""),
                "source": point.payload.get("source_file", ""),
                "category": point.payload.get("category", "general"),
                "score": point.score,
            }
            for point in response.points
        ]
