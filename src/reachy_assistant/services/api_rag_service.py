import os

import httpx

from reachy_assistant.services.rag_service import RagAnswer
from reachy_assistant.rag.rag_engine import RagResult, RagHit


class ApiRagService:

    def __init__(self):
        self.base_url = os.getenv("BACKEND_URL", "http://localhost:8000")

    def ask(self, question: str) -> RagAnswer:
        resp = httpx.post(
            f"{self.base_url}/rag/ask",
            json={"question": question},
            timeout=120
        )
        resp.raise_for_status()

        data = resp.json()

        hits = [
            RagHit(
                file=h["file"],
                score=float(h["score"]),
                text="" # backend returns no text content here
            )
            for h in data.get("hits", [])
        ]

        return RagAnswer(
            answer=data["answer"],
            rag=RagResult(hits=hits),
            used_query=question,
        )


