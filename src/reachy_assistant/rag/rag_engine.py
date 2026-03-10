"""
Retrieval-del av RAG.

Ansvar:
- Laste vektorindeks
- Lage embeddings
- Hente relevante dokumenter (top-k)

Genererer ikke tekst.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding


@dataclass
class RagHit:
    file: str
    score: float
    text: str


@dataclass
class RagResult:
    hits: list[RagHit]


class RagEngine:
    """
    Laster en ferdig bygd LlamaIndex index fra disk og gjør retrieval (vector search).
    Bygger ikke index.
    """

    def __init__(
        self,
        *,
        index_dir: Path,
        ollama_base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        top_k: int = 3,
    ) -> None:
        self.index_dir = index_dir
        self.ollama_base_url = ollama_base_url
        self.embed_model = embed_model
        self.top_k = top_k

        self._retriever = None

    def _load(self) -> None:
        if self._retriever is not None:
            return

        if not self.index_dir.exists():
            raise FileNotFoundError(f"RAG index ikke funnet: {self.index_dir}")

        Settings.embed_model = OllamaEmbedding(
            model_name=self.embed_model,
            base_url=self.ollama_base_url,
        )

        storage = StorageContext.from_defaults(persist_dir=str(self.index_dir))
        index = load_index_from_storage(storage)
        self._retriever = index.as_retriever(similarity_top_k=self.top_k)

    def retrieve(self, query: str) -> RagResult:
        self._load()
        results = self._retriever.retrieve(query)

        hits: list[RagHit] = []
        for r in results:
            meta = r.node.metadata or {}
            fname = meta.get("file_name") or meta.get("filename") or "ukjent"
            hits.append(
                RagHit(
                    file=fname,
                    score=float(r.score or 0.0),
                    text=r.node.get_content(),
                )
            )
        return RagResult(hits=hits)