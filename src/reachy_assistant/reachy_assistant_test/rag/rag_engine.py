from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llama_index.core import StorageContext, Settings, load_index_from_storage
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
    def __init__(
        self,
        index_dir: str | Path = "data/rag_index",
        embed_model: str = "nomic-embed-text",
        ollama_base_url: str = "http://localhost:11434",
        top_k: int = 5,
    ):
        self.index_dir = Path(index_dir)
        self.top_k = top_k

        Settings.embed_model = OllamaEmbedding(
            model_name=embed_model,
            base_url=ollama_base_url,
        )

        if not self.index_dir.exists():
            raise FileNotFoundError(f"Fant ikke RAG-index: {self.index_dir}")

        storage_context = StorageContext.from_defaults(persist_dir=str(self.index_dir))
        self.index = load_index_from_storage(storage_context)
        self.retriever = self.index.as_retriever(similarity_top_k=self.top_k)

    def retrieve(self, query: str) -> RagResult:
        query = query.strip()
        if not query:
            return RagResult(hits=[])

        results = self.retriever.retrieve(query)
        hits: list[RagHit] = []

        for r in results:
            meta = r.node.metadata or {}
            source = meta.get("file_name") or meta.get("filename") or "ukjent fil"
            score = float(r.score or 0.0)
            text = r.node.get_content().strip()
            hits.append(RagHit(file=source, score=score, text=text))

        return RagResult(hits=hits)