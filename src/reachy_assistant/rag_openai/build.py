from __future__ import annotations

import hashlib
import os
from pathlib import Path

from reachy_assistant.rag_openai.chunking import build_chunks
from reachy_assistant.rag_openai.embeddings import OpenAIEmbeddings
from reachy_assistant.rag_openai.store import QdrantVectorStore
from reachy_assistant.realtime.config import RealtimeConfig


DEFAULT_DOCS_DIR = Path("data/rag_sources")
DEFAULT_STORE_DIR = Path("data/rag_openai_store")


def point_id(source_file: str, chunk_index: int, raw: str) -> str:
    payload = f"{source_file}:{chunk_index}:{raw}".encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def build_index(
    docs_dir: Path = DEFAULT_DOCS_DIR,
    store_dir: Path = DEFAULT_STORE_DIR,
    chunk_size: int = 500,
    overlap: int = 100,
) -> int:
    RealtimeConfig._load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required to build rag_openai.")

    records = build_chunks(docs_dir, size=chunk_size, overlap=overlap)
    if not records:
        raise ValueError(f"No documents found in {docs_dir}")

    embeddings = OpenAIEmbeddings(api_key=api_key)
    vectors = embeddings.embed([record.text for record in records])
    store = QdrantVectorStore(path=str(store_dir))

    from qdrant_client.models import PointStruct

    points = []
    for record, vector in zip(records, vectors, strict=True):
        points.append(
            PointStruct(
                id=point_id(record.source_file, record.chunk_index, record.text),
                vector=vector,
                payload={
                    "source_file": record.source_file,
                    "chunk_index": record.chunk_index,
                    "category": record.category,
                    "text": record.text,
                },
            )
        )

    store.upsert(points)
    return len(points)


def main() -> None:
    count = build_index()
    print(f"Indexed {count} chunks into rag_openai.")


if __name__ == "__main__":
    main()
