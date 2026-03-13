from __future__ import annotations

from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter

try:
    from llama_index.embeddings.ollama import OllamaEmbedding
    USE_OLLAMA_EMBED = True
except Exception:
    USE_OLLAMA_EMBED = False

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "rag_sources"
INDEX_DIR = ROOT / "data" / "rag_index"

OLLAMA_EMBED_MODEL = "nomic-embed-text"
HF_EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def main():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Fant ikke {DOCS_DIR}. Legg dokumentene dine der.")

    if USE_OLLAMA_EMBED:
        Settings.embed_model = OllamaEmbedding(
            model_name=OLLAMA_EMBED_MODEL,
            base_url="http://localhost:11434",
        )
        print(f"Bruker OllamaEmbedding: {OLLAMA_EMBED_MODEL}")
    else:
        Settings.embed_model = HuggingFaceEmbedding(model_name=HF_EMBED_MODEL)
        print(f"Bruker HuggingFaceEmbedding: {HF_EMBED_MODEL}")

    docs = SimpleDirectoryReader(str(DOCS_DIR), recursive=True).load_data()
    if not docs:
        raise ValueError(f"Ingen dokumenter funnet i {DOCS_DIR}")

    splitter = SentenceSplitter(chunk_size=700, chunk_overlap=100)
    index = VectorStoreIndex.from_documents(docs, transformations=[splitter])

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    print("\nIndex bygget og lagret!")
    print(f" - Dokumenter: {len(docs)}")
    print(f" - Lagret i: {INDEX_DIR.resolve()}")


if __name__ == "__main__":
    main()