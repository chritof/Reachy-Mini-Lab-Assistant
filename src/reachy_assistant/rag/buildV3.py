from pathlib import Path

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DOCS_DIR = Path("data/rag_sources")
INDEX_DIR = Path("data/rag_index")

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 96

def main():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Fant ikke {DOCS_DIR}")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    docs = SimpleDirectoryReader(str(DOCS_DIR), recursive=True).load_data()
    if not docs:
        raise ValueError(f"Ingen dokumenter funnet i {DOCS_DIR}")

    splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    index = VectorStoreIndex.from_documents(docs, transformations=[splitter])

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    print("Index bygget og lagret")
    print(f"   Dokumenter: {len(docs)}")
    print(f"   Chunk size: {CHUNK_SIZE}, overlap: {CHUNK_OVERLAP}")
    print(f"   Embed model: {EMBED_MODEL}")
    print(f"   Lagret i: {INDEX_DIR.resolve()}")

if __name__ == "__main__":
    main()