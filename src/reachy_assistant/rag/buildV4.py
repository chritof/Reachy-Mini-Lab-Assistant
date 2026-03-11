from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
import shutil

ROOT = Path(__file__).resolve().parents[3]
DOCS_DIR = ROOT / "data" / "rag_sources"
INDEX_DIR = ROOT / "data" / "rag_index"

def main():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Fant ikke {DOCS_DIR}")

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
    )

    docs = SimpleDirectoryReader(str(DOCS_DIR), recursive=True).load_data()
    if not docs:
        raise ValueError(f"Ingen dokumenter funnet i {DOCS_DIR}")

    print("Dokumenter som indekseres:")
    for d in docs:
        print("-", d.metadata.get("file_name"))

    splitter = SentenceSplitter(chunk_size=250, chunk_overlap=40)
    index = VectorStoreIndex.from_documents(docs, transformations=[splitter])

    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    print(f"Index bygget i {INDEX_DIR.resolve()}")

if __name__ == "__main__":
    main()