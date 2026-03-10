from pathlib import Path
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding

ROOT = Path(__file__).resolve().parents[1]  # repo root
INDEX_DIR = ROOT / "data" / "rag_index"

def main():
    Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text", base_url="http://localhost:11434")

    storage = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = load_index_from_storage(storage)

    retriever = index.as_retriever(similarity_top_k=3)
    results = retriever.retrieve("Hva handler dokumentasjonen om?")

    print("Hits:", len(results))
    for r in results:
        meta = r.node.metadata or {}
        print("-", meta.get("file_name", "ukjent"), "score=", r.score)

if __name__ == "__main__":
    main()