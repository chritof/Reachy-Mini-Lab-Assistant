from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DOCS_DIR = Path("data/rag_sources")
INDEX_DIR = Path("data/rag_index")
#info:
#BAAI/bge-small-en-v1.5 er best, men
#sentence-transformers/all-MiniLM-L6-v2 er raskest
#PS: må bruke lik embedding fra query gitt av bruker (llm...v1,v1..vn)
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def main():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Fant ikke {DOCS_DIR}")

    
    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    docs = SimpleDirectoryReader(str(DOCS_DIR), recursive=True).load_data()
    if not docs:
        raise ValueError(f"Ingen dokumenter funnet i {DOCS_DIR}")

    splitter = SentenceSplitter(separator=" ", chunk_size=1024, chunk_overlap=128)

    
    index = VectorStoreIndex.from_documents(docs, transformations=[splitter])

    
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))

    print(f"Index bygget og lagret. Dokumenter: {len(docs)}")
    print(f"Lagret i: {INDEX_DIR.resolve()}")

if __name__ == "__main__":
    main()