# denne koden skal embedde txt filene (evt PDF og andre fieler)
# lagere de nye vectorfilene i en ny mappe under /data

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pathlib import Path

DOCS_DIR = Path("data/rag_sources")
INDEX_DIR = Path("data/rag_index")


def main():
    # vil inneholde en liste med documet-objects
    documents = SimpleDirectoryReader(str(DOCS_DIR)).load_data()

    # Må si til llama at vi skal bruke vår egen embedding modell
    # hvis ikke vil llama kjøre default modellen som er en online open AI modell
    Settings.embed_model = HuggingFaceEmbedding(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    # deler dokumentene opp i biter (noder)
    # lager embeddings for hver bit (vektor)
    # lagrer dem i et søkbart vector-indeks
    # returnerer et objekt som kan brukes til semantisk søk
    index = VectorStoreIndex.from_documents(documents)

    # printes hvis alt gikk fint!
    print("lasting av dokumenter: Vellykket!")


    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(INDEX_DIR))

if __name__ == "__main__":
    main()
