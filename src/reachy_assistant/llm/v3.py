from pathlib import Path
import requests

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

DOCS_DIR = Path("data/rag_sources")  
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral:latest"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

SYSTEM_PROMPT = (
    "Du er en labassistent. Svar på norsk.\n"
    "Bruk kun informasjon fra KONTEKST. Hvis du ikke finner svaret der, si: "
    "'Jeg finner ikke dette i dokumentasjonen.'"
)

def call_ollama(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"KONTEKST:\n{context}\n\nSPØRSMÅL: {question}\nSVAR:"},
    ]
    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]

def main():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Fant ikke {DOCS_DIR}. Legg dokumentene dine der.")

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    docs = SimpleDirectoryReader(str(DOCS_DIR), recursive=True).load_data()
    if not docs:
        raise ValueError(f"Ingen filer funnet i {DOCS_DIR}")

    index = VectorStoreIndex.from_documents(docs)
    retriever = index.as_retriever(similarity_top_k=3)

    print(" Klar! (RAG uten lagring) Skriv 'exit' for å avslutte.")
    while True:
        q = input("\nSpørsmål: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        results = retriever.retrieve(q)
        if not results:
            print("\nSvar: Jeg finner ikke dette i dokumentasjonen.")
            continue

        context = results[0].node.get_content()

        answer = call_ollama(q, context)
        print("\nSvar:\n", answer)

if __name__ == "__main__":
    main()