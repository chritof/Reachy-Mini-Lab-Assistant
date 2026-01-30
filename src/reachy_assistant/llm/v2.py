# Denne modellen er en oppgradering fra V1.
# Modellen sender mer tydlige svar til LLM.

from pathlib import Path
import requests
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# CONFIG
# alt dette kan endres ved behov:
INDEX_DIR = Path("data/index")
MODEL = "mistral"
OLLAMA_URL = "http://localhost:11434/api/chat"


SYSTEM_PROMPT = (
   "Du er en hjelpsom labassistent. Svar på norsk. "
    "Svar kort og konkret. "
    "Bruk kun informasjon fra KONTEKST. "
    "Hvis svaret ikke finnes i konteksten, si: 'Jeg finner ikke dette i dokumentasjonen.' "
    "Avslutt med et realistisk eksempel."
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



# modell:
def main():
    #må være samme embeddingmodell som når vi bygde index!
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    index = load_index_from_storage(StorageContext.from_defaults(persist_dir=str(INDEX_DIR)))

    print("loading: vellykket!")

    retriever = index.as_retriever()

    while True:
        q = input("\nSpørsmål: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        results = call_ollama(q)
        if not results:
            print("\nSvar: Jeg fant ingenting relevant i dokumentene.")
            continue

        print("\n--- Treff ---")
        for i, r in enumerate(results, start=1):
            meta = r.node.metadata or {}
            source = meta.get("file_name") or meta.get("filename") or "ukjent fil"
            score = float(r.score or 0.0)
            text = r.node.get_content()

            print(f"\n[{i}] Source: {source} | score={score:.3f}")
            print(text)

if __name__ == "__main__":
    main()