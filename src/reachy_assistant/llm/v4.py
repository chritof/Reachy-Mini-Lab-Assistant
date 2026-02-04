from pathlib import Path
import requests

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

INDEX_DIR = Path("data/rag_index")

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "mistral:latest"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

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
    #dette er det llm see!
    print(f"----------------++-------------")
    print(f"KONTEKST: {context} \n")
    print(f"SPØRSMPL: {question} \n")
    print(f"----------------++-------------")


    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"]

def main():
    if not INDEX_DIR.exists():
        print("Fant ikke data/index. Kjør buildV2.py først.")
        return

    Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)

    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = load_index_from_storage(storage_context)

    retriever = index.as_retriever(similarity_top_k=3)

    print("Klar! Skriv exit for å avslutte.")
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