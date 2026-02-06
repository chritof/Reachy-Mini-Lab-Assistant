# MÅL: sende med de siste 3 meldingen til llm.
# Når bruker sender den 4 meldingne, så slettes den siste

# Strategi 1: lagre spørsmål i en liste, fjerner den siste meldingen
# hvis listen er full.

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

MAX_MESSAGES = 4

chat_history = [
    {"role": "system", "content": SYSTEM_PROMPT}
]

def call_ollama(question: str, context: str) -> str:

    
    if len(chat_history) > MAX_MESSAGES:
        chat_history.pop(1)
        chat_history.append({
        "role": "user", "content": f"KONTEKST:\n{context}\n\nSPØRSMÅL: {question}\nSVAR:"
    })
    else:
        chat_history.append({
        "role": "user", "content": f"KONTEKST:\n{context}\n\nSPØRSMÅL: {question}\nSVAR:"
    })
    
    #dette er det llm see!
    print("----------------++-------------")
    print("DET LLM SER (historikk):\n")
    for m in chat_history:
        print(f"{m['role']}: {m['content']}\n")
    print("----------------++-------------")


    # Send hele historikken til Ollama
    r = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": chat_history, "stream": False},
        timeout=120,
    )
    r.raise_for_status()

    # Hent selve svaret fra Ollama
    answer = r.json()["message"]["content"]

    # Legg modellens svar inn i historikken (så den husker det neste gang)
    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    return answer

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