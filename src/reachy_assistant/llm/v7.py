from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

INDEX_DIR = Path("data/rag_index")

OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "mistral:latest"         
EMBED_MODEL = "nomic-embed-text"      

TOP_K = 3                            
MAX_TURNS_MEMORY = 3                  

SYSTEM_PROMPT = (
    "Du er en labassistent. Svar på norsk.\n"
    "Bruk KUN informasjon fra KONTEKST.\n"
    "Hvis du ikke finner svaret i KONTEKST, skriv nøyaktig: "
    "'Jeg finner ikke dette i dokumentasjonen.'"
)

chat_history: list[ChatMessage] = [
    ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT)
]


def trim_history():
    """
    Behold:
      - system-meldingen (alltid)
      - siste MAX_TURNS_MEMORY runder (user+assistant) = 2 * MAX_TURNS_MEMORY meldinger
    """
    global chat_history

    keep_last = 1 + (MAX_TURNS_MEMORY * 2)
    if len(chat_history) > keep_last:
        chat_history = [chat_history[0]] + chat_history[-(MAX_TURNS_MEMORY * 2):]


def build_context(results) -> tuple[str, list[tuple[str, float]]]:
    """
    Lager en kontekststreng + kildeliste fra retriever-resultater.
    """
    context_parts = []
    sources = []

    for r in results:
        meta = r.node.metadata or {}
        source = meta.get("file_name") or meta.get("filename") or "ukjent fil"
        score = float(r.score or 0.0)
        text = r.node.get_content()

        sources.append((source, score))
        context_parts.append(f"[KILDE: {source} | score={score:.3f}]\n{text}")

    context = "\n\n---\n\n".join(context_parts)
    return context, sources


def main():
    Settings.embed_model = OllamaEmbedding(
        model_name=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    print(f"Bruker OllamaEmbedding: {EMBED_MODEL}")

    if not INDEX_DIR.exists():
        print(f"Fant ikke {INDEX_DIR}. Kjør build-scriptet først.")
        return

    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))
    index = load_index_from_storage(storage_context)

    retriever = index.as_retriever(similarity_top_k=TOP_K)

    llm = Ollama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=120,
    )

    print("Klar! (RAG + minne) Skriv 'exit' for å avslutte.")

    while True:
        q = input("\nSpørsmål: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        
        results = retriever.retrieve(q)
        if not results:
            print("\nSvar: Jeg finner ikke dette i dokumentasjonen.")
            continue

        context, sources = build_context(results)

       
        user_content = (
            f"KONTEKST:\n{context}\n\n"
            f"SPØRSMÅL: {q}\n"
            f"SVAR:"
        )

        chat_history.append(
            ChatMessage(role=MessageRole.USER, content=user_content)
        )

        trim_history()

        print("\n----------------++-------------")
        print("DET LLM SER (historikk):\n")
        for m in chat_history:
            role = m.role.value
            print(f"{role}:\n{m.content}\n")
        print("----------------++-------------\n")

        
        response = llm.chat(chat_history)
        answer = response.message.content

        chat_history.append(
            ChatMessage(role=MessageRole.ASSISTANT, content=answer)
        )
        trim_history()

        print("\nSvar:\n", answer)
        print("\nKilder:")
        for s, sc in sources:
            print(f"- {s} (score={sc:.3f})")


if __name__ == "__main__":
    main()