from __future__ import annotations

from dataclasses import dataclass

from reachy_assistant.rag.rag_engine import RagEngine, RagResult


@dataclass
class RagAnswer:
    answer: str
    rag: RagResult


@dataclass
class RagService:
    """
    Orkestrerer:
      - retrieve hits
      - bygge kontekst-prompt
      - kalle LLM
      - returnere svar + kilder
    """

    rag_engine: RagEngine
    llm_engine: object  # LLMEngine, men hold det enkelt

    system_prompt: str = (
        "Du er en labassistent. Svar på norsk.\n"
        "Bruk KUN informasjon fra KONTEKST.\n"
        "Hvis du ikke finner svaret i KONTEKST, skriv nøyaktig: "
        "'Jeg finner ikke dette i dokumentasjonen.'"
    )

    def ask(self, question: str) -> RagAnswer:
        rag = self.rag_engine.retrieve(question)

        if not rag.hits:
            return RagAnswer(
                answer="Jeg finner ikke dette i dokumentasjonen.",
                rag=rag,
            )

        # Bygg kontekst (top_k chunks)
        context_parts = []
        for h in rag.hits:
            context_parts.append(f"[KILDE: {h.file} | score={h.score:.3f}]\n{h.text}")

        context = "\n\n---\n\n".join(context_parts)

        user_prompt = f"KONTEKST:\n{context}\n\nSPØRSMÅL:\n{question}\n\nSVAR:"

        answer = self.llm_engine.chat(system=self.system_prompt, user=user_prompt)

        return RagAnswer(answer=answer, rag=rag)