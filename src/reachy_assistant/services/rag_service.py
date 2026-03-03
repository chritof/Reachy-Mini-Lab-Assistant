"""
RAG-orkestrering.

Kobler sammen:
- RagEngine (henter relevant kontekst)
- LLMEngine (genererer svar)

Flyt:
Spørsmål -> hent kontekst -> kall LLM -> returner svar
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz import fuzz, process

from reachy_assistant.rag.rag_engine import RagEngine, RagResult, RagHit


@dataclass
class RagAnswer:
    answer: str
    rag: RagResult
    used_query: str  # nyttig for debugging


@dataclass
class RagService:
    rag_engine: RagEngine
    llm_engine: object  # LLMEngine

    system_prompt: str = (
        "Du er en labassistent. Svar på norsk.\n"
        "Svar kort (maks 3-4 setninger) med det viktigste først"
        "Bruk KONTEKST som primærkilde.\n"
        "Hvis du ikke finner svaret i KONTEKST, skriv nøyaktig: "
        "'Jeg finner ikke dette i dokumentasjonen.'"
    )

    # "memory"
    last_topic: Optional[str] = None  # f.eks. "Apple iPad 6" (fra beste hit)
    last_topic_file: Optional[str] = None  # f.eks. "Apple iPad 6.txt"

    _FILLER_PATTERNS = [
        r"\bkan du\b",
        r"\bkunne du\b",
        r"\bvennligst\b",
        r"\bfortell( meg)?\b",
        r"\bsnakk(e)?( litt)? om\b",
        r"\bhva står det om\b",
        r"\bhva (er|handler) (det )?om\b",
        r"\bsi( litt)? om\b",
        r"\bhar dere\b",
        r"\bhvis dere har\b",
    ]

    _REFERENCE_WORDS = [
        "den", "det", "denne", "dette", "den der", "det der", "den tingen", "produktet", "den igjen"
    ]

    def _normalize_query(self, q: str) -> str:
        q2 = q.lower().strip()
        q2 = re.sub(r"[?!.:,;]", " ", q2)
        for pat in self._FILLER_PATTERNS:
            q2 = re.sub(pat, " ", q2)
        q2 = re.sub(r"\s+", " ", q2).strip()
        return q2

    def _is_reference_question(self, q: str) -> bool:
        qn = q.lower()
        return any(w in qn for w in self._REFERENCE_WORDS)

    def _rewrite_with_context(self, question: str) -> str:
        """
        Hvis spørsmålet ser ut som 'den/det' og vi har en last_topic,
        så prepender vi temaet slik at retrieval blir stabil.
        """
        if self.last_topic and self._is_reference_question(question):
            return f"{self.last_topic}: {question}"
        return question

    def _best_title_guess(self, q: str, titles: list[str]) -> str | None:
        if not titles:
            return None
        match = process.extractOne(q, titles, scorer=fuzz.token_set_ratio)
        if match and match[1] >= 70:
            return match[0]
        return None

    def _merge_hits(self, hit_lists: list[list[RagHit]]) -> list[RagHit]:
        best: dict[str, RagHit] = {}
        for hits in hit_lists:
            for h in hits:
                prev = best.get(h.file)
                if prev is None or h.score > prev.score:
                    best[h.file] = h
        return sorted(best.values(), key=lambda h: h.score, reverse=True)

    def ask(self, question: str) -> RagAnswer:
        # 1) Rewrite for retrieval (context-aware)
        rewritten = self._rewrite_with_context(question)

        # 2) Robust multi-retrieve (som før)
        r1 = self.rag_engine.retrieve(rewritten)
        norm = self._normalize_query(rewritten)
        titles = [h.file.rsplit(".", 1)[0] for h in r1.hits]
        guess = self._best_title_guess(norm, titles)

        queries: list[str] = [rewritten]
        if norm and norm != rewritten:
            queries.append(norm)
        if guess:
            queries.append(guess)

        results: list[list[RagHit]] = [r1.hits]
        for q in queries[1:]:
            rr = self.rag_engine.retrieve(q)
            results.append(rr.hits)

        hits = self._merge_hits(results)

        if not hits:
            return RagAnswer(
                answer="Jeg finner ikke dette i dokumentasjonen.",
                rag=RagResult(hits=[]),
                used_query=rewritten,
            )

        # 3) Oppdater "memory" basert på beste hit
        best = hits[0]
        self.last_topic_file = best.file
        self.last_topic = best.file.rsplit(".", 1)[0]  # uten .txt

        # 4) Bygg kontekst + kall LLM
        top_n = 5
        hits = hits[:top_n]

        context = "\n\n---\n\n".join(
            [f"[KILDE: {h.file} | score={h.score:.3f}]\n{h.text}" for h in hits]
        )

        user_prompt = (
            f"KONTEKST:\n{context}\n\n"
            f"SPØRSMÅL:\n{question}\n\n"
            f"SVAR:"
        )

        answer = self.llm_engine.chat(system=self.system_prompt, user=user_prompt)

        return RagAnswer(answer=answer, rag=RagResult(hits=hits), used_query=rewritten)