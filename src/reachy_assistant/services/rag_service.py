from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz, process

from reachy_assistant.rag.rag_engine import RagEngine, RagResult, RagHit


@dataclass
class RagAnswer:
    answer: str
    rag: RagResult
    used_query: str


@dataclass
class RagService:
    rag_engine: object
    llm_engine: object

    system_prompt = """
    Du er en assistent for Læringslab ved Høgskulen på Vestlandet.

    Du svarer på spørsmål fra studenter og ansatte om læringslabben og utstyr.

    Regler:
    - Svar alltid på norsk
    - Bruk kun informasjon fra KONTEKST
    - Ikke finn på informasjon
    - Svar kort (1–3 setninger)

    Hvis svaret ikke finnes i dokumentasjonen, svar:
    "Jeg finner ikke dette i dokumentasjonen."
    """

    last_topic: Optional[str] = None
    last_topic_file: Optional[str] = None

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
        "den", "det", "denne", "dette", "den der", "det der",
        "den tingen", "produktet", "den igjen"
    ]

    def _normalize_query(self, q: str) -> str:
        q2 = (q or "").lower().strip()
        q2 = re.sub(r"[?!.:,;]", " ", q2)
        for pat in self._FILLER_PATTERNS:
            q2 = re.sub(pat, " ", q2)
        q2 = re.sub(r"\s+", " ", q2).strip()
        return q2

    def _is_reference_question(self, q: str) -> bool:
        qn = (q or "").lower()
        return any(w in qn for w in self._REFERENCE_WORDS)

    def _rewrite_with_context(self, question: str) -> str:
        if self.last_topic and self._is_reference_question(question):
            return f"{self.last_topic}: {question}"
        return question

    def _best_title_guess(self, q: str, titles: list[str]) -> str | None:
        if not titles:
            return None
        match = process.extractOne(q, titles, scorer=fuzz.token_set_ratio)
        if match and match[1] >= 75:
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
        question = (question or "").strip()
        if not question:
            return RagAnswer(
                answer="Jeg finner ikke dette i dokumentasjonen.",
                rag=RagResult(hits=[]),
                used_query="",
            )

        rewritten = self._rewrite_with_context(question)

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

        # Debug
        print(f"QUESTION: {question}")
        for h in hits[:3]:
            print(f"HIT: {h.file} | score={h.score:.3f}")

        if not hits:
            return RagAnswer(
                answer="Jeg finner ikke dette i dokumentasjonen.",
                rag=RagResult(hits=[]),
                used_query=rewritten,
            )

        best = hits[0]
        self.last_topic_file = best.file
        self.last_topic = best.file.rsplit(".", 1)[0]

        # Hard cutoff: hvis beste treff er for svakt, ikke bruk LLM
        if best.score < 0.50:
            return RagAnswer(
                answer="Jeg finner ikke dette i dokumentasjonen.",
                rag=RagResult(hits=[best]),
                used_query=rewritten,
            )

        # Bruk bare beste treff for å unngå støy
        hits = [best]

        context = "\n\n".join(hit.text for hit in hits)

        user_prompt = f"""
        KONTEKST FRA LÆRINGSLABBENS DOKUMENTASJON:

        {context}

        SPØRSMÅL:
        {question}

        Svar kun basert på informasjonen i KONTEKST.
        Hvis svaret ikke finnes der, skriv:
        Jeg finner ikke dette i dokumentasjonen.
        """

        answer = self.llm_engine.chat(system=self.system_prompt, user=user_prompt).strip()

        if not answer:
            answer = "Jeg finner ikke dette i dokumentasjonen."

        return RagAnswer(
            answer=answer,
            rag=RagResult(hits=hits),
            used_query=rewritten,
        )