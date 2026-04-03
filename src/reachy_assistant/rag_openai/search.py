from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass
class OpenAIRagSearch:
    vector_store: object
    embeddings: object

    def search(
        self,
        query: str,
        category: str | None = None,
        limit: int = 3,
    ) -> dict[str, Any]:
        query = (query or "").strip()
        if not query:
            return {"error": "query is required"}

        query_vector = self.embeddings.embed_one(query)
        wants_loan = self._wants_loan(query)
        search_limit = max(limit * 3, 6)
        raw_results = self.vector_store.search(query_vector, category=category, limit=search_limit)
        if not raw_results and category:
            raw_results = self.vector_store.search(query_vector, category=None, limit=search_limit)
        results = self._rerank_results(query, raw_results, limit=limit)

        if not results:
            return {"answer": "Jeg finner ikke dette i dokumentasjonen."}

        snippets = []
        sources = []
        for item in results:
            source = item["source"]
            text = " ".join((item.get("text") or "").split())
            snippet = text[:350].strip()
            snippets.append(f"[{source}] {snippet}")
            sources.append(source)

        context = "\n\n---\n\n".join(
            f"[{item['source']}]\n{item['text']}" for item in results
        )
        return {
            "answer": self._build_answer(
                query=query,
                results=results,
                wants_loan=wants_loan,
                fallback=" ".join(snippets[:2]),
            ),
            "results": results,
            "context": context,
            "sources": sources,
        }

    def _rerank_results(
        self,
        query: str,
        results: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        query_terms = self._terms(query)
        wants_loan = self._wants_loan(query)

        rescored: list[tuple[float, dict[str, Any]]] = []
        for item in results:
            source_terms = self._terms(str(item.get("source", "")))
            text_terms = self._terms(str(item.get("text", ""))[:250])
            lexical_overlap = len(query_terms & (source_terms | text_terms))
            equipment_boost = 0.0
            if lexical_overlap >= 2:
                equipment_boost += 0.25
            elif lexical_overlap == 1:
                equipment_boost += 0.1

            if wants_loan and "utlaan" in str(item.get("source", "")):
                equipment_boost += 0.12

            rescored.append((float(item.get("score", 0.0)) + equipment_boost, item))

        rescored.sort(key=lambda pair: pair[0], reverse=True)

        selected: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        saw_equipment = False
        saw_utlaan = False
        for score, item in rescored:
            item = dict(item)
            item["score"] = score
            source = str(item.get("source", ""))
            if source in seen_sources:
                continue
            selected.append(item)
            seen_sources.add(source)
            saw_equipment = saw_equipment or ("utlaan" not in source and len(self._terms(source) & query_terms) > 0)
            saw_utlaan = saw_utlaan or ("utlaan" in source)
            if len(selected) >= limit:
                break

        if wants_loan and saw_equipment and not saw_utlaan:
            for score, item in rescored:
                source = str(item.get("source", ""))
                if "utlaan" in source and source not in seen_sources and selected:
                    extra = dict(item)
                    extra["score"] = score
                    selected[-1] = extra
                    break

        return selected[:limit]

    def _build_answer(
        self,
        query: str,
        results: list[dict[str, Any]],
        *,
        wants_loan: bool,
        fallback: str,
    ) -> str:
        if not wants_loan:
            return fallback

        equipment = self._primary_equipment_result(query, results)
        if equipment is None:
            return fallback

        source = str(equipment.get("source", ""))
        title = self._extract_title(str(equipment.get("text", ""))) or self._humanize_source(source)
        status = self._extract_utlaan_status(results)

        if "ikke til utl" in self._normalized_text(status):
            return (
                f"{title} er ikke til utlån. "
                "Den skal brukes på stedet eller etter særskilt avtale. "
                "Se også utlaan.txt for de generelle reglene."
            )

        return (
            f"{title} kan normalt lånes i Læringslaben hvis ikke annet er oppgitt. "
            "De generelle reglene sier at utstyr kan lånes av studenter og ansatte, "
            "at noe utstyr kan kreve opplæring, og at alt skal leveres tilbake i samme stand. "
            "Spør ansatte om tilgjengelighet og eventuelle krav før utlån."
        )

    def _primary_equipment_result(self, query: str, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        query_terms = self._terms(query)
        best: dict[str, Any] | None = None
        best_overlap = -1
        for item in results:
            source = str(item.get("source", ""))
            if "utlaan" in source:
                continue
            overlap = len(self._terms(source) & query_terms)
            if overlap > best_overlap:
                best = item
                best_overlap = overlap
        return best or (results[0] if results else None)

    @staticmethod
    def _extract_title(text: str) -> str:
        match = re.search(r"Tittel:\s*(.+)", text)
        return match.group(1).strip() if match else ""

    @staticmethod
    def _extract_utlaan_status(results: list[dict[str, Any]]) -> str:
        for item in results:
            text = str(item.get("text", ""))
            match = re.search(r"Utl[åa]nsstatus:\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
            match = re.search(r"UtlÃ¥nsstatus:\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ""

    @staticmethod
    def _humanize_source(source: str) -> str:
        stem = source.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].removesuffix(".txt")
        return stem.replace("_", " ")

    def _wants_loan(self, query: str) -> bool:
        normalized = self._normalized_text(query)
        return any(token in normalized for token in ("lan", "lane", "utlan", "borrow", "loan"))

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9æøåÆØÅ]+", (text or "").lower()))

    @staticmethod
    def _normalized_text(text: str) -> str:
        text = (text or "").lower().replace("Ã¥", "å").replace("Ã¸", "ø").replace("Ã¦", "æ")
        text = (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
        return text
