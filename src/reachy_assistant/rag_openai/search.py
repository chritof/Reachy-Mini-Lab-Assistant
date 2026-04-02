from __future__ import annotations

import re
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
        raw_results = self.vector_store.search(query_vector, category=category, limit=max(limit * 3, 6))
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
            "answer": " ".join(snippets[:2]),
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
        wants_loan = any(term in {"lån", "låne", "utlån", "borrow", "loan"} for term in query_terms)

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

    @staticmethod
    def _terms(text: str) -> set[str]:
        return set(re.findall(r"[a-zA-Z0-9æøåÆØÅ]+", (text or "").lower()))
