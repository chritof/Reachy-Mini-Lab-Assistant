from __future__ import annotations


MODEL = "text-embedding-3-small"


class OpenAIEmbeddings:
    def __init__(self, api_key: str, model: str = MODEL) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIEmbeddings.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the 'openai' package to use rag_openai embeddings.") from exc

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in response.data]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
