"""
OpenAI-backed LLM wrapper for the original conversation pipeline.

Keeps the same `chat(system=..., user=...) -> str` interface as the Ollama engine
so it can be swapped into `RagService` without changing the pipeline itself.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


@dataclass
class OpenAIChatGPTEngine:
    model: str = "gpt-4o-mini"
    timeout: float = 120.0

    def _client(self) -> OpenAI:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAIChatGPTEngine.")
        return OpenAI(api_key=api_key, timeout=self.timeout)

    def chat(self, *, system: str, user: str) -> str:
        client = self._client()
        response = client.responses.create(
            model=os.getenv("OPENAI_CHAT_MODEL", self.model),
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )

        text = getattr(response, "output_text", "") or ""
        return text.strip()
