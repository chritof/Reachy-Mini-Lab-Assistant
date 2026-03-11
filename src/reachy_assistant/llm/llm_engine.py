"""
LLM-wrapper.

Sender prompt til LLM-backend
og returnerer svar.

Inneholder kun kommunikasjon med LLM.
"""
from __future__ import annotations

from dataclasses import dataclass

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMEngine:

    base_url: str = "http://localhost:11434"
    request_timeout: int = 120

    def _client(self) -> Ollama:
        return Ollama(
            model=os.getenv("ollama_model", "qwen2.5:14b"),
            base_url=self.base_url,
            request_timeout=self.request_timeout,
        )

    def chat(self, *, system: str, user: str) -> str:
        llm = self._client()
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=system),
            ChatMessage(role=MessageRole.USER, content=user),
        ]
        resp = llm.chat(messages)
        return resp.message.content.strip()