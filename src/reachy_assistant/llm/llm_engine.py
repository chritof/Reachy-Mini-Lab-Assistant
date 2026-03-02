from __future__ import annotations

from dataclasses import dataclass

from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole


@dataclass
class LLMEngine:
    """
    Tynn wrapper rundt Ollama-chat (qwen2.5:14b eller annet).
    """

    model: str = "qwen2.5:14b"
    base_url: str = "http://localhost:11434"
    request_timeout: int = 120

    def _client(self) -> Ollama:
        return Ollama(
            model=self.model,
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