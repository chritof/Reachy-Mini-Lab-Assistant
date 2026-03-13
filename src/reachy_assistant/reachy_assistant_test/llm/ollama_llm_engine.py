from __future__ import annotations

from llama_index.llms.ollama import Ollama


class OllamaLLMEngine:
    def __init__(
        self,
        model: str = "mistral:latest",
        base_url: str = "http://localhost:11434",
        request_timeout: int = 120,
    ):
        self.client = Ollama(
            model=model,
            base_url=base_url,
            request_timeout=request_timeout,
        )

    def chat(self, system: str, user: str) -> str:
        prompt = f"SYSTEM:\n{system}\n\nUSER:\n{user}\n\nASSISTANT:\n"
        response = self.client.complete(prompt)
        return str(response.text).strip()