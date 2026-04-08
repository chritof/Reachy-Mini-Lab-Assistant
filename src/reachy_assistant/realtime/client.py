"""
Tynn wrapper rundt OpenAI sin realtime-klient for å opprette sesjoner.
"""

from __future__ import annotations

from openai import AsyncOpenAI

from reachy_assistant.realtime.config import RealtimeConfig


class OpenAIRealtimeClient:
    def __init__(self, config: RealtimeConfig) -> None:
        self.config = config
        self._client = AsyncOpenAI(api_key=config.api_key)

    def connect(self):
        return self._client.beta.realtime.connect(model=self.config.model)
