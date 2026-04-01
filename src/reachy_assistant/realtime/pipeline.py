from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass
class RealtimeConversationPipeline:
    engine: object

    def run(self) -> None:
        try:
            asyncio.run(self.engine.run())
        except KeyboardInterrupt:
            if hasattr(self.engine, "stop"):
                self.engine.stop()
