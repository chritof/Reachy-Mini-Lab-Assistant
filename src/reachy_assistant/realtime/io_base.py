from __future__ import annotations

from typing import AsyncIterator, Protocol

import numpy as np


class RealtimeSessionRestart(Exception):
    """Signaliserer at den aktive realtime-sessionen skal startes på nytt."""


class RealtimeAudioSource(Protocol):
    sample_rate: int

    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def chunks(self) -> AsyncIterator[np.ndarray]: ...


class RealtimeAudioSink(Protocol):
    async def open(self, sample_rate: int) -> None: ...

    async def write(self, audio: np.ndarray, sample_rate: int) -> None: ...

    async def clear(self) -> None: ...

    async def close(self) -> None: ...
