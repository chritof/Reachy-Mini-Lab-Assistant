from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import numpy as np

from reachy_assistant.realtime.audio import ensure_mono, resample_audio
from reachy_assistant.realtime.io_base import RealtimeAudioSink, RealtimeAudioSource


@dataclass
class ReachyRealtimeAudioSource(RealtimeAudioSource):
    mini: object
    poll_interval: float = 0.01
    sample_rate: int = field(init=False)
    _active: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.sample_rate = int(self.mini.media.get_input_audio_samplerate())

    async def open(self) -> None:
        await asyncio.to_thread(self.mini.media.start_recording)
        self._active = True

    async def close(self) -> None:
        if self._active:
            await asyncio.to_thread(self.mini.media.stop_recording)
            self._active = False

    async def chunks(self):
        while True:
            samples = await asyncio.to_thread(self.mini.media.get_audio_sample)
            if samples is None:
                await asyncio.sleep(self.poll_interval)
                continue
            chunk = ensure_mono(np.asarray(samples, dtype=np.float32))
            if chunk.size == 0:
                await asyncio.sleep(self.poll_interval)
                continue
            yield chunk


@dataclass
class ReachyRealtimeAudioSink(RealtimeAudioSink):
    mini: object
    chunk_size: int = 2048
    _sample_rate: int | None = field(init=False, default=None)

    async def open(self, sample_rate: int) -> None:
        if self._sample_rate == sample_rate:
            return
        self._sample_rate = sample_rate
        await asyncio.to_thread(self.mini.media.start_playing)

    async def write(self, audio: np.ndarray, sample_rate: int) -> None:
        target_rate = int(self.mini.media.get_output_audio_samplerate())
        if self._sample_rate != target_rate:
            await self.open(target_rate)
        chunk = resample_audio(audio, sample_rate, target_rate)
        chunk = np.clip(chunk, -1.0, 1.0).astype(np.float32)

        for index in range(0, len(chunk), self.chunk_size):
            piece = chunk[index:index + self.chunk_size]
            await asyncio.to_thread(self.mini.media.push_audio_sample, piece)
            await asyncio.sleep(len(piece) / target_rate)

    async def close(self) -> None:
        if self._sample_rate is not None:
            await asyncio.to_thread(self.mini.media.stop_playing)
            self._sample_rate = None
