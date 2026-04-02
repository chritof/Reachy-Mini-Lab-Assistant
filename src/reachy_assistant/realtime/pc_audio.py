from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd

from reachy_assistant.realtime.audio import ensure_mono
from reachy_assistant.realtime.io_base import RealtimeAudioSink, RealtimeAudioSource


@dataclass
class PcRealtimeAudioSource(RealtimeAudioSource):
    sample_rate: int = 24000
    chunk_frames: int = 2400
    channels: int = 1
    _queue: asyncio.Queue[np.ndarray] = field(init=False)
    _stream: sd.InputStream | None = field(init=False, default=None)
    _loop: asyncio.AbstractEventLoop | None = field(init=False, default=None)

    async def open(self) -> None:
        self._queue = asyncio.Queue()
        self._loop = asyncio.get_running_loop()

        def callback(indata, frames, time_info, status) -> None:
            del frames, time_info
            if status:
                print(f"[realtime][pc-input] {status}")
            chunk = ensure_mono(np.array(indata, dtype=np.float32, copy=True))
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._queue.put_nowait, chunk)

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_frames,
            callback=callback,
        )
        self._stream.start()

    async def close(self) -> None:
        if self._stream is not None:
            await asyncio.to_thread(self._stream.stop)
            await asyncio.to_thread(self._stream.close)
            self._stream = None

    async def chunks(self):
        while True:
            yield await self._queue.get()


@dataclass
class PcRealtimeAudioSink(RealtimeAudioSink):
    channels: int = 1
    _stream: sd.OutputStream | None = field(init=False, default=None)
    _sample_rate: int | None = field(init=False, default=None)
    _write_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    async def open(self, sample_rate: int) -> None:
        async with self._write_lock:
            if self._stream is not None and self._sample_rate == sample_rate:
                return
            await self._close_locked()
            self._sample_rate = sample_rate
            self._stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=self.channels,
                dtype="float32",
                blocksize=0,
            )
            self._stream.start()

    async def write(self, audio: np.ndarray, sample_rate: int) -> None:
        async with self._write_lock:
            if self._stream is None or self._sample_rate != sample_rate:
                await self._open_locked(sample_rate)
            assert self._stream is not None
            chunk = ensure_mono(audio).reshape(-1, 1)
            await asyncio.to_thread(self._stream.write, chunk)

    async def clear(self) -> None:
        async with self._write_lock:
            if self._stream is None or self._sample_rate is None:
                return
            await asyncio.to_thread(self._stream.abort)
            await asyncio.to_thread(self._stream.close)
            sample_rate = self._sample_rate
            self._stream = None
            self._sample_rate = None
            await self._open_locked(sample_rate)

    async def close(self) -> None:
        async with self._write_lock:
            await self._close_locked()

    async def _open_locked(self, sample_rate: int) -> None:
        self._sample_rate = sample_rate
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=0,
        )
        self._stream.start()

    async def _close_locked(self) -> None:
        if self._stream is not None:
            await asyncio.to_thread(self._stream.stop)
            await asyncio.to_thread(self._stream.close)
            self._stream = None
            self._sample_rate = None
