"""
Audio source-wrapper som holder roboten sovende til openWakeWord aktiverer den.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from openwakeword.model import Model

from reachy_assistant.realtime.audio import ensure_mono, float32_to_pcm16_bytes, resample_audio
from reachy_assistant.realtime.io_base import RealtimeAudioSource, RealtimeSessionRestart


@dataclass
class OpenWakewordAudioSource(RealtimeAudioSource):
    base_source: RealtimeAudioSource
    model_path: str
    threshold: float = 0.5
    debounce_time: float = 0.8
    preroll_ms: int = 1200
    inactivity_timeout_sec: float = 20.0
    activity_level_threshold: float = 0.015
    on_wake: Callable[[], None] | None = None
    on_sleep: Callable[[], None] | None = None
    sample_rate: int = field(init=False)
    starts_asleep: bool = field(init=False, default=True)
    _model: Model | None = field(init=False, default=None)
    _model_name: str = field(init=False)
    _active: bool = field(init=False, default=False)
    _last_activity_at: float = field(init=False, default=0.0)
    _preroll_chunks: deque[np.ndarray] = field(init=False)
    _preroll_samples: int = field(init=False, default=0)
    _max_preroll_samples: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.sample_rate = int(self.base_source.sample_rate)
        self._model_name = Path(self.model_path).stem
        self._preroll_chunks = deque()
        self._max_preroll_samples = max(
            int(self.sample_rate * max(self.preroll_ms, 0) / 1000),
            0,
        )

    async def open(self) -> None:
        await self.base_source.open()
        self._model = Model(
            wakeword_models=[self.model_path],
            inference_framework="onnx",
        )
        self.reset_session_state()
        print(f"[wakeword] openWakeWord loaded: {self._model_name}")

    async def close(self) -> None:
        await self.base_source.close()

    async def chunks(self):
        async for chunk in self.base_source.chunks():
            mono = ensure_mono(chunk).astype(np.float32, copy=False)
            if mono.size == 0:
                continue

            if not self._active:
                self._store_preroll(mono)

            if not self._active:
                if self._detect_wakeword(mono):
                    self._active = True
                    self._last_activity_at = time.monotonic()
                    print(f"[wakeword] Detected {self._model_name}.")
                    if self.on_wake is not None:
                        self.on_wake()
                    wake_audio = self._consume_preroll()
                    if wake_audio.size:
                        yield wake_audio
                continue

            now = time.monotonic()
            level = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            if level >= self.activity_level_threshold:
                self._last_activity_at = now
            elif now - self._last_activity_at > self.inactivity_timeout_sec:
                self.reset_session_state()
                print("[wakeword] Inactivity timeout reached. Returning to sleep mode.")
                if self.on_sleep is not None:
                    self.on_sleep()
                raise RealtimeSessionRestart("Wakeword session expired; starting a fresh session.")

            yield mono

    def _store_preroll(self, chunk: np.ndarray) -> None:
        if self._max_preroll_samples <= 0:
            return
        buffered = np.array(chunk, copy=True)
        self._preroll_chunks.append(buffered)
        self._preroll_samples += buffered.size
        while self._preroll_samples > self._max_preroll_samples and self._preroll_chunks:
            removed = self._preroll_chunks.popleft()
            self._preroll_samples -= removed.size

    def _consume_preroll(self) -> np.ndarray:
        if not self._preroll_chunks:
            return np.empty(0, dtype=np.float32)
        combined = np.concatenate(tuple(self._preroll_chunks)).astype(np.float32, copy=False)
        self._preroll_chunks.clear()
        self._preroll_samples = 0
        return combined

    def reset_session_state(self) -> None:
        self._active = False
        self._last_activity_at = time.monotonic()
        self._preroll_chunks.clear()
        self._preroll_samples = 0
        if self._model is not None:
            self._model.reset()

    def _detect_wakeword(self, chunk: np.ndarray) -> bool:
        if self._model is None:
            return False

        pcm16 = np.frombuffer(
            float32_to_pcm16_bytes(resample_audio(chunk, self.sample_rate, 16000)),
            dtype=np.int16,
        )
        if pcm16.size == 0:
            return False

        scores = self._model.predict(
            pcm16,
            threshold={self._model_name: self.threshold},
            debounce_time=self.debounce_time,
        )
        score = float(scores.get(self._model_name, 0.0))
        return score >= self.threshold
