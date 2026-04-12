"""
Audio source-wrapper som holder roboten sovende til openWakeWord aktiverer den.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from reachy_assistant.realtime.audio import ensure_mono, float32_to_pcm16_bytes, resample_audio
from reachy_assistant.realtime.io_base import RealtimeAudioSource, RealtimeSessionRestart

if TYPE_CHECKING:
    from openwakeword.model import Model


@dataclass
class OpenWakewordAudioSource(RealtimeAudioSource):
    base_source: RealtimeAudioSource
    model_path: str
    sleep_model_path: str | None = None
    threshold: float = 0.5
    sleep_threshold: float | None = None
    debounce_time: float = 0.8
    preroll_ms: int = 1200
    inactivity_timeout_sec: float = 4.0
    activity_level_threshold: float = 0.015
    on_wake: Callable[[], None] | None = None
    on_sleep: Callable[[], None] | None = None
    sample_rate: int = field(init=False)
    starts_asleep: bool = field(init=False, default=True)
    _model: "Model | None" = field(init=False, default=None)
    _wake_model_name: str = field(init=False)
    _sleep_model_name: str | None = field(init=False, default=None)
    _active: bool = field(init=False, default=False)
    _last_activity_at: float = field(init=False, default=0.0)
    _preroll_chunks: deque[np.ndarray] = field(init=False)
    _preroll_samples: int = field(init=False, default=0)
    _max_preroll_samples: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.sample_rate = int(self.base_source.sample_rate)
        self._wake_model_name = Path(self.model_path).stem
        self._sleep_model_name = Path(self.sleep_model_path).stem if self.sleep_model_path else None
        self._preroll_chunks = deque()
        self._max_preroll_samples = max(
            int(self.sample_rate * max(self.preroll_ms, 0) / 1000),
            0,
        )

    async def open(self) -> None:
        from openwakeword.model import Model

        await self.base_source.open()
        wakeword_models = [self.model_path]
        if self.sleep_model_path:
            wakeword_models.append(self.sleep_model_path)
        self._model = Model(
            wakeword_models=wakeword_models,
            inference_framework="onnx",
        )
        self.reset_session_state()
        print(f"[wakeword] openWakeWord loaded: {self._wake_model_name}")
        if self._sleep_model_name:
            print(f"[wakeword] Sleep word loaded: {self._sleep_model_name}")

    async def close(self) -> None:
        await self.base_source.close()

    def is_awake(self) -> bool:
        return self._active

    def touch_activity(self) -> None:
        if self._active:
            self._last_activity_at = time.monotonic()

    async def chunks(self):
        async for chunk in self.base_source.chunks():
            mono = ensure_mono(chunk).astype(np.float32, copy=False)
            if mono.size == 0:
                continue

            if not self._active:
                self._store_preroll(mono)

            if not self._active:
                if self._detect_model(mono, self._wake_model_name, self.threshold):
                    self._active = True
                    self._last_activity_at = time.monotonic()
                    print(f"[wakeword] Detected {self._wake_model_name}.")
                    self._run_callback(self.on_wake, "wake")
                    self._consume_preroll()
                continue

            if self._sleep_model_name and self._detect_model(
                mono,
                self._sleep_model_name,
                self.sleep_threshold if self.sleep_threshold is not None else self.threshold,
            ):
                print(f"[wakeword] Detected {self._sleep_model_name}. Returning to sleep mode.")
                self.reset_session_state()
                self._run_callback(self.on_sleep, "sleep")
                raise RealtimeSessionRestart("Sleep word detected; starting a fresh wakeword session.")

            now = time.monotonic()
            level = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
            if level >= self.activity_level_threshold:
                self._last_activity_at = now
            elif now - self._last_activity_at > self.inactivity_timeout_sec:
                self.reset_session_state()
                print("[wakeword] Inactivity timeout reached. Returning to sleep mode.")
                self._run_callback(self.on_sleep, "sleep")
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

    def _run_callback(self, callback: Callable[[], None] | None, label: str) -> None:
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            print(f"[wakeword] {label} callback failed: {exc}")

    def _detect_model(self, chunk: np.ndarray, model_name: str | None, threshold: float) -> bool:
        if self._model is None:
            return False
        if not model_name:
            return False

        pcm16 = np.frombuffer(
            float32_to_pcm16_bytes(resample_audio(chunk, self.sample_rate, 16000)),
            dtype=np.int16,
        )
        if pcm16.size == 0:
            return False

        scores = self._model.predict(
            pcm16,
            threshold={model_name: threshold},
            debounce_time=self.debounce_time,
        )
        score = float(scores.get(model_name, 0.0))
        return score >= threshold
