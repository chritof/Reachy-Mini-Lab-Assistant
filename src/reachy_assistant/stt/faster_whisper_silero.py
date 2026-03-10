"""
STT-implementasjon.

Bruker Faster-Whisper for transkribering
og Silero VAD for å finne tale i lyd.

Inneholder lavnivå talegjenkjenning.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from faster_whisper import WhisperModel
from scipy.io.wavfile import read as wav_read
from scipy.signal import resample_poly


@dataclass
class Segment:
    start: int
    end: int


class FasterWhisperSileroSTT:
    def __init__(
        self,
        *,
        whisper_model: str = "medium",
        device: str = "cpu",
        compute_type: str = "int8",
        target_sr: int = 16000,
        language: str = "no",
        vad_threshold: float = 0.5,
        min_speech_ms: int = 250,
        padding_ms: int = 200,
        beam_size: int = 5,
        temperature: float = 0.0,
    ) -> None:
        self.target_sr = target_sr
        self.language = language

        self.vad_threshold = vad_threshold
        self.min_speech_ms = min_speech_ms
        self.padding_ms = padding_ms

        self.beam_size = beam_size
        self.temperature = temperature

        # cache models (don’t reload per call)
        self._whisper = WhisperModel(whisper_model, device=device, compute_type=compute_type)
        self._vad_model, self._vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        (self._get_speech_timestamps, _, _, _, _) = self._vad_utils

    def load_wav_float32_mono(self, wav_path: Path) -> tuple[int, np.ndarray]:
        sr, audio = wav_read(str(wav_path))
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        else:
            audio = audio.astype(np.float32)

        audio = audio - float(np.mean(audio))
        return int(sr), audio

    def resample_to_16k(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        if orig_sr == self.target_sr:
            return audio.astype(np.float32)

        g = np.gcd(orig_sr, self.target_sr)
        up = self.target_sr // g
        down = orig_sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)

        peak = float(np.max(np.abs(audio)) + 1e-9)
        if peak > 0:
            audio = np.clip(audio / peak * 0.9, -1.0, 1.0)

        return audio

    def vad_segments(self, audio_16k: np.ndarray) -> list[Segment]:
        wav = torch.from_numpy(audio_16k)
        ts = self._get_speech_timestamps(
            wav,
            self._vad_model,
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.min_speech_ms,
            sampling_rate=self.target_sr,
        )
        if not ts:
            return []

        pad = int(self.padding_ms * self.target_sr / 1000)
        segs = []
        for t in ts:
            s = max(0, int(t["start"]) - pad)
            e = min(len(audio_16k), int(t["end"]) + pad)
            if e > s:
                segs.append(Segment(s, e))

        segs.sort(key=lambda x: x.start)
        merged = []
        cur = segs[0]
        for seg in segs[1:]:
            if seg.start <= cur.end:
                cur.end = max(cur.end, seg.end)
            else:
                merged.append(cur)
                cur = seg
        merged.append(cur)
        return merged

    def extract_voiced(self, audio_16k: np.ndarray, segs: list[Segment]) -> np.ndarray:
        if not segs:
            return np.array([], dtype=np.float32)
        return np.concatenate([audio_16k[s.start:s.end] for s in segs]).astype(np.float32)

    def transcribe(self, audio_16k_voiced: np.ndarray, language: Optional[str] = None) -> str:
        if audio_16k_voiced.size == 0:
            return ""

        segments, _info = self._whisper.transcribe(
            audio_16k_voiced,
            language=language or self.language,
            vad_filter=False,
            temperature=self.temperature,
            beam_size=self.beam_size,
        )
        return " ".join(seg.text.strip() for seg in segments if seg.text).strip()

    def transcribe_wav(self, wav_path: Path, language: Optional[str] = None) -> str:
        sr, audio = self.load_wav_float32_mono(wav_path)
        audio_16k = self.resample_to_16k(audio, sr)
        segs = self.vad_segments(audio_16k)
        voiced = self.extract_voiced(audio_16k, segs)
        if voiced.size < int(self.target_sr * 0.2):
            return ""
        return self.transcribe(voiced, language=language)