from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

TARGET_SR = 16000
VAD_THRESHOLD = 0.5
MIN_SPEECH_MS = 250
PADDING_MS = 200


@dataclass
class Segment:
    start: int
    end: int


def get_vad_segments(audio_16k: np.ndarray) -> list[Segment]:
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )

    (get_speech_timestamps, _, _, _, _) = utils
    wav = torch.from_numpy(audio_16k)

    ts = get_speech_timestamps(
        wav,
        model,
        threshold=VAD_THRESHOLD,
        min_speech_duration_ms=MIN_SPEECH_MS,
        sampling_rate=TARGET_SR,
    )

    if not ts:
        return []

    pad = int(PADDING_MS * TARGET_SR / 1000)
    segments: list[Segment] = []

    for t in ts:
        s = max(0, int(t["start"]) - pad)
        e = min(len(audio_16k), int(t["end"]) + pad)
        if e > s:
            segments.append(Segment(start=s, end=e))

    return segments


def extract_voiced_audio(audio_16k: np.ndarray, segments: list[Segment]) -> np.ndarray:
    if not segments:
        return np.array([], dtype=np.float32)

    parts = [audio_16k[s.start:s.end] for s in segments]
    return np.concatenate(parts).astype(np.float32)