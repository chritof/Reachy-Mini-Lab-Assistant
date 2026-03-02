import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from scipy.io.wavfile import write as wav_write
from scipy.signal import resample_poly

import torch

from reachy_mini import ReachyMini
from faster_whisper import WhisperModel


# -----------------------------
# KONFIG
# -----------------------------
DURATION_SEC = 600.0          # ta opp litt mer enn du tror du trenger
TARGET_SR = 16000           # Whisper liker 16kHz
LANGUAGE = "no"

# Beste kvalitet lokalt:
# - "large-v3" best, men treg og tung (kan være for mye på CPU)
# - "medium" ofte "sweet spot" for norsk + støy
WHISPER_MODEL = "medium"     # prøv "large-v3" hvis maskinen tåler det

# compute_type:
# - CPU: "int8" er raskest, "int8_float16" litt bedre, "float32" mest nøyaktig (treg)
COMPUTE_TYPE = "int8"

# VAD
VAD_THRESHOLD = 0.5         # høyere = strengere, mindre falsk tale
MIN_SPEECH_MS = 250         # ignorer veldig korte "blips"
PADDING_MS = 200            # behold litt før/etter tale


# -----------------------------
# 1) RECORD AUDIO FRA REACHY MINI
# -----------------------------
def record_audio_wav(path: Path, duration_sec: float) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)

    with ReachyMini(media_backend="default") as reachy:
        reachy.media.start_recording()
        time.sleep(0.2)  # warmup

        t0 = time.time()
        chunks = []
        while time.time() - t0 < duration_sec:
            sample = reachy.media.get_audio_sample()
            if sample is not None:
                chunks.append(sample)
            else:
                time.sleep(0.01)

        sr = int(reachy.media.get_input_audio_samplerate())
        reachy.media.stop_recording()

    if not chunks:
        raise RuntimeError("No audio frames received from Reachy Mini mic")

    audio = np.concatenate(chunks, axis=0)

    # typisk float32 [-1,1] -> int16 WAV
    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
        audio_i16 = (audio * 32767.0).astype(np.int16)
    else:
        audio_i16 = audio.astype(np.int16)

    wav_write(str(path), sr, audio_i16)
    return path, sr


# -----------------------------
# 2) LAST WAV -> float32 mono @16k
# -----------------------------
def load_wav_as_float32_mono_16k(wav_path: Path, orig_sr: int) -> np.ndarray:
    # Vi vet allerede sample-rate fra Reachy, så vi leser bare bytes med numpy via scipy wav? (raskest)
    # Men siden vi skrev WAV med scipy, kan vi lese igjen:
    from scipy.io.wavfile import read as wav_read
    sr, audio = wav_read(str(wav_path))
    if sr != orig_sr:
        # ikke kritisk, men vi forventer sr==orig_sr
        orig_sr = sr

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    # int16 -> float32 [-1,1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    # Fjern DC offset
    audio = audio - float(np.mean(audio))

    # Resample til 16kHz
    if orig_sr != TARGET_SR:
        g = np.gcd(orig_sr, TARGET_SR)
        up = TARGET_SR // g
        down = orig_sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)

    # Mild normalisering (ikke aggressiv)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak > 0:
        audio = np.clip(audio / peak * 0.9, -1.0, 1.0)

    return audio


# -----------------------------
# 3) VAD (Silero) -> finn tale-segmenter
# -----------------------------
@dataclass
class Segment:
    start: int  # sample index @16k
    end: int    # sample index @16k


def get_vad_segments(audio_16k: np.ndarray) -> list[Segment]:
    # Silero VAD via torch.hub
    # Laster første gang (cache etterpå)
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    (get_speech_timestamps, _, _, _, _) = utils

    # torch tensor, mono
    wav = torch.from_numpy(audio_16k)

    # Silero returnerer timestamps i samples @16k
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
    segments = []
    for t in ts:
        s = max(0, int(t["start"]) - pad)
        e = min(len(audio_16k), int(t["end"]) + pad)
        if e > s:
            segments.append(Segment(s, e))

    # Slå sammen overlappende segmenter
    merged = []
    cur = segments[0]
    for seg in segments[1:]:
        if seg.start <= cur.end:
            cur.end = max(cur.end, seg.end)
        else:
            merged.append(cur)
            cur = seg
    merged.append(cur)

    return merged


def extract_voiced_audio(audio_16k: np.ndarray, segments: list[Segment]) -> np.ndarray:
    if not segments:
        return np.array([], dtype=np.float32)
    parts = [audio_16k[s.start:s.end] for s in segments]
    return np.concatenate(parts).astype(np.float32)


# -----------------------------
# 4) TRANSCRIBE med faster-whisper
# -----------------------------
def transcribe(audio_16k_voiced: np.ndarray) -> str:
    if audio_16k_voiced.size == 0:
        return ""

    # faster-whisper tar np.float32 @16k helt fint
    model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type=COMPUTE_TYPE)

    segments, info = model.transcribe(
        audio_16k_voiced,
        language=LANGUAGE,
        vad_filter=False,  # vi gjør VAD selv
        temperature=0.0,
        beam_size=5,
    )

    text_parts = []
    for seg in segments:
        if seg.text:
            text_parts.append(seg.text.strip())

    return " ".join(text_parts).strip()


# -----------------------------
# MAIN
# -----------------------------
def main():
    root = Path(__file__).resolve().parent
    wav_path = root / "data" / "audio" / "recordings" / "reachy_best.wav"

    print(f"🎙️  Tar opp {DURATION_SEC:.1f}s fra Reachy Mini...")
    wav_path, orig_sr = record_audio_wav(wav_path, DURATION_SEC)
    print("✅ Lagret:", wav_path)

    audio_16k = load_wav_as_float32_mono_16k(wav_path, orig_sr)

    # VAD
    segments = get_vad_segments(audio_16k)
    print(f"🧠 VAD fant {len(segments)} tale-segment(er).")

    voiced = extract_voiced_audio(audio_16k, segments)

    # Guard: hvis nesten ingen tale
    if voiced.size < TARGET_SR * 0.2:  # < 0.2 sek
        print("⚠️  Fant nesten ingen tale. (Sjekk mic-permissions / input level)")
        print("TRANSKRIPT: ")
        print("")
        return

    print(f"🧠 Transkriberer lokalt med faster-whisper ({WHISPER_MODEL}, {COMPUTE_TYPE})...")
    text = transcribe(voiced)

    print("\n--- TRANSKRIPT ---")
    print(text)
    print("------------------")


if __name__ == "__main__":
    main()