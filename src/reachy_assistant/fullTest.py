import time
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write, read as wav_read
from scipy.signal import resample_poly

import whisper
from reachy_mini import ReachyMini


# -----------------------------
# KONFIG
# -----------------------------
MODEL_SIZE = "small"   # "small" -> "medium" -> "large" (best)
TARGET_SR = 16000       # Whisper liker 16kHz


# -----------------------------
# 1) LAST MODELL ÉN GANG
# -----------------------------
_model = whisper.load_model(MODEL_SIZE)


# -----------------------------
# 2) RECORD AUDIO FRA REACHY MINI
# -----------------------------
def record_audio(recorded_audio_path: Path, duration_sec: float = 5.0) -> Path:
    recorded_audio_path.parent.mkdir(parents=True, exist_ok=True)

    with ReachyMini(media_backend="default") as reachy:
        reachy.media.start_recording()
        time.sleep(0.2)  # liten warmup

        t0 = time.time()
        chunks = []

        while time.time() - t0 < duration_sec:
            sample = reachy.media.get_audio_sample()
            if sample is not None:
                chunks.append(sample)
            else:
                time.sleep(0.01)

        sample_rate = reachy.media.get_input_audio_samplerate()
        reachy.media.stop_recording()

    if not chunks:
        raise RuntimeError("No audio frames received from Reachy Mini mic")

    samples = np.concatenate(chunks, axis=0)

    # Debug
    print("sample_rate:", sample_rate)
    print("raw dtype:", samples.dtype, "shape:", samples.shape)
    print("raw min/max:", float(np.min(samples)), float(np.max(samples)))

    # float32 [-1,1] -> int16
    if np.issubdtype(samples.dtype, np.floating):
        samples = np.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).astype(np.int16)
    else:
        samples = samples.astype(np.int16)

    # Lagre WAV
    write(recorded_audio_path, sample_rate, samples)
    return recorded_audio_path


# -----------------------------
# 3) PREPROSESSERING: mono + 16kHz + float32 [-1,1]
# -----------------------------
def load_wav_as_whisper_audio(path: Path, target_sr: int = TARGET_SR) -> np.ndarray:
    sr, audio = wav_read(str(path))

    # audio kan være (N,) eller (N,2)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # stereo -> mono

    # int16 -> float32 [-1,1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32)

    # fjern DC offset (kan hjelpe litt)
    audio = audio - np.mean(audio)

    # resample til 16kHz hvis nødvendig
    if sr != target_sr:
        # resample_poly er bra og stabil
        g = np.gcd(sr, target_sr)
        up = target_sr // g
        down = sr // g
        audio = resample_poly(audio, up, down).astype(np.float32)

    # lett normalisering (hindrer for lavt nivå)
    peak = np.max(np.abs(audio)) + 1e-9
    if peak < 0.05:  # veldig lavt signal -> boost litt
        audio = audio / peak * 0.2

    audio = np.clip(audio, -1.0, 1.0)
    return audio


# -----------------------------
# 4) TRANSCRIBE (BEDRE INNSTILLINGER)
# -----------------------------
def transcribe_file(audio_path: Path, language: str = "no") -> dict:
    audio = load_wav_as_whisper_audio(audio_path)

    # Viktige parametre:
    # - temperature=0: mer stabil/“seriøs”
    # - beam_size/best_of: bedre nøyaktighet (litt tregere)
    # - condition_on_previous_text=False: kan redusere “hallusinasjon” på korte klipp
    result = _model.transcribe(
        audio,
        language=language,
        task="transcribe",
        fp16=False,
        temperature=0.0,
        beam_size=5,
        best_of=5,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )
    return result


# -----------------------------
# MAIN
# -----------------------------
def main():
    root = Path(__file__).resolve().parent
    out_path = root / "data" / "audio" / "recordings" / "reachy-test.wav"

    print("🎙️  Tar opp 5 sekunder...")
    recorded = record_audio(out_path, duration_sec=5.0)
    print("✅ Lagret:", recorded.resolve())

    print(f"🧠 Transkriberer med Whisper ({MODEL_SIZE})...")
    res = transcribe_file(recorded, language="no")
    print("\n--- TRANSKRIPT ---")
    print(res["text"].strip())
    print("------------------")


if __name__ == "__main__":
    main()