import sys
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.io.wavfile import read as wav_read
from scipy.signal import resample_poly

from reachy_mini import ReachyMini

# -----------------------------
# KONFIG
# -----------------------------
# Legg voice-filer her:
#   data/tts/no_NO-talesyntese-medium.onnx
#   data/tts/no_NO-talesyntese-medium.onnx.json
PIPER_VOICE = Path("data/tts/no_NO-talesyntese-medium.onnx")  # <-- tilpass ved behov

# Streaming
CHUNK = 1024  # samples per push


def make_tts_wav(text: str, out_wav: Path) -> Path:
    """
    Lager wav med piper lokalt.
    Kjører piper via: python -m piper (robust i conda).
    """
    out_wav.parent.mkdir(parents=True, exist_ok=True)

    # Sjekk voice og json
    if not PIPER_VOICE.exists():
        raise FileNotFoundError(
            f"Fant ikke piper-voice: {PIPER_VOICE}\n"
            f"Legg en norsk piper .onnx (og .json) i data/tts/ og oppdater PIPER_VOICE."
        )

    voice_json = Path(str(PIPER_VOICE) + ".json")
    if not voice_json.exists():
        raise FileNotFoundError(
            f"Fant ikke voice-config: {voice_json}\n"
            f"Piper trenger vanligvis både .onnx og .onnx.json i samme mappe."
        )

    # Bruk python-modulen (slipper 'piper' i PATH)
    cmd = [
        sys.executable, "-m", "piper",
        "--model", str(PIPER_VOICE),
        "--output_file", str(out_wav),
    ]

    p = subprocess.run(
        cmd,
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if p.returncode != 0:
        raise RuntimeError(
            "Piper feilet.\n"
            f"STDERR:\n{p.stderr.decode('utf-8', errors='replace')}"
        )

    if not out_wav.exists() or out_wav.stat().st_size == 0:
        raise RuntimeError("TTS laget ingen wav (tom fil).")

    return out_wav


def wav_to_float32_mono(wav_path: Path) -> tuple[int, np.ndarray]:
    """
    Leser wav og returnerer (sr, mono_float32[-1,1]).
    """
    sr, audio = wav_read(str(wav_path))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.int32:
        audio = audio.astype(np.float32) / 2147483648.0
    else:
        audio = audio.astype(np.float32)

    audio = audio - float(np.mean(audio))
    audio = np.clip(audio, -1.0, 1.0)

    return sr, audio


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio.astype(np.float32)

    g = np.gcd(sr, target_sr)
    up = target_sr // g
    down = sr // g
    return resample_poly(audio, up, down).astype(np.float32)


def stream_audio_to_reachy(r: ReachyMini, audio: np.ndarray, sr: int) -> None:
    """
    Streamer float32 mono til Reachy.
    """
    if audio.size == 0:
        raise RuntimeError("Ingen audio-samples å spille.")

    # Normaliser moderat (for lav lyd = stille)
    peak = float(np.max(np.abs(audio)) + 1e-9)
    if peak < 0.02:
        # veldig lavt signal -> boost litt
        audio = audio / peak * 0.25
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

    # Start stream
    r.media.start_playing()

    # Pacing ~realtime
    for i in range(0, len(audio), CHUNK):
        chunk = audio[i:i + CHUNK]
        r.media.push_audio_sample(chunk)
        time.sleep(len(chunk) / sr)  # realtime pacing

    # La buffer spille ferdig
    time.sleep(0.4)
    r.media.stop_playing()


def say(text: str) -> None:
    wav_path = Path("data/audio/tts/reachy_tts.wav")

    # 1) lag wav
    make_tts_wav(text, wav_path)
    print("✅ Lagde TTS wav:", wav_path.resolve())

    # 2) koble til reachy og finn output SR
    with ReachyMini(media_backend="default") as r:
        out_sr = int(r.media.get_output_audio_samplerate())
        print("🔊 Reachy output samplerate:", out_sr)

        # 3) les wav -> float mono
        wav_sr, audio = wav_to_float32_mono(wav_path)
        print("📄 WAV sr:", wav_sr, "| samples:", len(audio), "| min/max:", float(audio.min()), float(audio.max()))

        # Hvis audio er nesten null => piper lagde “stille”
        if np.max(np.abs(audio)) < 0.001:
            print("⚠️ WAV ser nesten stille ut. Sjekk piper-voice / piper-output.")
            return

        # 4) resample til reachy SR
        audio = resample(audio, wav_sr, out_sr)

        # 5) stream
        stream_audio_to_reachy(r, audio, out_sr)

    print("✅ Spilte av TTS via streaming")


def main():
    print("Skriv/past inn teksten Reachy skal si. Enter:")
    text = input("> ").strip()
    if not text:
        print("Ingen tekst.")
        return
    say(text)


if __name__ == "__main__":
    main()