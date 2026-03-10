# audio/recorder.py
# henter rå lyd fra Reachy og lager en WAV-fil
# denne filen har én jobb!:
# - snakke med reachy
# - hente rå lyd
# - lagre som en .wav fil

# brukes til å vite hvor lenge vi vil ta opp lyd.
# + små pauser
import time
# for å håndtere filer
from pathlib import Path
# vi bruker numpy for å slå sammen lyd-chunks, sjekke datatyper
# og konvertere lydformat
import numpy as np
# brukes for å lagre wav fil
from scipy.io.wavfile import write as wav_write
# dette er API-et som snakker dirkete til reachy
from reachy_mini import ReachyMini

# path = hvor vi skal lagre lydfilen
# duration_sec = hvor lenge vi skal ta opp lyd
# 
def record_audio_wav(path: Path, duration_sec: float) -> tuple[Path, int]:
    # finn mappen til filen.
    # lag den hvis den ikke finnes
    # ikke krasj hvis den allerede finnes
    path.parent.mkdir(parents=True, exist_ok=True)

    # kobler til reachy
    with ReachyMini(media_backend="default") as reachy:
        # ta opp lyd fra mikrofonen
        reachy.media.start_recording()
        time.sleep(0.2)

        # starttid
        t0 = time.time()
        # liste som skal lagre små lydbiter
        chunks = []


        while time.time() - t0 < duration_sec:
            # retunerer en liten Numpy med lyddata
            sample = reachy.media.get_audio_sample()
            if sample is not None:
                chunks.append(sample)
            else:
                time.sleep(0.01)
        # henter vi sample rate (f.eks 48000 Hz)
        sr = int(reachy.media.get_input_audio_samplerate())
        # stopper opptaket
        reachy.media.stop_recording()

    if not chunks:
        raise RuntimeError("No audio frames received")
    # slår sammen alle chunksene
    audio = np.concatenate(chunks, axis=0)
    
    if np.issubdtype(audio.dtype, np.floating):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767.0).astype(np.int16)
    else:
        audio = audio.astype(np.int16)

    wav_write(str(path), sr, audio)
    return path, sr