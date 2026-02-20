import time
import numpy as np
from reachy_mini import ReachyMini

def main():
    with ReachyMini(media_backend="default") as r:
        sr = int(r.media.get_output_audio_samplerate())
        print("Reachy output samplerate:", sr)

        # 0.8 sek beep
        dur = 2.8
        t = np.linspace(0, dur, int(sr * dur), endpoint=False)
        beep = 0.2 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Start output stream + push samples
        r.media.start_playing()
        chunk = 1024
        for i in range(0, len(beep), chunk):
            r.media.push_audio_sample(beep[i:i+chunk])
            time.sleep(chunk / sr * 0.5)
        time.sleep(0.2)
        r.media.stop_playing()

if __name__ == "__main__":
    main()