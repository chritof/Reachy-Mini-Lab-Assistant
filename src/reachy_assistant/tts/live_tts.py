import asyncio
import subprocess
from pathlib import Path
import tempfile
import edge_tts

def speak_live(text: str, voice: str = "nb-NO-PernilleNeural", mini=None):
    """
    Hvis mini er gitt: (valgfritt) kan du senere implementere streaming til Reachy.
    Hvis mini er None: spiller av på PC ved å lage mp3 og åpne den.
    """
    async def _make_mp3(out_path: Path):
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(str(out_path))

    with tempfile.TemporaryDirectory() as td:
        mp3_path = Path(td) / "tts.mp3"
        asyncio.run(_make_mp3(mp3_path))

        if mini is None:
            # Spill av lokalt på Windows
            subprocess.run(["start", str(mp3_path)], shell=True)
        else:
            # TODO: her kan du bruke mini.media.start_playing/push_audio_sample
            # Men: dette krever at Reachy Mini faktisk er tilgjengelig.
            subprocess.run(["start", str(mp3_path)], shell=True)