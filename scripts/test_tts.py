import asyncio
import edge_tts
import subprocess
from pathlib import Path

TEXT = "Hei! Dette er en test av text to speech uten Reachy."
VOICE = "nb-NO-PernilleNeural"
OUT_MP3 = Path("test.mp3")

async def main():
    communicate = edge_tts.Communicate(TEXT, VOICE)
    await communicate.save(str(OUT_MP3))

asyncio.run(main())

print("MP3 laget. Spiller av...")

# Spill av på Windows
subprocess.run(["start", str(OUT_MP3)], shell=True)