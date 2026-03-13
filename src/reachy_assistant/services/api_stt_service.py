import os
from pathlib import Path
import httpx
"""
Service-lag for STT til å kalle STT-modell på backend-API (VM).
Logikken ligger i en STT-engine.

"""

class ApiSTTService:

    def __init__(self):
        self.base_url = os.getenv("BACKEND_URL", "http://localhost:8001")

    def transcribe_wav(self, wav_path: Path) -> str:
        with wav_path.open("rb") as f:
            files = {"audio": (wav_path.name, f, "audio/wav")}

            resp = httpx.post(
                f"{self.base_url}/stt/transcribe",
                files=files,
                timeout=120
            )

        resp.raise_for_status()
        return resp.json()["text"]
