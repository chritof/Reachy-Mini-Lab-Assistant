import os
import httpx
"""
Service-lag for TTS som kjører på backend-API (VM).
"""

class ApiTTSService:

    def __init__(self):
        self.base_url = os.getenv("BACKEND_URL", "http://localhost:8001")

    def synthesize(self, text: str) -> bytes:
        resp = httpx.post(
            f"{self.base_url}/tts/synthesize",
            json={"text": text},
            timeout=120,
        )

        resp.raise_for_status()
        return resp.content
