from __future__ import annotations

import subprocess
from pathlib import Path


class PiperTTSEngine:
    def __init__(self, model_path: str, config_path: str | None = None):
        self.model_path = model_path
        self.config_path = config_path

    def synthesize_wav_bytes(self, text: str) -> bytes:
        if not text.strip():
            return b""

        output_path = Path("data/audio/tts/reachy_response.wav")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "piper",
            "--model", self.model_path,
            "--output_file", str(output_path),
        ]
        if self.config_path:
            cmd += ["--config", self.config_path]

        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        _ = proc
        return output_path.read_bytes()