from pathlib import Path
import tempfile

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from reachy_assistant.services.stt_service import STTService
from backend.app.api.deps import get_stt_service

router = APIRouter(prefix="/stt", tags=["stt"])

@router.post("/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    svc: STTService = Depends(get_stt_service),
):
    if not audio.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Please upload a .wav file")

    suffix = Path(audio.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(await audio.read())

    try:
        text = svc.transcribe_wav(tmp_path)
        return {"text": text}
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass