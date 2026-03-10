"""
HTTP-endepunkter for TTS.

Tar imot tekst,
kaller TTSService,
returnerer WAV-lyd.
"""
from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel

from backend.app.api.deps import get_tts_service
from reachy_assistant.services.tts_service import TTSService

router = APIRouter(prefix="/tts", tags=["tts"])

class TTSRequest(BaseModel):
    text: str

@router.post("/synthesize")
def synthesize(req: TTSRequest, svc: TTSService = Depends(get_tts_service)):
    wav = svc.synthesize(req.text)
    return Response(content=wav, media_type="audio/wav")