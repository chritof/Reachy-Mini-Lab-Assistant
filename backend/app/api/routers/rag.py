from fastapi import APIRouter, Depends
from pydantic import BaseModel

from reachy_assistant.services.rag_service import RagService
from backend.app.api.deps import get_rag_service

router = APIRouter(prefix="/rag", tags=["rag"])


class AskRequest(BaseModel):
    question: str


@router.post("/ask")
def ask(req: AskRequest, svc: RagService = Depends(get_rag_service)):
    res = svc.ask(req.question)
    return {
        "answer": res.answer,
        "hits": [
            {"file": h.file, "score": h.score}
            for h in res.rag.hits
        ],
    }