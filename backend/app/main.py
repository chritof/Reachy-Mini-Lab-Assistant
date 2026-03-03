from fastapi import FastAPI
from backend.app.api.routers.health import router as health_router
from backend.app.api.routers.stt import router as stt_router
from backend.app.api.routers.rag import router as rag_router
from backend.app.api.routers.tts import router as tts_router

app = FastAPI(title="Reachy Backend API")

app.include_router(health_router)
app.include_router(stt_router)
app.include_router(rag_router)
app.include_router(tts_router)