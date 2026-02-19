from fastapi import FastAPI, HTTPException
import httpx
from .settings import settings

app = FastAPI()

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/llm/chat")
async def llm_chat(payload: dict):
    # Requires JSON body with "messages"
    if "messages" not in payload:
        raise HTTPException(status_code=422, detail="Missing required field: messages")

    req = {
        "model": payload.get("model") or settings.OLLAMA_MODEL,
        "messages": payload["messages"],
        "stream": False,
    }

    options = {}
    if "temperature" in payload:
        options["temperature"] = payload["temperature"]
    if "num_predict" in payload:
        options["num_predict"] = payload["num_predict"]
    if options:
        req["options"] = options

    async with httpx.AsyncClient(timeout=300) as client:
        r = await client.post(f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat", json=req)
        r.raise_for_status()
        data = r.json()

    return {"text": data["message"]["content"]}
