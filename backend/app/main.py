from fastapi import FastAPI, Header, HTTPException, UploadFile, File
import httpx

from .settings import settings

app = FastAPI(title="Reachy VM Backend", version="0.1.0")

def require_key(x_api_key: str | None) -> None:
    if settings.API_KEY and x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


@app.get("/health")
def health():
    return {"ok": True}

@app.post("/llm/chat")
async def llm_chat(payload: dict, x_api_key: str | None = Header(default=None)):
    """
    payload:
    {
      "messages": [{"role":"user","content":"..."}],
      "temperature": 0.2,          # optional
      "num_predict": 200           # optional (token-ish limit)
    }
    """

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

