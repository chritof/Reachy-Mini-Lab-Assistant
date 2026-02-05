# Bachelorprosjekt - Reachy Mini AI-assistent

Repoet inneholder arbeidet med bachelorprosjektet der vi utvikler en AI-assistent for bruk i Læringslabben på HVL, basert på Reachy Mini-roboten.

---

## Mål med prosjektet

Målet er å utvikle en fungerende prototype som:
  - kan samhandle med brukere via relevant tale
  - kan utføre enkle bevegelser

---

## Prosjektstruktur (Foreløpig)
```
Reachy-Mini-Lab-Assistant
├───data
│   └───rag_sources
│   └───audio
├───docs
│   ├───decisions
│   ├───diagrams
│   └───prototypes
├───src
│   └───reachy_assistant
└───tests
```

---

## Teknologistack (Foreløpig)

Prosjektet utvikles som en Python-applikasjon med disse teknologiene:

- Python
- Reachy Mini SDK
- LLM via Ollama (mistral) -> cloud LLM (ChatGPT Nano-5)
- STT (openai-whisper)
- Retrieval-Augmented Generation (RAG) (LlamaIndex)

Kommer sannsynligvis mer/endringer