# Reachy Mini Lab Assistant

Dette repoet inneholder den endelige Python-løsningen for bachelorprosjektet
vårt: en talebasert assistent for Reachy Mini i Læringslab ved HVL.

Sluttløsningen er bygget rundt OpenAI Realtime API, lokal RAG for
dokumentoppslag, wakeword-aktivering og Reachy-spesifikk lyd- og
bevegelsesstyring.

## Struktur

Den leveringsrelevante koden ligger hovedsakelig i:

- `src/reachy_assistant/realtime/`: sanntidsbasert samtalemotor og lydflyt
- `src/reachy_assistant/rag_openai/`: lokal dokumentindeksering og søk i Qdrant
- `src/reachy_assistant/robot/motion/`: Reachy Mini-bevegelser og tilstandsstyring
- `src/reachy_assistant/robot/pipeline/realtime_pipeline.py`: sammensetting av sluttløsningen
- `tests/`: tester for den gjeldende realtime-arkitekturen

## Oppsett

Kopier `.env.example` til `.env` og sett minst:

```env
OPENAI_API_KEY=
```

Installer avhengigheter med:

```powershell
pip install -r requirements.txt
```

## Kjøring

PC-variant:

```powershell
python -m reachy_assistant.realtime.main
```

Reachy Mini-variant:

```powershell
python -m reachy_assistant.main
```

## Tester

Repoet inneholder hovedsakelig enhetstester for realtime-, wakeword- og
RAG-komponentene.
