from pathlib import Path
from reachy_assistant.rag.rag_engine import RagEngine

ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "data" / "rag_index"

rag_engine = RagEngine(index_dir=INDEX_DIR)

result = rag_engine.retrieve("ipad")

for h in result.hits:
    print(h.file, h.score)