from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    source_file: str
    chunk_index: int
    text: str
    category: str


def infer_category(path: Path) -> str:
    return path.parent.name if path.parent.name else "general"


def chunk_text(text: str, size: int = 500, overlap: int = 100) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = start + size
        chunks.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def iter_content_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}:
            yield path


def build_chunks(root: Path, size: int = 500, overlap: int = 100) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for path in iter_content_files(root):
        content = path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_text(content, size=size, overlap=overlap)):
            records.append(
                ChunkRecord(
                    source_file=str(path.relative_to(root)),
                    chunk_index=index,
                    text=chunk,
                    category=infer_category(path),
                )
            )
    return records
