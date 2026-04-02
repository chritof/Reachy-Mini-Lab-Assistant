from pathlib import Path

from reachy_assistant.rag_openai.build import point_id
from reachy_assistant.rag_openai.chunking import build_chunks, chunk_text
from reachy_assistant.rag_openai.search import OpenAIRagSearch


def test_chunk_text_splits_with_overlap() -> None:
    text = "a" * 900
    chunks = chunk_text(text, size=500, overlap=100)

    assert len(chunks) == 2
    assert len(chunks[0]) == 500
    assert len(chunks[1]) == 500
    assert chunks[0][-100:] == chunks[1][:100]


def test_build_chunks_reads_text_files(tmp_path: Path) -> None:
    docs = tmp_path / "equipment"
    docs.mkdir()
    (docs / "ipad.txt").write_text("iPad information " * 50, encoding="utf-8")

    records = build_chunks(tmp_path, size=120, overlap=20)

    assert records
    assert records[0].source_file == "equipment\\ipad.txt"
    assert records[0].category == "equipment"


def test_point_id_is_stable() -> None:
    a = point_id("file.txt", 0, "hello")
    b = point_id("file.txt", 0, "hello")
    c = point_id("file.txt", 1, "hello")

    assert a == b
    assert a != c


class FakeEmbeddings:
    def embed_one(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def search(self, query_vector, category=None, limit=3):
        assert query_vector == [0.1, 0.2, 0.3]
        assert category == "equipment"
        assert limit >= 2
        return [
            {
                "source": "equipment/ipad.txt",
                "text": "This is the iPad entry.",
                "category": "equipment",
                "score": 0.9,
            }
        ]


def test_openai_rag_search_returns_context() -> None:
    search = OpenAIRagSearch(vector_store=FakeVectorStore(), embeddings=FakeEmbeddings())

    result = search.search("Where is the iPad?", category="equipment", limit=2)

    assert "results" in result
    assert "[equipment/ipad.txt]" in result["context"]
