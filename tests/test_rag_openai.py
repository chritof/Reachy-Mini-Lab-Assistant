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


def test_opening_hours_query_boosts_opening_hours_source() -> None:
    search = OpenAIRagSearch(vector_store=FakeVectorStore(), embeddings=FakeEmbeddings())

    results = search._rerank_results(
        "Hva er åpningstidene i Bergen?",
        [
            {
                "source": "general/lringslab.txt",
                "text": "Bergen campus overview without structured hours first.",
                "score": 0.95,
            },
            {
                "source": "general/aapningstider_laeringslab.txt",
                "text": "Bergen: mandag-torsdag 09:00-17:00, fredag 09:00-15:00",
                "score": 0.7,
            },
        ],
        limit=1,
        wants_opening_hours=True,
    )

    assert results[0]["source"] == "general/aapningstider_laeringslab.txt"


def test_staff_query_boosts_staff_source() -> None:
    search = OpenAIRagSearch(vector_store=FakeVectorStore(), embeddings=FakeEmbeddings())

    results = search._rerank_results(
        "Hvem jobber i Laeringslab i Bergen?",
        [
            {
                "source": "general/lringslab.txt",
                "text": "Generell informasjon om Laeringslab og campus.",
                "score": 0.95,
            },
            {
                "source": "general/ansatte_i_laeringslab.txt",
                "text": "Faste verter per campus: Bergen: Xavier, Robin og Kaspar.",
                "score": 0.7,
            },
        ],
        limit=1,
        wants_staff=True,
    )

    assert results[0]["source"] == "general/ansatte_i_laeringslab.txt"


def test_staff_answer_lists_known_staff() -> None:
    search = OpenAIRagSearch(vector_store=FakeVectorStore(), embeddings=FakeEmbeddings())

    answer = search._build_answer(
        query="Hvem jobber i Læringslab?",
        results=[
            {
                "source": "general/ansatte_i_laeringslab.txt",
                "text": (
                    "Tittel: Ansatte og verter i Læringslab\n"
                    "Faste verter per campus:\n"
                    "- Bergen: Xavier, Robin og Kaspar\n"
                    "- Stord: Robin\n"
                    "- Haugesund: Thomas\n"
                ),
                "score": 0.9,
            }
        ],
        wants_loan=False,
        wants_staff=True,
        fallback="fallback",
    )

    assert "Xavier, Robin og Kaspar" in answer
    assert "Stord: Robin" in answer


def test_loan_answer_leads_with_cheqroom() -> None:
    search = OpenAIRagSearch(vector_store=FakeVectorStore(), embeddings=FakeEmbeddings())

    answer = search._build_answer(
        query="Hva kan jeg låne?",
        results=[
            {
                "source": "equipment/ipad.txt",
                "text": "Tittel: iPad\nUtlånsstatus: Kan normalt lånes.",
                "score": 0.9,
            },
            {
                "source": "general/utlaan.txt",
                "text": "Cheqroom brukes for utlån og reservasjon.",
                "score": 0.8,
            },
        ],
        wants_loan=True,
        wants_staff=False,
        fallback="fallback",
    )

    assert answer.startswith("For utlån og reservasjon bruker Læringslab normalt Cheqroom.")
