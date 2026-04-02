import json

from reachy_assistant.realtime.rag_tool import OpenAIRagRealtimeTool


class FakeSearcher:
    def search(self, query: str, category: str | None = None, limit: int = 3):
        return {
            "query": query,
            "category": category,
            "limit": limit,
        }


def test_rag_tool_execute_returns_serialized_search_result() -> None:
    tool = OpenAIRagRealtimeTool(searcher=FakeSearcher())

    result = tool.execute('{"query":"ipad","category":"equipment","limit":2}')

    assert json.loads(result) == {
        "query": "ipad",
        "category": "equipment",
        "limit": 2,
    }


def test_rag_tool_execute_handles_invalid_json() -> None:
    tool = OpenAIRagRealtimeTool(searcher=FakeSearcher())

    result = tool.execute("{bad json")

    assert json.loads(result)["error"] == "Invalid JSON arguments."
