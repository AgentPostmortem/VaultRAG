"""Response serialization must preserve malformed citation evidence without coercion."""

from vaultrag.main import AskResponse


def test_dropped_citation_entries_survive_response_serialization():
    entries = ["1", "two", 2.5, -1, True, False, None, [], {"source": 1}]
    response = AskResponse(
        answer="No verifiable citation.",
        answered=False,
        query_id=1,
        dropped_citations=entries,
    )

    restored = AskResponse.model_validate_json(response.model_dump_json())
    assert restored.dropped_citations == entries
    assert [type(entry) for entry in restored.dropped_citations] == [
        type(entry) for entry in entries
    ]
