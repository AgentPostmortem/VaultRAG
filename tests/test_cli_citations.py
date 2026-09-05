"""CLI rendering regression tests, isolated from database retrieval."""

from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from rich.console import Console

from vaultrag import cli
from vaultrag.generate import Answer


@pytest.mark.parametrize("entry", ["[/]", "[red]bad[/red]"])
async def test_ask_prints_dropped_citations_literally(monkeypatch, entry):
    output = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=output, width=120, color_system=None))
    monkeypatch.setattr(cli, "get_embedder", Mock(return_value=Mock(embed=lambda _: [[0.0]])))
    monkeypatch.setattr(cli, "_llm", Mock(return_value=None))
    monkeypatch.setattr(cli.psycopg.AsyncConnection, "connect", AsyncMock(return_value=AsyncMock()))
    principal = SimpleNamespace(user_id="alice", principals=["alice"])
    monkeypatch.setattr(cli, "resolve_principal", AsyncMock(return_value=principal))
    hit = SimpleNamespace(doc_id="doc-1", title="Handbook", score=0.5)
    monkeypatch.setattr(cli, "search", AsyncMock(return_value=[hit]))
    monkeypatch.setattr(cli, "detect_conflicts", Mock(return_value=[]))
    monkeypatch.setattr(cli, "detect_stale", Mock(return_value=[]))
    monkeypatch.setattr(
        cli,
        "generate",
        Mock(return_value=Answer(text="No source.", answered=False, dropped_citations=[entry])),
    )

    assert await cli._ask(SimpleNamespace(user="alice", question="bonus?", limit=5)) == 0
    assert repr([entry]) in output.getvalue()
