from io import StringIO

import pytest
from rich.console import Console

from vaultrag import cli
from vaultrag.config import StartupConfigError, get_settings


def test_invalid_embedder_raises_structured_startup_error(monkeypatch):
    monkeypatch.setenv("EMBEDDER", "typo")

    with pytest.raises(StartupConfigError) as excinfo:
        get_settings()

    error = excinfo.value
    assert error.variable == "EMBEDDER"
    assert error.value == "typo"
    assert error.valid_values == ("local", "fake")
    assert "Invalid EMBEDDER='typo'" in str(error)
    assert "valid values: local, fake" in str(error)


def test_invalid_llm_raises_structured_startup_error(monkeypatch):
    monkeypatch.setenv("LLM", "typo")

    with pytest.raises(StartupConfigError) as excinfo:
        get_settings()

    error = excinfo.value
    assert error.variable == "LLM"
    assert error.value == "typo"
    assert error.valid_values == ("groq", "fake")
    assert "Invalid LLM='typo'" in str(error)
    assert "valid values: groq, fake" in str(error)


def test_cli_reports_invalid_embedder_without_traceback(monkeypatch):
    output = StringIO()
    monkeypatch.setenv("EMBEDDER", "typo")
    monkeypatch.setattr(cli, "console", Console(file=output, width=120, color_system=None))

    assert cli.main(["health"]) == 2

    rendered = output.getvalue()
    assert "configuration error" in rendered
    assert "Invalid EMBEDDER='typo'" in rendered
    assert "valid values: local, fake" in rendered
    assert "Traceback" not in rendered
