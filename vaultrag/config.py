"""Settings, read from the environment. No secrets in code, ever."""

from __future__ import annotations

import os
from dataclasses import dataclass

VALID_EMBEDDERS = ("local", "fake")
VALID_LLMS = ("groq", "fake")


class StartupConfigError(ValueError):
    """Raised when startup configuration is invalid."""

    def __init__(self, variable: str, value: str, valid_values: tuple[str, ...]) -> None:
        self.variable = variable
        self.value = value
        self.valid_values = valid_values
        super().__init__(
            f"Invalid {variable}={value!r}; valid values: {', '.join(valid_values)}"
        )


@dataclass(frozen=True)
class Settings:
    database_url: str
    embedder: str
    embed_model: str
    llm: str
    groq_api_key: str | None
    groq_model: str


def _validated_choice(variable: str, value: str, valid_values: tuple[str, ...]) -> str:
    if value in valid_values:
        return value
    raise StartupConfigError(variable, value, valid_values)


def get_settings() -> Settings:
    return Settings(
        database_url=os.getenv(
            "DATABASE_URL", "postgresql://vaultrag:***@localhost:5433/vaultrag"
        ),
        embedder=_validated_choice("EMBEDDER", os.getenv("EMBEDDER", "local"), VALID_EMBEDDERS),
        embed_model=os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        llm=_validated_choice("LLM", os.getenv("LLM", "groq"), VALID_LLMS),
        groq_api_key=os.getenv("GROQ_API_KEY"),
        groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )
