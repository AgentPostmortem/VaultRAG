"""Embedding providers.

Two implementations, one interface:

- LocalEmbedder: sentence-transformers, runs on CPU, no API key, free forever.
- FakeEmbedder: deterministic hash-based vectors. Used for tests.

The local embedder detects the actual model dimension and verifies that it
matches the database schema dimension.
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


DIMS = 384


class Embedder(Protocol):
    @property
    def dims(self) -> int: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class FakeEmbedder:
    """Deterministic, offline, instant embedder for tests."""

    dims = DIMS

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._one(t) for t in texts]

    def _one(self, text: str) -> list[float]:
        vec: list[float] = []
        counter = 0

        while len(vec) < DIMS:
            h = hashlib.sha256(f"{text}:{counter}".encode()).digest()
            vec.extend(b / 255.0 - 0.5 for b in h)
            counter += 1

        vec = vec[:DIMS]
        return _normalize(vec)


class LocalEmbedder:
    """sentence-transformers on CPU.

    The model is loaded lazily and its actual embedding dimension is checked
    against the dimension expected by the database schema.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self._model_name = model_name
        self._model = None
        self._dims: int | None = None

    @property
    def dims(self) -> int:
        if self._dims is None:
            self._load()

        assert self._dims is not None
        return self._dims

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
            self._dims = self._model.get_sentence_embedding_dimension()

            if self._dims != DIMS:
                raise ValueError(
                    f"Embedding model {self._model_name!r} produces "
                    f"{self._dims}-dimensional vectors, but VaultRAG expects "
                    f"{DIMS} dimensions. The embedding column in "
                    f"vaultrag/schema.sql is vector({DIMS}). Changing "
                    f"EMBED_MODEL requires updating the schema and "
                    f"re-ingesting the corpus."
                )

        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._load()

        arr = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        return [list(map(float, row)) for row in arr]


def _normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def get_embedder(
    kind: str = "local",
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Embedder:
    if kind == "fake":
        return FakeEmbedder()

    if kind == "local":
        return LocalEmbedder(model)

    raise ValueError(
        f"unknown embedder: {kind!r} (expected 'local' or 'fake')"
    )