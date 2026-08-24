from types import SimpleNamespace

import pytest

from vaultrag.embeddings import DIMS, LocalEmbedder


class FakeSentenceTransformer:
    def __init__(self, model_name: str, dims: int) -> None:
        self.model_name = model_name
        self._dims = dims

    def get_sentence_embedding_dimension(self) -> int:
        return self._dims

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        return [[0.0] * self._dims for _ in texts]


def install_fake_sentence_transformers(monkeypatch, dims: int):
    def constructor(model_name: str):
        return FakeSentenceTransformer(model_name, dims)

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=constructor),
    )


def test_local_embedder_detects_model_dimensions(monkeypatch):
    install_fake_sentence_transformers(monkeypatch, DIMS)

    embedder = LocalEmbedder("fake-model")

    assert embedder.dims == DIMS


def test_local_embedder_embed_works_with_valid_dimensions(monkeypatch):
    install_fake_sentence_transformers(monkeypatch, DIMS)

    embedder = LocalEmbedder("fake-model")

    result = embedder.embed(["hello", "world"])

    assert len(result) == 2
    assert all(len(vector) == DIMS for vector in result)


def test_local_embedder_rejects_wrong_dimensions_on_repeated_access(monkeypatch):
    install_fake_sentence_transformers(monkeypatch, 768)

    embedder = LocalEmbedder("sentence-transformers/all-mpnet-base-v2")

    for _ in range(2):
        with pytest.raises(
            ValueError,
            match=r"all-mpnet-base-v2.*768.*384.*vector\(384\)",
        ):
            embedder.dims


def test_local_embedder_error_mentions_schema(monkeypatch):
    install_fake_sentence_transformers(monkeypatch, 1024)

    embedder = LocalEmbedder("another-model")

    with pytest.raises(ValueError, match=r"vector\(384\)"):
        embedder.dims