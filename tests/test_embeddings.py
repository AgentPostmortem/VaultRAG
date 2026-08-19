from types import SimpleNamespace

import pytest

from vaultrag.embeddings import DIMS, LocalEmbedder


class FakeSentenceTransformer:
    def __init__(self, model_name: str, dims: int) -> None:
        self.model_name = model_name
        self._dims = dims

    def get_sentence_embedding_dimension(self) -> int:
        return self._dims


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


def test_local_embedder_rejects_wrong_dimensions(monkeypatch):
    install_fake_sentence_transformers(monkeypatch, 768)

    model_name = "sentence-transformers/all-mpnet-base-v2"
    embedder = LocalEmbedder(model_name)

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