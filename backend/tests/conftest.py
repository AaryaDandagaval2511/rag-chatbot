"""Shared pytest fixtures.

Every test runs OFFLINE: network connections are blocked for the whole test session, so
an accidental model download, telemetry call or API request fails loudly instead of
silently depending on the internet.
"""

import socket

import pytest

from app.config import Settings
from app.services.embeddings import HashEmbeddings
from app.services.vectorstore import get_client, get_collection
from tests.support import TEST_DIMENSIONS, SpyEmbeddings


@pytest.fixture(autouse=True)
def _block_network(monkeypatch):
    def guard(self, address, *args, **kwargs):
        if self.family == socket.AF_UNIX:  # local inter-process sockets are fine
            return original(self, address, *args, **kwargs)
        raise RuntimeError(f"Network access attempted during tests: {address!r}")

    original = socket.socket.connect
    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket.socket, "connect_ex", guard)


@pytest.fixture
def test_settings(tmp_path):
    """Settings isolated from the developer's real .env and vector store."""
    return Settings(
        _env_file=None,
        chroma_persist_dir=tmp_path / "vectorstore",
        chroma_collection="test_documents",
        embedding_provider="hash",
        embedding_dimensions=TEST_DIMENSIONS,
    )


@pytest.fixture
def embeddings():
    return HashEmbeddings(TEST_DIMENSIONS)


@pytest.fixture
def collection(tmp_path):
    """A fresh ChromaDB collection in a temporary folder, deleted after the test."""
    client = get_client(tmp_path / "vectorstore")
    return get_collection(
        client, "test_documents", model=HashEmbeddings.MODEL_NAME, dimensions=TEST_DIMENSIONS
    )


@pytest.fixture
def spy_embeddings():
    return SpyEmbeddings()
