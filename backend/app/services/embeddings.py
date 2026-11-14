"""Embedding providers: turn text into vectors.

The rest of the pipeline depends only on LangChain's `Embeddings` interface
(`embed_documents` for stored chunks, `embed_query` for questions). Any provider that
implements it, such as langchain-ollama's `OllamaEmbeddings`, can be plugged in through
`get_embedding_provider()` without touching the ingestor, vector store or retriever.

Currently available:
    "hash": HashEmbeddings, an offline, deterministic MOCK for development and tests.

Adding a real provider later (e.g. Ollama):
    1. pip install langchain-ollama
    2. add "ollama" to Settings.embedding_provider in config.py
    3. add a branch below returning OllamaEmbeddings(...) with its model name and dimensions
    (nomic-embed-text also expects "search_document: " / "search_query: " prefixes, which
    belong in that provider's embed_documents / embed_query.)
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings

from app.config import Settings

_WORD = re.compile(r"\w+")


class HashEmbeddings(Embeddings):
    """A deterministic MOCK embedding based on word hashing ("feature hashing").

    How it works: lowercase the text, split it into words, map each word to one of
    `dimensions` slots with a stable hash, count the words per slot, then scale the
    vector to length 1 (so cosine similarity is well-defined).

    What it IS good for: testing pipeline mechanics (IDs, storage, metadata, ranking
    code paths) offline, with no models or network. Texts that share words get similar
    vectors, so it behaves like a crude keyword match.

    What it is NOT: semantic. "vacation days" and "paid leave" share no words, so their
    vectors are unrelated. Never use it to judge real retrieval quality.
    """

    MODEL_NAME = "hash-v1"  # change this if the algorithm changes: stored vectors would differ

    def __init__(self, dimensions: int = 256) -> None:
        if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
            raise ValueError(f"dimensions must be a positive integer, got {dimensions!r}.")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        if not text or not text.strip():
            raise ValueError("Cannot embed empty or whitespace-only text.")

        # Text with no word characters (e.g. "→ ✓") is treated as a single token.
        tokens = _WORD.findall(text.lower()) or [text.strip()]

        vector = [0.0] * self.dimensions
        for token in tokens:
            vector[self._slot(token)] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))
        return [v / norm for v in vector]

    def _slot(self, token: str) -> int:
        # Python's built-in hash() is randomized per process, so it would give different
        # vectors on every run. hashlib is stable across runs and machines.
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dimensions


@dataclass(frozen=True)
class EmbeddingProvider:
    """An embedding implementation plus the identity recorded in the vector store.

    `model` and `dimensions` are stored on the ChromaDB collection, so vectors from a
    different model can never be mixed into it (see services/vectorstore.py).
    """

    embeddings: Embeddings
    model: str
    dimensions: int


def get_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the embedding provider selected in settings."""
    if settings.embedding_provider == "hash":
        dims = settings.embedding_dimensions
        return EmbeddingProvider(HashEmbeddings(dims), HashEmbeddings.MODEL_NAME, dims)

    # Unreachable while Settings only allows "hash"; kept as a guard for future providers.
    raise ValueError(f"Unknown embedding provider: {settings.embedding_provider!r}")
