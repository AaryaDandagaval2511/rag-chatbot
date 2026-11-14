"""Tests for the MOCK embedding provider (HashEmbeddings) and the provider factory.

These verify the mock's mechanical properties only. HashEmbeddings is lexical (word
overlap), not semantic. See tests/README.md.
"""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.services.embeddings import HashEmbeddings, get_embedding_provider

BACKEND_DIR = Path(__file__).resolve().parents[1]


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b))  # vectors are already unit length


@pytest.fixture
def emb():
    return HashEmbeddings(256)


def test_implements_langchain_embeddings_interface(emb):
    """Same interface as real providers such as OllamaEmbeddings."""
    assert isinstance(emb, Embeddings)


def test_vector_has_requested_dimensions(emb):
    assert len(emb.embed_query("hello world")) == 256
    assert len(HashEmbeddings(8).embed_query("hello world")) == 8


def test_vectors_are_unit_length(emb):
    for text in ["one", "a longer sentence with several words", "→ ✓"]:
        vector = emb.embed_query(text)
        assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)


def test_deterministic_within_a_process(emb):
    assert emb.embed_query("Paid leave policy") == HashEmbeddings(256).embed_query(
        "Paid leave policy"
    )


def test_deterministic_across_processes():
    """Python's built-in hash() changes between runs (PYTHONHASHSEED); ours must not,
    or vectors stored yesterday wouldn't match queries made today."""
    code = "from app.services.embeddings import HashEmbeddings; print(HashEmbeddings(16).embed_query('stable text'))"
    outputs = set()
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
            cwd=BACKEND_DIR,  # so the child process can import `app`
        )
        outputs.add(result.stdout)
    assert len(outputs) == 1


def test_embed_query_matches_embed_documents(emb):
    assert emb.embed_query("same text") == emb.embed_documents(["same text"])[0]


def test_embed_documents_returns_one_vector_per_text(emb):
    assert len(emb.embed_documents(["a", "b", "c"])) == 3


def test_case_insensitive(emb):
    assert emb.embed_query("Annual Leave") == emb.embed_query("annual leave")


def test_word_order_is_ignored(emb):
    """A known limitation of bag-of-words: order carries no meaning."""
    assert emb.embed_query("dog bites man") == emb.embed_query("man bites dog")


def test_shared_words_score_higher_than_unrelated_text(emb):
    """LEXICAL similarity: overlapping words -> closer vectors."""
    query = emb.embed_query("employee annual leave policy")
    related = emb.embed_query("the annual leave policy for every employee")
    unrelated = emb.embed_query("quarterly revenue grew in the northern region")

    assert cosine(query, related) > cosine(query, unrelated)


def test_mock_is_not_semantic(emb):
    """Documents the limitation: synonyms with no shared words are NOT similar.
    A real embedding model would score these two as close."""
    similarity = cosine(emb.embed_query("vacation days"), emb.embed_query("paid leave"))

    assert similarity < 0.2


def test_symbol_only_text_is_embedded(emb):
    assert len(emb.embed_query("→ ✓ │")) == 256


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_text_is_rejected(emb, text):
    with pytest.raises(ValueError, match="empty"):
        emb.embed_query(text)


@pytest.mark.parametrize("dimensions", [0, -1, 2.5, True, "64"])
def test_invalid_dimensions_are_rejected(dimensions):
    with pytest.raises(ValueError, match="positive integer"):
        HashEmbeddings(dimensions)


def test_provider_factory_builds_hash_provider():
    settings = Settings(_env_file=None, embedding_provider="hash", embedding_dimensions=32)

    provider = get_embedding_provider(settings)

    assert isinstance(provider.embeddings, HashEmbeddings)
    assert provider.model == "hash-v1"
    assert provider.dimensions == 32
    assert len(provider.embeddings.embed_query("text")) == 32
