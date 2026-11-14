"""Tests for app.services.vectorstore: ChromaDB configuration and operations.

All tests use a temporary database folder and hand-written or mock vectors.
"""

import math

import chromadb
import pytest
from chromadb.config import Settings as ChromaSettings

from app.services.embeddings import HashEmbeddings
from app.services.vectorstore import (
    EmbeddingModelMismatchError,
    VectorStoreError,
    delete_chunks,
    get_chunk_metadata,
    get_client,
    get_collection,
    open_collection,
    upsert_chunks,
)
from tests.support import TEST_DIMENSIONS

MODEL = HashEmbeddings.MODEL_NAME


def unit(*values):
    """A unit-length vector padded with zeros to TEST_DIMENSIONS."""
    vector = list(values) + [0.0] * (TEST_DIMENSIONS - len(values))
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


def add(collection, ids, vectors=None, metadatas=None):
    vectors = vectors or [unit(1, i + 1) for i in range(len(ids))]
    upsert_chunks(
        collection,
        ids=ids,
        documents=[f"text of {i}" for i in ids],
        metadatas=metadatas or [{"file_name": "a.pdf"} for _ in ids],
        embeddings=vectors,
    )


# --------------------------------------------------------------------------------------
# Client and collection configuration
# --------------------------------------------------------------------------------------


def test_get_client_creates_folder(tmp_path):
    folder = tmp_path / "nested" / "store"

    get_client(folder)

    assert folder.is_dir()


def test_collection_uses_cosine_distance(collection):
    assert collection.configuration["hnsw"]["space"] == "cosine"


def test_cosine_distances_have_expected_values(collection):
    """Proof the metric is really cosine: identical -> 0, perpendicular -> 1.
    (With ChromaDB's default squared-L2 metric, perpendicular unit vectors give 2.)"""
    add(collection, ["same", "perpendicular"], vectors=[unit(1, 0), unit(0, 1)])

    result = collection.query(query_embeddings=[unit(1, 0)], n_results=2, include=["distances"])

    assert result["ids"][0] == ["same", "perpendicular"]
    assert result["distances"][0][0] == pytest.approx(0.0, abs=1e-6)
    assert result["distances"][0][1] == pytest.approx(1.0, abs=1e-6)


def test_collection_records_embedding_model(collection):
    assert collection.metadata == {
        "embedding_model": MODEL,
        "embedding_dimensions": TEST_DIMENSIONS,
    }


def test_data_persists_across_client_instances(tmp_path):
    folder = tmp_path / "store"
    first = get_collection(get_client(folder), "docs", model=MODEL, dimensions=TEST_DIMENSIONS)
    add(first, ["c1", "c2"])

    reopened = get_collection(get_client(folder), "docs", model=MODEL, dimensions=TEST_DIMENSIONS)

    assert reopened.count() == 2
    assert reopened.configuration["hnsw"]["space"] == "cosine"


def test_different_model_name_is_rejected(tmp_path):
    client = get_client(tmp_path / "store")
    get_collection(client, "docs", model=MODEL, dimensions=TEST_DIMENSIONS)

    with pytest.raises(EmbeddingModelMismatchError, match="Re-index"):
        get_collection(client, "docs", model="other-model", dimensions=TEST_DIMENSIONS)


def test_different_dimensions_are_rejected(tmp_path):
    client = get_client(tmp_path / "store")
    get_collection(client, "docs", model=MODEL, dimensions=TEST_DIMENSIONS)

    with pytest.raises(EmbeddingModelMismatchError):
        get_collection(client, "docs", model=MODEL, dimensions=TEST_DIMENSIONS * 2)


def test_mismatch_check_does_not_alter_stored_identity(tmp_path):
    client = get_client(tmp_path / "store")
    get_collection(client, "docs", model=MODEL, dimensions=TEST_DIMENSIONS)
    with pytest.raises(EmbeddingModelMismatchError):
        get_collection(client, "docs", model="other-model", dimensions=TEST_DIMENSIONS)

    reopened = get_collection(client, "docs", model=MODEL, dimensions=TEST_DIMENSIONS)

    assert reopened.metadata["embedding_model"] == MODEL


def test_collection_with_wrong_distance_metric_is_rejected(tmp_path):
    """A collection created elsewhere with L2 can't silently be used as cosine."""
    folder = tmp_path / "store"
    raw = chromadb.PersistentClient(
        path=str(folder), settings=ChromaSettings(anonymized_telemetry=False)
    )
    raw.create_collection(
        "docs",
        configuration={"hnsw": {"space": "l2"}},
        metadata={"embedding_model": MODEL, "embedding_dimensions": TEST_DIMENSIONS},
        embedding_function=None,
    )

    with pytest.raises(VectorStoreError, match="distance metric"):
        get_collection(get_client(folder), "docs", model=MODEL, dimensions=TEST_DIMENSIONS)


def test_open_collection_uses_settings(test_settings):
    collection = open_collection(test_settings)

    assert collection.name == "test_documents"
    assert test_settings.chroma_persist_dir.is_dir()
    assert collection.metadata["embedding_dimensions"] == TEST_DIMENSIONS


# --------------------------------------------------------------------------------------
# No built-in embedding function (which would download a model)
# --------------------------------------------------------------------------------------


def test_collection_cannot_embed_text_by_itself(collection):
    """We always pass our own vectors. If ChromaDB could embed text itself, it would
    download its default model (~80 MB) on first use."""
    with pytest.raises(ValueError, match="embedding function"):
        collection.query(query_texts=["hello"], n_results=1)


def test_upserting_text_without_vectors_fails(collection):
    with pytest.raises(ValueError, match="embedding function"):
        collection.upsert(ids=["x"], documents=["no vector supplied"])


def test_wrong_vector_size_is_rejected_even_in_empty_collection(collection):
    """Regression: ChromaDB itself would accept these (the first insert sets the size)
    and lock the collection to 3 dimensions, despite its metadata saying 64."""
    with pytest.raises(ValueError, match="expects 64"):
        add(collection, ["x"], vectors=[[1.0, 0.0, 0.0]])

    assert collection.count() == 0


def test_wrong_vector_size_in_a_later_item_rejects_whole_write(collection):
    with pytest.raises(ValueError, match="dimensions"):
        add(collection, ["ok", "bad"], vectors=[unit(1, 0), [1.0, 0.0]])

    assert collection.count() == 0


# --------------------------------------------------------------------------------------
# Upsert, read, delete
# --------------------------------------------------------------------------------------


def test_upsert_stores_documents_metadata_and_vectors(collection):
    vector = unit(3, 4)
    add(collection, ["c1"], vectors=[vector], metadatas=[{"file_name": "a.pdf", "page": 2}])

    stored = collection.get(ids=["c1"], include=["documents", "metadatas", "embeddings"])

    assert stored["documents"] == ["text of c1"]
    assert stored["metadatas"] == [{"file_name": "a.pdf", "page": 2}]
    assert list(stored["embeddings"][0]) == pytest.approx(vector, abs=1e-6)


def test_upsert_with_same_ids_overwrites_instead_of_duplicating(collection):
    add(collection, ["c1", "c2"], metadatas=[{"v": 1}, {"v": 1}])
    add(collection, ["c1", "c2"], metadatas=[{"v": 2}, {"v": 2}])

    assert collection.count() == 2
    assert get_chunk_metadata(collection, {"v": 2}).keys() == {"c1", "c2"}


def test_upsert_splits_large_writes_into_batches(collection, monkeypatch):
    calls = []
    original = collection.upsert
    monkeypatch.setattr(
        collection, "upsert", lambda **kw: (calls.append(len(kw["ids"])), original(**kw))
    )
    ids = [f"c{i}" for i in range(5)]

    upsert_chunks(
        collection,
        ids=ids,
        documents=ids,
        metadatas=[{"i": i} for i in range(5)],
        embeddings=[unit(1, i + 1) for i in range(5)],
        batch_size=2,
    )

    assert calls == [2, 2, 1]
    assert collection.count() == 5


def test_upsert_rejects_mismatched_lengths(collection):
    with pytest.raises(ValueError, match="same length"):
        upsert_chunks(
            collection, ids=["a", "b"], documents=["only one"], metadatas=[{}, {}], embeddings=[]
        )


def test_get_chunk_metadata_filters_by_metadata(collection):
    add(
        collection,
        ["a1", "a2", "b1"],
        metadatas=[{"file_name": "a.pdf"}, {"file_name": "a.pdf"}, {"file_name": "b.pdf"}],
    )

    found = get_chunk_metadata(collection, {"file_name": "a.pdf"})

    assert set(found) == {"a1", "a2"}
    assert found["a1"] == {"file_name": "a.pdf"}


def test_get_chunk_metadata_with_no_matches_is_empty(collection):
    assert get_chunk_metadata(collection, {"file_name": "nothing.pdf"}) == {}


def test_delete_chunks(collection):
    add(collection, ["c1", "c2", "c3"])

    deleted = delete_chunks(collection, ["c1", "c3", "c1"])  # duplicates are ignored

    assert deleted == 2
    assert collection.get()["ids"] == ["c2"]


def test_delete_with_no_ids_is_a_no_op(collection):
    add(collection, ["c1"])

    assert delete_chunks(collection, []) == 0
    assert collection.count() == 1
