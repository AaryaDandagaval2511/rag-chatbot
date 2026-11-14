"""ChromaDB access: client, collection configuration, and the few operations we need.

We use the `chromadb` library directly (not a LangChain wrapper) so that IDs, upserts,
metadata filters and collection settings are all visible here.

Shared by the ingestor (writes) and, later, the retriever (reads). Both MUST use the same
collection settings and embedding model, which is why this lives in one place.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings as ChromaSettings

from app.config import Settings
from app.services.embeddings import EmbeddingProvider, get_embedding_provider

logger = logging.getLogger(__name__)

DISTANCE_METRIC = "cosine"
# ChromaDB rejects single writes above its limit (5461 in v1.5). Stay well below it.
UPSERT_BATCH_SIZE = 1000


class VectorStoreError(Exception):
    """Base class for vector store problems."""


class EmbeddingModelMismatchError(VectorStoreError):
    """The collection was built with a different embedding model or dimension."""


def get_client(persist_dir: str | Path) -> ClientAPI:
    """Open (or create) a ChromaDB database stored in `persist_dir`."""
    path = Path(persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    # Telemetry off: no usage data sent anywhere, and the app works fully offline.
    return chromadb.PersistentClient(
        path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
    )


def get_collection(
    client: ClientAPI, name: str, *, model: str, dimensions: int
) -> Collection:
    """Open the collection, creating it on first use, and verify it is compatible.

    On creation we fix two things that can't be changed later without rebuilding:
      * the distance metric (cosine; ChromaDB's default is L2)
      * the embedding model identity, stored in the collection's metadata

    On every open we compare the stored identity with the current one. ChromaDB's
    get_or_create silently keeps the OLD metadata, so without this check a model change
    would mix incompatible vectors and return nonsense results without any error.
    """
    collection = client.get_or_create_collection(
        name=name,
        configuration={"hnsw": {"space": DISTANCE_METRIC}},
        metadata={"embedding_model": model, "embedding_dimensions": dimensions},
        # Must be explicit: ChromaDB's default would embed text itself with a built-in
        # model, downloading ~80 MB on first use. We always supply our own vectors.
        embedding_function=None,
    )
    _check_compatible(collection, model=model, dimensions=dimensions)
    return collection


def open_collection(
    settings: Settings, provider: EmbeddingProvider | None = None
) -> Collection:
    """Convenience: the collection configured in settings, for the configured model."""
    provider = provider or get_embedding_provider(settings)
    client = get_client(settings.chroma_persist_dir)
    return get_collection(
        client,
        settings.chroma_collection,
        model=provider.model,
        dimensions=provider.dimensions,
    )


def get_chunk_metadata(collection: Collection, where: dict[str, Any]) -> dict[str, dict]:
    """Return {chunk_id: metadata} for every stored chunk matching a metadata filter."""
    result = collection.get(where=where, include=["metadatas"])
    return dict(zip(result["ids"], result["metadatas"]))


def upsert_chunks(
    collection: Collection,
    *,
    ids: Sequence[str],
    documents: Sequence[str],
    metadatas: Sequence[dict],
    embeddings: Sequence[Sequence[float]],
    batch_size: int = UPSERT_BATCH_SIZE,
) -> None:
    """Insert chunks, or overwrite chunks whose IDs already exist ("upsert").

    Upsert + deterministic IDs = idempotent writes: indexing the same file twice leaves
    exactly one copy of each chunk.
    """
    lengths = {len(ids), len(documents), len(metadatas), len(embeddings)}
    if len(lengths) != 1:
        raise ValueError(
            "ids, documents, metadatas and embeddings must have the same length, got "
            f"{len(ids)}, {len(documents)}, {len(metadatas)}, {len(embeddings)}."
        )

    # ChromaDB only enforces vector size AFTER the first insert (the first vectors decide
    # it). So an empty collection would silently accept wrong-sized vectors and lock
    # itself to that size. Check against the size recorded at creation instead.
    expected = (collection.metadata or {}).get("embedding_dimensions")
    if expected is not None:
        bad = next((len(v) for v in embeddings if len(v) != expected), None)
        if bad is not None:
            raise ValueError(
                f"Embedding has {bad} dimensions, but collection '{collection.name}' "
                f"expects {expected}."
            )

    for start in range(0, len(ids), batch_size):
        end = start + batch_size
        collection.upsert(
            ids=list(ids[start:end]),
            documents=list(documents[start:end]),
            metadatas=list(metadatas[start:end]),
            embeddings=[list(vector) for vector in embeddings[start:end]],
        )


def delete_chunks(collection: Collection, ids: Iterable[str]) -> int:
    """Delete chunks by ID. Returns how many IDs were requested for deletion."""
    ids = sorted(set(ids))
    if ids:
        collection.delete(ids=ids)
    return len(ids)


def _check_compatible(collection: Collection, *, model: str, dimensions: int) -> None:
    stored = collection.metadata or {}
    stored_model = stored.get("embedding_model")
    stored_dims = stored.get("embedding_dimensions")
    if stored_model != model or stored_dims != dimensions:
        raise EmbeddingModelMismatchError(
            f"Collection '{collection.name}' was built with embedding model "
            f"{stored_model!r} ({stored_dims} dims), but the current model is {model!r} "
            f"({dimensions} dims). Vectors from different models can't be compared. "
            "Re-index: delete the vector store folder (or use a new collection name) "
            "and ingest the documents again."
        )

    space = (collection.configuration or {}).get("hnsw", {}).get("space")
    if space != DISTANCE_METRIC:
        raise VectorStoreError(
            f"Collection '{collection.name}' uses distance metric {space!r}, expected "
            f"{DISTANCE_METRIC!r}. The metric is fixed at creation; re-create the collection."
        )
