"""Tests for app.services.ingestor.index_pdf: IDs, duplicates, updates and failures.

These are PIPELINE MECHANICS tests. They run with mock embeddings and a temporary
ChromaDB folder, and say nothing about retrieval quality. See tests/README.md.
"""

import pytest

import app.services.vectorstore as vectorstore
from app.services.ingestor import (
    IndexingError,
    ScannedPDFError,
    compute_doc_id,
    index_pdf,
    load_and_chunk_pdf,
)
from tests.pdf_factory import make_pdf

SIZE, OVERLAP = 200, 20  # small chunks so test PDFs produce several of them


def long_text(topic, sentences=25):
    return " ".join(f"Fact {i} about {topic} is stated here." for i in range(sentences))


@pytest.fixture
def report_pdf(tmp_path):
    return make_pdf(tmp_path / "report.pdf", [long_text("budgets"), long_text("hiring")])


def index(path, collection, embeddings, size=SIZE, overlap=OVERLAP):
    return index_pdf(
        path, collection=collection, embeddings=embeddings, chunk_size=size, chunk_overlap=overlap
    )


def stored(collection, **where):
    """{id: metadata} for stored chunks, optionally filtered by one metadata field."""
    result = collection.get(where=where or None, include=["metadatas"])
    return dict(zip(result["ids"], result["metadatas"]))


# --------------------------------------------------------------------------------------
# Document IDs
# --------------------------------------------------------------------------------------


def test_doc_id_is_16_hex_characters(report_pdf):
    doc_id = compute_doc_id(report_pdf)

    assert len(doc_id) == 16
    assert int(doc_id, 16) >= 0  # parses as hexadecimal


def test_doc_id_depends_on_content_not_name_or_location(tmp_path, report_pdf):
    copy = tmp_path / "elsewhere" / "renamed.pdf"
    copy.parent.mkdir()
    copy.write_bytes(report_pdf.read_bytes())

    assert compute_doc_id(copy) == compute_doc_id(report_pdf)


def test_different_content_gives_different_doc_id(tmp_path):
    a = make_pdf(tmp_path / "a.pdf", ["Version one"])
    b = make_pdf(tmp_path / "b.pdf", ["Version two"])

    assert compute_doc_id(a) != compute_doc_id(b)


# --------------------------------------------------------------------------------------
# First indexing
# --------------------------------------------------------------------------------------


def test_first_index_stores_every_chunk(report_pdf, collection, embeddings):
    expected_chunks = load_and_chunk_pdf(report_pdf, SIZE, OVERLAP)

    result = index(report_pdf, collection, embeddings)

    assert result.status == "indexed"
    assert result.file_name == "report.pdf"
    assert result.chunk_count == len(expected_chunks) == collection.count()
    assert result.chunks_deleted == 0


def test_chunk_ids_are_doc_id_and_chunk_index(report_pdf, collection, embeddings):
    result = index(report_pdf, collection, embeddings)

    expected = {f"{result.doc_id}-{i}" for i in range(result.chunk_count)}
    assert set(collection.get()["ids"]) == expected
    assert result.doc_id == compute_doc_id(report_pdf)


def test_text_and_vectors_are_stored(report_pdf, collection, embeddings):
    chunks = load_and_chunk_pdf(report_pdf, SIZE, OVERLAP)
    result = index(report_pdf, collection, embeddings)

    first = collection.get(ids=[f"{result.doc_id}-0"], include=["documents", "embeddings"])

    assert first["documents"] == [chunks[0].page_content]
    assert list(first["embeddings"][0]) == pytest.approx(
        embeddings.embed_query(chunks[0].page_content), abs=1e-6
    )


def test_metadata_round_trips_through_chromadb(report_pdf, collection, embeddings):
    result = index(report_pdf, collection, embeddings)

    meta = stored(collection)[f"{result.doc_id}-0"]

    assert meta == {
        "source": str(report_pdf),
        "file_name": "report.pdf",
        "total_pages": 2,
        "page": 1,
        "start_index": 0,
        "chunk_index": 0,
        "doc_id": result.doc_id,
        "chunk_size": SIZE,
        "chunk_overlap": OVERLAP,
    }


def test_every_chunk_records_its_page(report_pdf, collection, embeddings):
    index(report_pdf, collection, embeddings)

    pages = {meta["page"] for meta in stored(collection).values()}

    assert pages == {1, 2}


def test_different_files_coexist(tmp_path, collection, embeddings):
    a = make_pdf(tmp_path / "a.pdf", [long_text("apples")])
    b = make_pdf(tmp_path / "b.pdf", [long_text("bananas")])

    ra = index(a, collection, embeddings)
    rb = index(b, collection, embeddings)

    assert collection.count() == ra.chunk_count + rb.chunk_count
    assert len(stored(collection, file_name="a.pdf")) == ra.chunk_count
    assert len(stored(collection, file_name="b.pdf")) == rb.chunk_count


# --------------------------------------------------------------------------------------
# Duplicate detection
# --------------------------------------------------------------------------------------


def test_reindexing_unchanged_file_is_skipped_without_embedding(
    report_pdf, collection, spy_embeddings
):
    first = index(report_pdf, collection, spy_embeddings)
    calls_after_first = spy_embeddings.calls

    second = index(report_pdf, collection, spy_embeddings)

    assert second.status == "skipped"
    assert spy_embeddings.calls == calls_after_first  # the expensive step never ran
    assert collection.count() == first.chunk_count  # no duplicates


def test_identical_content_under_another_name_is_skipped(
    tmp_path, report_pdf, collection, spy_embeddings
):
    copy = tmp_path / "copy_of_report.pdf"
    copy.write_bytes(report_pdf.read_bytes())
    first = index(report_pdf, collection, spy_embeddings)

    result = index(copy, collection, spy_embeddings)

    assert result.status == "skipped"
    assert spy_embeddings.calls == 1
    assert collection.count() == first.chunk_count
    # Stored once, under the name it was first indexed with (a documented limitation).
    assert all(m["file_name"] == "report.pdf" for m in stored(collection).values())


# --------------------------------------------------------------------------------------
# Updated files and changed settings
# --------------------------------------------------------------------------------------


def test_new_version_of_same_file_name_replaces_old_chunks(tmp_path, collection, embeddings):
    path = tmp_path / "policy.pdf"
    make_pdf(path, [long_text("old rules", 40)])
    old = index(path, collection, embeddings)

    make_pdf(path, [long_text("new rules", 10)])  # same name, different (shorter) content
    new = index(path, collection, embeddings)

    assert new.status == "updated"
    assert new.doc_id != old.doc_id
    assert new.chunks_deleted == old.chunk_count
    assert stored(collection, doc_id=old.doc_id) == {}
    assert len(stored(collection, doc_id=new.doc_id)) == new.chunk_count == collection.count()


def test_changed_chunk_settings_reindex_and_remove_stale_chunks(
    report_pdf, collection, spy_embeddings
):
    small = index(report_pdf, collection, spy_embeddings, size=150, overlap=10)

    large = index(report_pdf, collection, spy_embeddings, size=400, overlap=40)

    assert large.status == "updated"
    assert spy_embeddings.calls == 2  # re-embedded with the new chunking
    assert large.chunk_count < small.chunk_count
    assert collection.count() == large.chunk_count
    assert {m["chunk_size"] for m in stored(collection).values()} == {400}


def test_same_chunk_count_but_different_settings_still_reindexes(
    tmp_path, collection, spy_embeddings
):
    """IDs alone can't tell these apart (both produce chunk 0 only), so the stored
    chunk settings must be compared too."""
    pdf = make_pdf(tmp_path / "short.pdf", ["A short page."])
    index(pdf, collection, spy_embeddings, size=100, overlap=10)

    result = index(pdf, collection, spy_embeddings, size=100, overlap=20)

    assert result.status == "updated"
    assert spy_embeddings.calls == 2
    assert stored(collection)[f"{result.doc_id}-0"]["chunk_overlap"] == 20


def test_interrupted_earlier_run_is_completed(report_pdf, collection, spy_embeddings):
    """Simulate a crash after only some chunks were written: the skip check must not be
    fooled, and re-running heals the store."""
    full = index(report_pdf, collection, spy_embeddings)
    missing = [f"{full.doc_id}-{i}" for i in range(1, full.chunk_count)]
    collection.delete(ids=missing)
    assert collection.count() == 1

    healed = index(report_pdf, collection, spy_embeddings)

    assert healed.status == "updated"
    assert collection.count() == full.chunk_count


def test_file_replaced_by_already_indexed_content_drops_its_old_chunks(
    tmp_path, collection, spy_embeddings
):
    a = make_pdf(tmp_path / "a.pdf", [long_text("alpha")])
    b = tmp_path / "b.pdf"
    make_pdf(b, [long_text("beta")])
    ra = index(a, collection, spy_embeddings)
    rb_old = index(b, collection, spy_embeddings)

    b.write_bytes(a.read_bytes())  # b.pdf now holds exactly a.pdf's content
    result = index(b, collection, spy_embeddings)

    assert result.status == "skipped"  # nothing new to embed
    assert result.chunks_deleted == rb_old.chunk_count  # but b's old chunks are gone
    assert stored(collection, doc_id=rb_old.doc_id) == {}
    assert collection.count() == ra.chunk_count


# --------------------------------------------------------------------------------------
# Failures leave the store consistent
# --------------------------------------------------------------------------------------


def test_unloadable_pdf_writes_nothing(tmp_path, collection, embeddings):
    scan = make_pdf(tmp_path / "scan.pdf", [None])

    with pytest.raises(ScannedPDFError):
        index(scan, collection, embeddings)

    assert collection.count() == 0


def test_invalid_chunk_settings_write_nothing(report_pdf, collection, embeddings):
    with pytest.raises(ValueError):
        index(report_pdf, collection, embeddings, size=100, overlap=100)

    assert collection.count() == 0


class FailingEmbeddings:
    def embed_documents(self, texts):
        raise ConnectionError("embedding service unavailable")


def test_embedding_failure_keeps_previous_version(tmp_path, collection, embeddings):
    path = tmp_path / "policy.pdf"
    make_pdf(path, [long_text("old rules")])
    old = index(path, collection, embeddings)

    make_pdf(path, [long_text("new rules")])
    with pytest.raises(ConnectionError):
        index(path, collection, FailingEmbeddings())

    assert len(stored(collection, doc_id=old.doc_id)) == old.chunk_count


def test_storage_failure_keeps_previous_version(tmp_path, collection, embeddings, monkeypatch):
    """Write-before-delete: if writing the new chunks fails, the old ones are untouched."""
    path = tmp_path / "policy.pdf"
    make_pdf(path, [long_text("old rules")])
    old = index(path, collection, embeddings)

    def broken_upsert(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(vectorstore, "upsert_chunks", broken_upsert)
    make_pdf(path, [long_text("new rules")])
    with pytest.raises(RuntimeError, match="disk full"):
        index(path, collection, embeddings)

    assert len(stored(collection, doc_id=old.doc_id)) == old.chunk_count


class WrongCountEmbeddings:
    def embed_documents(self, texts):
        return [[1.0] + [0.0] * 63]  # one vector, however many texts


def test_wrong_number_of_vectors_raises_and_writes_nothing(report_pdf, collection):
    with pytest.raises(IndexingError, match="vectors for"):
        index(report_pdf, collection, WrongCountEmbeddings())

    assert collection.count() == 0
