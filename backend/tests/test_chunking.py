"""Tests for chunking: app.services.ingestor.chunk_documents and load_and_chunk_pdf.

Most tests build Document objects directly. Chunking only needs text plus metadata,
so no PDF is required. The integration tests at the end go through a real PDF file.
"""

import logging

import pytest
from langchain_core.documents import Document

from app.services.ingestor import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    ScannedPDFError,
    chunk_documents,
    load_and_chunk_pdf,
)
from tests.pdf_factory import make_pdf


def page(text, page_number=1, source="docs/report.pdf"):
    """A page Document shaped like load_pdf()'s output."""
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "file_name": source.rsplit("/", 1)[-1],
            "total_pages": 3,
            "page": page_number,
        },
    )


def numbered_words(count):
    """'w0 w1 w2 ...': every word is unique, so overlaps are easy to detect."""
    return " ".join(f"w{i}" for i in range(count))


# --------------------------------------------------------------------------------------
# Basic splitting
# --------------------------------------------------------------------------------------


def test_short_page_becomes_single_chunk():
    chunks = chunk_documents([page("A short page.")])

    assert len(chunks) == 1
    assert chunks[0].page_content == "A short page."


def test_long_page_is_split_and_respects_chunk_size():
    chunks = chunk_documents([page(numbered_words(500))], chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)


def test_text_without_any_separators_is_still_split():
    """Last-resort fallback: a 2500-character 'word' gets cut character by character."""
    chunks = chunk_documents([page("x" * 2500)], chunk_size=1000, chunk_overlap=0)

    assert [len(c.page_content) for c in chunks] == [1000, 1000, 500]


def test_no_text_is_lost():
    """Every word of the page appears in at least one chunk."""
    text = numbered_words(400)
    chunks = chunk_documents([page(text)], chunk_size=150, chunk_overlap=30)

    words_in_chunks = {w for c in chunks for w in c.page_content.split()}
    assert words_in_chunks == set(text.split())


# --------------------------------------------------------------------------------------
# Overlap
# --------------------------------------------------------------------------------------


def test_consecutive_chunks_overlap():
    chunks = chunk_documents([page(numbered_words(300))], chunk_size=100, chunk_overlap=30)

    for previous, current in zip(chunks, chunks[1:]):
        first_word_of_current = current.page_content.split()[0]
        assert first_word_of_current in previous.page_content.split()


def test_zero_overlap_means_no_repeated_words():
    chunks = chunk_documents([page(numbered_words(300))], chunk_size=100, chunk_overlap=0)

    all_words = [w for c in chunks for w in c.page_content.split()]
    assert len(all_words) == len(set(all_words))


# --------------------------------------------------------------------------------------
# Where the splitter cuts
# --------------------------------------------------------------------------------------


def test_prefers_paragraph_breaks():
    para_a = "Alpha sentence one. Alpha sentence two."
    para_b = "Beta sentence one. Beta sentence two."
    chunks = chunk_documents([page(f"{para_a}\n\n{para_b}")], chunk_size=50, chunk_overlap=0)

    assert [c.page_content for c in chunks] == [para_a, para_b]


def test_prefers_sentence_end_over_pdf_line_wrap():
    """pypdf inserts '\\n' at every printed line wrap, often mid-sentence. The cut should
    land after the full stop, not at the line wrap."""
    text = "The first sentence wraps\nonto a second line. The second sentence is here."
    chunks = chunk_documents([page(text)], chunk_size=50, chunk_overlap=0)

    assert chunks[0].page_content == "The first sentence wraps\nonto a second line."
    assert chunks[1].page_content == "The second sentence is here."


def test_full_stop_stays_with_its_sentence():
    text = " ".join(f"Sentence number {i} is here." for i in range(20))
    chunks = chunk_documents([page(text)], chunk_size=120, chunk_overlap=0)

    assert all(c.page_content.endswith(".") for c in chunks)
    assert not any(c.page_content.startswith(".") for c in chunks)


def test_chunks_never_span_pages():
    pages = [page(numbered_words(200), page_number=1), page(numbered_words(200), page_number=2)]
    chunks = chunk_documents(pages, chunk_size=150, chunk_overlap=20)

    page_text = {p.metadata["page"]: p.page_content for p in pages}
    for chunk in chunks:
        assert chunk.page_content in page_text[chunk.metadata["page"]]


def test_whitespace_only_page_produces_no_chunks():
    chunks = chunk_documents([page("   \n\n   "), page("Real text.", page_number=2)])

    assert [c.page_content for c in chunks] == ["Real text."]


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------


def test_chunk_inherits_page_metadata_and_adds_chunk_fields():
    chunk = chunk_documents([page("Some text.", page_number=2)])[0]

    assert chunk.metadata == {
        "source": "docs/report.pdf",
        "file_name": "report.pdf",
        "total_pages": 3,
        "page": 2,
        "start_index": 0,
        "chunk_index": 0,
    }


def test_start_index_points_at_chunk_text_in_page():
    text = numbered_words(300)
    chunks = chunk_documents([page(text)], chunk_size=100, chunk_overlap=25)

    for chunk in chunks:
        start = chunk.metadata["start_index"]
        assert text[start : start + len(chunk.page_content)] == chunk.page_content


def test_chunk_index_is_sequential_across_pages_of_one_file():
    pages = [page(numbered_words(100), page_number=n) for n in (1, 2, 3)]
    chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=0)

    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_chunk_index_restarts_for_each_source_file():
    pages = [
        page(numbered_words(60), source="a.pdf"),
        page(numbered_words(60), source="b.pdf"),
    ]
    chunks = chunk_documents(pages, chunk_size=100, chunk_overlap=0)

    for source in ("a.pdf", "b.pdf"):
        indexes = [c.metadata["chunk_index"] for c in chunks if c.metadata["source"] == source]
        assert indexes == list(range(len(indexes)))


def test_chunks_do_not_share_metadata_dicts():
    chunks = chunk_documents([page(numbered_words(200))], chunk_size=100, chunk_overlap=0)
    chunks[0].metadata["page"] = 99

    assert chunks[1].metadata["page"] == 1


def test_input_pages_are_not_modified():
    original = page(numbered_words(200))
    before_text, before_meta = original.page_content, dict(original.metadata)

    chunk_documents([original], chunk_size=100, chunk_overlap=0)

    assert original.page_content == before_text
    assert original.metadata == before_meta


# --------------------------------------------------------------------------------------
# Input handling and validation
# --------------------------------------------------------------------------------------


def test_empty_input_returns_empty_list_with_warning(caplog):
    with caplog.at_level(logging.WARNING):
        assert chunk_documents([]) == []
    assert "no documents" in caplog.text


def test_accepts_any_iterable_such_as_a_generator():
    chunks = chunk_documents(page(f"Page {n} text.", page_number=n) for n in (1, 2))

    assert len(chunks) == 2


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap", "message"),
    [
        (0, 0, "must be positive"),
        (-100, 0, "must be positive"),
        (100, -1, "cannot be negative"),
        (100, 100, "must be smaller"),
        (100, 150, "must be smaller"),
        (100.5, 10, "must be an integer"),
        ("1000", 10, "must be an integer"),
        (True, 0, "must be an integer"),
        (100, None, "must be an integer"),
    ],
)
def test_invalid_parameters_raise_value_error(chunk_size, chunk_overlap, message):
    with pytest.raises(ValueError, match=message):
        chunk_documents([page("text")], chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_invalid_parameters_rejected_even_for_empty_input():
    """Bad settings should fail loudly, not hide behind an empty input."""
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=0)


def test_defaults_are_sensible():
    assert 0 < DEFAULT_CHUNK_OVERLAP < DEFAULT_CHUNK_SIZE


def test_logs_summary(caplog):
    with caplog.at_level(logging.INFO):
        chunk_documents([page("One."), page("Two.", page_number=2)])
    assert "Split 2 page(s) into 2 chunk(s)" in caplog.text


# --------------------------------------------------------------------------------------
# Integration: PDF file -> chunks
# --------------------------------------------------------------------------------------


def test_load_and_chunk_pdf_end_to_end(tmp_path):
    long_page = " ".join(f"Fact number {i} about the topic." for i in range(60))
    pdf = make_pdf(tmp_path / "report.pdf", [long_page, "A short second page."])

    chunks = load_and_chunk_pdf(pdf, chunk_size=300, chunk_overlap=50)

    assert len(chunks) > 2
    assert {c.metadata["page"] for c in chunks} == {1, 2}
    assert all(c.metadata["file_name"] == "report.pdf" for c in chunks)
    assert chunks[-1].page_content == "A short second page."
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_load_and_chunk_pdf_propagates_loading_errors(tmp_path):
    pdf = make_pdf(tmp_path / "scan.pdf", [None])

    with pytest.raises(ScannedPDFError):
        load_and_chunk_pdf(pdf)


def test_load_and_chunk_pdf_validates_params_before_loading(tmp_path):
    """Bad settings fail fast, before any file work. The path doesn't even exist: if the
    PDF were loaded first, this would raise InvalidPDFError instead of ValueError."""
    with pytest.raises(ValueError):
        load_and_chunk_pdf(tmp_path / "missing.pdf", chunk_size=10, chunk_overlap=10)
