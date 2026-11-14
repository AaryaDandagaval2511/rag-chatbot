"""Tests for app.services.ingestor.load_pdf.

Each test builds its own tiny PDF in a temporary folder (pytest's `tmp_path`),
so the tests don't depend on any files outside the repo.
"""

import logging
import os

import pytest
from langchain_core.documents import Document
from pypdf import PdfWriter

from app.services.ingestor import (
    EmptyPDFError,
    InvalidPDFError,
    PDFLoadError,
    ScannedPDFError,
    load_pdf,
)

# --------------------------------------------------------------------------------------
# Helpers to build test PDFs
# --------------------------------------------------------------------------------------


def make_pdf(path, pages, title=None):
    """Write a minimal, valid PDF by hand.

    `pages` is a list where each item is the text for that page, or None for a blank
    page with no text layer (which is what a scanned page looks like to a text extractor).
    """
    objects = []  # PDF object bodies; object number = index + 1

    def add(body: str) -> int:
        objects.append(body)
        return len(objects)

    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages_obj = add("PLACEHOLDER")  # filled in once we know the page object numbers
    page_ids = []
    for text in pages:
        if text is None:
            contents = ""
        else:
            safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            stream = f"BT /F1 12 Tf 72 720 Td ({safe}) Tj ET"
            stream_id = add(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
            contents = f" /Contents {stream_id} 0 R"
        page_ids.append(
            add(
                f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 612 792]"
                f" /Resources << /Font << /F1 {font} 0 R >> >>{contents} >>"
            )
        )
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects[pages_obj - 1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>"
    catalog = add(f"<< /Type /Catalog /Pages {pages_obj} 0 R >>")
    info = add(f"<< /Title ({title}) >>") if title else None

    # Serialize, recording each object's byte offset for the cross-reference table.
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref_start = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    trailer = f"<< /Size {len(objects) + 1} /Root {catalog} 0 R"
    trailer += f" /Info {info} 0 R >>" if info else " >>"
    out += f"trailer\n{trailer}\nstartxref\n{xref_start}\n%%EOF\n".encode()

    path.write_bytes(out)
    return path


# --------------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------------


def test_loads_one_document_per_page(tmp_path):
    pdf = make_pdf(tmp_path / "report.pdf", ["Hello page one", "Second page text"])

    docs = load_pdf(pdf)

    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    assert docs[0].page_content == "Hello page one"
    assert docs[1].page_content == "Second page text"


def test_metadata_is_populated(tmp_path):
    pdf = make_pdf(tmp_path / "report.pdf", ["Page A", "Page B"], title="Annual Report")

    docs = load_pdf(pdf)

    assert docs[0].metadata == {
        "source": str(pdf),
        "file_name": "report.pdf",
        "total_pages": 2,
        "title": "Annual Report",
        "page": 1,  # 1-based
    }
    assert docs[1].metadata["page"] == 2


def test_missing_optional_metadata_is_omitted_not_none(tmp_path):
    pdf = make_pdf(tmp_path / "plain.pdf", ["Some text"])

    metadata = load_pdf(pdf)[0].metadata

    assert "title" not in metadata
    assert "author" not in metadata
    assert None not in metadata.values()


def test_pages_do_not_share_metadata_dict(tmp_path):
    pdf = make_pdf(tmp_path / "doc.pdf", ["One", "Two"])

    docs = load_pdf(pdf)
    docs[0].metadata["page"] = 99

    assert docs[1].metadata["page"] == 2


def test_accepts_str_path_and_uppercase_extension(tmp_path):
    pdf = make_pdf(tmp_path / "UPPER.PDF", ["Text"])

    assert len(load_pdf(str(pdf))) == 1


# --------------------------------------------------------------------------------------
# Blank and scanned pages
# --------------------------------------------------------------------------------------


def test_blank_pages_are_skipped_with_warning(tmp_path, caplog):
    pdf = make_pdf(tmp_path / "mixed.pdf", ["First", None, "Third"])

    with caplog.at_level(logging.WARNING):
        docs = load_pdf(pdf)

    assert [d.metadata["page"] for d in docs] == [1, 3]
    assert docs[0].metadata["total_pages"] == 3
    assert "Skipped 1 page(s)" in caplog.text


def test_pdf_with_no_text_raises_scanned_error(tmp_path):
    pdf = make_pdf(tmp_path / "scan.pdf", [None, None])

    with pytest.raises(ScannedPDFError, match="scanned"):
        load_pdf(pdf)


def test_whitespace_only_page_counts_as_empty(tmp_path):
    pdf = make_pdf(tmp_path / "spaces.pdf", ["     "])

    with pytest.raises(ScannedPDFError):
        load_pdf(pdf)


# --------------------------------------------------------------------------------------
# Empty and invalid files
# --------------------------------------------------------------------------------------


def test_zero_page_pdf_raises_empty_error(tmp_path):
    pdf = tmp_path / "empty.pdf"
    with pdf.open("wb") as f:
        PdfWriter().write(f)

    with pytest.raises(EmptyPDFError):
        load_pdf(pdf)


def test_missing_file_raises_invalid(tmp_path):
    with pytest.raises(InvalidPDFError, match="not found"):
        load_pdf(tmp_path / "nope.pdf")


def test_directory_path_raises_invalid(tmp_path):
    folder = tmp_path / "folder.pdf"
    folder.mkdir()

    with pytest.raises(InvalidPDFError, match="not found"):
        load_pdf(folder)


def test_wrong_extension_raises_invalid(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")

    with pytest.raises(InvalidPDFError, match=".pdf extension"):
        load_pdf(txt)


def test_renamed_non_pdf_raises_invalid(tmp_path):
    fake = tmp_path / "fake.pdf"
    fake.write_text("This is just text pretending to be a PDF.")

    with pytest.raises(InvalidPDFError, match="missing PDF header"):
        load_pdf(fake)


def test_zero_byte_file_raises_invalid(tmp_path):
    empty = tmp_path / "zero.pdf"
    empty.write_bytes(b"")

    with pytest.raises(InvalidPDFError):
        load_pdf(empty)


def test_corrupted_pdf_raises_invalid(tmp_path):
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4\nthis is not a real pdf body at all\n")

    with pytest.raises(InvalidPDFError):
        load_pdf(broken)


def test_broken_page_tree_raises_invalid(tmp_path):
    """Regression: pypdf parses the page tree lazily, so this used to escape as a raw
    pypdf PdfReadError instead of our InvalidPDFError."""
    header = b"%PDF-1.4\n"
    catalog = b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
    not_a_page_tree = b"2 0 obj\n42\nendobj\n"  # /Pages must be a dictionary, not a number
    body = header + catalog + not_a_page_tree
    xref_start = len(body)
    body += (
        b"xref\n0 3\n0000000000 65535 f \n"
        + f"{len(header):010d} 00000 n \n".encode()
        + f"{len(header) + len(catalog):010d} 00000 n \n".encode()
        + b"trailer\n<< /Size 3 /Root 1 0 R >>\n"
        + f"startxref\n{xref_start}\n%%EOF\n".encode()
    )
    bad = tmp_path / "badtree.pdf"
    bad.write_bytes(body)

    with pytest.raises(InvalidPDFError, match="corrupted or unreadable"):
        load_pdf(bad)


@pytest.mark.skipif(
    hasattr(os, "geteuid") and os.geteuid() == 0,
    reason="root ignores file permissions (e.g. inside Docker), so this can't fail there",
)
def test_unreadable_file_raises_invalid(tmp_path):
    """Regression: used to escape as a raw PermissionError."""
    pdf = make_pdf(tmp_path / "locked_down.pdf", ["Text"])
    pdf.chmod(0o000)
    try:
        with pytest.raises(InvalidPDFError, match="Cannot read"):
            load_pdf(pdf)
    finally:
        pdf.chmod(0o644)  # let pytest clean up the temp folder


def test_password_protected_pdf_raises_invalid(tmp_path):
    source = make_pdf(tmp_path / "plain.pdf", ["Secret text"])
    writer = PdfWriter(clone_from=source)
    writer.encrypt(user_password="hunter2", algorithm="RC4-128")
    locked = tmp_path / "locked.pdf"
    with locked.open("wb") as f:
        writer.write(f)

    with pytest.raises(InvalidPDFError, match="password-protected"):
        load_pdf(locked)


def test_owner_only_encryption_is_readable(tmp_path):
    """Encrypted with an empty user password (only restricts permissions) -> still loads."""
    source = make_pdf(tmp_path / "plain.pdf", ["Readable text"])
    writer = PdfWriter(clone_from=source)
    writer.encrypt(user_password="", owner_password="owner", algorithm="RC4-128")
    restricted = tmp_path / "restricted.pdf"
    with restricted.open("wb") as f:
        writer.write(f)

    docs = load_pdf(restricted)

    assert docs[0].page_content == "Readable text"


def test_all_errors_share_base_class():
    for error in (InvalidPDFError, EmptyPDFError, ScannedPDFError):
        assert issubclass(error, PDFLoadError)
