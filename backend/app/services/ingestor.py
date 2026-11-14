"""PDF ingestion: load a PDF and turn each page into a LangChain Document.

This is the first stage of the ingestion pipeline:

    PDF file  ->  [load_pdf]  ->  list[Document]  ->  (later) chunking -> embedding -> ChromaDB

We call `pypdf` directly instead of LangChain's `PyPDFLoader`. PyPDFLoader lived in
`langchain-community`, which was sunset in 2026. It was a thin wrapper anyway: it ran pypdf
and wrapped each page in a `Document`, which is exactly what this module does, with
error handling we control.
"""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger(__name__)

# The PDF spec allows the "%PDF-" header to appear anywhere in the first 1024 bytes.
_PDF_MAGIC = b"%PDF-"
_HEADER_SEARCH_BYTES = 1024


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------
# One base class lets callers catch "any PDF problem" in one place, while the subclasses
# let a future FastAPI route map each case to a specific, helpful HTTP error message.


class PDFLoadError(Exception):
    """Base class for every error raised while loading a PDF."""


class InvalidPDFError(PDFLoadError):
    """The file is missing, not a PDF, corrupted, or password-protected."""


class EmptyPDFError(PDFLoadError):
    """The PDF is valid but has zero pages."""


class ScannedPDFError(PDFLoadError):
    """The PDF has pages but no extractable text (likely scanned / image-only)."""


# --------------------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------------------


def load_pdf(file_path: str | Path) -> list[Document]:
    """Load a PDF and return one Document per page that contains text.

    Each Document has:
        page_content: the page's extracted text (whitespace-trimmed)
        metadata:     source, file_name, page (1-based), total_pages,
                      plus title/author when the PDF provides them

    Pages with no text are skipped (with a warning), because empty pages would only
    produce empty chunks and useless embeddings later.

    Raises:
        InvalidPDFError: file missing, not a PDF, corrupted, or encrypted.
        EmptyPDFError:   the PDF has no pages.
        ScannedPDFError: no page has extractable text.
    """
    path = Path(file_path)
    _validate_file(path)
    reader, total_pages = _open_reader(path)

    if total_pages == 0:
        raise EmptyPDFError(f"'{path.name}' contains no pages.")

    base_metadata = _document_metadata(reader, path, total_pages)

    documents: list[Document] = []
    skipped_pages: list[int] = []

    for index, page in enumerate(reader.pages):
        page_number = index + 1  # 1-based, so it matches what a human sees in a PDF viewer
        text = _extract_page_text(page, page_number, path.name)

        if not text:
            skipped_pages.append(page_number)
            continue

        documents.append(
            Document(
                page_content=text,
                # Copy the shared metadata so each page gets its own dict.
                metadata={**base_metadata, "page": page_number},
            )
        )

    if not documents:
        raise ScannedPDFError(
            f"No extractable text found in '{path.name}' ({total_pages} page(s)). "
            "It is probably a scanned or image-only PDF. OCR is not supported yet."
        )

    if skipped_pages:
        logger.warning(
            "Skipped %d page(s) with no text in '%s': %s",
            len(skipped_pages),
            path.name,
            skipped_pages,
        )

    logger.info(
        "Loaded '%s': %d of %d page(s) contain text.",
        path.name,
        len(documents),
        total_pages,
    )
    return documents


# --------------------------------------------------------------------------------------
# Helpers (private: the leading underscore means "not part of this module's API")
# --------------------------------------------------------------------------------------


def _validate_file(path: Path) -> None:
    """Cheap checks before handing the file to the PDF parser."""
    if not path.is_file():
        raise InvalidPDFError(f"File not found: '{path}'.")

    if path.suffix.lower() != ".pdf":
        raise InvalidPDFError(f"'{path.name}' does not have a .pdf extension.")

    # The extension can lie (e.g. a renamed .txt). Real PDFs start with "%PDF-".
    try:
        with path.open("rb") as f:
            header = f.read(_HEADER_SEARCH_BYTES)
    except OSError as exc:  # e.g. PermissionError
        raise InvalidPDFError(f"Cannot read '{path.name}': {exc}") from exc
    if _PDF_MAGIC not in header:
        raise InvalidPDFError(f"'{path.name}' is not a valid PDF (missing PDF header).")


def _open_reader(path: Path) -> tuple[PdfReader, int]:
    """Open the PDF with pypdf and return (reader, page count).

    pypdf parses lazily: creating a PdfReader only reads the file's table of contents.
    The page tree isn't parsed until `reader.pages` is first used, so a broken page tree
    would fail *later*, outside this error handling. We count the pages here on purpose,
    so every structural error surfaces in one place as an InvalidPDFError.
    """
    try:
        reader = PdfReader(path)

        if reader.is_encrypted:
            # Many "encrypted" PDFs only restrict printing/editing and open with an empty
            # password. Try that first; anything that needs a real password is rejected.
            # This must happen before reading pages, which are encrypted too.
            try:
                unlocked = reader.decrypt("")
            except (PyPdfError, NotImplementedError) as exc:
                raise InvalidPDFError(
                    f"'{path.name}' is encrypted and could not be opened: {exc}"
                ) from exc
            if not unlocked:
                raise InvalidPDFError(f"'{path.name}' is password-protected.")

        total_pages = len(reader.pages)  # forces the page tree to be parsed now
    except PyPdfError as exc:  # base class of every pypdf error, incl. PdfReadError
        # `from exc` keeps the original pypdf error attached for debugging.
        raise InvalidPDFError(f"'{path.name}' is corrupted or unreadable: {exc}") from exc

    return reader, total_pages


def _document_metadata(reader: PdfReader, path: Path, total_pages: int) -> dict:
    """Metadata shared by every page of this document."""
    metadata: dict = {
        "source": str(path),
        "file_name": path.name,
        "total_pages": total_pages,
    }

    # Optional fields embedded in the PDF itself. Only add them if they have a value:
    # vector databases (like ChromaDB, later) reject None as a metadata value.
    try:
        info = reader.metadata
    except PyPdfError:
        info = None  # A broken info dictionary shouldn't stop us from reading the text.
    if info:
        if info.title:
            metadata["title"] = str(info.title).strip()
        if info.author:
            metadata["author"] = str(info.author).strip()

    return metadata


def _extract_page_text(page, page_number: int, file_name: str) -> str:
    """Extract and trim one page's text. A single broken page shouldn't fail the whole PDF."""
    try:
        text = page.extract_text() or ""
    except Exception as exc:  # pypdf can raise many different errors on malformed pages
        logger.warning(
            "Could not extract text from page %d of '%s': %s", page_number, file_name, exc
        )
        return ""
    return text.strip()


# --------------------------------------------------------------------------------------
# Manual check:  cd backend && python -m app.services.ingestor path/to/file.pdf
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if len(sys.argv) != 2:
        print("Usage: python -m app.services.ingestor <path-to-pdf>")
        sys.exit(1)

    try:
        docs = load_pdf(sys.argv[1])
    except PDFLoadError as err:
        print(f"Error ({type(err).__name__}): {err}")
        sys.exit(1)

    for doc in docs:
        preview = doc.page_content[:150].replace("\n", " ")
        print(f"\n--- page {doc.metadata['page']} / {doc.metadata['total_pages']} ---")
        print(f"metadata: {doc.metadata}")
        print(f"text:     {preview}{'...' if len(doc.page_content) > 150 else ''}")
