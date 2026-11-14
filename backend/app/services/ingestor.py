"""PDF ingestion: load a PDF, then split its pages into retrieval-sized chunks.

This module covers the first two stages of the ingestion pipeline:

    PDF file -> [load_pdf] -> page Documents -> [chunk_documents] -> chunk Documents
             -> (later) embedding -> ChromaDB

`load_and_chunk_pdf` runs both stages in one call.

We call `pypdf` directly instead of LangChain's `PyPDFLoader`. PyPDFLoader lived in
`langchain-community`, which was sunset in 2026. It was a thin wrapper anyway: it ran pypdf
and wrapped each page in a `Document`, which is exactly what this module does, with
error handling we control.
"""

from __future__ import annotations

import logging
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from pypdf.errors import PyPdfError

logger = logging.getLogger(__name__)

# The PDF spec allows the "%PDF-" header to appear anywhere in the first 1024 bytes.
_PDF_MAGIC = b"%PDF-"
_HEADER_SEARCH_BYTES = 1024

# --- Chunking defaults ----------------------------------------------------------------
# Sizes are in CHARACTERS, not tokens (roughly 4 characters per English token).
# 1000 characters is about 150-250 words: usually one complete idea. That's big enough to
# answer a question on its own, but small enough that its embedding stays about one topic.
DEFAULT_CHUNK_SIZE = 1000
# 20% overlap: a sentence that falls on a boundary appears whole in at least one chunk.
DEFAULT_CHUNK_OVERLAP = 200

# Where the splitter is allowed to cut, from most to least preferred. It uses the first
# separator that keeps chunks under the size limit, and only falls back to the next one
# for pieces that are still too long.
#   "\n\n" paragraph break -> ". " sentence end -> "\n" line break -> " " word -> "" any char
# Sentence ends come BEFORE single newlines on purpose: pypdf puts "\n" at every printed
# line wrap, usually mid-sentence, so cutting there would split sentences in half.
_CHUNK_SEPARATORS = ["\n\n", ". ", "\n", " ", ""]


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
        page_content: the page's extracted text, Unicode-normalized (NFKC) and trimmed
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


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split page Documents into smaller, overlapping chunk Documents.

    Each page is split on its own, so a chunk never spans two pages. That keeps the
    `page` metadata exact, which later citations ("page 7 of report.pdf") depend on.

    Every chunk keeps its page's metadata and gains:
        chunk_index: position of the chunk within its source file (0, 1, 2, ...)
        start_index: character offset where the chunk starts within its page's text

    Raises:
        ValueError: if chunk_size or chunk_overlap is invalid.
    """
    _validate_chunk_params(chunk_size, chunk_overlap)

    documents = list(documents)
    if not documents:
        logger.warning("chunk_documents() received no documents; returning no chunks.")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_CHUNK_SEPARATORS,
        keep_separator="end",  # keep the "." with its sentence, not at the next chunk's start
        add_start_index=True,  # record each chunk's offset within its page
        strip_whitespace=True,
    )
    # split_documents copies each page's metadata into every chunk it produces, so the
    # input pages are left untouched.
    chunks = splitter.split_documents(documents)

    # Number chunks per source file, so chunks from different PDFs each start at 0.
    next_index: Counter[str] = Counter()
    for chunk in chunks:
        source = str(chunk.metadata.get("source", ""))
        chunk.metadata["chunk_index"] = next_index[source]
        next_index[source] += 1

    logger.info(
        "Split %d page(s) into %d chunk(s) (chunk_size=%d, chunk_overlap=%d).",
        len(documents),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def load_and_chunk_pdf(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Load a PDF and split it into chunks: the full ingestion pipeline so far.

    Raises the same errors as `load_pdf` and `chunk_documents`.
    """
    # Check the settings first, so a config mistake fails instantly instead of after
    # parsing a possibly large PDF.
    _validate_chunk_params(chunk_size, chunk_overlap)
    return chunk_documents(load_pdf(file_path), chunk_size, chunk_overlap)


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
            metadata["title"] = _normalize_text(str(info.title)).strip()
        if info.author:
            metadata["author"] = _normalize_text(str(info.author)).strip()

    return metadata


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Reject bad settings up front with a clear message.

    These are programming or configuration mistakes, not problems with a PDF, so they
    raise ValueError rather than PDFLoadError.
    """
    # bool is a subclass of int in Python, so exclude it explicitly (chunk_size=True is a bug).
    for name, value in (("chunk_size", chunk_size), ("chunk_overlap", chunk_overlap)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer, got {value!r}.")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap cannot be negative, got {chunk_overlap}.")
    # Overlap equal to the size would make every chunk repeat the previous one.
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be smaller than chunk_size ({chunk_size})."
        )


def _extract_page_text(page, page_number: int, file_name: str) -> str:
    """Extract and trim one page's text. A single broken page shouldn't fail the whole PDF."""
    try:
        text = page.extract_text() or ""
    except Exception as exc:  # pypdf can raise many different errors on malformed pages
        logger.warning(
            "Could not extract text from page %d of '%s': %s", page_number, file_name, exc
        )
        return ""
    return _normalize_text(text).strip()


def _normalize_text(text: str) -> str:
    """Replace "compatibility" characters with their plain equivalents (Unicode NFKC).

    Why: PDFs often store typographic ligatures as single characters, so pypdf returns
    "deﬁnes" (with one "ﬁ" character, U+FB01) instead of "defines". That text would never
    match a search for "defines", and it gives slightly different embeddings. NFKC fixes
    ligatures (ﬁ ﬂ ﬀ ﬃ ﬄ), non-breaking spaces, full-width letters and "…".

    It leaves accented letters (café), curly quotes, dashes and symbols unchanged.
    Trade-off: superscripts and subscripts become plain digits ("x²" -> "x2"), which is
    acceptable for question answering over documents.
    """
    return unicodedata.normalize("NFKC", text)


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
        chunks = load_and_chunk_pdf(sys.argv[1])
    except PDFLoadError as err:
        print(f"Error ({type(err).__name__}): {err}")
        sys.exit(1)

    for chunk in chunks:
        meta = chunk.metadata
        preview = chunk.page_content[:120].replace("\n", " ")
        print(
            f"\n--- chunk {meta['chunk_index']} | page {meta['page']}/{meta['total_pages']}"
            f" | start {meta['start_index']} | {len(chunk.page_content)} chars ---"
        )
        print(f"text: {preview}{'...' if len(chunk.page_content) > 120 else ''}")
