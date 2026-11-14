"""Build small, valid PDF files by hand for tests (no PDF library needed)."""


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
