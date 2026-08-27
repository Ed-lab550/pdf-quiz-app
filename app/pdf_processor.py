"""
Handles Step 1 of the pipeline:
  - Extract text from a PDF (typed PDFs)
  - Fall back to OCR for scanned/image-based PDFs
  - Detect whether the PDF is notes/textbook content or a past-questions paper

Design notes:
  - We never guess silently. Every function returns enough info for the
    caller (main.py) to log/inspect what path was taken, which matters
    when debugging why a quiz came out wrong.
  - OCR is only triggered when normal extraction yields too little text,
    so typed PDFs stay fast and cheap.
"""
import io
import re
from dataclasses import dataclass
from typing import List, Tuple

import pdfplumber
from pdf2image import convert_from_bytes
import pytesseract

from app.models import PdfKind

# If a page yields fewer than this many characters of extracted text,
# we treat it as an image/scanned page and OCR it instead.
MIN_CHARS_PER_PAGE = 40

# Regex patterns that strongly suggest a "past questions" exam paper
# rather than plain textbook notes.
PAST_QUESTION_PATTERNS = [
    r"\bSECTION\s+[A-D]\b",
    r"\bINSTRUCTION(S)?\s*:",
    r"^\s*\d{1,2}[\.\)]\s+.{0,200}\?",   # "1. What is ...?"
    r"\([a-d]\)\s",                       # "(a) ... (b) ..."
    r"\bTIME\s*ALLOWED\b",
    r"\bOBJ(ECTIVE)?\s*TEST\b",
]


@dataclass
class PageResult:
    page_number: int          # 1-indexed
    text: str
    was_ocr: bool


@dataclass
class ExtractionResult:
    pages: List[PageResult]
    full_text: str
    pdf_kind: PdfKind


def _ocr_page(pdf_bytes: bytes, page_number: int) -> str:
    """OCR a single page. page_number is 1-indexed."""
    images = convert_from_bytes(
        pdf_bytes, first_page=page_number, last_page=page_number, dpi=300
    )
    if not images:
        return ""
    return pytesseract.image_to_string(images[0])


def extract_text(pdf_bytes: bytes) -> List[PageResult]:
    """
    Extract text page by page. Uses pdfplumber first; any page with too
    little text is re-processed with OCR (handles scanned/mixed PDFs).
    """
    results: List[PageResult] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            if len(text) < MIN_CHARS_PER_PAGE:
                ocr_text = _ocr_page(pdf_bytes, i).strip()
                # Keep whichever is longer — sometimes pdfplumber gets
                # a little text even on a mostly-scanned page.
                if len(ocr_text) > len(text):
                    results.append(PageResult(i, ocr_text, was_ocr=True))
                    continue
            results.append(PageResult(i, text, was_ocr=False))
    return results


def detect_pdf_kind(pages: List[PageResult]) -> PdfKind:
    """
    Decide whether this looks like a past-questions paper or
    plain notes/textbook content. Scanned status is tracked separately
    per page (was_ocr) so callers know OCR was used regardless of kind.
    """
    full_text = "\n".join(p.text for p in pages)
    hits = 0
    for pattern in PAST_QUESTION_PATTERNS:
        if re.search(pattern, full_text, flags=re.IGNORECASE | re.MULTILINE):
            hits += 1

    # Require at least 2 independent signals before calling it a past
    # question paper — a single "(a)" or "SECTION A" could appear in notes.
    if hits >= 2:
        return PdfKind.PAST_QUESTIONS

    if any(p.was_ocr for p in pages):
        return PdfKind.SCANNED

    return PdfKind.TYPED_NOTES


def process_pdf(pdf_bytes: bytes) -> ExtractionResult:
    pages = extract_text(pdf_bytes)
    kind = detect_pdf_kind(pages)
    full_text = "\n\n".join(f"[PAGE {p.page_number}]\n{p.text}" for p in pages)
    return ExtractionResult(pages=pages, full_text=full_text, pdf_kind=kind)
