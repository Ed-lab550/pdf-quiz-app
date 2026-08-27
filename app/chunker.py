"""
Step 2: Chunking.

Goal: break extracted text into topic-sized pieces so quiz generation
stays focused and each question can be traced back to a page.

Strategy:
  1. Try heading-based splitting first (lines that look like headings:
     short, title-cased or ALL CAPS, not ending in a full stop).
  2. If that produces too few/too uneven chunks (common in badly-OCR'd
     text), fall back to fixed-size word chunking while still tracking
     which page each chunk started/ended on.
"""
import re
from typing import List

from app.models import TextChunk
from app.pdf_processor import PageResult

HEADING_RE = re.compile(
    r"^(?=.{3,80}$)(?!.*[.?!]$)([A-Z][A-Za-z0-9 ,\-&()]+)$"
)

FALLBACK_WORDS_PER_CHUNK = 350


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if not HEADING_RE.match(line):
        return False
    # Mostly capitalized words, not a full sentence
    words = line.split()
    if len(words) > 10:
        return False
    return True


def _heading_split(pages: List[PageResult]) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    current_heading = None
    current_text: List[str] = []
    current_start_page = pages[0].page_number if pages else 1
    chunk_id = 0

    def flush(end_page: int):
        nonlocal chunk_id, current_text, current_heading, current_start_page
        text = "\n".join(current_text).strip()
        if len(text) > 30:  # skip near-empty fragments
            chunks.append(
                TextChunk(
                    id=chunk_id,
                    heading=current_heading,
                    text=text,
                    page_start=current_start_page,
                    page_end=end_page,
                )
            )
            chunk_id += 1
        current_text = []

    for page in pages:
        for line in page.text.splitlines():
            if _looks_like_heading(line):
                if current_text:
                    flush(page.page_number)
                current_heading = line.strip()
                current_start_page = page.page_number
            else:
                current_text.append(line)

    flush(pages[-1].page_number if pages else current_start_page)
    return chunks


def _fixed_size_split(pages: List[PageResult]) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    buffer_words: List[str] = []
    start_page = pages[0].page_number if pages else 1
    chunk_id = 0

    for page in pages:
        for word in page.text.split():
            buffer_words.append(word)
            if len(buffer_words) >= FALLBACK_WORDS_PER_CHUNK:
                chunks.append(
                    TextChunk(
                        id=chunk_id,
                        heading=None,
                        text=" ".join(buffer_words),
                        page_start=start_page,
                        page_end=page.page_number,
                    )
                )
                chunk_id += 1
                buffer_words = []
                start_page = page.page_number

    if buffer_words:
        chunks.append(
            TextChunk(
                id=chunk_id,
                heading=None,
                text=" ".join(buffer_words),
                page_start=start_page,
                page_end=pages[-1].page_number if pages else start_page,
            )
        )
    return chunks


def chunk_pages(pages: List[PageResult]) -> List[TextChunk]:
    if not pages:
        return []

    heading_chunks = _heading_split(pages)
    # Heuristic sanity check: if heading splitting produced almost
    # nothing (e.g. 1 giant chunk) or an excessive number of tiny
    # fragments, fall back to fixed-size chunking instead.
    total_words = sum(len(p.text.split()) for p in pages)
    if not heading_chunks or (
        len(heading_chunks) < max(1, total_words // 1200)
    ):
        return _fixed_size_split(pages)

    return heading_chunks
