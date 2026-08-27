"""
Pydantic schemas shared across the app.
"""
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class PdfKind(str, Enum):
    TYPED_NOTES = "typed_notes"
    SCANNED = "scanned"
    PAST_QUESTIONS = "past_questions"


class MCQOption(BaseModel):
    label: str          # "A", "B", "C", "D"
    text: str


class QuizQuestion(BaseModel):
    question: str
    options: List[MCQOption]
    correct_label: str
    explanation: Optional[str] = None
    source_chunk_id: Optional[int] = None
    source_page: Optional[int] = None
    origin: str          # "generated" or "extracted"


class QuizResponse(BaseModel):
    pdf_kind: PdfKind
    total_questions: int
    questions: List[QuizQuestion]


class TextChunk(BaseModel):
    id: int
    heading: Optional[str] = None
    text: str
    page_start: int
    page_end: int
