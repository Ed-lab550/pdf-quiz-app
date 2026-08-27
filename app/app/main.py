"""
FastAPI backend for Feature 2: PDF upload -> auto-generated quiz.

Endpoint flow:
  POST /quiz/from-pdf
    1. Read uploaded PDF bytes
    2. Extract text (OCR fallback for scanned pages) -> pdf_processor
    3. Detect kind: typed_notes | scanned | past_questions
    4. If past_questions -> extract real questions verbatim
       Else -> chunk text -> generate grounded MCQs per chunk
    5. Return a QuizResponse

Run locally:
    export GEMINI_API_KEY=AIza...
    uvicorn app.main:app --reload
"""
import logging
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.chunker import chunk_pages
from app.models import MCQOption, PdfKind, QuizQuestion, QuizResponse
from app.pdf_processor import process_pdf
from app.quiz_generator import extract_from_past_questions, generate_from_chunk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_quiz")

app = FastAPI(title="PDF Quiz Generator")

# Loosen this in production to your actual frontend origin(s).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_PDF_SIZE_MB = 25
MAX_CHUNKS_TO_PROCESS = 15  # safety cap so one huge PDF can't blow up cost/time


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/quiz/from-pdf", response_model=QuizResponse)
async def quiz_from_pdf(
    file: UploadFile = File(...),
    exam_body: Optional[str] = Form(None),   # e.g. "WAEC", "JAMB", "MBBS"
    subject: Optional[str] = Form(None),     # e.g. "Biology", "Anatomy"
    questions_per_chunk: int = Form(3),
):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported.")

    pdf_bytes = await file.read()
    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise HTTPException(400, f"PDF too large ({size_mb:.1f}MB). Max is {MAX_PDF_SIZE_MB}MB.")

    logger.info("Processing PDF: %s (%.2f MB)", file.filename, size_mb)

    try:
        extraction = process_pdf(pdf_bytes)
    except Exception as e:
        logger.exception("Extraction failed")
        raise HTTPException(500, f"Could not read PDF: {e}")

    if not extraction.full_text.strip():
        raise HTTPException(422, "No readable text found in this PDF, even after OCR.")

    all_questions: list[QuizQuestion] = []

    if extraction.pdf_kind == PdfKind.PAST_QUESTIONS:
        # Extract real questions rather than generating new ones.
        try:
            raw_questions = extract_from_past_questions(extraction.full_text)
        except Exception as e:
            logger.exception("Extraction of past questions failed")
            raise HTTPException(502, f"Question extraction failed: {e}")

        for q in raw_questions:
            if not q.get("correct_label"):
                # Skip questions where we can't verify the correct answer —
                # better to omit than to guess and mislead a student.
                continue
            all_questions.append(
                QuizQuestion(
                    question=q["question"],
                    options=[MCQOption(**opt) for opt in q["options"]],
                    correct_label=q["correct_label"],
                    explanation=q.get("explanation"),
                    origin="extracted",
                )
            )
    else:
        chunks = chunk_pages(extraction.pages)[:MAX_CHUNKS_TO_PROCESS]
        for chunk in chunks:
            try:
                raw_questions = generate_from_chunk(
                    chunk.text, exam_body, subject, questions_per_chunk
                )
            except Exception as e:
                # One bad chunk shouldn't kill the whole quiz — log and continue.
                logger.warning("Generation failed for chunk %s: %s", chunk.id, e)
                continue

            for q in raw_questions:
                all_questions.append(
                    QuizQuestion(
                        question=q["question"],
                        options=[MCQOption(**opt) for opt in q["options"]],
                        correct_label=q["correct_label"],
                        explanation=q.get("explanation"),
                        source_chunk_id=chunk.id,
                        source_page=chunk.page_start,
                        origin="generated",
                    )
                )

    if not all_questions:
        raise HTTPException(
            422, "Could not generate any questions from this PDF's content."
        )

    return QuizResponse(
        pdf_kind=extraction.pdf_kind,
        total_questions=len(all_questions),
        questions=all_questions,
    )
