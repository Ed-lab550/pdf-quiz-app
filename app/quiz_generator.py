"""
Step 3 & 4: Turn chunks into quiz questions.

Uses Google's Gemini API (free tier: Flash / Flash-Lite models) instead
of a paid-only provider. Swap GEMINI_MODEL below if you later upgrade.

Two distinct paths, matching the product decision:
  - generate_from_chunk(): for typed_notes / scanned content.
    The model is instructed to ONLY use facts present in the given
    text, to reduce hallucination. Returns strict JSON we can parse.
  - extract_from_past_questions(): for past_questions PDFs.
    The model extracts real questions verbatim/near-verbatim rather
    than inventing new ones, preserving exam authenticity.

Both return a list of dicts matching the QuizQuestion schema fields
(minus source_chunk_id/source_page, which the caller fills in).
"""
import json
import os
from typing import List, Optional

from google import genai
from google.genai import types

# Free-tier eligible model as of 2026. If you later move to a paid
# project, you can swap this for "gemini-2.5-pro" without changing
# anything else in this file.
GEMINI_MODEL = "gemini-flash-latest"

_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it before starting the server."
            )
        _client = genai.Client(api_key=api_key)
    return _client


GENERATE_SYSTEM_PROMPT = """You are a quiz-writing assistant for an exam-focused study app (e.g. WAEC, JAMB, MBBS style).

Rules you must follow strictly:
1. Only use facts that are explicitly stated in the passage given to you. Do not add outside knowledge, even if you know it's true.
2. If the passage does not contain enough clear factual content to write a question, return an empty "questions" array. Do not force a question out of thin material.
3. Write questions in the tone/format typical of the specified exam body (e.g. WAEC/JAMB style is direct and fact-testing; MBBS style may include clinical correlation if the passage supports it).
4. Each question must have exactly 4 options, only one correct.
5. Wrong options (distractors) must be plausible within the same topic — not random or obviously silly.
6. Provide a one-sentence explanation for the correct answer, grounded only in the passage.
7. Respond with ONLY valid JSON, no markdown fences, no commentary, matching this shape exactly:

{
  "questions": [
    {
      "question": "string",
      "options": [{"label": "A", "text": "string"}, {"label": "B", "text": "string"}, {"label": "C", "text": "string"}, {"label": "D", "text": "string"}],
      "correct_label": "A",
      "explanation": "string"
    }
  ]
}
"""

EXTRACT_SYSTEM_PROMPT = """You are extracting real exam questions from a past-questions paper (already written by an exam body such as WAEC, JAMB, or a university).

Rules you must follow strictly:
1. Do NOT invent new questions. Only extract questions that literally appear in the given text.
2. Preserve the original wording of each question and its options as closely as possible.
3. If the correct answer is not indicated in the text, leave "correct_label" as null — do not guess.
4. If a question does not have exactly 4 options in the source, skip it (this pipeline only supports 4-option MCQs for now).
5. Respond with ONLY valid JSON, no markdown fences, no commentary, matching this shape exactly:

{
  "questions": [
    {
      "question": "string",
      "options": [{"label": "A", "text": "string"}, {"label": "B", "text": "string"}, {"label": "C", "text": "string"}, {"label": "D", "text": "string"}],
      "correct_label": "A or null",
      "explanation": null
    }
  ]
}
"""


def _call_gemini(system_prompt: str, user_content: str) -> dict:
    client = get_client()
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            temperature=0.4,
        ),
    )
    raw_text = (response.text or "").strip()

    # Defensive cleanup in case the model wraps output in fences anyway.
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.lower().startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output: {raw_text[:500]}"
        ) from e


def generate_from_chunk(
    chunk_text: str, exam_body: Optional[str], subject: Optional[str], num_questions: int = 3
) -> List[dict]:
    context_line = ""
    if exam_body or subject:
        context_line = f"Target exam body: {exam_body or 'general'}. Subject: {subject or 'unspecified'}.\n\n"

    user_content = (
        f"{context_line}"
        f"Generate up to {num_questions} multiple-choice questions from the passage below. "
        f"Follow all system rules exactly.\n\n"
        f"PASSAGE:\n{chunk_text}"
    )
    data = _call_gemini(GENERATE_SYSTEM_PROMPT, user_content)
    return data.get("questions", [])


def extract_from_past_questions(section_text: str) -> List[dict]:
    user_content = f"Extract all valid 4-option MCQs from this past questions text:\n\n{section_text}"
    data = _call_gemini(EXTRACT_SYSTEM_PROMPT, user_content)
    return data.get("questions", [])
