"""Read papers, pull out the literature-review fields, and index them for chat.

Two ways in:
  * an uploaded PDF            -> ingest_upload()
  * a paper found by search    -> ingest_library_paper()  (downloads the open-access PDF
                                  if there is one, otherwise uses the abstract)
Both save the paper to the topic and build its chat (RAG) index.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Tuple

import fitz  # PyMuPDF
import requests

import config
import db
import llm
import rag
from models import Extraction, Paper, clean_doi

EXTRACT_PROMPT = """You are a research assistant reading an academic paper for a literature review.
Extract the fields below from the paper. Rules:
- Use only what the paper says. If something is not in the paper, return "" (or [] for lists).
- Write each text field as 2-4 clear sentences, specific (name methods, datasets, numbers).
- "results": include the key quantitative results when given (accuracy, F1, speed-up...).
- "future_work": what the authors say should be done next, plus open problems they mention.
- "limitations": weaknesses the authors admit, or obvious ones from their setup.
- "year": 4 digits. "doi": only if printed in the paper.

{source_note}
"""


# ---------- PDF handling ----------

def pdf_pages(pdf_bytes: bytes) -> List[str]:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return [re.sub(r"[ \t]+", " ", page.get_text("text")) for page in doc]


def _cut_references(text: str) -> str:
    """Drop the reference list: it's long and useless for extraction."""
    m = list(re.finditer(r"\n\s*(references|bibliography)\s*\n", text, flags=re.I))
    if m and m[-1].start() > len(text) * 0.5:
        return text[:m[-1].start()]
    return text


def download_pdf(url: str) -> Optional[bytes]:
    """Fetch an open-access PDF. Returns None if it isn't really a PDF or is too big."""
    try:
        r = requests.get(url, timeout=config.HTTP_TIMEOUT, stream=True,
                         headers={"User-Agent": "research-agent/0.1 (personal research tool)"})
        r.raise_for_status()
        data = r.raw.read(config.MAX_PDF_MB * 1024 * 1024 + 1, decode_content=True)
    except requests.RequestException:
        return None
    if not data.startswith(b"%PDF") or len(data) > config.MAX_PDF_MB * 1024 * 1024:
        return None
    return data


# ---------- Extraction ----------

def _to_extraction(out: dict) -> Extraction:
    clean = {}
    for k, v in (out or {}).items():
        if k not in Extraction.model_fields:
            continue
        is_list = k in ("authors", "datasets", "key_contributions")
        if is_list and isinstance(v, str):
            v = [x.strip() for x in re.split(r"[;,]", v) if x.strip()]
        elif not is_list and not isinstance(v, str):
            v = "" if v is None else str(v)
        clean[k] = v
    return Extraction(**clean)


def _run_extraction(contents: list, cache_id: str) -> Extraction:
    cached = db.cache_get("extract", "gemini", cache_id)
    if cached:
        return Extraction(**cached)
    ex = _to_extraction(llm.gemini_json(contents, schema=Extraction))
    if ex.problem_statement or ex.methodology:   # only cache useful answers
        db.cache_set(ex.model_dump(), "extract", "gemini", cache_id)
    return ex


def _condensed(text: str) -> str:
    """Start (abstract, intro, method) + end (results, conclusion) of the paper."""
    n = config.FALLBACK_TEXT_CHARS
    if len(text) <= n:
        return text
    return text[:n // 2] + "\n\n[... middle of the paper omitted ...]\n\n" + text[-n // 2:]


def _groq_extraction(text: str) -> Extraction:
    fields = ", ".join(f'"{k}"' for k in Extraction.model_fields)
    prompt = (EXTRACT_PROMPT.format(source_note="Paper text (start and end only):")
              + f"\nReturn one JSON object with exactly these keys: {fields}. "
                "authors, datasets and key_contributions are lists of strings; all others are strings.\n\n"
              + _condensed(text))
    return _to_extraction(llm.groq_json_any(prompt))


def extract_from_pdf(pdf_bytes: bytes, pages: List[str]) -> Tuple[Extraction, str]:
    """Returns (extraction, how): how is 'full text', or 'partial text' when Gemini
    was busy and Groq read a condensed version instead."""
    cache_id = hashlib.sha256(pdf_bytes).hexdigest()
    text = "\n".join(pages).strip()
    if len(text) >= config.MIN_TEXT_CHARS:
        text = _cut_references(text)[:config.MAX_PAPER_CHARS]
        prompt = EXTRACT_PROMPT.format(source_note="Paper text:") + "\n" + text
        try:
            return _run_extraction([prompt], cache_id), "full text"
        except llm.GeminiBusy:
            if not config.GROQ_API_KEY:
                raise
            return _groq_extraction(text), "partial text"
    # Little or no text layer -> probably scanned. Only Gemini can read the PDF itself.
    prompt = EXTRACT_PROMPT.format(source_note="The paper is attached as a PDF.")
    return _run_extraction([llm.pdf_part(pdf_bytes), prompt], cache_id), "full text"


def extract_from_abstract(paper: Paper) -> Extraction:
    note = ("Only the title and abstract are available (no full text). Fill what the abstract "
            "supports; leave the rest empty.\n\n"
            f"Title: {paper.title}\nAbstract: {paper.abstract or 'n/a'}")
    try:
        return _run_extraction([EXTRACT_PROMPT.format(source_note=note)], "abs:" + paper.key)
    except llm.GeminiBusy:
        if not config.GROQ_API_KEY:
            raise
        fields = ", ".join(f'"{k}"' for k in Extraction.model_fields)
        return _to_extraction(llm.groq_json_any(
            EXTRACT_PROMPT.format(source_note=note) + f"\nReturn one JSON object with keys: {fields}."))


def _year(value: str) -> Optional[int]:
    m = re.search(r"(19|20)\d{2}", value or "")
    return int(m.group()) if m else None


def _index(paper: Paper, pages: Optional[List[str]]) -> str:
    """Build the chat index; never lose the extraction if indexing fails."""
    try:
        rag.index_paper(paper, pages)
        return "chat-ready" if pages else "chat-ready (summary only)"
    except Exception as e:
        print(f"[research-agent] indexing failed for {paper.title[:60]}: {e}")
        return "not chat-ready yet (service busy; use Prepare papers for chat later)"


# ---------- Public entry points ----------

def ingest_upload(topic_id: int, pdf_bytes: bytes, filename: str) -> Tuple[Paper, str]:
    """Uploaded PDF -> extraction -> saved to topic -> chat index. Returns (paper, note)."""
    pages = pdf_pages(pdf_bytes)
    ex, how = extract_from_pdf(pdf_bytes, pages)
    fallback_title = re.sub(r"[_-]+", " ", filename.rsplit(".", 1)[0]).strip()
    paper = Paper(
        title=ex.title.strip() or fallback_title,
        authors=ex.authors,
        year=_year(ex.year),
        venue=ex.venue or None,
        doi=clean_doi(ex.doi),
        sources=["upload"],
        extraction=ex,
        extracted_from=how,
    )
    db.save_paper(topic_id, paper)
    has_text = len("".join(pages).strip()) >= config.MIN_TEXT_CHARS
    note = _index(paper, pages if has_text else None)
    if how == "partial text":
        note = "quick read of the key sections (re-upload later for a full read); " + note
    return paper, note


def ingest_library_paper(topic_id: int, paper: Paper) -> Tuple[Paper, str]:
    """Paper from search (or already in the library) -> extraction (if missing) + chat index.
    Uses the open-access PDF when it can be downloaded, otherwise the abstract."""
    pdf = download_pdf(paper.pdf_url) if paper.pdf_url else None
    if pdf:
        pages = pdf_pages(pdf)
        if not paper.extraction or paper.extracted_from != "full text":
            paper.extraction, paper.extracted_from = extract_from_pdf(pdf, pages)
        db.save_paper(topic_id, paper)
        return paper, "full text, " + _index(paper, pages)

    if not paper.extraction and paper.abstract:
        paper.extraction = extract_from_abstract(paper)
        paper.extracted_from = "abstract only"
    if not paper.extraction and not paper.abstract:
        return paper, "skipped: no PDF or abstract available. Upload the PDF instead."
    db.save_paper(topic_id, paper)
    how = "abstract only (PDF not accessible)" if paper.extracted_from == "abstract only" else "existing extraction"
    return paper, how + ", " + _index(paper, None)
