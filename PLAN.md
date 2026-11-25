# Research Agent — Project Plan

A personal research copilot: give it a topic, it finds relevant (IEEE-focused) papers,
you add or upload papers, it extracts the literature-review fields of each one, and it writes
a one-page synopsis of your own research idea grounded in those papers.

## Constraints
- **Free only.** No paid APIs, no paid hosting.
- **No local LLMs.** All LLM work goes through free API tiers (Groq, Google AI Studio).
- **Single user**, runs locally. **Domain:** IEEE. **UI:** Streamlit.

## Tech stack

| Layer | Choice |
|---|---|
| Fast LLM (relevance ranking) | Groq `openai/gpt-oss-120b` |
| Long-context LLM (paper extraction, synopsis) | Gemini Flash (Google AI Studio free tier) |
| Paper search | OpenAlex, Semantic Scholar, arXiv, IEEE Xplore (optional key) |
| PDF reading | PyMuPDF text; scanned PDFs are sent to Gemini directly |
| Storage | SQLite (topics, papers + extractions, synopses, API cache) |
| Export | python-docx (synopsis), CSV (review matrix), Markdown |
| UI | Streamlit with custom theme/CSS |

## Status

### ✅ Done
- **Discover:** multi-source search, dedupe, IEEE / open-access filters, AI relevance ranking,
  add one or all results to the library.
- **Upload:** up to 30 PDFs per batch, progress per file, scanned-PDF fallback.
- **Extraction:** title, authors, year, venue, DOI, problem statement, methodology, results,
  future work, limitations, datasets, key contributions. Uses the full text when the PDF is
  uploaded or open access, otherwise the abstract (marked "abstract" in the UI).
- **Library:** filter, extract one / all, export a literature review matrix (CSV).
- **Synopsis:** your idea + chosen papers → one page in a fixed format (below), preview,
  edit, save, history, download .docx / .md.

- **Chat (RAG):** section/page-aware chunks + summary chunk per paper, Gemini embeddings in
  SQLite, hybrid search (vectors + BM25, Reciprocal Rank Fusion, max 3 chunks per paper),
  follow-up question rewriting, cited answers with a sources panel, scope to chosen papers,
  saved history per topic. Groq answers; Gemini is the fallback.
- **Keys** are read only from the local `.env`.

### Synopsis format (fixed)
1. Title + Keywords
2. Background & Motivation (~80 words)
3. Literature Review (~150 words, cites papers)
4. Research Gap (~70 words, cites papers)
5. Problem Statement (~50 words)
6. Objectives (3–4 bullets)
7. Proposed Methodology (~130 words)
8. Expected Outcomes (3 bullets)
9. Novelty (~50 words)
10. References (IEEE style, built from paper metadata, only papers actually cited)

### Next (later)
- Research-gap explorer across all papers.
- Slide deck from the synopsis.
