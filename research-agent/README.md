#  Research Agent

Find IEEE-focused papers on a topic, upload your own PDFs (up to 30 at a time), extract the
problem / method / results / future work of every paper, and generate a one-page synopsis of
your research idea grounded in that literature. Uses only free APIs. See `PLAN.md`.

## Setup
```bash
cd research-agent
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# Put your keys in .env (GROQ_API_KEY and GOOGLE_API_KEY), then:
.venv/bin/streamlit run app.py
```
Keys are read only from the local `.env` file. Restart the app after changing them.

## How to use
1. **Create a topic** in the sidebar and describe what you already know.
2. **🔎 Discover**: search, then add papers to your library.
3. ** Upload PDFs**: add papers the APIs can't access (paywalled IEEE papers etc.).
   Uploaded papers are extracted and indexed for chat automatically.
4. ** Library**: click **Extract all**, then download the review matrix (CSV) if you want it.
5. ** Chat**: ask questions about your papers; answers cite the paper, section and page.
6. ** Synopsis**: describe your idea, pick papers, generate, edit, and download as Word.

## How chat works (RAG)
Each paper's full text is split into ~1,200-character chunks that remember their section and
page, plus one summary chunk from the extracted fields. Chunks are embedded with Gemini and stored
in SQLite. A question is matched with hybrid search (vector similarity + BM25 keywords), and the
top chunks go to Groq (Gemini if Groq is rate-limited), which answers only from them with
citations. Papers without an accessible PDF are searchable through their abstract and extracted
fields only.

## Free keys
| Key | Where | Needed for |
|---|---|---|
| `GROQ_API_KEY` | console.groq.com/keys | AI relevance ranking |
| `GOOGLE_API_KEY` | aistudio.google.com/apikey | Extraction and synopsis |
| `SEMANTIC_SCHOLAR_API_KEY` | semanticscholar.org/product/api | Optional: avoids HTTP 429 from Semantic Scholar |
| `IEEE_API_KEY` | developer.ieee.org | Optional: IEEE Xplore as a direct source |

Free tiers have rate limits. The app waits and retries, and caches every result, so
re-running the same search or re-uploading the same PDF costs nothing.
Free-tier Gemini prompts may be used by Google to improve its products, so avoid uploading
confidential unpublished work.
