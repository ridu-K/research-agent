"""Chat with your library (RAG).

Indexing:  paper -> chunks (full text split by section/page + one "summary" chunk built from
           the extracted fields) -> Gemini embeddings -> SQLite `chunks` table.
Answering: question -> (rewrite follow-ups into a standalone question) -> hybrid search
           (vector similarity + BM25 keywords, fused with Reciprocal Rank Fusion)
           -> top chunks as numbered sources -> LLM answer that cites [1], [2] ...
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

import config
import db
import llm
from models import Paper

# ---------- Chunking ----------

# Section headings in IEEE-style papers: "I. INTRODUCTION", "III. PROPOSED METHOD",
# "2 Related Work", "4.1 Datasets", "Abstract", "Index Terms" ...
_HEADING = re.compile(
    r"^\s*(?:"
    r"(?:[IVX]{1,5})\.\s+[A-Z][A-Z0-9 ,:&/\-]{2,60}"            # I. INTRODUCTION
    r"|\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+[A-Z][A-Za-z][A-Za-z ,:&/\-]{1,60}"  # 2 Related Work / 4.1 Data
    r"|(?:Abstract|ABSTRACT|Index Terms|INDEX TERMS|Conclusions?|CONCLUSIONS?)\b.{0,40}"
    r")\s*$"
)
_REFERENCES = re.compile(r"^\s*(?:[IVX]+\.\s*|\d+\.?\s*)?(references|bibliography)\s*$", re.I)


_LONE_NUMBER = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,2}\.?|[IVX]{1,5}\.)$")


def _is_heading(line: str) -> bool:
    return len(line) <= 70 and len(line.split()) <= 9 and bool(_HEADING.match(line))


def _merge_split_headings(page_text: str) -> List[str]:
    """PDF text often puts a heading's number on its own line ("3.2" / "Attention").
    Re-join them; drop other lone numbers (page numbers, table cells)."""
    out: List[str] = []
    pending = None
    for raw in page_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _LONE_NUMBER.match(line):
            pending = line
            continue
        if pending:
            joined = f"{pending} {line}"
            out.append(joined if _is_heading(joined) else line)
            pending = None
        else:
            out.append(line)
    return out


def chunk_pages(pages: List[str]) -> List[dict]:
    """Split page texts into ~CHUNK_CHARS chunks that remember their section and page.
    Stops at the reference list (if it's in the second half of the paper)."""
    lines: List[Tuple[int, str, str]] = []     # (page, section, line)
    section = "Front matter"
    total = len(pages)
    for page_no, text in enumerate(pages, 1):
        for line in _merge_split_headings(text):
            if _REFERENCES.match(line) and page_no > total / 2:
                break
            if _is_heading(line):
                section = re.sub(r"\s+", " ", line)
                continue
            lines.append((page_no, section, line))
        else:
            continue
        break   # hit references

    chunks: List[dict] = []
    buf: List[Tuple[int, str, str]] = []
    size = 0

    def flush():
        if buf:
            text = re.sub(r"-\n(\w)", r"\1", "\n".join(l for _, _, l in buf))  # re-join hyphenated words
            text = re.sub(r"\s*\n\s*", " ", text).strip()
            if len(text) > 80:
                chunks.append({"page": buf[0][0], "section": buf[0][1], "text": text})

    for item in lines:
        # Start a new chunk at a section change if the current one is reasonably full
        if buf and item[1] != buf[-1][1] and size > config.CHUNK_CHARS * 0.4:
            flush()
            buf, size = [], 0
        buf.append(item)
        size += len(item[2]) + 1
        if size >= config.CHUNK_CHARS:
            flush()
            # keep the last few lines as overlap
            keep, kept = [], 0
            for it in reversed(buf):
                if kept >= config.CHUNK_OVERLAP:
                    break
                keep.insert(0, it)
                kept += len(it[2]) + 1
            buf, size = keep, kept
    flush()
    return chunks


def summary_text(p: Paper) -> str:
    """One chunk describing the whole paper: good for 'what does paper X do' questions."""
    parts = [f"Title: {p.title}",
             f"Authors: {', '.join(p.authors[:6])}" if p.authors else "",
             f"Year: {p.year}" if p.year else "",
             f"Venue: {p.venue}" if p.venue else ""]
    if p.abstract:
        parts.append(f"Abstract: {p.abstract}")
    ex = p.extraction
    if ex:
        parts += [f"Problem: {ex.problem_statement}", f"Methodology: {ex.methodology}",
                  f"Results: {ex.results}", f"Future work: {ex.future_work}",
                  f"Limitations: {ex.limitations}",
                  f"Datasets: {', '.join(ex.datasets)}" if ex.datasets else "",
                  f"Contributions: {'; '.join(ex.key_contributions)}" if ex.key_contributions else ""]
    return "\n".join(x for x in parts if x and not x.endswith(": "))


# ---------- Indexing ----------

def _to_blob(vec) -> bytes:
    v = np.asarray(vec, dtype=np.float32)
    n = np.linalg.norm(v)
    return (v / n if n else v).tobytes()


def index_paper(paper: Paper, pages: Optional[List[str]] = None) -> int:
    """(Re)build the chunks for one paper. Returns the number of chunks stored."""
    items = [{"kind": "summary", "section": "Summary (extracted)", "page": None,
              "text": summary_text(paper)}]
    for c in chunk_pages(pages or []):
        items.append({"kind": "text", **c})
    vectors = llm.embed([f"{paper.title}\n{it['section'] or ''}\n{it['text']}" for it in items])
    with db.connect() as c:
        c.execute("DELETE FROM chunks WHERE paper_key = ?", (paper.key,))
        c.executemany(
            "INSERT INTO chunks(paper_key, kind, section, page, text, emb) VALUES(?, ?, ?, ?, ?, ?)",
            [(paper.key, it["kind"], it["section"], it["page"], it["text"], _to_blob(v))
             for it, v in zip(items, vectors)])
    return len(items)


def index_status(topic_id: int) -> Dict[str, str]:
    """paper_key -> 'full text' | 'summary' for papers in the topic that are chat-ready."""
    with db.connect() as c:
        rows = c.execute(
            "SELECT c.paper_key, SUM(c.kind = 'text') FROM chunks c "
            "JOIN topic_papers tp ON tp.paper_key = c.paper_key WHERE tp.topic_id = ? "
            "GROUP BY c.paper_key", (topic_id,)).fetchall()
    return {k: ("full text" if n else "summary") for k, n in rows}


# ---------- Retrieval ----------

def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def retrieve(topic_id: int, query: str, paper_keys: Optional[List[str]] = None,
             k: int = config.TOP_K) -> List[dict]:
    with db.connect() as c:
        rows = c.execute(
            "SELECT c.paper_key, c.kind, c.section, c.page, c.text, c.emb FROM chunks c "
            "JOIN topic_papers tp ON tp.paper_key = c.paper_key WHERE tp.topic_id = ?",
            (topic_id,)).fetchall()
    if paper_keys:
        wanted = set(paper_keys)
        rows = [r for r in rows if r[0] in wanted]
    if not rows:
        return []

    # Vector ranking
    matrix = np.vstack([np.frombuffer(r[5], dtype=np.float32) for r in rows])
    q = np.asarray(llm.embed([query], task="RETRIEVAL_QUERY")[0], dtype=np.float32)
    q /= (np.linalg.norm(q) or 1)
    vec_rank = np.argsort(-(matrix @ q))

    # Keyword ranking (catches exact terms: dataset names, acronyms, metrics)
    bm25 = BM25Okapi([_tokens(r[4]) or ["_"] for r in rows])
    kw_rank = np.argsort(-bm25.get_scores(_tokens(query)))

    # Reciprocal Rank Fusion
    score = np.zeros(len(rows))
    for ranking in (vec_rank, kw_rank):
        for pos, idx in enumerate(ranking):
            score[idx] += 1.0 / (60 + pos)

    picked, per_paper = [], {}
    for idx in np.argsort(-score):
        key = rows[idx][0]
        if per_paper.get(key, 0) >= config.MAX_CHUNKS_PER_PAPER:
            continue
        per_paper[key] = per_paper.get(key, 0) + 1
        r = rows[idx]
        picked.append({"paper_key": r[0], "kind": r[1], "section": r[2], "page": r[3], "text": r[4]})
        if len(picked) >= k:
            break
    return picked


# ---------- Answering ----------

SYSTEM = """You are a research assistant helping a researcher understand THEIR paper library.
Answer using ONLY the numbered sources provided. Rules:
- Cite every claim right after it with the source number(s), e.g. "... uses CICIDS2017 [2][5]."
- If the sources don't contain the answer, say clearly: "I couldn't find this in your library."
  Then say what kind of paper would answer it. Never use outside knowledge as if it came from the papers.
- When comparing papers, use a markdown table (one row per paper).
- Refer to papers by a short title, not "source 3".
- Be concise, specific and technical (methods, datasets, metrics, numbers)."""

REWRITE = """Rewrite the user's latest question as a standalone search query, resolving
references like "it", "they", "that paper" using the conversation. Return only the query.

Conversation:
{history}

Latest question: {question}"""


def _history_msgs(history: List[dict]) -> List[dict]:
    recent = history[-2 * config.HISTORY_TURNS:]
    return [{"role": m["role"], "content": m["content"]} for m in recent]


def answer(topic_id: int, question: str, papers: Dict[str, Paper], history: List[dict],
           paper_keys: Optional[List[str]] = None) -> Tuple[str, List[dict]]:
    """Returns (answer markdown, sources). `papers` maps paper_key -> Paper (for titles)."""
    query = question
    if history:
        convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in _history_msgs(history))
        try:
            query = llm.chat([{"role": "user", "content": REWRITE.format(
                history=convo, question=question)}], temperature=0).strip() or question
        except Exception:
            query = question

    chunks = retrieve(topic_id, query, paper_keys)
    if not chunks:
        return ("Your library has no chat-ready papers yet. Use **Prepare papers for chat** "
                "above, or upload PDFs."), []

    sources, blocks = [], []
    for i, ch in enumerate(chunks, 1):
        p = papers.get(ch["paper_key"])
        title = p.title if p else ch["paper_key"]
        where = ch["section"] or ""
        if ch["page"]:
            where += f", p.{ch['page']}"
        blocks.append(f"[{i}] Paper: {title} ({p.year if p and p.year else 'n.d.'}) | {where}\n{ch['text']}")
        sources.append({"n": i, "title": title, "year": p.year if p else None, "where": where,
                        "kind": ch["kind"], "text": ch["text"]})

    messages = [{"role": "system", "content": SYSTEM}] + _history_msgs(history) + [{
        "role": "user",
        "content": "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}",
    }]
    reply = llm.chat(messages)
    # Some models cite as 【1】 or 【1†L3-L5】: normalise to [1]
    reply = re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", reply)

    cited = {int(n) for n in re.findall(r"\[(\d+)\]", reply)}
    used = [s for s in sources if s["n"] in cited] or sources
    return reply, used
