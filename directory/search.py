"""Discovery pipeline: search all sources in parallel -> merge duplicates ->
filter -> score relevance with the LLM -> sort."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import config
import db
import llm
from discovery import arxiv, ieee, openalex, semantic_scholar
from models import Paper, normalize_title

SOURCES = {
    "openalex": openalex.search,
    "semantic_scholar": semantic_scholar.search,
    "arxiv": arxiv.search,
    "ieee": ieee.search,
}


@dataclass
class SearchResult:
    papers: List[Paper]
    errors: Dict[str, str] = field(default_factory=dict)   # source -> error message


def _ids(p: Paper) -> List[str]:
    ids = ["title:" + normalize_title(p.title)]
    if p.doi:
        ids.append("doi:" + p.doi.lower())
    if p.arxiv_id:
        ids.append("arxiv:" + p.arxiv_id.lower())
    return ids


def _merge(a: Paper, b: Paper) -> Paper:
    """Fill gaps in `a` with values from `b`; keep the longer abstract."""
    data = a.model_dump()
    for k, v in b.model_dump().items():
        if k == "sources":
            data[k] = sorted(set(a.sources) | set(b.sources))
        elif k == "abstract":
            if v and len(v) > len(data[k] or ""):
                data[k] = v
        elif k == "citation_count":
            data[k] = max(filter(None, [a.citation_count, b.citation_count]), default=None)
        elif data.get(k) in (None, "", []):
            data[k] = v
    return Paper(**data)


def dedupe(papers: List[Paper]) -> List[Paper]:
    merged: List[Paper] = []
    index: Dict[str, int] = {}   # any identifier -> position in `merged`
    for p in papers:
        hit = next((index[i] for i in _ids(p) if i in index), None)
        if hit is None:
            merged.append(p)
            hit = len(merged) - 1
        else:
            merged[hit] = _merge(merged[hit], p)
        for i in _ids(merged[hit]):
            index[i] = hit
    return merged


RANK_PROMPT = """You are helping a researcher find papers for a literature review.

Research topic: {topic}
What the researcher already knows / is looking for: {context}

Score each paper's relevance to this topic from 0 (unrelated) to 10 (core paper).
Return JSON: {{"scores": [{{"id": <id>, "score": <0-10>, "reason": "<max 20 words>"}}]}}

Papers:
{papers}"""


def rank(papers: List[Paper], topic: str, context: str) -> List[Paper]:
    todo = []
    for i, p in enumerate(papers):
        cached = db.cache_get("rank", topic, context, p.key)
        if cached:
            p.relevance, p.relevance_reason = cached["score"], cached["reason"]
        else:
            todo.append(i)

    for start in range(0, len(todo), config.RANK_BATCH_SIZE):
        batch = todo[start:start + config.RANK_BATCH_SIZE]
        listing = "\n\n".join(
            f"id={i}\nTitle: {papers[i].title}\n"
            f"Abstract: {(papers[i].abstract or 'n/a')[:config.ABSTRACT_CHARS_FOR_RANKING]}"
            for i in batch
        )
        out = llm.chat_json([{"role": "user", "content": RANK_PROMPT.format(
            topic=topic, context=context or "n/a", papers=listing)}])
        for s in out.get("scores", []):
            try:
                i = int(s["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if i in batch:
                papers[i].relevance = max(0, min(10, int(s.get("score", 0))))
                papers[i].relevance_reason = s.get("reason", "")
                db.cache_set({"score": papers[i].relevance, "reason": papers[i].relevance_reason},
                             "rank", topic, context, papers[i].key)

    return sorted(papers, key=lambda p: (p.relevance or 0, p.citation_count or 0), reverse=True)


def search(topic: str, context: str = "", year_from: Optional[int] = None,
           ieee_only: bool = False, open_access_only: bool = False,
           use_llm_ranking: bool = True) -> SearchResult:
    query = topic.strip()
    results: List[Paper] = []
    errors: Dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=len(SOURCES)) as pool:
        futures = {name: pool.submit(fn, query, config.RESULTS_PER_SOURCE, year_from)
                   for name, fn in SOURCES.items()}
        for name, fut in futures.items():
            try:
                results.extend(fut.result())
            except Exception as e:
                errors[name] = str(e)[:200]

    papers = dedupe(results)
    if ieee_only:
        papers = [p for p in papers if p.is_ieee]
    if open_access_only:
        papers = [p for p in papers if p.is_open_access]

    if use_llm_ranking and papers:
        try:
            papers = rank(papers, topic, context)
        except Exception as e:
            errors["ranking"] = str(e)[:200]
    if not use_llm_ranking or "ranking" in errors:
        papers.sort(key=lambda p: p.citation_count or 0, reverse=True)
    return SearchResult(papers=papers, errors=errors)
