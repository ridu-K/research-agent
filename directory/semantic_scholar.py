"""Semantic Scholar: free. Works without a key (shared rate limit); a free key
gives your own limit."""
from __future__ import annotations

from typing import List, Optional

import config
from discovery.http import get
from models import Paper, clean_doi

FIELDS = "title,abstract,year,authors,venue,externalIds,openAccessPdf,citationCount,url"


def search(query: str, limit: int = config.RESULTS_PER_SOURCE,
           year_from: Optional[int] = None) -> List[Paper]:
    params = {"query": query, "limit": limit, "fields": FIELDS}
    if year_from:
        params["year"] = f"{year_from}-"
    headers = {"x-api-key": config.SEMANTIC_SCHOLAR_API_KEY} if config.SEMANTIC_SCHOLAR_API_KEY else None
    data = get("https://api.semanticscholar.org/graph/v1/paper/search", params, headers)

    papers = []
    for p in data.get("data") or []:
        if not p.get("title"):
            continue
        ids = p.get("externalIds") or {}
        papers.append(Paper(
            title=p["title"],
            authors=[a["name"] for a in p.get("authors") or [] if a.get("name")],
            year=p.get("year"),
            venue=p.get("venue") or None,
            abstract=p.get("abstract"),
            doi=clean_doi(ids.get("DOI")),
            arxiv_id=ids.get("ArXiv"),
            url=p.get("url"),
            pdf_url=(p.get("openAccessPdf") or {}).get("url") or None,
            citation_count=p.get("citationCount"),
            sources=["semantic_scholar"],
        ))
    return papers
