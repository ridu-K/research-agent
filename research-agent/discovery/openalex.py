"""OpenAlex: free, no key, very broad coverage (includes IEEE metadata)."""
from __future__ import annotations

from typing import List, Optional

import config
from discovery.http import get
from models import Paper, clean_doi


def _abstract(inverted: Optional[dict]) -> Optional[str]:
    """OpenAlex stores abstracts as {word: [positions]}. Rebuild the text."""
    if not inverted:
        return None
    words = sorted((pos, w) for w, positions in inverted.items() for pos in positions)
    return " ".join(w for _, w in words)


def search(query: str, limit: int = config.RESULTS_PER_SOURCE,
           year_from: Optional[int] = None) -> List[Paper]:
    params = {"search": query, "per_page": limit}
    if year_from:
        params["filter"] = f"from_publication_date:{year_from}-01-01"
    if config.CONTACT_EMAIL:
        params["mailto"] = config.CONTACT_EMAIL
    data = get("https://api.openalex.org/works", params)

    papers = []
    for w in data.get("results", []):
        if not w.get("title"):
            continue
        loc = w.get("primary_location") or {}
        best_oa = w.get("best_oa_location") or {}
        arxiv_id = None
        for l in w.get("locations") or []:
            u = (l.get("landing_page_url") or "")
            if "arxiv.org/abs/" in u:
                arxiv_id = u.rsplit("/abs/", 1)[1]
        papers.append(Paper(
            title=w["title"],
            authors=[a["author"]["display_name"] for a in w.get("authorships", [])
                     if a.get("author")],
            year=w.get("publication_year"),
            venue=(loc.get("source") or {}).get("display_name"),
            abstract=_abstract(w.get("abstract_inverted_index")),
            doi=clean_doi(w.get("doi")),
            arxiv_id=arxiv_id,
            url=loc.get("landing_page_url") or w.get("doi"),
            pdf_url=best_oa.get("pdf_url"),
            citation_count=w.get("cited_by_count"),
            sources=["openalex"],
        ))
    return papers
