"""IEEE Xplore API: free key from https://developer.ieee.org (limited calls/day).
Optional: if no key is set, this source is skipped and OpenAlex/Semantic Scholar
still return IEEE papers (DOI prefix 10.1109)."""
from __future__ import annotations

from typing import List, Optional

import config
from discovery.http import get
from models import Paper, clean_doi


def search(query: str, limit: int = config.RESULTS_PER_SOURCE,
           year_from: Optional[int] = None) -> List[Paper]:
    if not config.IEEE_API_KEY:
        return []
    params = {"querytext": query, "max_records": limit, "apikey": config.IEEE_API_KEY}
    if year_from:
        params["start_year"] = year_from
    data = get("https://ieeexploreapi.ieee.org/api/v1/search/articles", params)

    papers = []
    for a in data.get("articles") or []:
        if not a.get("title"):
            continue
        is_oa = (a.get("access_type") or "").upper() == "OPEN_ACCESS"
        year = a.get("publication_year")
        papers.append(Paper(
            title=a["title"],
            authors=[x.get("full_name") for x in (a.get("authors") or {}).get("authors", [])
                     if x.get("full_name")],
            year=int(year) if year else None,
            venue=a.get("publication_title"),
            abstract=a.get("abstract"),
            doi=clean_doi(a.get("doi")),
            url=a.get("html_url"),
            pdf_url=a.get("pdf_url") if is_oa else None,
            citation_count=a.get("citing_paper_count"),
            sources=["ieee"],
        ))
    return papers
