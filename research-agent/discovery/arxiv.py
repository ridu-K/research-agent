"""arXiv: free, no key. Preprints, always with a free PDF."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import List, Optional

import config
from discovery.http import get
from models import Paper, clean_doi

NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def search(query: str, limit: int = config.RESULTS_PER_SOURCE,
           year_from: Optional[int] = None) -> List[Paper]:
    params = {"search_query": f"all:{query}", "max_results": limit, "sortBy": "relevance"}
    xml = get("https://export.arxiv.org/api/query", params, as_json=False)
    root = ET.fromstring(xml)

    papers = []
    for e in root.findall("a:entry", NS):
        title = re.sub(r"\s+", " ", e.findtext("a:title", "", NS)).strip()
        if not title:
            continue
        abs_url = e.findtext("a:id", "", NS)
        arxiv_id = re.sub(r"v\d+$", "", abs_url.rsplit("/abs/", 1)[-1])
        year = int(e.findtext("a:published", "0000", NS)[:4]) or None
        if year_from and year and year < year_from:
            continue
        pdf = next((l.get("href") for l in e.findall("a:link", NS)
                    if l.get("title") == "pdf"), None)
        papers.append(Paper(
            title=title,
            authors=[a.findtext("a:name", "", NS) for a in e.findall("a:author", NS)],
            year=year,
            venue=e.findtext("arxiv:journal_ref", None, NS) or "arXiv",
            abstract=re.sub(r"\s+", " ", e.findtext("a:summary", "", NS)).strip(),
            doi=clean_doi(e.findtext("arxiv:doi", None, NS)),
            arxiv_id=arxiv_id,
            url=abs_url,
            pdf_url=pdf,
            sources=["arxiv"],
        ))
    return papers
