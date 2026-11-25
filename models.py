"""Data shapes used across the app."""
from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field

import config


class Extraction(BaseModel):
    """What we pull out of every paper. Also used as Gemini's output schema,
    so keep field types simple (str / list of str)."""
    title: str = ""
    authors: List[str] = Field(default_factory=list)
    year: str = ""
    venue: str = ""
    doi: str = ""
    problem_statement: str = ""
    methodology: str = ""
    results: str = ""
    future_work: str = ""
    limitations: str = ""
    datasets: List[str] = Field(default_factory=list)
    key_contributions: List[str] = Field(default_factory=list)


class Paper(BaseModel):
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url: Optional[str] = None
    pdf_url: Optional[str] = None        # open-access PDF, if any
    citation_count: Optional[int] = None
    sources: List[str] = Field(default_factory=list)  # which APIs found it ("upload" for PDFs)

    # Filled by the relevance ranker
    relevance: Optional[int] = None      # 0-10
    relevance_reason: Optional[str] = None

    # Filled by extraction
    extraction: Optional[Extraction] = None
    extracted_from: Optional[str] = None  # "full text" | "partial text" | "abstract only"

    @property
    def key(self) -> str:
        """Stable ID used for dedupe and storage: DOI > arXiv ID > normalized title."""
        if self.doi:
            return "doi:" + self.doi.lower()
        if self.arxiv_id:
            return "arxiv:" + self.arxiv_id.lower()
        return "title:" + normalize_title(self.title)

    @property
    def is_ieee(self) -> bool:
        if self.doi and self.doi.lower().startswith(config.IEEE_DOI_PREFIX):
            return True
        return bool(self.venue and "ieee" in self.venue.lower())

    @property
    def is_open_access(self) -> bool:
        return bool(self.pdf_url)

    @property
    def is_uploaded(self) -> bool:
        return "upload" in self.sources

    def ieee_reference(self) -> str:
        """IEEE-style reference string, built from metadata (never by the LLM)."""
        def short(name: str) -> str:
            parts = name.split()
            if len(parts) < 2:
                return name
            return " ".join(p[0] + "." for p in parts[:-1]) + " " + parts[-1]

        names = [short(a) for a in self.authors[:3]]
        authors = ", ".join(names) + (" et al." if len(self.authors) > 3 else "")
        bits = [authors + ","] if authors else []
        bits.append(f"“{self.title},”")
        if self.venue:
            bits.append(f"{self.venue},")
        bits.append(f"{self.year}." if self.year else "n.d.")
        ref = " ".join(bits)
        if self.doi:
            ref += f" doi: {self.doi}."
        return ref


def normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def clean_doi(doi: Optional[str]) -> Optional[str]:
    """'https://doi.org/10.1109/X' -> '10.1109/x'"""
    if not doi:
        return None
    doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", doi.strip(), flags=re.I)
    return doi.lower() or None
