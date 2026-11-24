"""One-page research synopsis: literature (your library) + your idea -> fixed format.

FORMAT (always the same, ~650 words + references, fits one A4 page):
  # Title
  **Keywords:** ...
  ## 1. Background & Motivation
  ## 2. Literature Review          (cites papers as [1], [2] ...)
  ## 3. Research Gap               (cites papers)
  ## 4. Problem Statement
  ## 5. Objectives                 (3-4 bullets)
  ## 6. Proposed Methodology
  ## 7. Expected Outcomes          (3 bullets)
  ## 8. Novelty
  ## References                    (IEEE style, built from metadata, only cited papers)
"""
from __future__ import annotations

import io
import re
from typing import Dict, List

from pydantic import BaseModel, Field

import config
import llm
from models import Paper

SYNOPSIS_TYPES = ["Research proposal", "Thesis / dissertation", "Project proposal",
                  "Conference paper idea", "Grant / funding pitch"]


class SynopsisDraft(BaseModel):
    title: str = ""
    keywords: List[str] = Field(default_factory=list)
    background: str = ""
    literature_review: str = ""
    research_gap: str = ""
    problem_statement: str = ""
    objectives: List[str] = Field(default_factory=list)
    methodology: str = ""
    expected_outcomes: List[str] = Field(default_factory=list)
    novelty: str = ""


PROMPT = """You are an experienced research advisor. Write a ONE-PAGE synopsis ({kind}) for the
researcher's idea below, grounded in the literature provided.

Research topic: {topic}
Researcher's idea:
\"\"\"{idea}\"\"\"
{extra}

Literature (each paper has an id in square brackets):
{literature}

Rules:
- Total length about {words} words. Section budgets (words): background 80, literature_review 150,
  research_gap 70, problem_statement 50, objectives 3-4 bullets of max 20 words, methodology 130,
  expected_outcomes 3 bullets of max 20 words, novelty 50, keywords 4-6 terms.
- Cite papers ONLY with their ids, one id per bracket, e.g. "[2]" or "[2], [5]". Never invent papers.
- literature_review: group papers by approach and compare them; don't list them one by one.
- research_gap: must follow from the literature's limitations and future work, with citations.
- methodology: concrete steps, datasets, baselines and evaluation metrics where sensible.
- novelty: say exactly what is new compared with the cited work.
- Formal academic tone, no filler, no markdown inside fields.
"""


def _literature_block(papers: List[Paper], per_paper: int = 1500) -> str:
    out = []
    for i, p in enumerate(papers, 1):
        head = f"[{i}] {p.title} ({p.year or 'n.d.'}{', ' + p.venue if p.venue else ''})"
        ex = p.extraction
        if ex and (ex.problem_statement or ex.methodology):
            body = (f"Problem: {ex.problem_statement}\nMethod: {ex.methodology}\n"
                    f"Results: {ex.results}\nLimitations: {ex.limitations}\n"
                    f"Future work: {ex.future_work}")
        else:
            body = f"Abstract: {(p.abstract or 'n/a')}"
        out.append(head + "\n" + body[:per_paper])
    return "\n\n".join(out)


def _renumber(sections: Dict[str, str], n_papers: int):
    """Renumber citations in order of first appearance and drop invalid ids.
    Returns (sections, ordered list of original ids)."""
    order: List[int] = []
    pattern = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")

    def ids_in(group: str) -> List[int]:
        ids: List[int] = []
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[–-]", part):
                a, b = [int(x) for x in re.split(r"\s*[–-]\s*", part)]
                ids.extend(range(a, b + 1))
            else:
                ids.append(int(part))
        return [i for i in ids if 1 <= i <= n_papers]

    for text in sections.values():
        for m in pattern.finditer(text):
            for i in ids_in(m.group(1)):
                if i not in order:
                    order.append(i)
    new_num = {old: k for k, old in enumerate(order, 1)}

    def repl(m):
        ids = ids_in(m.group(1))
        return ", ".join(f"[{new_num[i]}]" for i in ids) if ids else ""

    fixed = {k: re.sub(r"\s+([.,;])", r"\1", pattern.sub(repl, v)) for k, v in sections.items()}
    return fixed, order


def to_markdown(d: SynopsisDraft, papers: List[Paper]) -> str:
    text_fields = {
        "background": d.background, "literature_review": d.literature_review,
        "research_gap": d.research_gap, "problem_statement": d.problem_statement,
        "methodology": d.methodology, "novelty": d.novelty,
        **{f"obj{i}": o for i, o in enumerate(d.objectives)},
        **{f"out{i}": o for i, o in enumerate(d.expected_outcomes)},
    }
    s, order = _renumber(text_fields, len(papers))
    objectives = "\n".join(f"- {s[f'obj{i}']}" for i in range(len(d.objectives)))
    outcomes = "\n".join(f"- {s[f'out{i}']}" for i in range(len(d.expected_outcomes)))
    refs = "\n".join(f"[{k}] {papers[i - 1].ieee_reference()}  " for k, i in enumerate(order, 1))

    return f"""# {d.title.strip() or 'Research Synopsis'}

**Keywords:** {', '.join(d.keywords)}

## 1. Background & Motivation
{s['background']}

## 2. Literature Review
{s['literature_review']}

## 3. Research Gap
{s['research_gap']}

## 4. Problem Statement
{s['problem_statement']}

## 5. Objectives
{objectives}

## 6. Proposed Methodology
{s['methodology']}

## 7. Expected Outcomes
{outcomes}

## 8. Novelty
{s['novelty']}

## References
{refs or '_No papers cited._'}
"""


def generate(topic: str, idea: str, papers: List[Paper], kind: str = SYNOPSIS_TYPES[0],
             extra: str = "") -> str:
    """Returns the synopsis as markdown in the fixed format."""
    fmt = dict(kind=kind, topic=topic, idea=idea.strip(), words=config.SYNOPSIS_WORDS,
               extra=f"Extra instructions: {extra.strip()}" if extra.strip() else "")
    try:
        out = llm.gemini_json(PROMPT.format(literature=_literature_block(papers), **fmt),
                              schema=SynopsisDraft, temperature=0.4)
    except llm.GeminiBusy:
        if not config.GROQ_API_KEY:
            raise
        # Groq free tier: ~8k tokens/min, so send a shorter summary of each paper.
        per_paper = max(250, min(1500, config.FALLBACK_TEXT_CHARS * 3 // 4 // max(1, len(papers))))
        fields = ", ".join(f'"{k}"' for k in SynopsisDraft.model_fields)
        prompt = (PROMPT.format(literature=_literature_block(papers, per_paper), **fmt)
                  + f"\nReturn one JSON object with exactly these keys: {fields}. "
                    "keywords, objectives and expected_outcomes are lists of strings; the rest are strings.")
        out = llm.groq_json_any(prompt, max_tokens=3500)
    clean = {}
    for k, v in (out or {}).items():
        if k in SynopsisDraft.model_fields:
            is_list = k in ("keywords", "objectives", "expected_outcomes")
            if is_list and isinstance(v, str):
                v = [x.strip(" -•") for x in re.split(r"[\n;]", v) if x.strip(" -•")]
            elif not is_list and isinstance(v, list):
                v = " ".join(map(str, v))
            clean[k] = v
    draft = SynopsisDraft(**clean)
    if not (draft.problem_statement or draft.methodology):
        raise RuntimeError("The model returned an empty synopsis. Try again in a minute.")
    return to_markdown(draft, papers)


def word_count(markdown: str) -> int:
    body = markdown.split("## References")[0]
    return len(re.findall(r"\b\w+\b", re.sub(r"[#*_>-]", " ", body)))


# ---------- Export: markdown -> one-page .docx ----------

def _add_runs(par, text: str):
    """Support **bold** inside a line."""
    for i, chunk in enumerate(re.split(r"\*\*(.+?)\*\*", text)):
        if chunk:
            par.add_run(chunk).bold = (i % 2 == 1)


def to_docx(markdown: str) -> bytes:
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    for sec in doc.sections:
        sec.top_margin = sec.bottom_margin = Inches(0.6)
        sec.left_margin = sec.right_margin = Inches(0.7)
    normal = doc.styles["Normal"]
    normal.font.name, normal.font.size = "Calibri", Pt(10)
    normal.paragraph_format.space_after = Pt(2)

    accent = RGBColor(0x37, 0x30, 0xA3)
    in_refs = False
    for line in markdown.splitlines():
        line = line.rstrip()
        if not line:
            continue
        if line.startswith("# "):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(line[2:])
            r.bold, r.font.size, r.font.color.rgb = True, Pt(14), accent
        elif line.startswith("## "):
            in_refs = line[3:].strip().lower() == "references"
            p = doc.add_paragraph()
            p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(5), Pt(1)
            r = p.add_run(line[3:])
            r.bold, r.font.size, r.font.color.rgb = True, Pt(11), accent
        elif line.startswith("- "):
            p = doc.add_paragraph(style="List Bullet")
            p.paragraph_format.space_after = Pt(0)
            _add_runs(p, line[2:])
        else:
            p = doc.add_paragraph()
            _add_runs(p, line.strip().strip("_"))
            if in_refs:
                for r in p.runs:
                    r.font.size = Pt(8.5)
                p.paragraph_format.space_after = Pt(0)
            else:
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
