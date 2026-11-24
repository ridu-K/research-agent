"""Research Agent: Streamlit UI.   Run:  streamlit run app.py"""
from __future__ import annotations

import warnings

# Harmless on macOS's built-in Python 3.9: old LibreSSL + Google's "Python 3.9 is EOL" notice.
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL.*")
warnings.filterwarnings("ignore", category=FutureWarning, module=r"google\..*")

import csv
import html
import io
import re

import streamlit as st

import config
import db
import llm
import papers as paper_tools
import rag
import synopsis
from discovery.search import search
from models import Paper
from ui_style import CSS, badge, empty_state, field, hero

st.set_page_config(page_title="Research Agent", page_icon="📚", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


# ====================== Sidebar ======================
with st.sidebar:
    st.markdown('<div class="brand">📚 <span>Research Agent</span></div>', unsafe_allow_html=True)
    st.caption("Discover papers · extract insights · write your synopsis")

    # ---- API keys (read-only status: keys live in the local .env file) ----
    groq_ok, google_ok = bool(config.GROQ_API_KEY), bool(config.GOOGLE_API_KEY)
    with st.expander(("✅" if groq_ok and google_ok else "🔑") + " API keys",
                     expanded=not (groq_ok and google_ok)):
        st.markdown(("🟢" if groq_ok else "🔴") + " Groq key  \n" +
                    ("🟢" if google_ok else "🔴") + " Google AI Studio key")
        if not (groq_ok and google_ok):
            st.caption(f"Add the missing keys to `{config.ENV_PATH}` and restart the app. "
                       "Free keys: [Groq](https://console.groq.com/keys) · "
                       "[Google AI Studio](https://aistudio.google.com/apikey)")

    st.divider()

    # ---- Topic workspace ----
    topics = db.list_topics()
    names = [t["name"] for t in topics]
    if "select_topic" in st.session_state:          # set right after creating a topic
        st.session_state["topic_choice"] = st.session_state.pop("select_topic")
    elif names and st.session_state.get("topic_choice") not in names + ["➕ New topic"]:
        st.session_state["topic_choice"] = names[0]   # open the most recent topic
    choice = st.selectbox("📂 Topic workspace", ["➕ New topic"] + names, key="topic_choice")

    if choice == "➕ New topic":
        with st.form("new_topic"):
            name = st.text_input("Topic name", placeholder="e.g. Federated learning for IoT intrusion detection")
            context = st.text_area("What you already know / what you're looking for", height=130,
                                   placeholder="Background, key terms, the angle you care about...")
            created = st.form_submit_button("Create topic", type="primary", use_container_width=True)
        if created and name.strip():
            db.upsert_topic(name, context)
            st.session_state["select_topic"] = name.strip()
            st.rerun()
    else:
        topic = next(t for t in topics if t["name"] == choice)
        with st.expander("✏️ Topic context"):
            new_ctx = st.text_area("Context", topic["context"], height=140, label_visibility="collapsed")
            if st.button("Save context", use_container_width=True):
                db.upsert_topic(topic["name"], new_ctx)
                st.rerun()
        with st.expander("🗑️ Delete topic"):
            if st.checkbox("Yes, delete this topic and its synopses"):
                if st.button("Delete", type="primary", use_container_width=True):
                    db.delete_topic(topic["id"])
                    st.session_state.pop("topic_choice", None)
                    st.rerun()


# ====================== Welcome (no topic yet) ======================
if choice == "➕ New topic":
    st.markdown(hero("Welcome to Research Agent",
                     "Create a topic in the sidebar to get started."), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    steps = [("🔎 Discover", "Search OpenAlex, Semantic Scholar, arXiv and IEEE for papers."),
             ("📤 Upload", f"Drop in up to {config.MAX_UPLOADS} PDFs at a time."),
             ("🧠 Extract", "Get the problem, method, results and future work of every paper."),
             ("📝 Synopsis", "Describe your idea and get a one-page synopsis grounded in the papers.")]
    for col, (t, d) in zip((c1, c2, c3, c4), steps):
        with col.container(border=True):
            st.markdown(f"**{t}**")
            st.caption(d)
    st.stop()


library = db.topic_papers(topic["id"])
library_keys = {p.key for p in library}
extracted = [p for p in library if p.extraction]
indexed = rag.index_status(topic["id"])            # paper_key -> 'full text' | 'summary'
not_ready = [p for p in library if not p.extraction or p.key not in indexed]


def need(google: bool = False, groq: bool = False) -> bool:
    """Show a friendly message if a key is missing. Returns True if OK."""
    missing = []
    if google and not config.GOOGLE_API_KEY:
        missing.append("Google AI Studio")
    if groq and not config.GROQ_API_KEY:
        missing.append("Groq")
    if missing:
        names = " and ".join(missing)
        st.warning(f"{names} key missing. Add it to `.env` in the project folder and restart "
                   "the app.", icon="🔑")
    return not missing


def friendly(e: Exception) -> str:
    """Plain-language error for the UI (full details still print in the terminal)."""
    print(f"[research-agent] {type(e).__name__}: {e}")
    msg = str(e).lower()
    if isinstance(e, llm.MissingKey):
        return "An API key is missing in `.env`. Add it and restart the app."
    if isinstance(e, llm.GeminiBusy) or any(s in msg for s in (
            "busy", "429", "503", "unavailable", "high demand", "rate", "quota", "exhausted",
            "timed out", "timeout")):
        return "The free AI service is very busy right now. Please try again in a minute or two."
    if "api key" in msg or "401" in msg or "403" in msg or "permission" in msg:
        return "An API key was rejected. Check the keys in `.env` and restart the app."
    return "Something went wrong. Please try again."


# ====================== Header ======================
st.markdown(hero(topic["name"], (topic["context"] or "No context yet: add some in the sidebar "
                                 "to improve ranking and the synopsis.")[:220]),
            unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
m1.metric("📁 Papers in library", len(library))
m2.metric("🧠 Extracted", f"{len(extracted)}/{len(library)}")
m3.metric("💬 Chat-ready", f"{len(indexed)}/{len(library)}")
m4.metric("🏛️ IEEE", sum(p.is_ieee for p in library))
st.write("")


# ====================== Paper card ======================
def paper_card(p: Paper, idx: str, in_library: bool):
    badges = []
    if p.relevance is not None:
        badges.append(badge(f"⭐ {p.relevance}/10", "score"))
    if p.is_uploaded:
        badges.append(badge("Uploaded PDF", "up"))
    if p.is_ieee:
        badges.append(badge("IEEE", "ieee"))
    if not p.is_uploaded:
        badges.append(badge("🔓 Open PDF", "oa") if p.is_open_access else badge("🔒 Metadata only", "lock"))
    if p.citation_count:
        badges.append(badge(f"{p.citation_count} citations", "cite"))
    if p.extraction:
        badges.append(badge("✓ Extracted", "done") if p.extracted_from == "full text"
                      else badge("✓ Extracted (partial: re-upload later for a full read)", "abs")
                      if p.extracted_from == "partial text"
                      else badge("✓ Extracted (abstract)", "abs"))
    if in_library and p.key in indexed:
        badges.append(badge("💬 Chat-ready" + (" (summary)" if indexed[p.key] == "summary" else ""),
                            "cite"))

    authors = ", ".join(p.authors[:4]) + (" et al." if len(p.authors) > 4 else "")
    with st.container(border=True):
        left, right = st.columns([7, 1.3])
        left.markdown(
            f'<div class="ptitle">{html.escape(p.title)}</div>'
            f'<div class="pmeta">{html.escape(authors or "Unknown authors")} · '
            f'{html.escape(p.venue or "Unknown venue")} · {p.year or "n.d."}</div>'
            + "".join(badges)
            + (f'<div class="preason">{html.escape(p.relevance_reason)}</div>' if p.relevance_reason else ""),
            unsafe_allow_html=True)

        if in_library:
            ready = p.extraction and p.key in indexed
            if not ready and right.button("🧠 Extract", key=f"ex-{idx}", use_container_width=True):
                if need(google=True):
                    with st.spinner("Reading the paper..."):
                        try:
                            p, note = paper_tools.ingest_library_paper(topic["id"], p)
                            st.toast(f"Done: {note}")
                            st.rerun()
                        except Exception as e:
                            st.error(friendly(e))
            if right.button("Remove", key=f"rm-{idx}", use_container_width=True):
                db.remove_paper(topic["id"], p.key)
                st.rerun()
        elif right.button("➕ Add", key=f"add-{idx}", type="primary", use_container_width=True):
            db.save_paper(topic["id"], p)
            st.toast("Added to library")
            st.rerun()

        if p.extraction:
            ex = p.extraction
            with st.expander("📋 Literature review fields"):
                a, b = st.columns(2)
                a.markdown(field("Problem statement", ex.problem_statement or "—")
                           + field("Methodology", ex.methodology or "—")
                           + field("Results", ex.results or "—"), unsafe_allow_html=True)
                b.markdown(field("Future work", ex.future_work or "—")
                           + field("Limitations", ex.limitations or "—")
                           + field("Datasets", ", ".join(ex.datasets) or "—")
                           + field("Key contributions", "; ".join(ex.key_contributions) or "—"),
                           unsafe_allow_html=True)
        if p.abstract or p.url or p.pdf_url:
            with st.expander("Abstract & links"):
                st.write(p.abstract or "No abstract available.")
                links = [f"[Page]({p.url})" if p.url else None,
                         f"[PDF]({p.pdf_url})" if p.pdf_url else None,
                         f"DOI: `{p.doi}`" if p.doi else None]
                st.markdown(" · ".join(x for x in links if x))


def prepare_papers(todo):
    """Extract + index a batch of library papers, with a progress bar."""
    bar = st.progress(0.0)
    for i, p in enumerate(todo, 1):
        bar.progress((i - 1) / len(todo), text=f"{i}/{len(todo)}: {p.title[:70]}")
        try:
            paper_tools.ingest_library_paper(topic["id"], p)
        except Exception as e:
            st.caption(f"❌ {p.title[:60]}: {friendly(e)}")
    st.rerun()


tab_discover, tab_upload, tab_library, tab_chat, tab_syn = st.tabs(
    ["🔎 Discover", "📤 Upload PDFs", f"📁 Library ({len(library)})", "💬 Chat", "📝 Synopsis"])


# ====================== Discover ======================
with tab_discover:
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        query = c1.text_input("Search query", value=topic["name"])
        year_from = c2.number_input("From year", min_value=1950, max_value=2100, value=2018, step=1)
        f1, f2, f3, f4 = st.columns([1, 1, 1.4, 1])
        ieee_only = f1.toggle("IEEE only")
        oa_only = f2.toggle("Open access only")
        use_rank = f3.toggle("AI relevance ranking", value=bool(config.GROQ_API_KEY),
                             help="Scores each paper 0-10 against your topic + context.")
        go = f4.button("🔎 Search", type="primary", use_container_width=True)

    if go:
        if not use_rank or need(groq=True):
            with st.spinner("Searching OpenAlex, Semantic Scholar, arXiv and IEEE..."):
                st.session_state["results"] = search(
                    query, topic["context"], year_from=int(year_from),
                    ieee_only=ieee_only, open_access_only=oa_only, use_llm_ranking=use_rank)

    res = st.session_state.get("results")
    if res:
        for src, err in res.errors.items():
            hint = " (add a free Semantic Scholar key in `.env` to avoid this)" if "semantic" in src else ""
            st.caption(f"⚠️ {src} unavailable: {err[:90]}{hint}")
        new = [p for p in res.papers if p.key not in library_keys]
        c1, c2 = st.columns([3, 1])
        c1.markdown(f"**{len(res.papers)} papers found** · {len(new)} not in your library yet")
        min_score = c2.slider("Min relevance", 0, 10, 0) if use_rank else 0
        shown = [p for p in res.papers if (p.relevance or 0) >= min_score]
        if new and c2.button(f"➕ Add all shown ({len([p for p in shown if p.key not in library_keys])})",
                             use_container_width=True):
            for p in shown:
                if p.key not in library_keys:
                    db.save_paper(topic["id"], p)
            st.rerun()
        for i, p in enumerate(shown):
            paper_card(p, f"d{i}", p.key in library_keys)
    else:
        st.markdown(empty_state("Search the literature",
                                "Results come from free scholarly APIs. Paywalled papers show "
                                "metadata only; upload their PDFs in the next tab."),
                    unsafe_allow_html=True)


# ====================== Upload ======================
with tab_upload:
    st.markdown(f"#### 📤 Upload papers (up to {config.MAX_UPLOADS} PDFs at a time)")
    st.caption("Each PDF is read, its title, problem, methodology, results, future work and "
               "limitations are extracted, and its full text is indexed for Chat. "
               "Scanned PDFs work too.")
    up_key = f"uploader-{st.session_state.get('upload_round', 0)}"
    files = st.file_uploader("Drop PDFs here", type=["pdf"], accept_multiple_files=True,
                             key=up_key, label_visibility="collapsed")
    if files:
        if len(files) > config.MAX_UPLOADS:
            st.warning(f"You selected {len(files)} files. Only the first {config.MAX_UPLOADS} will be processed.")
            files = files[:config.MAX_UPLOADS]
        if st.button(f"🧠 Process {len(files)} paper(s)", type="primary") and need(google=True):
            bar = st.progress(0.0, text="Starting...")
            log = st.container(border=True)
            ok = 0
            for i, f in enumerate(files, 1):
                bar.progress((i - 1) / len(files), text=f"Reading {f.name} ({i}/{len(files)})")
                try:
                    p, note = paper_tools.ingest_upload(topic["id"], f.getvalue(), f.name)
                    ok += 1
                    log.markdown(f"✅ **{f.name}** → {p.title}  \n<small>{note}</small>",
                                 unsafe_allow_html=True)
                except Exception as e:
                    log.markdown(f"❌ **{f.name}**: {friendly(e)}")
            bar.progress(1.0, text=f"Done: {ok}/{len(files)} papers added to the library")
            st.session_state["upload_round"] = st.session_state.get("upload_round", 0) + 1
            if ok:
                st.button("Show in library ↻")


# ====================== Library ======================
with tab_library:
    if not library:
        st.markdown(empty_state("Your library is empty",
                                "Add papers from Discover or upload PDFs."), unsafe_allow_html=True)
    else:
        c1, c2, c3 = st.columns([3, 1.3, 1.2])
        q = c1.text_input("Filter", placeholder="Filter by title, author, venue...",
                          label_visibility="collapsed")
        if not_ready and c2.button(f"🧠 Extract all ({len(not_ready)})", type="primary",
                                   use_container_width=True) and need(google=True):
            prepare_papers(not_ready)

        # Literature table as CSV (handy for a review matrix in Excel)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Title", "Authors", "Year", "Venue", "DOI", "Problem", "Methodology",
                    "Results", "Future work", "Limitations", "Datasets", "Extracted from"])
        for p in library:
            ex = p.extraction
            w.writerow([p.title, "; ".join(p.authors), p.year or "", p.venue or "", p.doi or "",
                        ex.problem_statement if ex else "", ex.methodology if ex else "",
                        ex.results if ex else "", ex.future_work if ex else "",
                        ex.limitations if ex else "", "; ".join(ex.datasets) if ex else "",
                        p.extracted_from or ""])
        c3.download_button("⬇️ Review matrix (CSV)", buf.getvalue(), file_name="literature_review.csv",
                           mime="text/csv", use_container_width=True)

        ql = q.lower().strip()
        for i, p in enumerate(library):
            hay = " ".join([p.title, " ".join(p.authors), p.venue or ""]).lower()
            if not ql or ql in hay:
                paper_card(p, f"l{i}", True)


# ====================== Chat (RAG) ======================
SUGGESTIONS = ["Summarise the main approaches in my library",
               "Which datasets and metrics are used most?",
               "Compare the methodologies in a table",
               "What future work do the papers suggest?"]


def show_sources(sources):
    with st.expander(f"📎 Sources ({len(sources)})"):
        for s in sources:
            kind = "full text" if s["kind"] == "text" else "extracted summary"
            st.markdown(f"**[{s['n']}] {s['title']}** ({s['year'] or 'n.d.'}) · "
                        f"_{s['where'] or kind}_ · <small>{kind}</small>", unsafe_allow_html=True)
            st.caption(s["text"][:450] + ("..." if len(s["text"]) > 450 else ""))


with tab_chat:
    if not library:
        st.markdown(empty_state("Nothing to chat about yet",
                                "Add or upload papers first. Chat answers only from your library."),
                    unsafe_allow_html=True)
    else:
        full = sum(v == "full text" for v in indexed.values())
        c1, c2, c3 = st.columns([3, 1.6, 0.9])
        c1.markdown(f"**💬 Chat with your library** · {len(indexed)}/{len(library)} papers ready "
                    f"({full} with full text)")
        if not_ready and c2.button(f"⚡ Prepare {len(not_ready)} paper(s) for chat",
                                   type="primary", use_container_width=True) and need(google=True):
            prepare_papers(not_ready)
        if c3.button("🧹 Clear", use_container_width=True, help="Clear this topic's chat history"):
            db.clear_chat(topic["id"])
            st.rerun()

        by_key = {p.key: p for p in library}
        scope_opts = {f"{p.title[:80]} ({p.year or 'n.d.'})": p.key for p in library if p.key in indexed}
        scope = st.multiselect("Ask about", list(scope_opts), placeholder="All chat-ready papers",
                               help="Leave empty to search the whole library.")
        if any(v == "summary" for v in indexed.values()):
            st.caption("ℹ️ Papers marked 'summary' were indexed from their extracted fields/abstract only. "
                       "Upload their PDFs for full-text answers.")

        history = db.chat_history(topic["id"])
        if not history:
            st.caption("Try one of these:")
            cols = st.columns(len(SUGGESTIONS))
            for col, s in zip(cols, SUGGESTIONS):
                if col.button(s, use_container_width=True):
                    st.session_state["pending_q"] = s
                    st.rerun()

        for m in history:
            with st.chat_message(m["role"], avatar="🧑‍🔬" if m["role"] == "user" else "📚"):
                st.markdown(m["content"])
                if m["sources"]:
                    show_sources(m["sources"])

        question = st.chat_input("Ask a question about your papers...") or st.session_state.pop("pending_q", None)
        if question and need(google=True):
            with st.chat_message("user", avatar="🧑‍🔬"):
                st.markdown(question)
            with st.chat_message("assistant", avatar="📚"):
                with st.spinner("Searching your papers..."):
                    try:
                        reply, sources = rag.answer(topic["id"], question, by_key, history,
                                                    [scope_opts[s] for s in scope] or None)
                    except Exception as e:
                        reply = None
                        st.error(f"Couldn't answer: {friendly(e)} Your question wasn't saved.")
            if reply is not None:
                db.add_message(topic["id"], "user", question)
                db.add_message(topic["id"], "assistant", reply, sources)
                st.rerun()


# ====================== Synopsis ======================
with tab_syn:
    st.markdown("#### 📝 One-page synopsis")
    st.caption("Describe your idea. The synopsis is written from your library's literature in a "
               "fixed format: Background · Literature Review · Research Gap · Problem Statement · "
               "Objectives · Methodology · Expected Outcomes · Novelty · References (IEEE).")

    if not library:
        st.markdown(empty_state("No literature yet",
                                "Add or upload papers first. The synopsis is grounded in them."),
                    unsafe_allow_html=True)
    else:
        with st.form("synopsis_form"):
            idea = st.text_area("💡 Your research idea", height=150,
                                placeholder="What do you want to do, why, and roughly how? "
                                            "A few sentences or bullet points is enough.")
            c1, c2 = st.columns(2)
            kind = c1.selectbox("Synopsis type", synopsis.SYNOPSIS_TYPES)
            extra = c2.text_input("Extra instructions (optional)",
                                  placeholder="e.g. focus on edge devices; target IEEE IoT Journal")
            labels = {f"{p.title[:90]} ({p.year or 'n.d.'})": p for p in library}
            default = [l for l, p in labels.items() if p.extraction] or list(labels)
            chosen = st.multiselect("Papers to use", list(labels), default=default,
                                    help="Defaults to extracted papers. Unextracted ones use their abstract.")
            submit = st.form_submit_button("✨ Generate synopsis", type="primary")

        if submit:
            if not idea.strip():
                st.warning("Describe your idea first.")
            elif not chosen:
                st.warning("Pick at least one paper.")
            elif need(google=True):
                with st.spinner("Writing your synopsis..."):
                    try:
                        md = synopsis.generate(topic["name"], idea, [labels[c] for c in chosen],
                                               kind=kind, extra=extra)
                        title = re.sub(r"^#\s*", "", md.strip().splitlines()[0])
                        st.session_state["syn_id"] = db.save_synopsis(topic["id"], title, md)
                    except Exception as e:
                        st.error(f"Could not generate the synopsis. {friendly(e)}")

        history = db.list_synopses(topic["id"])
        if history:
            ids = [h["id"] for h in history]
            if st.session_state.get("syn_id") not in ids:
                st.session_state["syn_id"] = ids[0]
            cur = st.selectbox("Synopsis", ids, index=ids.index(st.session_state["syn_id"]),
                               format_func=lambda i: next(f"{h['title'][:80]}  ·  {h['created_at'][:16]}"
                                                          for h in history if h["id"] == i))
            st.session_state["syn_id"] = cur
            md = next(h["markdown"] for h in history if h["id"] == cur)

            words = synopsis.word_count(md)
            fits = words <= config.SYNOPSIS_WORDS + 80
            st.caption(("🟢" if fits else "🟠") + f" {words} words "
                       + ("(fits on one page)" if fits else "(may run past one page: trim in Edit)"))

            view, edit = st.tabs(["👁️ Preview", "✏️ Edit"])
            with view:
                with st.container(key="synopsis_page"):
                    st.markdown(md)
            with edit:
                edited = st.text_area("Markdown", md, height=520, key=f"edit-{cur}",
                                      label_visibility="collapsed")
                e1, e2 = st.columns([1, 5])
                if e1.button("💾 Save", type="primary"):
                    db.update_synopsis(cur, edited)
                    st.toast("Saved")
                    st.rerun()
                e2.caption("Keep the `#` / `##` headings and `- ` bullets so the Word export stays formatted.")

            d1, d2, d3, _ = st.columns([1.2, 1.2, 1, 3])
            safe = re.sub(r"[^A-Za-z0-9]+", "_", md.splitlines()[0].lstrip("# "))[:60] or "synopsis"
            d1.download_button("⬇️ Word (.docx)", synopsis.to_docx(md), file_name=f"{safe}.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               use_container_width=True)
            d2.download_button("⬇️ Markdown", md, file_name=f"{safe}.md", use_container_width=True)
            if d3.button("🗑️ Delete", use_container_width=True):
                db.delete_synopsis(cur)
                st.session_state.pop("syn_id", None)
                st.rerun()
