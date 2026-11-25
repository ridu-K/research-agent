"""SQLite store: topics, papers saved under each topic, synopses,
and a small key-value cache so we never repeat the same API call."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from typing import List, Optional

import config
from models import Paper

SCHEMA = """
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    context TEXT DEFAULT '',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS papers (
    key TEXT PRIMARY KEY,
    data TEXT NOT NULL                 -- Paper as JSON (includes extraction)
);
CREATE TABLE IF NOT EXISTS topic_papers (
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    paper_key TEXT REFERENCES papers(key) ON DELETE CASCADE,
    added_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (topic_id, paper_key)
);
CREATE TABLE IF NOT EXISTS synopses (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    markdown TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (           -- RAG index
    id INTEGER PRIMARY KEY,
    paper_key TEXT NOT NULL,
    kind TEXT NOT NULL,                         -- 'text' (full text) | 'summary' (extracted fields/abstract)
    section TEXT,
    page INTEGER,
    text TEXT NOT NULL,
    emb BLOB NOT NULL                           -- float32 vector, L2-normalised
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_key);
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY,
    topic_id INTEGER REFERENCES topics(id) ON DELETE CASCADE,
    role TEXT NOT NULL,                         -- 'user' | 'assistant'
    content TEXT NOT NULL,
    sources TEXT,                               -- JSON list for assistant messages
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS cache (
    k TEXT PRIMARY KEY,
    v TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def connect():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# --- Topics ---

def upsert_topic(name: str, context: str = "") -> int:
    with connect() as c:
        c.execute(
            "INSERT INTO topics(name, context) VALUES(?, ?) "
            "ON CONFLICT(name) DO UPDATE SET context = excluded.context",
            (name.strip(), context),
        )
        return c.execute("SELECT id FROM topics WHERE name = ?", (name.strip(),)).fetchone()[0]


def list_topics() -> List[dict]:
    with connect() as c:
        rows = c.execute("SELECT id, name, context FROM topics ORDER BY created_at DESC").fetchall()
    return [{"id": r[0], "name": r[1], "context": r[2]} for r in rows]


def delete_topic(topic_id: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM topics WHERE id = ?", (topic_id,))


# --- Papers ---

def save_paper(topic_id: int, paper: Paper) -> None:
    """Insert or update a paper and link it to the topic. An existing
    extraction is kept if the new copy doesn't have one."""
    with connect() as c:
        row = c.execute("SELECT data FROM papers WHERE key = ?", (paper.key,)).fetchone()
        if row and paper.extraction is None:
            old = Paper.model_validate_json(row[0])
            paper.extraction, paper.extracted_from = old.extraction, old.extracted_from
        c.execute(
            "INSERT INTO papers(key, data) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET data = excluded.data",
            (paper.key, paper.model_dump_json()),
        )
        c.execute(
            "INSERT OR IGNORE INTO topic_papers(topic_id, paper_key) VALUES(?, ?)",
            (topic_id, paper.key),
        )


def remove_paper(topic_id: int, key: str) -> None:
    with connect() as c:
        c.execute("DELETE FROM topic_papers WHERE topic_id = ? AND paper_key = ?", (topic_id, key))


def topic_papers(topic_id: int) -> List[Paper]:
    with connect() as c:
        rows = c.execute(
            "SELECT p.data FROM papers p JOIN topic_papers tp ON tp.paper_key = p.key "
            "WHERE tp.topic_id = ? ORDER BY tp.added_at", (topic_id,),
        ).fetchall()
    return [Paper.model_validate_json(r[0]) for r in rows]


# --- Synopses ---

def save_synopsis(topic_id: int, title: str, markdown: str) -> int:
    with connect() as c:
        cur = c.execute("INSERT INTO synopses(topic_id, title, markdown) VALUES(?, ?, ?)",
                        (topic_id, title, markdown))
        return cur.lastrowid


def update_synopsis(synopsis_id: int, markdown: str) -> None:
    with connect() as c:
        c.execute("UPDATE synopses SET markdown = ? WHERE id = ?", (markdown, synopsis_id))


def list_synopses(topic_id: int) -> List[dict]:
    with connect() as c:
        rows = c.execute("SELECT id, title, markdown, created_at FROM synopses "
                         "WHERE topic_id = ? ORDER BY id DESC", (topic_id,)).fetchall()
    return [{"id": r[0], "title": r[1], "markdown": r[2], "created_at": r[3]} for r in rows]


def delete_synopsis(synopsis_id: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM synopses WHERE id = ?", (synopsis_id,))


# --- Chat history ---

def add_message(topic_id: int, role: str, content: str, sources: Optional[list] = None) -> None:
    with connect() as c:
        c.execute("INSERT INTO chat(topic_id, role, content, sources) VALUES(?, ?, ?, ?)",
                  (topic_id, role, content, json.dumps(sources) if sources is not None else None))


def chat_history(topic_id: int) -> List[dict]:
    with connect() as c:
        rows = c.execute("SELECT role, content, sources FROM chat WHERE topic_id = ? ORDER BY id",
                         (topic_id,)).fetchall()
    return [{"role": r[0], "content": r[1], "sources": json.loads(r[2]) if r[2] else []}
            for r in rows]


def clear_chat(topic_id: int) -> None:
    with connect() as c:
        c.execute("DELETE FROM chat WHERE topic_id = ?", (topic_id,))


# --- Cache ---

def _cache_key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def cache_get(*parts) -> Optional[object]:
    with connect() as c:
        row = c.execute("SELECT v FROM cache WHERE k = ?", (_cache_key(*parts),)).fetchone()
    return json.loads(row[0]) if row else None


def cache_set(value: object, *parts) -> None:
    with connect() as c:
        c.execute(
            "INSERT OR REPLACE INTO cache(k, v) VALUES(?, ?)",
            (_cache_key(*parts), json.dumps(value)),
        )
