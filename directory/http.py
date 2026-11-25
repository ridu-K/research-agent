"""Shared HTTP helper: caching + retry on 429/5xx for the free scholarly APIs."""
from __future__ import annotations

import time
from typing import Optional

import requests

import config
import db

HEADERS = {"User-Agent": "research-agent/0.1 (personal research tool)"}


def get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
        as_json: bool = True, attempts: int = 4):
    """GET with an on-disk cache. Returns parsed JSON (or text if as_json=False)."""
    cached = db.cache_get("http", url, params or {})
    if cached is not None:
        return cached

    last_err = None
    for i in range(attempts):
        try:
            r = requests.get(url, params=params, headers={**HEADERS, **(headers or {})},
                             timeout=config.HTTP_TIMEOUT)
            if r.status_code == 429 or r.status_code >= 500:
                last_err = RuntimeError(f"HTTP {r.status_code} from {url}")
                time.sleep(2 ** (i + 1))
                continue
            r.raise_for_status()
            value = r.json() if as_json else r.text
            db.cache_set(value, "http", url, params or {})
            return value
        except requests.RequestException as e:
            last_err = e
            time.sleep(2 ** i)
    raise last_err
