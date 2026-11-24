"""Thin wrappers around the free APIs: Groq (fast chat) and Google Gemini
(long inputs, embeddings). Keys come from the local .env (see config.py)."""
from __future__ import annotations

import json
import time
from typing import List, Optional, Union

import config

_groq = None
_google = None


class MissingKey(RuntimeError):
    pass


class GeminiBusy(RuntimeError):
    """Every Gemini model is overloaded / unavailable right now."""


def _groq_client():
    global _groq
    if _groq is None:
        if not config.GROQ_API_KEY:
            raise MissingKey(f"GROQ_API_KEY is not set in {config.ENV_PATH}")
        from groq import Groq
        _groq = Groq(api_key=config.GROQ_API_KEY, timeout=60, max_retries=1)
    return _groq


def _google_client():
    global _google
    if _google is None:
        if not config.GOOGLE_API_KEY:
            raise MissingKey(f"GOOGLE_API_KEY is not set in {config.ENV_PATH}")
        from google import genai
        from google.genai import types
        # No SDK-level retries: we handle busy models ourselves by switching model.
        _google = genai.Client(api_key=config.GOOGLE_API_KEY, http_options=types.HttpOptions(
            timeout=120_000,   # ms: a hung request counts as "busy" and we move on
            retry_options=types.HttpRetryOptions(attempts=1)))
    return _google


_retired: set = set()        # models that returned 404 (not available for this key)
_busy_until: dict = {}       # model -> time when we may try it again
last_gemini_model: Optional[str] = None


def _is_busy(msg: str) -> bool:
    return any(s in msg for s in ("429", "500", "503", "504", "unavailable", "overloaded",
                                  "high demand", "rate", "quota", "exhausted", "timed out",
                                  "timeout", "deadline"))


def _gemini_call(make_request):
    """Run `make_request(model)` on the first Gemini model that works.
    404 -> model retired, skip for this session. 503/429 -> model busy, skip for 60 s.
    If every model is busy, wait and go round again (3 rounds)."""
    global last_gemini_model
    last_error = None
    for round_no in range(3):
        for model in config.GEMINI_MODELS:
            if model in _retired or _busy_until.get(model, 0) > time.time():
                continue
            try:
                result = make_request(model)
                last_gemini_model = model
                return result
            except MissingKey:
                raise
            except Exception as e:
                msg = str(e).lower()
                if "404" in msg or "not_found" in msg or "no longer available" in msg:
                    _retired.add(model)
                elif _is_busy(msg):
                    _busy_until[model] = time.time() + 60
                else:
                    raise
                last_error = e
        if all(m in _retired for m in config.GEMINI_MODELS):
            break
        time.sleep(10 * (round_no + 1))
        _busy_until.clear()
    raise GeminiBusy("All Gemini models are busy or unavailable right now; try again in a "
                       f"minute. Last error: {str(last_error)[:150]}")


def _with_retry(fn, attempts: int = 5):
    """Free tiers return 429 when you go too fast. Wait and retry."""
    for i in range(attempts):
        try:
            return fn()
        except MissingKey:
            raise
        except Exception as e:  # SDKs use different exception types for 429
            msg = str(e).lower()
            busy = any(s in msg for s in ("429", "rate", "quota", "exhausted", "503", "overloaded"))
            if i < attempts - 1 and busy:
                time.sleep(min(60, 5 * 2 ** i))
                continue
            raise


# ---------- Groq ----------

def chat_json(messages: List[dict], model: Optional[str] = None) -> dict:
    """Groq chat completion forced to return a JSON object."""
    resp = _with_retry(lambda: _groq_client().chat.completions.create(
        model=model or config.FAST_MODEL,
        messages=messages,
        temperature=0,
        response_format={"type": "json_object"},
    ))
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return {}


def groq_json_any(prompt: str, max_tokens: int = 2000) -> dict:
    """JSON answer from the first Groq model that accepts the request. Free tier is
    ~8k tokens/minute *per model*, so keep `prompt` short (~20k chars) and rotate models."""
    last = None
    for round_no in range(4):
        for model in config.GROQ_FALLBACK_MODELS:
            try:
                resp = _groq_client().chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], temperature=0,
                    max_tokens=max_tokens, response_format={"type": "json_object"})
                return json.loads(resp.choices[0].message.content or "{}")
            except MissingKey:
                raise
            except Exception as e:   # 429 / 413 / bad JSON -> next model
                last = e
        if "429" not in str(last) and "rate" not in str(last).lower():
            break
        time.sleep(20)   # per-minute limits reset; try again
    raise RuntimeError(f"Gemini is busy and the Groq fallback failed too: {str(last)[:150]}")


def chat(messages: List[dict], temperature: float = 0.2) -> str:
    """Plain-text chat. Uses Groq; falls back to Gemini if Groq fails
    (e.g. its free-tier limit is used up) and a Google key is set."""
    try:
        resp = _with_retry(lambda: _groq_client().chat.completions.create(
            model=config.FAST_MODEL, messages=messages, temperature=temperature,
        ), attempts=3)
        return resp.choices[0].message.content or ""
    except Exception:
        if not config.GOOGLE_API_KEY:
            raise
        prompt = "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in messages)
        return gemini_text(prompt + "\n\nASSISTANT:", temperature=temperature)


# ---------- Gemini ----------

def gemini_text(contents: Union[str, list], temperature: float = 0.2) -> str:
    from google.genai import types

    resp = _gemini_call(lambda model: _google_client().models.generate_content(
        model=model, contents=contents,
        config=types.GenerateContentConfig(temperature=temperature),
    ))
    return resp.text or ""


def gemini_json(contents: Union[str, list], schema: Optional[type] = None,
                temperature: float = 0) -> dict:
    """Gemini call for long inputs. `contents` is a prompt string, or a list of
    strings / PDF parts. If a Pydantic `schema` is given, the output follows it."""
    from google.genai import types

    cfg = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_schema=schema,
    )
    resp = _gemini_call(lambda model: _google_client().models.generate_content(
        model=model, contents=contents, config=cfg,
    ))
    try:
        return json.loads(resp.text)
    except (json.JSONDecodeError, TypeError):
        return {}


def pdf_part(pdf_bytes: bytes):
    """Wrap raw PDF bytes so Gemini can read the PDF directly (scanned papers)."""
    from google.genai import types
    return types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf")


def embed(texts: List[str], task: str = "RETRIEVAL_DOCUMENT") -> List[List[float]]:
    """Gemini embeddings. Use task='RETRIEVAL_QUERY' for questions."""
    from google.genai import types

    out: List[List[float]] = []
    for i in range(0, len(texts), 100):  # API batch limit
        batch = texts[i:i + 100]
        resp = _with_retry(lambda: _google_client().models.embed_content(
            model=config.EMBED_MODEL, contents=batch,
            config=types.EmbedContentConfig(task_type=task,
                                            output_dimensionality=config.EMBED_DIM),
        ))
        out.extend(e.values for e in resp.embeddings)
    return out
