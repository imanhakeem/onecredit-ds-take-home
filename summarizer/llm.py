"""A small client for any OpenAI-compatible chat API (Gemini by default), with a disk cache.

Every response is saved under runs/cache/, keyed by a hash of the exact request. Re-running a version, or re-scoring it,
never calls the model again unless the prompt or the model changed. That keeps a free tier's daily quota for new work.

Settings come from environment variables or a .env file at the repo root (see .env.example).
"""

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from .data import ROOT, RUNS

CACHE = RUNS / "cache"


def load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_env()

BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash")
RPM = float(os.environ.get("LLM_RPM", "8"))  # stay under the free tier's requests-per-minute limit
REASONING_EFFORT = os.environ.get("LLM_REASONING_EFFORT", "")  # e.g. "low"; empty means the provider's default

_last_call = 0.0


class LLMError(RuntimeError):
    pass


class QuotaExhausted(LLMError):
    pass


def _key() -> str:
    key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMError("No API key. Copy .env.example to .env and set LLM_API_KEY.")
    return key


def _request_body(messages: list[dict], model: str, temperature: float, reasoning_effort: str) -> dict:
    body: dict = {"model": model, "messages": messages, "temperature": temperature}
    if reasoning_effort:
        body["reasoning_effort"] = reasoning_effort
    return body


def cache_key(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:24]


def chat(
    messages: list[dict],
    model: str = MODEL,
    temperature: float = 0.2,
    offline: bool = False,
    reasoning_effort: str = REASONING_EFFORT,
) -> dict:
    """Return {"text", "usage", "cached", "key"}. With offline=True, only the cache is used."""
    global _last_call
    body = _request_body(messages, model, temperature, reasoning_effort)
    key = cache_key(body)
    path = CACHE / f"{key}.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        return {"text": strip_thoughts(saved["text"]), "usage": saved.get("usage"), "cached": True, "key": key}
    if offline:
        raise LLMError(f"Not in cache: {key}")

    for attempt in range(6):
        wait = 60 / RPM - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()
        req = urllib.request.Request(
            f"{BASE_URL}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_key()}",
                "User-Agent": "company-summaries/0.1",  # Groq's firewall rejects Python's default urllib agent
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 (fixed https endpoint)
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            if e.code == 429 and ("PerDay" in detail or "per day" in detail.lower()):
                # A daily cap won't reset for hours, so retrying is pointless. Stop and say so.
                raise QuotaExhausted(f"Daily free-tier quota used up for {model}. Try another model or tomorrow.") from e
            if e.code in (429, 500, 503) and attempt < 5:
                delay = min(120.0, float(e.headers.get("Retry-After") or 15 * 2**attempt))
                print(f"  HTTP {e.code}, retrying in {delay:.0f}s", flush=True)
                time.sleep(delay)
                continue
            raise LLMError(f"HTTP {e.code}: {detail[:500]}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt < 5:
                time.sleep(10 * 2**attempt)
                continue
            raise LLMError(str(e)) from e
    else:
        raise LLMError("Gave up after retries")

    raw = data["choices"][0]["message"].get("content") or ""
    text = strip_thoughts(raw)
    CACHE.mkdir(parents=True, exist_ok=True)
    saved = {"request": body, "text": text, "raw": raw, "usage": data.get("usage"), "model": data.get("model", model), "at": time.time()}
    path.write_text(json.dumps(saved, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"text": text, "usage": data.get("usage"), "cached": False, "key": key}


def strip_thoughts(text: str) -> str:
    """Some models (Gemma, Qwen) put their reasoning in <thought> or <think> tags before the answer. Keep the answer."""
    return re.sub(r"<(thought|think)>.*?</\1>", "", text, flags=re.S).strip()


if __name__ == "__main__":
    print(chat([{"role": "user", "content": "Reply with the single word: ok"}], temperature=0)["text"])
