"""VLM reader clients: OpenRouter (most models) and the direct Gemini API (free tier).

Both paths share one image encoder, so a read from either differs from a cached read in
exactly one respect -- the model. Every reader comparison in the write-up depends on that
being true.

Two operational details that cost us real time and are worth inheriting:

**Reasoning models return empty content.** A model with a thinking budget can spend the
entire completion allowance thinking and hand back ``""`` with ``finish_reason="length"``.
The obvious fix -- retry with reasoning disabled -- is rejected outright by Gemini 3 with
HTTP 400 "Reasoning is mandatory", so the fallback has to be *more* room rather than less
thinking. Only the failure path pays for this, so cached readers still reproduce exactly.

**Temperature is a real parameter and we left it at zero.** ``temperature`` defaults to 0
for reproducibility. Passing ``>0`` and reading the same image K times gives a candidate
pool whose dispersion is a confidence signal in its own right; our two-draw version paid,
and we never pushed past K=2. See docs/story.md.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from museumscat.prompts import PROMPT

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

DEFAULT_MODEL = "google/gemini-3.7-flash"
DEFAULT_TIMEOUT_S = 180
MAX_TOKENS = 512
MAX_WIDTH = 1568
JPEG_QUALITY = 90

# Transient failures are expected across a few thousand images: retry those and let
# anything else surface immediately.
RETRY_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
MAX_RETRIES = 5
BACKOFF_BASE_S = 2.0

NO_REASONING = {"reasoning": {"enabled": False}}

# Validated Gemini pool. Free-tier quota is per DAY and per MODEL, so rotating this pool
# yields roughly 120 free validated reads a day.
#
# NEVER add a "-lite" tier: gemini-3.5-flash-lite returned a collector's name as a
# locality and mangled Danish (Orholm for Ørholm) on blank labels.
GEMINI_POOL = (
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-2.5-flash",
)


def encode_image(image_path: str | Path, max_width: int = MAX_WIDTH) -> str:
    """Downscale to max_width and return base64 JPEG. Shared by every reader path."""
    from PIL import Image

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        if img.width > max_width:
            height = round(img.height * max_width / img.width)
            img = img.resize((max_width, height), Image.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _api_key(name: str) -> str:
    key = os.environ.get(name, "").strip()
    if key:
        return key
    raise RuntimeError(f"{name} is not set; export it before running a reader")


def _post(url: str, body: dict, headers: dict, timeout: int) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def query_openrouter(
    image_path: str | Path,
    model: str = DEFAULT_MODEL,
    prompt: str | None = None,
    *,
    max_width: int = MAX_WIDTH,
    timeout: int = DEFAULT_TIMEOUT_S,
    api_key: str | None = None,
    temperature: float = 0.0,
    disable_reasoning: bool = False,
) -> str:
    """One read through OpenRouter. Returns the model's raw text (JSON we then parse)."""
    key = api_key or _api_key("OPENROUTER_API_KEY")
    encoded = encode_image(image_path, max_width)
    base = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt if prompt is not None else PROMPT},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
            ],
        }],
        "temperature": temperature,
        "max_tokens": MAX_TOKENS,
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    def send(extra: dict | None) -> dict:
        body = dict(base)
        if extra:
            body.update(extra)
        return _post(OPENROUTER_URL, body, headers, timeout)

    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            payload = send(NO_REASONING if disable_reasoning else None)
            choices = payload.get("choices") or []
            if not choices:
                raise RuntimeError(f"no choices in response: {str(payload)[:300]}")
            content = choices[0]["message"].get("content") or ""
            if not content.strip() or choices[0].get("finish_reason") == "length":
                content = _recover_truncated(send, content)
            return content
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            if exc.code not in RETRY_STATUSES:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(BACKOFF_BASE_S * (2 ** attempt))
    raise RuntimeError(f"failed after {MAX_RETRIES} attempts: {last_error}")


def _recover_truncated(send, content: str) -> str:
    """A reasoning model burned the budget thinking. Turn thinking off, or buy more room.

    Gemini 3 rejects NO_REASONING with HTTP 400 "Reasoning is mandatory", so on that one
    error we retry with a larger token budget instead -- the think block is billed against
    max_tokens, and a truncated answer means it ran out mid-thought.
    """
    try:
        retry = send(NO_REASONING)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        if exc.code != 400 or "easoning is mandatory" not in detail:
            raise
        retry = send({"max_tokens": MAX_TOKENS * 8})
    choices = retry.get("choices") or []
    return (choices[0]["message"].get("content") or content) if choices else content


def query_gemini(
    image_path: str | Path,
    model: str = "gemini-3.7-flash",
    prompt: str | None = None,
    *,
    max_width: int = MAX_WIDTH,
    timeout: int = DEFAULT_TIMEOUT_S,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> str:
    """One read through the direct Gemini API (the free tier we used for validation)."""
    key = api_key or _api_key("GEMINI_API_KEY")
    encoded = encode_image(image_path, max_width)
    body = {
        "contents": [{
            "parts": [
                {"text": prompt if prompt is not None else PROMPT},
                {"inline_data": {"mime_type": "image/jpeg", "data": encoded}},
            ],
        }],
        "generationConfig": {"temperature": temperature, "maxOutputTokens": MAX_TOKENS * 8},
    }
    url = GEMINI_URL.format(model=model) + f"?key={key}"
    payload = _post(url, body, {"Content-Type": "application/json"}, timeout)
    candidates = payload.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"no candidates in response: {str(payload)[:300]}")
    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts)


def query(image_path: str | Path, model: str = DEFAULT_MODEL, **kwargs) -> str:
    """Route to the right backend. Bare model names go direct; slashed names to OpenRouter."""
    if "/" in model:
        return query_openrouter(image_path, model, **kwargs)
    return query_gemini(image_path, model, **kwargs)
