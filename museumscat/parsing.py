#Parse a reader's raw response into fields, and read back its legibility grades.



from __future__ import annotations

import json
import re

from museumscat.config import DATE_COL, LOCALITY_COL, MISSING

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

DATE_LEGIBILITY_KEY = "date_legibility"
LOCALITY_LEGIBILITY_KEY = "locality_legibility"


def parse_json(raw: str) -> dict:
    """Best-effort JSON extraction from a model response. Never raises."""
    if not raw:
        return {}
    text = _FENCE_RE.sub("", str(raw)).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        pass
    match = _JSON_RE.search(text)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _clean(value: object) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    return text or MISSING


def extract(parsed: dict) -> tuple[str, str, str, str]:
    """Return (date, locality, date_legibility, locality_legibility)."""
    return (
        _clean(parsed.get(DATE_COL)),
        _clean(parsed.get(LOCALITY_COL)),
        str(parsed.get(DATE_LEGIBILITY_KEY, "")).strip().lower(),
        str(parsed.get(LOCALITY_LEGIBILITY_KEY, "")).strip().lower(),
    )


def legibility(raw: str, field: str) -> str:

    key = DATE_LEGIBILITY_KEY if field.startswith("date") else LOCALITY_LEGIBILITY_KEY
    return str(parse_json(raw).get(key, "")).strip().lower()


def is_missing(value: object) -> bool:
    """True when a value is the literal MISSING sentinel, in any casing."""
    return str(value).strip().casefold() == MISSING.casefold()
