"""The reader prompt, and the legibility -> confidence mapping it feeds.

Two things about this prompt matter more than its wording.

**It asks for strict JSON with a per-field legibility self-report.** The legibility grade
is what the base confidence column is built from, and -- more importantly -- it is what
makes the self-contradiction cohort in ``cohorts.py`` possible. The prompt states that an
absent field must be reported as ``MISSING`` *and* graded ``absent``; a row that returns
``MISSING`` while grading the same field ``clear`` or ``partial`` has violated its own
instruction. That is a logical inconsistency, not a statistical enrichment, and it was the
single largest lever of the campaign.

**The verbatim conventions are ours and are deliberately different from the public
notebooks**, which fold letters onto the modern alphabet. This is a verbatim task: the
historical spellings are frequently correct, and normalising them destroys score.
"""

from __future__ import annotations

PROMPT = """You are transcribing a historical Danish museum specimen label (cropped to the label band).
Read ONLY the collection DATE and the LOCALITY (place of collection), and rate how clearly you can read each.

Return ONLY a JSON object, no other text, with exactly these keys:
{"verbatimDate": "<date or MISSING>", "date_legibility": "clear|partial|unreadable|absent", "verbatimLocality": "<locality or MISSING>", "locality_legibility": "clear|partial|unreadable|absent"}

Transcription rules:
- Copy the text VERBATIM. Do NOT fix or normalize spelling, letters, dates, or punctuation (keep the original exactly: a written aa stays aa, y-with-diaeresis stays as written).
- DATE = a real calendar date (day/month/year; may be Roman numerals, a Danish month word, or just a year). Never invent one from a name, a stamp, or a place.
- LOCALITY = a real place name (town, wood, coast, island; may carry a "DENMARK:"/district prefix or coordinates). A short abbreviation like "Kb", "Al", "Ti", "NS" IS a valid locality.
- These are NOT a date or locality -- ignore them: collector labels ("Coll.", "leg.", surnames such as Schiodte, Hoeg, West, Johansen, Hansen, Jacobsen), determiners ("det."), species/determination names (Aphodius, Onthophagus, Geotrupes, rufipes, fimetarius), museum/accession stamps ("Mus.", "Lev.", "NHMD", barcodes), the country word "Dania", and substrate notes ("kogodning").
- If several separate cards each carry a value, join them with " | " in reading order.
- If a field is absent, output "MISSING" for it and "absent" for its legibility.

Legibility: "clear" = every character certain; "partial" = some characters guessed; "unreadable" = a label is present but you cannot read it; "absent" = no such label exists (then the value must be MISSING)."""

# Legibility -> base confidence. A confident MISSING ("absent") ranks ABOVE a shaky guess
# ("unreadable"), which is what AURC rewards. Hand-set, not fitted.
#
# Note 0.85 for "absent": this is the second-highest band, which is exactly why a FALSE
# MISSING is the most expensive row shape in the metric -- a maximal error (NED 1.0)
# carried at high rank. See cohorts.py.
LEGIBILITY_TO_CONF = {
    "clear": 0.92,
    "partial": 0.55,
    "unreadable": 0.20,
    "absent": 0.85,
}
DEFAULT_CONF = 0.35


def confidence_of(legibility: str) -> float:
    """Map a legibility grade to its base confidence."""
    return LEGIBILITY_TO_CONF.get(str(legibility).strip().lower(), DEFAULT_CONF)
