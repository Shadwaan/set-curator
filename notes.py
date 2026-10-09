"""
Your notes from the sax page (sax_notes.json) as gentle hints for the curation.

    role words   "good sustainer", "good warmer", "a good moment"  push that role for the song (2 std; hedged 1)
                 (hedged -- "maybe", "probably", "at best", "?", "semi" -- counts half)
    "driving"    a small push towards Sustainers; "laid back" a small push towards Warmers
    energy       "low energy" / "mid energy" / "high energy" puts the song in that Sax and Aura energy tier
    listening    "hard to mix", "not a dance track", "not danceable": a song to listen to more than mix. In Set roles
                 it can only be an Opener, Moment or Closer; SC / Aura also lists it in "Listening (hard to mix)"

The push is in standard deviations of the role score (see roles.assign), so it nudges and the audio can
still disagree. A note that says "not", "no" or "isn't" is not read for roles.
"""

import json
import re

from cluster import HERE

NOTES = HERE / "sax_notes.json"
ROLE_WORDS = [
    (r"\bopen(er|ers|ing)\b", "1 Openers"), (r"\bwarm(er|ers|-?up)\b", "2 Warmers"), (r"\bmomentum\b", "3 Momentum builders"),
    (r"\bcrowd\b", "4 Crowd attractors"), (r"\bpeak\b", "5 Peak time"), (r"\bsustainers?\b", "6 Sustainers"),
    (r"\bmoments?\b", "7 Moments"), (r"\bclos(er|ers|ing)\b", "8 Closers"),
]
LOOSE = [(r"\bdriv(e|ing)\b", "6 Sustainers", 0.4), (r"\blaid[- ]back\b", "2 Warmers", 0.4)]
HEDGES = re.compile(r"\b(maybe|probably|at best|semi|kind of|sort of)\b|\?", re.I)
NEGATION = re.compile(r"\b(not|no|isn't|isnt|never|don't|dont)\b", re.I)
TIER_WORDS = [(r"\blow energy\b", "Low energy"), (r"\bmid(dle)? energy\b|\bmedium energy\b", "Mid energy"), (r"\bhigh energy\b", "High energy")]


def parse(text: str) -> tuple[list[tuple[str, float]], str | None]:
    """-> ([(role, push)], energy tier or None)"""
    scale = 0.5 if HEDGES.search(text) else 1.0
    pushes = []
    if not NEGATION.search(text):
        pushes += [(role, 2.0 * scale) for pat, role in ROLE_WORDS if re.search(pat, text, re.I)]
        pushes += [(role, push) for pat, role, push in LOOSE if re.search(pat, text, re.I)]
    tier = next((t for pat, t in TIER_WORDS if re.search(pat, text, re.I)), None)
    return pushes, tier


LISTENING = re.compile(r"hard to mix|can'?t mix|cannot mix|(?:wouldn'?t|would not) know how to mix|not (?:really )?a dance|not danceable|listening", re.I)


def listening(merged: dict[str, str] | None = None) -> set[str]:
    return {i for i, t in load(merged).items() if LISTENING.search(t)}


def load(merged: dict[str, str] | None = None) -> dict[str, str]:
    """{song id: note}; a note on a copy belongs to the song that was kept."""
    raw = json.loads(NOTES.read_text(encoding="utf-8")) if NOTES.exists() else {}
    return {(merged or {}).get(i, i): t for i, t in raw.items()}


def role_nudges(merged: dict[str, str] | None = None) -> dict[str, list[tuple[str, float]]]:
    out = {i: p for i, t in load(merged).items() if (p := parse(t)[0])}
    return out


def tier_overrides(merged: dict[str, str] | None = None) -> dict[str, str]:
    return {i: tier for i, t in load(merged).items() if (tier := parse(t)[1])}
