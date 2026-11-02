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
    (r"\bgroove[- ]?reset|\bsettler\b|\bcalm(?:s|ing)?(?: the)?(?: people| crowd| dancefloor)? down\b", "9 Groove resets"),
    (r"\bopen(er|ers|ing)\b", "1 Openers"), (r"\bwarm(er|ers|-?up)\b", "2 Warmers"), (r"\bmomentum\b", "3 Momentum builders"),
    (r"\bcrowd\b", "4 Crowd attractors"), (r"\bpeak\b", "5 Peak time"), (r"\bsustainers?\b", "6 Sustainers"),
    (r"\bmoments?\b", "7 Moments"), (r"\bclos(er|ers|ing)\b", "8 Closers"),
]
LOOSE = [(r"\bdriv(e|ing)\b", "6 Sustainers", 0.4), (r"\blaid[- ]back\b", "2 Warmers", 0.4)]
HEDGES = re.compile(r"\b(maybe|probably|at best|semi|kind of|sort of)\b|\?", re.I)
NEGATION = re.compile(r"\b(?:not|nto|no|isn'?t|isnt|doesn'?t|doesnt|wouldn'?t|wouldnt|never|don'?t|dont|nah)\b", re.I)


def positive_part(clause: str) -> str:
    """What a clause says before a "not ...": "its a momentum builder nto groove reset" -> "its a momentum builder"."""
    m = NEGATION.search(clause)
    return clause[:m.start()] if m else clause
TIER_WORDS = [(r"\blow energy\b", "Low energy"), (r"\bmid(dle)? energy\b|\bmedium energy\b", "Mid energy"), (r"\bhigh energy\b", "High energy")]


# energy words that come with a qualifier ("edging on high", "high for dub sets") are context, not a verdict
QUALIFIED = re.compile(r"\b(edging|bordering|lower end|higher side|upper end|near(?:er)?|unless|for (?:dub|lower)|dub sets|dub music|break sets|in dub)\b", re.I)


# "a settler after a peak", "before the drop": the word after these is a place in the set, not the song's role
CONTEXT = re.compile(r"\b(?:after|before|following|during|into|from|through)\s+(?:a|an|the|your|every|heavy)?\s*\w+(?:\s+(?:bass|drop|section))?", re.I)


def parse(text: str) -> tuple[list[tuple[str, float]], str | None]:
    """-> ([(role, push)], energy tier or None)"""
    text = CONTEXT.sub(" ", text)
    scale = 0.5 if HEDGES.search(text) else 1.0
    pushes = []
    body = " ".join(positive_part(c) for c in CLAUSES.split(text))          # a role named after a "not" is not a nudge
    pushes += [(role, 2.0 * scale) for pat, role in ROLE_WORDS if re.search(pat, body, re.I)]
    pushes += [(role, push) for pat, role, push in LOOSE if re.search(pat, body, re.I)]
    tier = None if QUALIFIED.search(text) else next((t for pat, t in TIER_WORDS if re.search(pat, text, re.I)), None)
    return pushes, tier


LISTENING = re.compile(r"hard to mix|can'?t mix|cannot mix|(?:wouldn'?t|would not) know how to mix|not (?:really )?a dance|not danceable|listening", re.I)


def listening(merged: dict[str, str] | None = None) -> set[str]:
    return {i for i, t in load(merged).items() if LISTENING.search(t)}


CLAUSES = re.compile(r"[,.;]|\bbut\b|\bor\b|\bas well\b|\balso\b|\band\b", re.I)
EXTRA_ROLES = HERE / "role_extra.json"      # roles I suggested for a song when your note was unclear (editable): {id: [roles]}


def explicit_roles(merged: dict[str, str] | None = None) -> dict[str, list[str]]:
    """The roles a song fills, in the order your note names them ("better sustainer, can be moment" -> Sustainers, Moments).
    A part of a note with "maybe", "probably", "at best", "?" or a "no / not" is not counted. The first is the song's main
    role; the others place it in those role folders too."""
    out: dict[str, list[str]] = {}
    for i, t in load(merged).items():
        found = []
        for clause in CLAUSES.split(CONTEXT.sub(" ", t)):
            clause = positive_part(clause)
            if HEDGES.search(clause):
                continue
            hits = sorted((m.start(), role) for pat, role in ROLE_WORDS for m in [re.search(pat, clause, re.I)] if m)
            found += [r for _, r in hits if r not in found]
        if found:
            out[i] = found
    answered = HERE / "sax_roles.json"
    if answered.exists():
        for i, roles_ in json.loads(answered.read_text()).items():
            k = (merged or {}).get(i, i)
            for r in roles_:
                if r not in out.get(k, []):
                    out.setdefault(k, []).append(r)
    if EXTRA_ROLES.exists():
        for i, roles_ in json.loads(EXTRA_ROLES.read_text()).items():
            out[(merged or {}).get(i, i)] = roles_
    import groove
    for i in groove.dropped(merged):                        # resets you agreed to drop
        if "9 Groove resets" in out.get(i, []):
            out[i] = [r for r in out[i] if r != "9 Groove resets"]
            if not out[i]:
                del out[i]
    return out


# your own vocabulary: a note that uses one of these words tags the song with it (shown in the Rekordbox comment)
VIBES = [("BANGKOK", re.compile(r"\bbangkok\b", re.I)), ("BEACHBAR", re.compile(r"\bbeach(?:\s*bar)?\b", re.I)),
         ("SWEATY", re.compile(r"\bsweaty\b", re.I))]


def vibe_tags(merged: dict[str, str] | None = None) -> dict[str, list[str]]:
    """{song id: ["BANGKOK", "SWEATY", ...]} from the words you used in your notes."""
    out = {}
    for i, t in load(merged).items():
        tags = [name for name, pat in VIBES if pat.search(t)]
        if tags:
            out[i] = tags
    return out


def load(merged: dict[str, str] | None = None) -> dict[str, str]:
    """{song id: note}; a note on a copy belongs to the song that was kept."""
    raw = json.loads(NOTES.read_text(encoding="utf-8")) if NOTES.exists() else {}
    return {(merged or {}).get(i, i): t for i, t in raw.items()}


def role_nudges(merged: dict[str, str] | None = None) -> dict[str, list[tuple[str, float]]]:
    out = {i: p for i, t in load(merged).items() if (p := parse(t)[0])}
    return out


def tier_overrides(merged: dict[str, str] | None = None) -> dict[str, str]:
    return {i: tier for i, t in load(merged).items() if (tier := parse(t)[1])}
