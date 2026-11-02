"""
Acid: the squelchy, resonant TB-303 bassline of acid house, acid techno and acid trance.

Three signals, none perfect alone:
  * Discogs: the share of a song's releases that carry an "Acid ..." style (discogs_styles.json, from discogs_styles.py)
  * the genre model's acid share (Discogs-400 "Acid" and "Acid House")
  * the word "acid" in the title or file name
Candidates go to the review page ("Acid check"); the songs you confirm (sax_acid.json) get the ACID tag and an SC / Acid folder.

    python acid.py          # -> acid_candidates.json
"""

import json
import re

import numpy as np

from cluster import HERE

STYLES = HERE / "discogs_styles.json"
ACID_STYLES = {"Acid", "Acid House", "Acid Trance", "Acid Techno"}
OUT = HERE / "acid_candidates.json"
ANSWERS = HERE / "sax_acid.json"


def confirmed(merged=None) -> list[str]:
    a = json.loads(ANSWERS.read_text()) if ANSWERS.exists() else {}
    return sorted({(merged or {}).get(i, i) for i, v in a.items() if "Acid" in v})


ROOM = HERE / "sax_303.json"                      # your answers in the "303 room check": {id: ["Room for a 303"] or []}
ROOM_CANDIDATES = HERE / "room303_candidates.json"
ROOM303 = re.compile(r"\b(?:play|space|room)\b[^.;]*\b303\b|\b303\b[^.;]*\b(?:over|on top)\b", re.I)
NOT_ACID = re.compile(r"\b(?:no|not)\s+acid\b|\bno\s+303", re.I)


def room_seeds(merged=None) -> dict[str, bool]:
    """{song id: certain?} for songs whose note says you can play a 303 over them ("not acid but can play a 303 over this").
    A note with "maybe" is not certain: it is offered in the 303 room check instead."""
    import notes
    out = {}
    for i, t in notes.load(merged).items():
        if re.search(r"\b303\b", t) and ROOM303.search(t):
            out[i] = not re.search(r"\b(maybe|probably|perhaps)\b|\?", t, re.I)
    return out


def room_members(merged=None) -> list[str]:
    said = json.loads(ROOM.read_text()) if ROOM.exists() else {}
    keep = {i for i, ok in room_seeds(merged).items() if ok} | {(merged or {}).get(i, i) for i, v in said.items() if v}
    return sorted(keep - set(confirmed(merged)))


def tags(merged=None) -> dict[str, list[str]]:
    """{song id: ["ACID"] or ["ACID-ROOM"]} for the comment."""
    out = {i: ["ACID"] for i in confirmed(merged)}
    for i in room_members(merged):
        out.setdefault(i, []).append("ACID-ROOM")
    return out


def room_candidates(top: int = 16) -> dict[str, list[str]]:
    """Songs that sound like the ones you can play a 303 over, that are not acid themselves."""
    import cluster
    import palette
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, merged = cluster.merge_copies(tracks, ids, vecs)
    sec = palette.load()
    reggae = json.loads((HERE / "reggae_all.json").read_text())
    said = json.loads(ROOM.read_text()) if ROOM.exists() else {}
    seeds = [i for i in room_members(merged) if i in sec]
    maybe = [i for i, ok in room_seeds(merged).items() if not ok and i not in said]
    pool = [i for i in songs if i in sec and reggae.get(i, 0) < 0.3 and i not in seeds and i not in set(confirmed(merged)) and i not in said and i not in maybe]
    score = sum(palette.similarity(sec, s, pool) for s in seeds) / len(seeds)
    z = (score - score.mean()) / score.std()
    out = {i: ["your note says 'maybe' you can play a 303 over it"] for i in maybe}
    for k in np.argsort(-score)[:top]:
        out[pool[k]] = [f"sounds like the songs you can play a 303 over ({z[k]:+.1f} sd)"]
    ROOM_CANDIDATES.write_text(json.dumps(out))
    return out


def signals() -> dict[str, dict]:
    import cluster
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, merged = cluster.merge_copies(tracks, ids, vecs)
    classes, probs = cluster.style_predictions(songs)
    cols = [k for k, c in enumerate(classes) if c in ("Electronic---Acid", "Electronic---Acid House")]
    styles = json.loads(STYLES.read_text()) if STYLES.exists() else {}
    out = {}
    for i in songs:
        d = styles.get(i, {})
        rel = max(d.get("releases", 0), 1)
        acid_rel = max((n for s, n in d.get("styles", {}).items() if s in ACID_STYLES), default=0)   # not Acid Jazz or Acid Rock
        word = "acid" in (tracks[i]["title"] + " " + tracks[i]["path"].split("/")[-1]).lower()
        out[i] = {"discogs": acid_rel / rel if d.get("releases") else 0.0, "discogs_n": acid_rel,
                  "model": float(sum(probs[i][k] for k in cols)), "word": word}
    return out


def candidates(top: int = 40) -> dict[str, list[str]]:
    import cluster
    tracks = cluster.load(["Deep Tech FLAC"])[0]
    sig = signals()
    done = json.loads(ANSWERS.read_text()) if ANSWERS.exists() else {}
    strength = lambda s: (2.0 if s["word"] else 0) + 2.0 * s["discogs"] + (1.5 if s["model"] >= 0.2 else 0) + 5 * min(s["model"], 0.2)
    ranked = sorted((i for i, s in sig.items() if i not in done and (s["word"] or s["discogs"] > 0 or s["model"] >= 0.08)),
                    key=lambda i: -strength(sig[i]))[:top]
    out = {}
    for i in ranked:
        s, why = sig[i], []
        if s["word"]:
            why.append("'acid' is in its title")
        if s["discogs"] > 0:
            why.append(f"Discogs tags {s['discogs_n']} of its releases with an Acid style")
        if s["model"] >= 0.08:
            why.append(f"the genre model gives acid {s['model']:.2f}")
        out[i] = why
    OUT.write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    import cluster
    tr = cluster.load(["Deep Tech FLAC"])[0]
    c = candidates()
    print(len(c), "acid candidates")
    for i, why in c.items():
        print(f"  {tr[i]['artist'][:22]:22} - {tr[i]['title'][:34]:34} | {'; '.join(why)}")
