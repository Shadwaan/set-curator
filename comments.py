"""
Tag every song's Rekordbox comment with its high end and the playlists it is from, so both show
in the track info on the deck:

    [OPEN | Dub Reggae Bass Addict, Dub | AURA | SAX, SAX-ROOM | BANGKOK, SWEATY | Dubstep] your own comment, if there was one

The last part is the song's sound style (genre).

SAX, SAX-MELODY and SAX-ROOM come from how you filed songs in the sax page (sax_filed.json), AURA from
your aura answers (sax_aura.json).

OPEN / BAL / CLOSED are thirds of the library by how much energy sits in the top of the spectrum
(highend.py). Running it again replaces the tag; your own comment text is kept.

    python comments.py             # preview
    python comments.py --apply     # write into Rekordbox (close it first)
"""

import argparse
import json
from collections import Counter

import numpy as np

import cluster
from cluster import HERE


def short(name: str) -> str:
    return name[:-5] if name.endswith(" AIFF") else name


SAX_TAGS = {"Has sax": "SAX", "Melody a sax could play": "SAX-MELODY", "Room for a sax": "SAX-ROOM"}


def your_tags() -> dict[str, list[str]]:
    """Tags from your own filing: {song id: ["SAX", "SAX-ROOM", "AURA"]}."""
    out: dict[str, list[str]] = {}
    f, a = HERE / "sax_filed.json", HERE / "sax_aura.json"
    for i, groups in (json.loads(f.read_text()).items() if f.exists() else []):
        out.setdefault(i, []).extend(SAX_TAGS[g] for g in SAX_TAGS if g in groups)
    for i, v in (json.loads(a.read_text()).items() if a.exists() else []):
        if v == ["Has the aura"]:
            out.setdefault(i, []).append("AURA")
    return out


def acid_tags(merged) -> dict[str, list[str]]:
    import acid
    return acid.tags(merged)


def playlists_of_song(own: str, kept: str, merged: dict[str, str], raw: dict) -> list[str]:
    """Your playlists for a song: this file's own first, in their existing order, then those of the song's other files."""
    members = [own, kept] + [c for c, k in merged.items() if k == kept]
    out: list[str] = []
    for m in members:
        for p in raw.get(m, {}).get("playlists", []):
            if p not in out:
                out.append(p)
    return out


VIBE_ORDER = ["BANGKOK", "BANGKOK?", "BEACHBAR", "BEACHBAR-CLUB", "SWEATY"]


def genre_tags(ids) -> dict[str, str]:
    """The sound style that sets each song apart from the rest of the library (Discogs-400 model on its stored embedding):
    the style with the highest share, damped by how common that style is, so a house track is not just "House". Last in the tag."""
    ids = list(ids)
    classes, probs = cluster.style_predictions(ids)
    lib = np.mean([probs[i] for i in ids], axis=0)
    from collections import Counter as C
    bare = C(c.split("---")[1] for c in classes)
    pretty = lambda c: c.split("---")[1] if bare[c.split("---")[1]] == 1 else c.replace("---", " ")
    return {i: pretty(classes[int(np.argmax(probs[i] / np.sqrt(lib + 1e-9)))]) for i in ids}


def build_prefixes():
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, merged = cluster.merge_copies(tracks, ids, vecs)
    raw = {t["id"]: t for t in json.loads((HERE / "tracks.json").read_text(encoding="utf-8"))}
    high = json.loads((HERE / "highend.json").read_text())
    score = cluster.openness()
    lo, hi = np.percentile(list(score.values()), [100 / 3, 200 / 3])
    tag = lambda i: "OPEN" if score[i] > hi else "CLOSED" if score[i] < lo else "BAL"
    mine = your_tags()
    acid_by = acid_tags(merged)
    import vibes as vibe_groups
    vibes = vibe_groups.tags(merged)                       # tags from the words in your notes and your Vibe check answers
    genres = genre_tags(list(songs))
    out = {}
    for i in list(songs) + list(merged):
        src = merged.get(i, i)                                 # a copy takes its twin's reading
        if src not in score:
            continue
        # a song stored as several files (one per playlist) lists the playlists of ALL its files on every file
        crates = [short(p) for p in playlists_of_song(i, src, merged, raw) if p != "Deep Tech FLAC"]
        mine_tags = mine.get(src, []) + [t for t in mine.get(i, []) if t not in mine.get(src, [])]
        parts = [tag(src)]
        if crates:
            parts.append(", ".join(crates))
        if "AURA" in mine_tags:
            parts.append("AURA")
        sax = [t for t in mine_tags if t.startswith("SAX")] + acid_by.get(src, [])
        if sax:
            parts.append(", ".join(sax))
        v = [t for t in VIBE_ORDER if t in vibes.get(src, [])]
        if any("beachbar" in c.lower() for c in crates):
            v = [t for t in v if t != "BEACHBAR"]       # the playlist name in the tag already says it
        if v:
            parts.append(", ".join(v))
        if genres.get(src):
            parts.append(genres[src])                          # the sound style, always the last part
        out[i] = "[" + " | ".join(parts) + "]"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write the comments into Rekordbox")
    args = ap.parse_args()
    prefixes = build_prefixes()
    print(f"{len(prefixes)} songs;", dict(Counter(v.strip("[]").split(" | ")[0] for v in prefixes.values())))
    tagged = Counter(t for v in prefixes.values() for part in v.strip("[]").split(" | ")[1:] for t in part.split(", ")
                     if t == "AURA" or t.startswith("SAX"))
    print("your tags:", dict(tagged))
    for i, v in list(prefixes.items())[:4]:
        print("  e.g.", v)
    if not args.apply:
        print("Preview only. Add --apply to write the comments (Rekordbox closed).")
        return
    from rekordbox_write import write_comments
    r = write_comments(prefixes)
    print(f"Comments changed on {r['changed']} songs ({r['unchanged']} already right). Backup: {r['backup']}")


if __name__ == "__main__":
    main()
