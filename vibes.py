"""
Your own vocabulary as groups: Bangkok groove, Beachbar (and whether a beachbar song also works in a club), Sweaty.

A song is in a group when you used the word in a note (notes.py) or ticked it in the review page's Vibe check
(sax_vibes.json). Candidates are found by sound: the songs closest to the ones already in the group.

    python vibes.py            # find candidates -> vibe_candidates.json (shown in the review page)
"""

import json
import sys

import numpy as np

from cluster import HERE

FILE = HERE / "sax_vibes.json"
CANDIDATES = HERE / "vibe_candidates.json"
NAMES = ["Bangkok groove", "Beachbar", "Also works in a club", "Sweaty"]
CLUB_OPTIONS = ["Also works in a club", "Beachbar only"]           # the Club check page: does a beachbar song have club energy?
CLUB_CANDIDATES = HERE / "club_candidates.json"
BEACHBAR_PLAYLIST = "beachbar classics (Zayan) AIFF"
TAG = {"Bangkok groove": "BANGKOK", "Beachbar": "BEACHBAR", "Also works in a club": "BEACHBAR-CLUB", "Sweaty": "SWEATY"}
NOTE_TAG = {"BANGKOK": "Bangkok groove", "BEACHBAR": "Beachbar", "SWEATY": "Sweaty"}


def filed() -> dict[str, list[str]]:
    return json.loads(FILE.read_text()) if FILE.exists() else {}


def members(merged: dict[str, str] | None = None) -> dict[str, list[str]]:
    """{group name: [song ids]} from your notes and your Vibe check answers (a copy counts as the song that was kept)."""
    import notes
    out: dict[str, set[str]] = {n: set() for n in NAMES}
    for i, tags in notes.vibe_tags(merged).items():
        for t in tags:
            out[NOTE_TAG[t]].add(i)
    for i, names in filed().items():
        for n in names:
            if n in out:
                out[n].add((merged or {}).get(i, i))
    return {n: sorted(v) for n, v in out.items()}


SUGGEST_Z = 1.8        # how much closer than average (in standard deviations) a song must sound to your confirmed ones


def suggestions(merged: dict[str, str] | None = None, top: int = 30) -> list[str]:
    """Songs that sound like your confirmed Bangkok groove songs but that you have not confirmed (and did not turn down).
    On your own answers, the songs shown at this level were ticked 85% of the time, so they are tagged "BANGKOK?"."""
    import cluster
    import palette
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, kept = cluster.merge_copies(tracks, ids, vecs)
    sec = palette.load()
    reggae = json.loads((HERE / "reggae_all.json").read_text())
    have = set(members(kept)["Bangkok groove"])
    refused = {kept.get(i, i) for i, a in filed().items() if "Bangkok groove" not in a}
    seeds = [i for i in have if i in sec]
    if len(seeds) < 3:
        return []
    pool = [i for i in songs if i in sec and reggae.get(i, 0) < 0.3 and i not in have and i not in refused]
    score = sum(palette.similarity(sec, s, pool) for s in seeds) / len(seeds)
    z = (score - score.mean()) / score.std()
    return [pool[k] for k in np.argsort(-score)[:top] if z[k] >= SUGGEST_Z]


def tags(merged: dict[str, str] | None = None) -> dict[str, list[str]]:
    """{song id: ["BANGKOK", "BANGKOK?", ...]} in a fixed order, for the comment. "BANGKOK?" is my guess, not your answer."""
    by: dict[str, list[str]] = {}
    for n in NAMES:
        for i in members(merged)[n]:
            by.setdefault(i, []).append(TAG[n])
    for i in suggestions(merged):
        by.setdefault(i, []).append("BANGKOK?")
    return by


def candidates(per_group: int = 14) -> dict[str, list[str]]:
    import cluster
    import palette
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, merged = cluster.merge_copies(tracks, ids, vecs)
    reggae = json.loads((HERE / "reggae_all.json").read_text())
    sec = palette.load()
    have = members(merged)
    seen = {i for v in have.values() for i in v} | set(filed())
    pool = [i for i in songs if i in sec and reggae.get(i, 0) < 0.3 and i not in seen]
    out: dict[str, list[str]] = {}
    for group in ("Bangkok groove",):                      # the Beachbar vibe is your beachbar playlist itself
        seeds = [i for i in have[group] if i in sec]
        if len(seeds) < 3:
            continue
        score = sum(palette.similarity(sec, s, pool) for s in seeds) / len(seeds)
        z = (score - score.mean()) / score.std()
        for k in np.argsort(-score)[:per_group]:
            out.setdefault(pool[k], []).append(f"sounds like your {group} songs ({z[k]:+.1f} sd)")
    CANDIDATES.write_text(json.dumps(out))
    return out


def club_candidates(top: int = 24) -> dict[str, list[str]]:
    """The beachbar playlist's songs with the most club energy on your energy scale (energy_model.json), highest first."""
    import cluster
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, _ = cluster.merge_copies(tracks, ids, vecs)
    f = json.loads((HERE / "energy_features.json").read_text())
    m = json.loads((HERE / "energy_model.json").read_text())
    names, mu, sd, w = m["names"], np.array(m["mu"]), np.array(m["sd"]), np.array(m["w"])

    def score(i):
        v = np.array([f[i][k] for k in names], float)
        return float(((np.where(np.isnan(v), mu, v) - mu) / sd) @ w + m["intercept"])

    cut = float(np.percentile([score(i) for i in f], 200 / 3))
    done = filed()
    pool = [i for i in songs if BEACHBAR_PLAYLIST in tracks[i]["playlists"] and i in f]
    out = {}
    for i in sorted(pool, key=lambda i: -score(i))[:top]:
        out[i] = [f"energy {score(i):.2f} ({'High' if score(i) > cut else 'Mid'} on your scale), {tracks[i]['bpm']:.0f} BPM, party feel {f[i]['party']:.2f}"]
    CLUB_CANDIDATES.write_text(json.dumps(out))
    return out


if __name__ == "__main__" and len(sys.argv) == 1:
    import cluster
    tr = cluster.load(["Deep Tech FLAC"])[0]
    c = candidates()
    cc = club_candidates()
    print(len(c), "Bangkok groove candidates;", len(cc), "beachbar songs for the club check")
    print()
    for i, why in c.items():
        print(f"  {tr[i]['artist'][:22]:22} - {tr[i]['title'][:32]:32} | {why[0]}")
