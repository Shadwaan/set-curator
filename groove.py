"""
Groove resets: songs that calm the floor down after a peak while keeping a groove going.

Seeds are the songs your notes or your answers call a groove reset. Candidates are the songs that sound closest to them
(palette similarity), limited to songs that are calm and steady enough on your own measurements, never dub reggae.

    python groove.py        # -> groove_candidates.json (shown in the review page as "Groove reset check")
"""

import json

import numpy as np

from cluster import HERE

OUT = HERE / "groove_candidates.json"
ANSWERS = HERE / "sax_roles.json"


def seeds(merged=None) -> list[str]:
    import notes
    return sorted(i for i, rl in notes.explicit_roles(merged).items() if "9 Groove resets" in rl)


DECISIONS = HERE / "groove_decisions.json"      # my proposals about the resets (shown in the page); your answers: sax_decisions.json
DECIDED = HERE / "sax_decisions.json"


def dropped(merged=None) -> set[str]:
    """Songs I proposed to drop from the groove resets AND that you agreed to drop (a reset should calm; these are High energy)."""
    if not DECISIONS.exists():
        return set()
    mine = json.loads(DECISIONS.read_text())
    said = json.loads(DECIDED.read_text()) if DECIDED.exists() else {}
    return {(merged or {}).get(i, i) for i, d in mine.items() if d["decision"] == "drop" and said.get(i) == ["Agree"]}


def candidates(top: int = 16) -> dict[str, list[str]]:
    import cluster
    import palette
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, merged = cluster.merge_copies(tracks, ids, vecs)
    sec = palette.load()
    reggae = json.loads((HERE / "reggae_all.json").read_text())
    f = json.loads((HERE / "energy_features.json").read_text())
    done = json.loads(ANSWERS.read_text()) if ANSWERS.exists() else {}
    have = seeds(merged)
    use = [i for i in have if i in sec]
    f_ = f
    pool = [i for i in songs if i in sec and i in f and reggae.get(i, 0) < 0.3 and i not in done and i not in have]
    calm = np.array([f[i]["relaxed"] - 0.5 * f[i]["aggressive"] for i in pool])          # calm and not hard
    steady = np.array([f[i]["steady"] for i in pool])
    keep = [k for k in range(len(pool)) if steady[k] <= 0.90 and steady[k] >= 0.74]       # your resets: 0.76-0.89; the rigid ones (0.92+) you turned down
    score = sum(palette.similarity(sec, s, pool) for s in use) / len(use)
    z = (score - score.mean()) / score.std()
    out = {}
    for k in sorted(keep, key=lambda k: -score[k])[:top]:
        out[pool[k]] = [f"sounds like your groove resets ({z[k]:+.1f} sd), calm and steady"]
    OUT.write_text(json.dumps(out))
    return out


if __name__ == "__main__":
    import cluster
    tr = cluster.load(["Deep Tech FLAC"])[0]
    print("seeds:", [tr[i]["title"][:30] for i in seeds()])
    for i, why in candidates().items():
        print(f"  {tr[i]['artist'][:22]:22} - {tr[i]['title'][:34]:34} | {why[0]}")
