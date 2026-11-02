"""
The left-field to house-side axis.

You described your library as a scale: your Dub playlist is full left-field music, Deep Tech sits between house and
left-field (nearer left-field), and beachbar, Classic and Oldies are the house-side end. Each song's position is how much
closer its sound is to the Dub songs than to the house-side songs (embeddings, z-scored over the library): positive =
left-field, negative = house-side. Checked against your playlists: Dub +0.97, Deep Tech +0.48, Half moon -0.54, beachbar -1.39.

    python leftfield.py      # -> leftfield_axis.json
"""

import json
import sys
from pathlib import Path

import numpy as np

from cluster import HERE

FILE = HERE / "leftfield_axis.json"
SPLIT = -0.4      # a reset at or above this reads as left-field: it sits in the gap between your house-side core (-0.87 and lower) and the rest


ANCHOR = HERE / "leftfield_anchor.json"   # the ruler as first set: the songs at each end, and the library it was z-scored over


def freeze(tracks_file: str, embeddings_file: str):
    """Pin the ruler to a past library (tracks.json and embeddings.npz from then), so a renamed or retired playlist
    cannot move older songs along the axis. Run once: python leftfield.py --freeze OLD_TRACKS.json OLD_EMBEDDINGS.npz"""
    old = json.loads(Path(tracks_file).read_text())
    have = {str(i) for i in np.load(embeddings_file)["ids"]}
    reggae = json.loads((HERE / "reggae_all.json").read_text())
    short = lambda t: {p[:-5] if p.endswith(" AIFF") else p for p in t["playlists"]}
    old = [t for t in old if t["id"] in have]
    out = {"left": [t["id"] for t in old if "Dub" in short(t) and reggae.get(t["id"], 0) < 0.3],
           "house": [t["id"] for t in old if short(t) & {"beachbar sets (classic)", "Classic", "Oldies"}],
           "retro": [t["id"] for t in old if short(t) & {"beachbar sets (classic)", "Classic", "Oldies"}],
           "calib": [t["id"] for t in old]}
    ANCHOR.write_text(json.dumps(out))
    print({k: len(v) for k, v in out.items()}, "->", ANCHOR.name)


def scores(tracks, ids, vecs, songs) -> dict[str, float]:
    """Position of each song on the axis. With leftfield_anchor.json the ends and the z-scale are the frozen ones;
    without it they come from today's playlists and today's library."""
    row = {i: n for n, i in enumerate(ids)}
    pl = lambda i: {p[:-5] if p.endswith(" AIFF") else p for p in tracks[i]["playlists"]}
    if ANCHOR.exists():
        a = json.loads(ANCHOR.read_text())
        left, house, calib = ([i for i in a[k] if i in row] for k in ("left", "house", "calib"))
    else:
        reggae = json.loads((HERE / "reggae_all.json").read_text())
        left = [i for i in songs if "Dub" in pl(i) and reggae.get(i, 0) < 0.3 and i in row]
        house = [i for i in songs if pl(i) & {"beachbar classics (Zayan)", "Classic", "Oldies"} and i in row]
        calib = None
    unit = lambda v: v / np.linalg.norm(v)
    cl, ch = unit(vecs[[row[i] for i in left]].mean(0)), unit(vecs[[row[i] for i in house]].mean(0))
    raw = {i: float(vecs[row[i]] @ cl - vecs[row[i]] @ ch) for i in songs if i in row}
    ref = np.array([raw[i] for i in calib if i in raw]) if calib else np.array(list(raw.values()))
    return {i: (x - ref.mean()) / ref.std() for i, x in raw.items()}


def compute() -> dict[str, float]:
    import cluster
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, _ = cluster.merge_copies(tracks, ids, vecs)
    out = scores(tracks, ids, vecs, songs)
    FILE.write_text(json.dumps(out))
    return out


def axis() -> dict[str, float]:
    return json.loads(FILE.read_text()) if FILE.exists() else compute()


def side(i: str, merged: dict[str, str] | None = None) -> str:
    z = axis().get((merged or {}).get(i, i))
    return "Left-field" if z is not None and z >= SPLIT else "House-side"


if __name__ == "__main__":
    if sys.argv[1:2] == ["--freeze"]:
        freeze(sys.argv[2], sys.argv[3])
    else:
        print(len(compute()), "songs placed on the axis")
