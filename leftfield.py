"""
The left-field to house-side axis.

You described your library as a scale: your Dub playlist is full left-field music, Deep Tech sits between house and
left-field (nearer left-field), and beachbar, Classic and Oldies are the house-side end. Each song's position is how much
closer its sound is to the Dub songs than to the house-side songs (embeddings, z-scored over the library): positive =
left-field, negative = house-side. Checked against your playlists: Dub +0.97, Deep Tech +0.48, Half moon -0.54, beachbar -1.39.

    python leftfield.py      # -> leftfield_axis.json
"""

import json

import numpy as np

from cluster import HERE

FILE = HERE / "leftfield_axis.json"
SPLIT = -0.4      # a reset at or above this reads as left-field: it sits in the gap between your house-side core (-0.87 and lower) and the rest


def compute() -> dict[str, float]:
    import cluster
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, _ = cluster.merge_copies(tracks, ids, vecs)
    row = {i: n for n, i in enumerate(ids)}
    reggae = json.loads((HERE / "reggae_all.json").read_text())
    pl = lambda i: {p[:-5] if p.endswith(" AIFF") else p for p in tracks[i]["playlists"]}
    left = [i for i in songs if "Dub" in pl(i) and reggae.get(i, 0) < 0.3 and i in row]
    house = [i for i in songs if pl(i) & {"beachbar classics (Zayan)", "Classic", "Oldies"} and i in row]
    unit = lambda v: v / np.linalg.norm(v)
    cl, ch = unit(vecs[[row[i] for i in left]].mean(0)), unit(vecs[[row[i] for i in house]].mean(0))
    raw = {i: float(vecs[row[i]] @ cl - vecs[row[i]] @ ch) for i in songs if i in row}
    v = np.array(list(raw.values()))
    out = {i: (x - v.mean()) / v.std() for i, x in raw.items()}
    FILE.write_text(json.dumps(out))
    return out


def axis() -> dict[str, float]:
    return json.loads(FILE.read_text()) if FILE.exists() else compute()


def side(i: str, merged: dict[str, str] | None = None) -> str:
    z = axis().get((merged or {}).get(i, i))
    return "Left-field" if z is not None and z >= SPLIT else "House-side"


if __name__ == "__main__":
    print(len(compute()), "songs placed on the axis")
