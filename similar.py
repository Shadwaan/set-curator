"""
Nearest neighbours by cosine similarity over embeddings.npz.
Needs only numpy, so it runs anywhere (Windows, WSL, macOS).

    python similar.py                # top matches for every embedded track
    python similar.py "Instant Death" --top 10
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?", help="part of an artist or title")
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    tracks = {t["id"]: t for t in json.loads((HERE / "tracks.json").read_text(encoding="utf-8"))}
    data = np.load(HERE / "embeddings.npz")
    ids = [str(i) for i in data["ids"]]
    vecs = data["vectors"]
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sims = vecs @ vecs.T

    def name(i):
        t = tracks.get(ids[i], {})
        return f"{t.get('artist', '?')} - {t.get('title', '?')}  [{t.get('bpm', 0):.0f} {t.get('key', '')}]"

    rows = range(len(ids))
    if args.query:
        q = args.query.lower()
        rows = [i for i in rows if q in name(i).lower()]

    for i in rows:
        print(name(i))
        for j in [j for j in np.argsort(-sims[i]) if j != i][: args.top]:
            print(f"   {sims[i, j]:.3f}  {name(j)}")
        print()


if __name__ == "__main__":
    main()
