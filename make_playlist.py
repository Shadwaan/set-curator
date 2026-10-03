"""
Build a playlist from one or more weighted seed tracks and (optionally) write
it into Rekordbox.

Each seed is "text" or "text:weight" (default weight 1), matched against
"Artist - Title". The seeds' unit-length embeddings are combined by weight into
one target, and the library tracks closest to it (cosine similarity) fill the
playlist, seeds first.

    python make_playlist.py "Instant Death:2" "Dub Echo" --size 20             # preview
    python make_playlist.py "Instant Death:2" "Dub Echo" --size 20 --name "SC Dub" --apply

Preview is the default; nothing touches Rekordbox without --apply.
"""

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def parse_seed(spec: str) -> tuple[str, float]:
    text, sep, weight = spec.rpartition(":")
    if sep:
        try:
            return text, float(weight)
        except ValueError:
            pass
    return spec, 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seeds", nargs="+", help='"artist or title[:weight]"')
    ap.add_argument("--size", type=int, default=20, help="tracks in the playlist, seeds included")
    ap.add_argument("--bpm", help="keep only tracks in this BPM range, e.g. 118-126")
    ap.add_argument("--name", help="Rekordbox playlist name (required with --apply)")
    ap.add_argument("--apply", action="store_true", help="write the playlist into Rekordbox")
    ap.add_argument("--append", action="store_true", help="add to the playlist if it already exists")
    args = ap.parse_args()

    tracks = {t["id"]: t for t in json.loads((HERE / "tracks.json").read_text(encoding="utf-8"))}
    data = np.load(HERE / "embeddings.npz")
    ids = [str(i) for i in data["ids"]]
    vecs = data["vectors"] / np.linalg.norm(data["vectors"], axis=1, keepdims=True)
    label = [f"{tracks[i]['artist']} - {tracks[i]['title']}" if i in tracks else i for i in ids]

    target = np.zeros(vecs.shape[1])
    seed_rows = []
    for spec in args.seeds:
        text, weight = parse_seed(spec)
        hits = [r for r, name in enumerate(label) if text.lower() in name.lower()]
        if len(hits) != 1:
            print(f"Seed '{text}' matches {len(hits)} embedded tracks" +
                  ("" if not hits else ":\n  " + "\n  ".join(label[r] for r in hits[:10])))
            raise SystemExit(1)
        seed_rows.append(hits[0])
        target += weight * vecs[hits[0]]
    target /= np.linalg.norm(target)

    scores = vecs @ target
    order = [r for r in np.argsort(-scores) if r not in seed_rows]
    if args.bpm:
        lo, hi = (float(x) for x in args.bpm.split("-"))
        order = [r for r in order if lo <= tracks.get(ids[r], {}).get("bpm", 0) <= hi]
    picked = seed_rows + order[: max(0, args.size - len(seed_rows))]

    for n, r in enumerate(picked, 1):
        t = tracks.get(ids[r], {})
        tag = "seed" if r in seed_rows else f"{scores[r]:.3f}"
        print(f"{n:3}. {tag:>5}  {label[r]}  [{t.get('bpm', 0):.0f} {t.get('key', '')}]")

    if not args.apply:
        print("\nPreview only. Add --name \"...\" --apply to write it into Rekordbox.")
        return
    if not args.name:
        raise SystemExit("--apply needs --name")

    from rekordbox_write import write_playlist
    result = write_playlist(args.name, [tracks[ids[r]] for r in picked], append=args.append)
    print(f"\n{'Created' if result['created'] else 'Updated'} '{args.name}': "
          f"{result['added']} tracks added. Backup: {result['backup']}")
    for t in result["missing"]:
        print(f"  not in this Rekordbox library: {t['artist']} - {t['title']}")


if __name__ == "__main__":
    main()
