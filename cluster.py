"""
Group the library into sound-alike clusters, one playlist per cluster, and
(optionally) write them all into Rekordbox via rekordbox_write.

Missing files, Rekordbox's built-in sampler sounds and duplicate copies of the
same audio are left out. Each
playlist is named after the existing playlist most specific to it plus its
BPM range (slowest-fastest), and its tracks are ordered by BPM.

    python cluster.py [--size 25]          # preview
    python cluster.py --apply              # write "SC 01 ...", "SC 02 ..." into Rekordbox

Preview is the default; nothing touches Rekordbox without --apply.
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

HERE = Path(__file__).parent
LOSSLESS = (".aiff", ".aif", ".wav", ".flac")


def load():
    tracks = {t["id"]: t for t in json.loads((HERE / "tracks.json").read_text(encoding="utf-8"))}
    data = np.load(HERE / "embeddings.npz")
    ids = [str(i) for i in data["ids"]]
    vecs = data["vectors"] / np.linalg.norm(data["vectors"], axis=1, keepdims=True)
    keep = [n for n, i in enumerate(ids) if i in tracks and os.path.exists(tracks[i]["path"])
            and "/rekordbox/Sampler/" not in tracks[i]["path"]]
    # lossless first, so it is the copy kept when the same audio exists twice
    keep.sort(key=lambda n: not tracks[ids[n]]["path"].lower().endswith(LOSSLESS))
    return tracks, [ids[n] for n in keep], vecs[keep]


def drop_duplicates(ids, vecs, threshold=0.999):
    sims = vecs @ vecs.T
    keep, dropped = [], []
    for n in range(len(ids)):
        if keep and sims[n, keep].max() >= threshold:
            dropped.append(ids[n])
        else:
            keep.append(n)
    return [ids[n] for n in keep], vecs[keep], dropped


def mix_bpm(group, tracks):
    """Each track's BPM as half, as-is or double -- whichever is closest to the
    group's median -- so a half-time 80 in a 160 group counts as 160."""
    centre = np.median([tracks[i]["bpm"] for i in group if tracks[i]["bpm"]] or [0])
    return {i: min((b / 2, b, b * 2), key=lambda x: abs(x - centre)) if (b := tracks[i]["bpm"]) else centre
            for i in group}


def tempo_bands(group, eff, width):
    bands = []
    for i in sorted(group, key=eff.get):
        if bands and eff[i] - eff[bands[-1][0]] <= width:
            bands[-1].append(i)
        else:
            bands.append([i])
    return bands


def place_leftovers(leftovers, bands, eff, tracks, vecs, row, width):
    """Put each track from a too-small band into the closest-sounding band it
    fits tempo-wise (as-is, double or half time) without widening that band past
    `width`. Returns the tracks that fit nowhere."""
    centroids = [vecs[[row[i] for i in b]].mean(axis=0) for b in bands]
    unplaced = []
    for i in leftovers:
        bpm, best = tracks[i]["bpm"], None
        for b, centroid in zip(bands, centroids):
            lo, hi = min(eff[j] for j in b), max(eff[j] for j in b)
            fit = next((x for x in (bpm, bpm * 2, bpm / 2) if bpm and hi - width <= x <= lo + width), None)
            if fit is not None:
                sim = vecs[row[i]] @ centroid
                if best is None or sim > best[0]:
                    best = (sim, b, fit)
        if best:
            best[1].append(i)
            eff[i] = best[2]
        else:
            unplaced.append(i)
    return unplaced


def label(members, tracks, playlist_sizes, eff):
    # count^2 / playlist size: favours a playlist this cluster holds a big share of,
    # so large playlists like "Dub" don't name every cluster
    counts = Counter(p for i in members for p in tracks[i]["playlists"])
    top = max(counts, key=lambda p: counts[p] ** 2 / playlist_sizes[p]) if counts else "Unsorted"
    lo, hi = round(min(eff[i] for i in members)), round(max(eff[i] for i in members))
    return f"{top} {lo}-{hi}" if lo != hi else f"{top} {lo}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=50, help="target tracks per sound group, before the tempo split")
    ap.add_argument("--bpm-width", type=float, default=8, help="max BPM spread within a playlist")
    ap.add_argument("--min", type=int, default=5, help="smaller tempo bands merge into a neighbour")
    ap.add_argument("--prefix", default="SC", help="playlist name prefix")
    ap.add_argument("--apply", action="store_true", help="write the playlists into Rekordbox")
    args = ap.parse_args()

    tracks, ids, vecs = load()
    ids, vecs, dropped = drop_duplicates(ids, vecs)
    for i in dropped:
        print(f"duplicate audio, skipped: {tracks[i]['path']}")

    k = max(2, round(len(ids) / args.size))
    labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(vecs)

    playlist_sizes = Counter(p for i in ids for p in tracks[i]["playlists"])
    eff, bands, leftovers = {}, [], []
    for c in range(k):
        group = [ids[n] for n in np.flatnonzero(labels == c)]
        group_eff = mix_bpm(group, tracks)
        eff.update(group_eff)
        for b in tempo_bands(group, group_eff, args.bpm_width):
            (bands.append(b) if len(b) >= args.min else leftovers.extend(b))
    row = {i: n for n, i in enumerate(ids)}
    unplaced = place_leftovers(leftovers, bands, eff, tracks, vecs, row, args.bpm_width)
    bands = [sorted(b, key=eff.get) for b in bands]
    bands.sort(key=lambda b: np.median([eff[i] for i in b]))
    clusters = [(f"{args.prefix} {n:02d} {label(b, tracks, playlist_sizes, eff)}", b)
                for n, b in enumerate(bands, 1)]
    if unplaced:
        for i in unplaced:
            eff[i] = tracks[i]["bpm"]
        clusters.append((f"{args.prefix} {len(clusters) + 1:02d} Other tempos",
                         sorted(unplaced, key=lambda i: tracks[i]["bpm"])))

    print(f"\n{len(ids)} tracks: {k} sound groups split into {len(clusters)} playlists\n")
    for name, members in clusters:
        examples = ", ".join(f"{tracks[i]['artist']} - {tracks[i]['title']}".strip(" -") for i in members[:3])
        # BPM as Rekordbox shows it; * = counted as half/double time
        bpms = " ".join(f"{tracks[i]['bpm']:.0f}{'*' if round(eff[i]) != round(tracks[i]['bpm']) else ''}"
                        for i in members)
        print(f"{name}  ({len(members)})\n    {examples}\n    BPM: {bpms}")

    if not args.apply:
        print("\nPreview only. Add --apply to write these playlists into Rekordbox.")
        return

    from rekordbox_write import write_playlists
    results = write_playlists([(name, [tracks[i] for i in members]) for name, members in clusters])
    print(f"\nCreated {len(results)} playlists, {sum(r['added'] for r in results)} tracks. "
          f"Backup: {results[0]['backup']}")
    for r in results:
        for t in r["missing"]:
            print(f"  not in this Rekordbox library: {t['artist']} - {t['title']}")


if __name__ == "__main__":
    main()
