"""
Group the library by sound, name each group by its Discogs style, split it
into tempo bands, and (optionally) write the result into Rekordbox as a folder
tree: SC / <style> / <BPM range>.

Missing files, Rekordbox's built-in sampler sounds and duplicate copies of the
same audio are left out. Tracks in each playlist are ordered by BPM.

    python cluster.py              # preview the tree
    python cluster.py --apply      # write it into Rekordbox

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
STYLE_MODEL = HERE / "models" / "genre_discogs400-discogs-effnet-1.pb"
STYLE_LABELS = HERE / "models" / "genre_discogs400-discogs-effnet-1.json"


def load(exclude_playlists=()):
    tracks = {t["id"]: t for t in json.loads((HERE / "tracks.json").read_text(encoding="utf-8"))}
    data = np.load(HERE / "embeddings.npz")
    ids = [str(i) for i in data["ids"]]
    vecs = data["vectors"] / np.linalg.norm(data["vectors"], axis=1, keepdims=True)
    keep = [n for n, i in enumerate(ids) if i in tracks and os.path.exists(tracks[i]["path"])
            and "/rekordbox/Sampler/" not in tracks[i]["path"]
            and not set(tracks[i]["playlists"]) & set(exclude_playlists)]
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


def style_predictions(ids):
    """Discogs-400 style activations per track, from the stored mean embeddings
    (the classifier is trained on per-frame embeddings, so this is approximate)."""
    from essentia.standard import TensorflowPredict2D
    classes = json.loads(STYLE_LABELS.read_text())["classes"]
    head = TensorflowPredict2D(graphFilename=str(STYLE_MODEL),
                               input="serving_default_model_Placeholder", output="PartitionedCall:0")
    data = np.load(HERE / "embeddings.npz")
    raw = {str(i): v for i, v in zip(data["ids"], data["vectors"])}
    probs = np.array(head(np.stack([raw[i] for i in ids]).astype(np.float32)))
    return classes, {i: p for i, p in zip(ids, probs)}


def style_names(groups, probs, classes):
    """A distinct style per group: score = group share^2 / library share, so a
    group is named for what sets it apart (Disco, Tech House) rather than the
    style everything shares (House); stronger claims win a contested name."""
    library = np.mean(list(probs.values()), axis=0)
    claims = []
    for g, members in enumerate(groups):
        share = np.mean([probs[i] for i in members], axis=0)
        claims += [(share[s] ** 2 / library[s], g, s) for s in np.argsort(-share)[:15]]
    names, taken = {}, set()
    for _, g, s in sorted(claims, reverse=True):
        if g not in names and s not in taken:
            names[g] = s
            taken.add(s)
    # "Reggae---Dub" -> "Dub", unless another parent genre has a "Dub" too
    bare = Counter(c.split("---")[1] for c in classes)
    pretty = lambda c: c.split("---")[1] if bare[c.split("---")[1]] == 1 else c.replace("---", " ")
    return [pretty(classes[names[g]]) for g in range(len(groups))]


def bpm_range(members, eff):
    lo, hi = round(min(eff[i] for i in members)), round(max(eff[i] for i in members))
    return f"{lo}-{hi}" if lo != hi else f"{lo}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=50, help="target tracks per sound group, before the tempo split")
    ap.add_argument("--bpm-width", type=float, default=8, help="max BPM spread within a playlist")
    ap.add_argument("--min", type=int, default=5, help="smaller tempo bands go to a neighbouring playlist")
    ap.add_argument("--exclude-playlist", action="append", default=[], metavar="NAME",
                    help="leave out tracks in this Rekordbox playlist (repeatable)")
    ap.add_argument("--folder", default="SC", help="name of the top-level Rekordbox folder")
    ap.add_argument("--apply", action="store_true", help="write the folder tree into Rekordbox")
    args = ap.parse_args()

    tracks, ids, vecs = load(args.exclude_playlist)
    ids, vecs, dropped = drop_duplicates(ids, vecs)
    for i in dropped:
        print(f"duplicate audio, skipped: {tracks[i]['path']}")

    k = max(2, round(len(ids) / args.size))
    labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(vecs)
    groups = [[ids[n] for n in np.flatnonzero(labels == c)] for c in range(k)]
    classes, probs = style_predictions(ids)
    styles = style_names(groups, probs, classes)

    eff, bands, band_group, leftovers = {}, [], [], []
    for g, group in enumerate(groups):
        group_eff = mix_bpm(group, tracks)
        eff.update(group_eff)
        for b in tempo_bands(group, group_eff, args.bpm_width):
            if len(b) >= args.min:
                bands.append(b)
                band_group.append(g)
            else:
                leftovers.extend(b)
    row = {i: n for n, i in enumerate(ids)}
    unplaced = place_leftovers(leftovers, bands, eff, tracks, vecs, row, args.bpm_width)

    folders = []
    for g in sorted(range(k), key=lambda g: styles[g]):
        playlists = sorted((sorted(b, key=eff.get) for b, bg in zip(bands, band_group) if bg == g),
                           key=lambda b: eff[b[0]])
        if playlists:
            folders.append({"name": styles[g], "children": [
                {"name": bpm_range(b, eff), "tracks": b} for b in playlists]})
    tree = {"name": args.folder, "children": list(folders)}
    if unplaced:
        for i in unplaced:
            eff[i] = tracks[i]["bpm"]
        tree["children"].append({"name": "Other tempos", "tracks": sorted(unplaced, key=lambda i: tracks[i]["bpm"])})

    def show(node, depth=0):
        pad = "    " * depth
        if "children" in node:
            print(f"{pad}{node['name']}/")
            for child in node["children"]:
                show(child, depth + 1)
            return
        members = node["tracks"]
        # BPM as Rekordbox shows it; * = counted as half/double time
        bpms = " ".join(f"{tracks[i]['bpm']:.0f}{'*' if round(eff[i]) != round(tracks[i]['bpm']) else ''}"
                        for i in members)
        examples = ", ".join(f"{tracks[i]['artist']} - {tracks[i]['title']}".strip(" -")[:40] for i in members[:2])
        print(f"{pad}{node['name']}  ({len(members)})  {examples}\n{pad}    BPM: {bpms}")

    n_playlists = sum(len(f["children"]) for f in folders) + bool(unplaced)
    print(f"\n{len(ids)} tracks: {len(folders)} style folders, {n_playlists} playlists\n")
    show(tree)

    if not args.apply:
        print("\nPreview only. Add --apply to write this folder into Rekordbox.")
        return

    from rekordbox_write import write_folder

    def to_tracks(node):
        if "children" in node:
            return {"name": node["name"], "children": [to_tracks(c) for c in node["children"]]}
        return {"name": node["name"], "tracks": [tracks[i] for i in node["tracks"]]}

    result = write_folder(to_tracks(tree))
    print(f"\nCreated '{args.folder}': {result['folders']} folders, {result['playlists']} playlists, "
          f"{result['added']} tracks. Backup: {result['backup']}")
    for t in result["missing"]:
        print(f"  not in this Rekordbox library: {t['artist']} - {t['title']}")


if __name__ == "__main__":
    main()
