"""
Cross-pollination, written as SC / Cross-pollination. Every group is a folder
whose playlists are just its tempo bands ("120-128"):

  Bridges/            for each pair of your playlists that genuinely overlap: the
                      tracks from either that sit closest to the other (whole-track
                      sound) plus tracks in both -- for moving a set between crates
  Bridges (palette)/  the same, judged by shared sounds (palette matching), with the
                      threshold set to let through the same share of tracks
  Moods/              the whole library by Essentia mood (Dark & Driving, Peak Time,
                      Laid-back, ...)
  (Mood palettes were merged into Set arcs, see roles.py --arcs)
  Relatives/          for each of your productions (under ~/Documents/Samples): the
                      production itself, then the 20 tracks sharing most of its
                      sound palette, closest first

    python crosspollinate.py                    # preview
    python crosspollinate.py --apply --replace  # (re)write SC / Cross-pollination
"""

import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np

import cluster
import palette
from cluster import HERE, bpm_range, label_of, load, mix_bpm, tempo_bands

MOODS = {"aggressive": "Dark & Driving", "happy": "Bright & Uplifting", "party": "Peak Time",
         "relaxed": "Laid-back", "sad": "Melancholic"}
PRODUCTIONS = str(Path.home() / "Documents" / "Samples") + "/"  # your own productions live here


def short(playlist: str) -> str:
    return playlist[:-5] if playlist.endswith(" AIFF") else playlist


def tempo_folder(name, members, tracks, width, min_size, keep_rest=False):
    """{"name": name, "children": tempo playlists named just "120-128"}, or None if no
    band is big enough. With keep_rest, songs in bands that are too small go to an
    "Other tempos" playlist instead of being left out."""
    eff = mix_bpm(members, tracks)
    bands = [{"name": bpm_range(b, eff), "tracks": b, "eff": eff}
             for b in tempo_bands(members, eff, width) if len(b) >= min_size]
    if keep_rest:
        placed = {i for b in bands for i in b["tracks"]}
        rest = sorted((i for i in members if i not in placed), key=lambda i: tracks[i]["bpm"])
        if rest:
            bands.append({"name": "Other tempos", "tracks": rest})
    return {"name": name, "children": bands} if bands else None


def palette_matrix(ids) -> np.ndarray:
    sections = palette.load()
    missing = [i for i in ids if i not in sections]
    if missing:
        raise SystemExit(f"{len(missing)} tracks have no palette analysis; run python palette.py build")
    return np.stack([palette.similarity(sections, i, ids) for i in ids])


def bridges(tracks, ids, vecs, threshold, width, method="averaged"):
    """method "averaged": whole-track fingerprints, `threshold` as given.
    method "palette": shared sounds section by section, with the threshold moved to
    the palette score that is exactly as selective as `threshold` is for averaged."""
    row = {i: n for n, i in enumerate(ids)}
    sims = vecs @ vecs.T
    same_audio = sims >= 0.999

    def best_cross(m):
        """Each track's best match among tracks sharing none of its playlists."""
        crates = [set(tracks[i]["playlists"]) for i in ids]
        out = []
        for a in range(len(ids)):
            other = [b for b in range(len(ids)) if not crates[a] & crates[b] and not same_audio[a, b]]
            if other:
                out.append(m[a, other].max())
        return np.array(out)

    if method == "palette":
        share = (best_cross(sims) >= threshold).mean()
        sims = palette_matrix(ids)
        threshold = float(np.quantile(best_cross(sims), 1 - share))
        print(f"palette bridges: threshold {threshold:.3f}, letting through the same {share:.0%} of tracks "
              f"as averaged does at its threshold")
    names = sorted({p for i in ids for p in tracks[i]["playlists"]}, key=str.lower)
    members = {p: {i for i in ids if p in tracks[i]["playlists"]} for p in names}
    out = []
    for a, b in combinations(names, 2):
        only_a, only_b = sorted(members[a] - members[b]), sorted(members[b] - members[a])
        if not only_a or not only_b:
            continue
        sub = sims[np.ix_([row[i] for i in only_a], [row[i] for i in only_b])]
        same = same_audio[np.ix_([row[i] for i in only_a], [row[i] for i in only_b])]
        sub = np.where(same, 0, sub)  # another copy of the same audio isn't a bridge
        near_a = [i for i, s in zip(only_a, sub.max(axis=1)) if s >= threshold]
        near_b = [i for i, s in zip(only_b, sub.max(axis=0)) if s >= threshold]
        shared = sorted(members[a] & members[b])
        if len(near_a) >= 2 and len(near_b) >= 2:
            folder = tempo_folder(f"{short(a)} × {short(b)}", near_a + near_b + shared, tracks, width, 4)
            if folder:
                out.append(folder)
    return out


def mood_scores(ids) -> np.ndarray:
    """Per track, each mood relative to your library (z-scores; columns in MOODS order).
    Most of the library is "party", so raw scores would say little."""
    from essentia.standard import TensorflowPredict2D
    data = np.load(HERE / "embeddings.npz")
    raw = {str(i): v for i, v in zip(data["ids"], data["vectors"])}
    x = np.stack([raw[i] for i in ids]).astype(np.float32)
    scores = []
    for mood in MOODS:
        meta = json.loads((HERE / "models" / f"mood_{mood}-discogs-effnet-1.json").read_text())
        head = TensorflowPredict2D(graphFilename=str(HERE / "models" / f"mood_{mood}-discogs-effnet-1.pb"),
                                   input="model/Placeholder", output="model/Softmax")
        scores.append(np.array(head(x))[:, meta["classes"].index(mood)])
    p = np.stack(scores, axis=1)
    return (p - p.mean(axis=0)) / p.std(axis=0)


def moods(tracks, ids, z, width, min_z=0.5):
    out = []
    for k, name in enumerate(MOODS.values()):
        group = [i for i, row in zip(ids, z) if row.argmax() == k and row[k] >= min_z]
        folder = tempo_folder(name, group, tracks, width, 5)
        if folder:
            out.append(folder)
    return out


def palette_families(tracks, ids, sims, z, size=40) -> list[dict]:
    """The whole library grouped by shared sound palette (spectral clustering on
    palette similarity; groups over 2x `size` are split again), largest first.
    Each family: {"mood": its strongest mood, "crate": the one playlist of yours most
    of its songs come from, "style": its Discogs style (distinct per family),
    "members": [ids]}."""
    from sklearn.cluster import SpectralClustering
    sims = (sims + sims.T) / 2
    lo, hi = np.percentile(sims, 5), np.percentile(sims, 99.5)
    affinity = np.clip((sims - lo) / (hi - lo), 0, 1) ** 3

    def split(rows, k):
        labels = SpectralClustering(n_clusters=k, affinity="precomputed", random_state=0).fit_predict(
            affinity[np.ix_(rows, rows)])
        return [[rows[n] for n in np.flatnonzero(labels == c)] for c in range(k)]

    groups, todo = [], split(list(range(len(ids))), max(2, round(len(ids) / size)))
    while todo:
        g = todo.pop()
        if len(g) > 2 * size:
            todo += split(g, round(len(g) / size))
        else:
            groups.append(g)

    crate_sizes = {}
    for i in ids:
        for c in tracks[i]["playlists"]:
            crate_sizes[c] = crate_sizes.get(c, 0) + 1
    groups.sort(key=len, reverse=True)
    classes, probs = cluster.style_predictions(ids)
    styles = cluster.style_names([[ids[r] for r in g] for g in groups], probs, classes)
    families = []
    for n, g in enumerate(groups):
        counts = {}
        for r in g:
            for c in tracks[ids[r]]["playlists"]:
                counts[c] = counts.get(c, 0) + 1
        # the playlist most specific to this family, among those holding a real share of it
        # (a tiny playlist that happens to fit entirely shouldn't name a big family)
        big = [c for c in counts if counts[c] >= 0.25 * len(g)] or list(counts)
        crate = max(big, key=lambda c: counts[c] ** 2 / crate_sizes[c]) if big else None
        families.append({"mood": list(MOODS.values())[int(z[g].mean(axis=0).argmax())],
                         "crate": short(crate) if crate else "No playlist", "style": styles[n],
                         "members": [ids[r] for r in g]})
    return families


def mood_tree(families, leaf) -> list[dict]:
    """Mood > your playlist > [style] > whatever `leaf(name, family)` builds. The
    style level only appears where one playlist has several families under one mood
    (so Laid-back > Dub Reggae Bass Addict > Reggae, Electronic Dub). Names never
    combine two playlists."""
    by_mood = {}
    for f in families:
        by_mood.setdefault(f["mood"], {}).setdefault(f["crate"], []).append(f)
    out = []
    for mood in MOODS.values():
        crates = []
        for crate, fams in sorted(by_mood.get(mood, {}).items(), key=lambda kv: -sum(len(f["members"]) for f in kv[1])):
            if len(fams) == 1:
                node = leaf(crate, fams[0])
            else:
                kids = [k for k in (leaf(f["style"], f) for f in fams) if k]
                node = {"name": crate, "children": kids} if kids else None
            if node:
                crates.append(node)
        if crates:
            out.append({"name": mood, "children": crates})
    return out


def palettes(tracks, ids, sims, z, width):
    families = palette_families(tracks, ids, sims, z)
    return mood_tree(families, lambda name, f: tempo_folder(name, f["members"], tracks, width, 4, keep_rest=True))


def relatives(tracks, ids, vecs, n=20):
    row = {i: r for r, i in enumerate(ids)}
    prods, _, _ = cluster.drop_duplicates([i for i in ids if tracks[i]["path"].startswith(PRODUCTIONS)],
                                          vecs[[row[i] for i in ids if tracks[i]["path"].startswith(PRODUCTIONS)]])
    sections = palette.load()
    out = []
    for p in prods:
        same = vecs @ vecs[row[p]]
        # palette match (shared sounds, section by section) where analysed, else the averaged sound
        sims = palette.similarity(sections, p, ids) if p in sections else same
        picked = []
        for r in np.argsort(-sims):
            if same[r] >= 0.999:
                continue  # the production itself, or another copy of it
            if not picked or (vecs[[row[i] for i in picked]] @ vecs[r]).max() < 0.999:
                picked.append(ids[r])
            if len(picked) == n:
                break
        # the production itself first, then its relatives
        out.append({"name": tracks[p]["title"].split(" - ")[-1] or label_of(tracks[p]), "tracks": [p] + picked})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bridge-threshold", type=float, default=0.90, help="similarity for a bridge track")
    ap.add_argument("--bpm-width", type=float, default=8, help="max BPM spread within a playlist")
    ap.add_argument("--exclude-playlist", action="append", default=["Deep Tech FLAC"], metavar="NAME")
    ap.add_argument("--apply", action="store_true", help="write into the existing SC folder")
    ap.add_argument("--replace", action="store_true", help="rebuild SC / Cross-pollination if it already exists")
    args = ap.parse_args()

    tracks, ids, vecs = load(args.exclude_playlist)
    ids, vecs, merged = cluster.merge_copies(tracks, ids, vecs)
    print(f"{len(merged)} extra copies of songs merged (one entry per song)")
    z = mood_scores(ids)
    tree = {"name": "Cross-pollination", "children": [
        {"name": "Bridges", "children": bridges(tracks, ids, vecs, args.bridge_threshold, args.bpm_width)},
        {"name": "Bridges (palette)", "children": bridges(tracks, ids, vecs, args.bridge_threshold, args.bpm_width,
                                                          "palette")},
        {"name": "Moods", "children": moods(tracks, ids, z, args.bpm_width)},
        {"name": "Relatives", "children": relatives(tracks, ids, vecs)},
    ]}

    def show(node, depth=0):
        pad = "    " * depth
        if "children" in node:
            n = sum(1 for _ in walk(node))
            print(f"{pad}{node['name']}/  ({n} playlists)")
            for child in node["children"]:
                show(child, depth + 1)
        else:
            eff = node.get("eff", {})
            star = sum(1 for i in node["tracks"] if i in eff and round(eff[i]) != round(tracks[i]["bpm"]))
            print(f"{pad}{node['name']}  ({len(node['tracks'])}{f', {star} half-time' if star else ''})")

    def walk(node):
        for child in node["children"]:
            yield from (walk(child) if "children" in child else [child])

    for folder in tree["children"]:
        show(folder)

    if not args.apply:
        print("\nPreview only. Add --apply to write SC / Cross-pollination into Rekordbox.")
        return

    from rekordbox_write import write_folder

    def to_tracks(node):
        if "children" in node:
            return {"name": node["name"], "children": [to_tracks(c) for c in node["children"]]}
        return {"name": node["name"], "tracks": [tracks[i] for i in node["tracks"]]}

    result = write_folder(to_tracks(tree), parent="SC", replace=args.replace)
    print(f"\nCreated: {result['folders']} folders, {result['playlists']} playlists, "
          f"{result['added']} tracks. Backup: {result['backup']}")


if __name__ == "__main__":
    main()
