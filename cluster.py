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


def excluded() -> set[str]:
    """Track ids kept out of every analysis and folder (excluded.json: {id: reason})."""
    f = HERE / "excluded.json"
    return set(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else set()


def load(exclude_playlists=()):
    tracks = {t["id"]: t for t in json.loads((HERE / "tracks.json").read_text(encoding="utf-8"))}
    data = np.load(HERE / "embeddings.npz")
    ids = [str(i) for i in data["ids"]]
    vecs = data["vectors"] / np.linalg.norm(data["vectors"], axis=1, keepdims=True)
    skip = excluded()
    keep = [n for n, i in enumerate(ids) if i in tracks and i not in skip and os.path.exists(tracks[i]["path"])
            and "/rekordbox/Sampler/" not in tracks[i]["path"]
            and not set(tracks[i]["playlists"]) & set(exclude_playlists)]
    # lossless first, so it is the copy kept when the same audio exists twice
    keep.sort(key=lambda n: not tracks[ids[n]]["path"].lower().endswith(LOSSLESS))
    return tracks, [ids[n] for n in keep], vecs[keep]


def merge_copies(tracks, ids, vecs):
    """One entry per song. The library can hold a song as separate files -- e.g.
    downloaded once per crate folder -- which Rekordbox treats as separate tracks.
    Copies = same artist and title with audio >= 0.94, or audio >= 0.995 whatever
    the name. Keeps one copy per song (lossless first, as load() orders them) and
    gives it the playlists of all its copies, so a song filed in two crates counts
    as being in both. Returns (ids, vecs, {dropped id: kept id}); `tracks` is
    updated in memory only."""
    import re
    def key(t):
        artist, title = t["artist"], t["title"]
        if not artist and " - " in title and not re.match(r"^(original|extended|radio|club|dub|instrumental|vocal|edit)?\s*(mix|edit|version|remix)$", title.split(" - ", 1)[1].strip(), re.I):
            artist, title = title.split(" - ", 1)
        n = lambda x: re.sub(r"[^a-z0-9]+", " ", (x or "").lower()).strip()
        return n(artist), n(title)
    sims = vecs @ vecs.T
    keep, merged = [], {}
    for n, i in enumerate(ids):
        twin = next((ids[k] for k in keep if sims[n, k] >= 0.995
                     or (sims[n, k] >= 0.94 and key(tracks[i]) == key(tracks[ids[k]]))), None)
        if twin is None:
            keep.append(n)
        else:
            merged[i] = twin
            tracks[twin] = {**tracks[twin], "playlists": sorted(set(tracks[twin]["playlists"]) | set(tracks[i]["playlists"]))}
    return [ids[n] for n in keep], vecs[keep], merged


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


_OPENNESS = None


def openness() -> dict[str, float]:
    """Each song's high-end openness (more open = higher), from highend.json; empty if not measured."""
    global _OPENNESS
    if _OPENNESS is None:
        f = HERE / "highend.json"
        _OPENNESS = {}
        if f.exists():
            h = json.loads(f.read_text())
            if h:
                p = np.array([v["presence"] for v in h.values()]); a = np.array([v["air"] for v in h.values()])
                zp, za = (p - p.mean()) / p.std(), (a - a.mean()) / a.std()
                _OPENNESS = {i: float((x + y) / 2) for i, x, y in zip(h, zp, za)}
    return _OPENNESS


def blend_key(eff):
    """Sort key for playlists: tempo first, then the high end (closed to open) among songs at the same
    tempo, so neighbouring songs blend without a jump in brightness."""
    op = openness()
    return lambda i: (round(eff[i]), op.get(i, 0.0))


def tempo_bands(group, eff, width):
    bands = []
    for i in sorted(group, key=blend_key(eff)):
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


def label_of(t):
    return f"{t['artist']} - {t['title']}".strip(" -")


def bpm_range(members, eff):
    lo, hi = round(min(eff[i] for i in members)), round(max(eff[i] for i in members))
    return f"{lo}-{hi}" if lo != hi else f"{lo}"


def split_by_sound_and_tempo(ids, vecs, tracks, probs, classes, size, width, min_size):
    """Group `ids` by sound, name each group by its style, split each into tempo
    bands. Returns ([(style, [band of ids sorted by tempo]), ...], unplaced ids, eff BPMs)."""
    k = max(1, round(len(ids) / size))
    labels = KMeans(n_clusters=k, n_init=20, random_state=0).fit_predict(vecs) if k > 1 else np.zeros(len(ids), int)
    groups = [[ids[n] for n in np.flatnonzero(labels == c)] for c in range(k)]
    styles = style_names(groups, probs, classes)

    eff, bands, band_group, leftovers = {}, [], [], []
    for g, group in enumerate(groups):
        group_eff = mix_bpm(group, tracks)
        eff.update(group_eff)
        for b in tempo_bands(group, group_eff, width):
            if len(b) >= min_size:
                bands.append(b)
                band_group.append(g)
            else:
                leftovers.extend(b)
    row = {i: n for n, i in enumerate(ids)}
    unplaced = place_leftovers(leftovers, bands, eff, tracks, vecs, row, width) if bands else leftovers
    if bands and 0 < len(unplaced) < min_size:
        # too few for an "Other tempos" playlist of their own: join the nearest tempo
        # band, as long as that band stays within 1.5x the usual spread
        still = []
        for i in unplaced:
            bpm = tracks[i]["bpm"] or 0
            fits = []
            for b in bands:
                lo, hi = min(eff[j] for j in b), max(eff[j] for j in b)
                for x in (bpm / 2, bpm, bpm * 2):
                    if bpm and max(hi, x) - min(lo, x) <= width * 1.5:
                        fits.append((abs(x - (lo + hi) / 2), x, b))
            if fits:
                _, eff[i], band = min(fits, key=lambda f: f[0])
                band.append(i)
            else:
                still.append(i)
        unplaced = still
    for i in unplaced:
        eff[i] = tracks[i]["bpm"]

    styled = []
    for g in sorted(range(k), key=lambda g: styles[g]):
        for b in sorted((sorted(b, key=eff.get) for b, bg in zip(bands, band_group) if bg == g),
                        key=lambda b: eff[b[0]]):
            styled.append((styles[g], b))
    return styled, sorted(unplaced, key=lambda i: tracks[i]["bpm"]), eff


def strong_matches(members, ids, vecs, tracks, threshold, limit=20):
    """Tracks from other playlists that sound very close to any of `members`."""
    row = {i: n for n, i in enumerate(ids)}
    inside = set(members)
    best = {}
    for m in members:
        sims = vecs @ vecs[row[m]]
        for n in np.flatnonzero((sims >= threshold) & (sims < 0.999)):
            other = ids[n]
            if other not in inside and sims[n] > best.get(other, (0, None))[0]:
                best[other] = (float(sims[n]), m)
    return sorted(best.items(), key=lambda kv: -kv[1][0])[:limit]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-playlist", action="store_true",
                    help="one folder per Rekordbox playlist, split inside by sound and tempo")
    ap.add_argument("--size", type=int, help="target tracks per sound group before the tempo split "
                                             "(default 50, or 25 with --per-playlist)")
    ap.add_argument("--bpm-width", type=float, default=8, help="max BPM spread within a playlist")
    ap.add_argument("--min", type=int, default=5, help="smaller tempo bands go to a neighbouring playlist")
    ap.add_argument("--min-split", type=int, default=12,
                    help="--per-playlist: smaller playlists stay whole, plus suggestions from other playlists")
    ap.add_argument("--suggest-threshold", type=float, default=0.92,
                    help="--per-playlist: similarity for a 'really strong' cross-playlist suggestion")
    ap.add_argument("--exclude-playlist", action="append", default=[], metavar="NAME",
                    help="leave out tracks in this Rekordbox playlist (repeatable)")
    ap.add_argument("--folder", default="SC", help="name of the top-level Rekordbox folder")
    ap.add_argument("--only", metavar="PLAYLIST", help="--per-playlist: rebuild just that playlist's folder inside the "
                    "existing folder, leaving everything else as it is")
    ap.add_argument("--apply", action="store_true", help="write the folder tree into Rekordbox")
    ap.add_argument("--replace", action="store_true", help="rebuild the folder if it already exists (same position)")
    args = ap.parse_args()
    size = args.size or (25 if args.per_playlist else 50)

    tracks, ids, vecs = load(args.exclude_playlist)
    classes, probs = style_predictions(ids)
    eff = {i: tracks[i]["bpm"] for i in ids}
    report = []

    if args.per_playlist:
        names = sorted({p for i in ids for p in tracks[i]["playlists"]} - set(args.exclude_playlist), key=str.lower)
        folders = []
        for name in names:
            member_rows = [n for n, i in enumerate(ids) if name in tracks[i]["playlists"]]
            p_ids, p_vecs, dropped = drop_duplicates([ids[n] for n in member_rows], vecs[member_rows])
            for i in dropped:
                print(f"{name}: duplicate audio, skipped: {tracks[i]['path']}")
            if len(p_ids) < args.min_split:
                children = [{"name": name, "tracks": sorted(p_ids, key=lambda i: tracks[i]["bpm"])}]
                matches = strong_matches(p_ids, ids, vecs, tracks, args.suggest_threshold)
                if matches:
                    children.append({"name": "Suggested from other playlists", "tracks": [o for o, _ in matches]})
                    for other, (sim, mine) in matches:
                        report.append(f"{name}: {label_of(tracks[other])} ({', '.join(tracks[other]['playlists'])})"
                                      f" ~ {label_of(tracks[mine])}  {sim:.2f}")
            else:
                styled, unplaced, p_eff = split_by_sound_and_tempo(
                    p_ids, p_vecs, tracks, probs, classes, size, args.bpm_width, args.min)
                children = [{"name": f"{style} {bpm_range(b, p_eff)}", "tracks": b, "eff": p_eff} for style, b in styled]
                if unplaced:
                    children.append({"name": "Other tempos", "tracks": unplaced})
            folders.append({"name": name, "children": children})
        tree = {"name": args.folder, "children": folders}
    else:
        ids, vecs, dropped = drop_duplicates(ids, vecs)
        for i in dropped:
            print(f"duplicate audio, skipped: {tracks[i]['path']}")
        styled, unplaced, eff = split_by_sound_and_tempo(
            ids, vecs, tracks, probs, classes, size, args.bpm_width, args.min)
        folders = []
        for style, b in styled:
            if not folders or folders[-1]["name"] != style:
                folders.append({"name": style, "children": []})
            folders[-1]["children"].append({"name": bpm_range(b, eff), "tracks": b})
        tree = {"name": args.folder, "children": list(folders)}
        if unplaced:
            tree["children"].append({"name": "Other tempos", "tracks": unplaced})

    def show(node, depth=0):
        pad = "    " * depth
        if "children" in node:
            print(f"{pad}{node['name']}/")
            for child in node["children"]:
                show(child, depth + 1)
            return
        members, node_eff = node["tracks"], node.get("eff", eff)
        # BPM as Rekordbox shows it; * = counted as half/double time
        bpms = " ".join(f"{tracks[i]['bpm']:.0f}{'*' if round(node_eff.get(i, 0)) != round(tracks[i]['bpm']) else ''}"
                        for i in members)
        examples = ", ".join(label_of(tracks[i])[:40] for i in members[:2])
        print(f"{pad}{node['name']}  ({len(members)})  {examples}\n{pad}    BPM: {bpms}")

    def count(node):
        return (1, 0) if "tracks" in node else tuple(map(sum, zip((0, 1), *map(count, node["children"]))))

    n_playlists, n_folders = count(tree)
    print(f"\n{n_folders - 1} folders, {n_playlists} playlists\n")
    show(tree)
    if report:
        print("\nStrong matches from other playlists (suggestions):")
        for line in report:
            print("  " + line)

    if not args.apply:
        print("\nPreview only. Add --apply to write this folder into Rekordbox.")
        return

    from rekordbox_write import write_folder

    def to_tracks(node):
        if "children" in node:
            return {"name": node["name"], "children": [to_tracks(c) for c in node["children"]]}
        return {"name": node["name"], "tracks": [tracks[i] for i in node["tracks"]]}

    if args.only:
        mine = [c for c in tree["children"] if c["name"] == args.only]
        if not mine:
            raise SystemExit(f"No playlist folder named '{args.only}' in the tree")
        result = write_folder(to_tracks(mine[0]), parent=args.folder, replace=True)
    else:
        result = write_folder(to_tracks(tree), replace=args.replace)
    print(f"\nCreated '{args.folder}': {result['folders']} folders, {result['playlists']} playlists, "
          f"{result['added']} tracks. Backup: {result['backup']}")
    for t in result["missing"]:
        print(f"  not in this Rekordbox library: {t['artist']} - {t['title']}")


if __name__ == "__main__":
    main()
