"""
Two flat crates, each a Rekordbox folder at the root holding one playlist per tempo range (nothing deeper):

  Left field   every song that sits nearer your Dub songs than your house-side songs (the left-field axis, leftfield.py)
  Sanctuary    a sunset opening set: retro / classic house (your Classic, Oldies and beachbar crates), with saxophone and guitar songs
               that fit it sprinkled in between; low-to-mid energy, 108-128 BPM

    python crates.py            # preview
    python crates.py --apply    # write into Rekordbox (it must be closed; backs up master.db first)
"""

import json
import sys

import numpy as np

import cluster
import organise
import roles
from cluster import HERE

LEFT_MIN = 0.5                       # axis position at or above this: clearly nearer your Dub songs (Dub sits at about +1.0, Deep Tech about +0.5)
LEFT_MIN_ZAYAN = 1.0                 # songs that are only in Zayan's playlists need to sit clearly further left than that
ZAYAN = {"beachbar", "beachbar classics (Zayan)", "Beachbar Unc", "Saville Row Disco"}
GUITAR_SHARE = 1 / 3                 # guitar songs are sprinkled in: at most this share of Sanctuary
CORE_MAX = 0.0                       # the retro house core must sit on the house-side of the axis
RETRO_CRATES = {"Classic AIFF", "Oldies AIFF", "beachbar classics (Zayan) AIFF"}
SUNSET_BPM = (106, 128)
SAX_MIN = 0.25                       # the "has sax" line from sax.py
GUITAR_TOP = 0.88                    # guitar songs: the top 12% of the library by guitar activation


def axis_now(tracks, ids, vecs, songs) -> dict:
    """leftfield.py's axis, recomputed over today's library but not saved (the saved one sets the groove-reset folders)."""
    import leftfield
    return leftfield.scores(tracks, ids, vecs, songs)


def guitar_scores(ids) -> dict:
    """Guitar activation per song from the instrument model's head on the stored mean embedding (approximate, fine for ranking)."""
    from essentia.standard import TensorflowPredict2D
    path = str(HERE / "models" / "mtg_jamendo_instrument-discogs-effnet-1")
    classes = json.loads(open(path + ".json").read())["classes"]
    cols = [classes.index(c) for c in ("guitar", "acousticguitar", "electricguitar", "classicalguitar")]
    head = TensorflowPredict2D(graphFilename=path + ".pb", input="model/Placeholder", output="model/Sigmoid")
    data = np.load(HERE / "embeddings.npz")
    raw = {str(i): v for i, v in zip(data["ids"], data["vectors"])}
    p = np.array(head(np.stack([raw[i] for i in ids]).astype(np.float32)))
    return {i: float(p[n, cols].max()) for n, i in enumerate(ids)}


def lanes_to_playlists(members, tracks, width=8):
    eff = cluster.mix_bpm(members, tracks)
    lanes = [b for b in cluster.tempo_bands(members, eff, width) if len(b) >= 3]
    out = [{"name": cluster.bpm_range(lane, eff), "tracks": sorted(lane, key=cluster.blend_key(eff))} for lane in lanes]
    placed = {i for lane in lanes for i in lane}
    rest = sorted((i for i in members if i not in placed), key=lambda i: tracks[i]["bpm"])
    if rest:
        out.append({"name": "Other tempos", "tracks": rest})
    return out


def build():
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, merged = cluster.merge_copies(tracks, ids, vecs)
    songs = [i for i in songs if tracks[i]["bpm"]]
    axis = axis_now(tracks, ids, vecs, songs)
    _, _, _, energy, reggae = organise.load_state()
    lo, hi = np.percentile(list(energy.values()), [100 / 3, 200 / 3])
    rated = json.loads((HERE / "sax_energy.json").read_text()) if (HERE / "sax_energy.json").exists() else {}
    tier = lambda i: ({"Low energy": "Low", "Mid energy": "Mid", "High energy": "High"}.get((rated.get(i) or [None])[0])
                      or ("Low" if energy[i] < lo else "High" if energy[i] > hi else "Mid"))
    sx = json.loads((HERE / "sax_scan.json").read_text())
    filed = json.loads((HERE / "sax_filed.json").read_text())
    guitar = guitar_scores(songs)
    g_line = float(np.quantile(list(guitar.values()), GUITAR_TOP))
    crates = lambda i: {p for p in tracks[i]["playlists"]}
    has_sax = lambda i: sx.get(i, {}).get("saxophone", 0) >= SAX_MIN or bool({"Has sax", "Melody a sax could play"} & set(filed.get(i) or []))

    short = lambda i: {p[:-5] if p.endswith(" AIFF") else p for p in tracks[i]["playlists"]}
    zayan_only = lambda i: bool(short(i)) and short(i) <= ZAYAN
    left = [i for i in songs if axis.get(i, -9) >= (LEFT_MIN_ZAYAN if zayan_only(i) else LEFT_MIN) and i in energy]
    fits = lambda i: SUNSET_BPM[0] <= roles.lane_bpm(tracks[i]["bpm"]) <= SUNSET_BPM[1] and tier(i) in ("Low", "Mid") and not reggae.get(i)
    import leftfield
    frozen = set(json.loads(leftfield.ANCHOR.read_text())["retro"]) if leftfield.ANCHOR.exists() else set()   # the retro core as first approved
    core = [i for i in songs if (crates(i) & RETRO_CRATES or i in frozen) and axis.get(i, 9) < CORE_MAX and fits(i)]
    in_core = set(core)
    sax = [i for i in songs if i not in in_core and has_sax(i) and axis.get(i, 9) < 0.4 and fits(i)]
    gtr = [i for i in songs if i not in in_core and i not in set(sax) and guitar.get(i, 0) >= g_line and axis.get(i, 9) < 0.4 and fits(i)]
    gtr = sorted(gtr, key=lambda i: -guitar[i])[: round((len(core) + len(sax)) * GUITAR_SHARE / (1 - GUITAR_SHARE))]
    return tracks, {"Left field": left, "Sanctuary": core + sax + gtr}, {"retro house core": core, "sax": sax, "guitar": gtr}, axis, guitar


def main():
    apply = "--apply" in sys.argv
    tracks, crates, parts, axis, guitar = build()
    trees = {}
    for name, members in crates.items():
        pls = lanes_to_playlists(sorted(set(members)), tracks)
        trees[name] = {"name": name, "children": [{"name": p["name"], "tracks": [tracks[i] for i in p["tracks"]]} for p in pls]}
        print(f"\n{name}: {len(set(members))} songs in {len(pls)} tempo playlists (lists in Rekordbox: {len(pls) + 1})")
        print("   " + ", ".join(f"{p['name']} ({len(p['tracks'])})" for p in pls))
    print("\nSanctuary is made of:", {k: len(v) for k, v in parts.items()})
    for k in ("sax", "guitar"):
        print(f"  {k}:", "; ".join(cluster.label_of(tracks[i])[:34] for i in parts[k][:6]))
    if not apply:
        print("\nPreview only. Add --apply to write into Rekordbox (closed).")
        return
    import rekordbox_write as w
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    for name, tree in trees.items():
        if only and name != only:
            continue
        r = w.write_folder(tree, replace=True)
        print(f"{name}: written. {r}")


if __name__ == "__main__":
    main()
