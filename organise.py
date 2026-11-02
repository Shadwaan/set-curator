"""
SC / Sax and SC / Aura, built from what you filed in the sax page (sax_filed.json, sax_aura.json).

    SC / Sax / Has sax | Melody a sax could play | Room for a sax
                 each: Low energy, Mid energy, High energy, All (low to high energy)
    SC / Sax / Dub & Reggae / the same three groups   (dub reggae kept in its own tree)
    SC / Aura / Low energy, Mid energy, High energy, All (low to high energy), and Listening (hard to mix) for songs
                 your notes flag as hard to mix

Energy is fitted to your own ratings in the sax page (energy_check.py fit; relaxed lowers it, party feel,
a dark mood, tempo and bass raise it) and split into thirds of the library; without ratings it falls back
to peak level, party feel, busyness and bass. Songs inside an energy playlist run by tempo, then from closed to open high end; the All playlist runs from your
lowest-energy song to your highest, by the fitted energy score.

    python organise.py            # preview
    python organise.py --apply    # write into Rekordbox (close it first)
"""

import argparse
import json

import numpy as np

import cluster
import cues
import roles
from cluster import HERE

GROUPS = ["Has sax", "Melody a sax could play", "Room for a sax"]
TIERS = ["Low energy", "Mid energy", "High energy"]


def energy_scores(ids, feats, models) -> dict[str, float]:
    z = lambda v: (np.asarray(v, float) - np.nanmean(v)) / np.nanstd(v)
    bass = [feats[i].get("bass") for i in ids]
    med = np.nanmedian(np.array([b for b in bass if b], dtype=float), axis=0)
    bass = np.array([b if b else med for b in bass], dtype=float)
    e = (0.5 * z([feats[i]["peak_level"] for i in ids]) + z(models["party"]) + z([feats[i]["busy"] for i in ids])
         + 0.8 * (z(bass[:, 0]) + z(bass[:, 1])) / 2) / 3.3
    return dict(zip(ids, e))


def model_energy(ids, fallback):
    """Energy fitted to your own ratings (energy_check.py fit -> energy_model.json), on the same scale for all songs.
    Songs it has no measurements for get the old score; run `energy_check.py sample` after adding songs."""
    model, feats = HERE / "energy_model.json", HERE / "energy_features.json"
    if not (model.exists() and feats.exists()):
        return fallback
    m = json.loads(model.read_text())
    f = json.loads(feats.read_text())
    names, mu, sd, w = m["names"], np.array(m["mu"]), np.array(m["sd"]), np.array(m["w"])
    out = dict(fallback)
    for i in ids:
        if i in f:
            x = np.array([f[i][k] for k in names], float)
            x = np.where(np.isnan(x), mu, x)
            out[i] = float(((x - mu) / sd) @ w + m["intercept"])
    return out


def load_state():
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, merged = cluster.merge_copies(tracks, ids, vecs)
    cue_data = json.loads(cues.OUT.read_text())
    ids = [i for i in songs if i in cue_data]
    feats = roles.audio_features(tracks, ids, cue_data)
    models = roles.model_scores(ids)
    energy = model_energy(ids, energy_scores(ids, feats, models))
    reggae_score = roles.reggae_scores(ids)
    reggae = {i: roles.is_reggae(i, reggae_score, tracks) for i in ids}
    return tracks, ids, merged, energy, reggae


def tiers_of(energy, ids, override=None):
    """Thirds of the whole library by energy; a note like "low energy" (notes.py) decides for that song."""
    lo, hi = np.percentile(list(energy.values()), [100 / 3, 200 / 3])
    return lambda i: (override or {}).get(i) or (TIERS[0] if energy[i] < lo else TIERS[2] if energy[i] > hi else TIERS[1])


def playlist(name, members, tracks, eff, key=None):
    if not members:
        return None
    members = sorted(members, key=key or cluster.blend_key(eff))
    return {"name": f"{name} {cluster.bpm_range(members, eff)}", "tracks": members}


def energy_folder(name, members, tracks, tier, energy):
    """{"name", "children"}: the three energy tiers and an All playlist, each tempo-ordered."""
    eff = cluster.mix_bpm(members, tracks)
    kids = [playlist(t, [i for i in members if tier(i) == t], tracks, eff) for t in TIERS]
    # "All" builds up by YOUR energy ranking (the score fitted to your ratings): lowest first, tempo only breaks ties
    blend = cluster.blend_key(eff)
    kids.append(playlist("All (low to high energy)", members, tracks, eff,
                         key=lambda i: (TIERS.index(tier(i)), round(energy[i], 2)) + blend(i)))
    return {"name": name, "children": [k for k in kids if k]}


def build():
    tracks, ids, merged, energy, reggae = load_state()
    import notes
    # your own word on a song's energy beats the formula: an Energy check rating first, then an unqualified note
    rated = json.loads((HERE / "sax_energy.json").read_text()) if (HERE / "sax_energy.json").exists() else {}
    override = dict(notes.tier_overrides(merged))
    override.update({merged.get(i, i): v[0] for i, v in rated.items() if v})
    tier = tiers_of(energy, ids, override)
    filed = json.loads((HERE / "sax_filed.json").read_text())
    aura = json.loads((HERE / "sax_aura.json").read_text())
    have = set(ids)
    have_ids = have
    lost = [i for i in set(filed) | set(aura) if i not in have and i not in merged]
    def song(i):
        return merged.get(i, i)                     # a filed copy counts as the song that was kept

    sax = {"name": "Sax", "children": []}
    reggae_tree = {"name": "Dub & Reggae", "children": []}
    for g in GROUPS:
        members = sorted({song(i) for i, v in filed.items() if g in v and song(i) in have})
        plain = [i for i in members if not reggae[i]]
        dub = [i for i in members if reggae[i]]
        sax["children"].append(energy_folder(g, plain, tracks, tier, energy))
        if dub:
            eff = cluster.mix_bpm(dub, tracks)
            reggae_tree["children"].append(playlist(g, dub, tracks, eff))
    if reggae_tree["children"]:
        sax["children"].append(reggae_tree)
    aura_ids = sorted({song(i) for i, v in aura.items() if v == ["Has the aura"] and song(i) in have and not reggae[song(i)]})
    aura_tree = energy_folder("Aura", aura_ids, tracks, tier, energy)
    import vibes as vibe_groups
    have = vibe_groups.members(merged)
    build.vibes = []
    import acid
    acid_ids = [i for i in acid.confirmed(merged) if i in have_ids and not reggae[i]]
    room_ids = [i for i in acid.room_members(merged) if i in have_ids and not reggae[i]]
    if acid_ids or room_ids:
        build.vibes.append({"name": "Acid", "children": [x for x in (energy_folder("Acid lines", acid_ids, tracks, tier, energy) if acid_ids else None,
                                                                    energy_folder("Room for a 303", room_ids, tracks, tier, energy) if room_ids else None) if x]})
    for group in ("Bangkok groove",):          # Beachbar is your beachbar playlist itself, so it gets no folder of its own
        ids_ = [i for i in have[group] if i in have_ids and not reggae[i]]
        if not ids_:
            continue
        folder = energy_folder(group, ids_, tracks, tier, energy)
        if group == "Beachbar":
            club = [i for i in have["Also works in a club"] if i in ids_]
            if club:
                folder["children"].append(playlist("Also works in a club", club, tracks, cluster.mix_bpm(club, tracks)))
        build.vibes.append(folder)
    hard = [i for i in aura_ids if i in notes.listening(merged)]
    if hard:
        aura_tree["children"].append(playlist("Listening (hard to mix)", hard, tracks, cluster.mix_bpm(hard, tracks)))
    return sax, aura_tree, tracks, lost, energy


def show(node, depth=0):
    if "children" in node:
        print("  " * depth + node["name"] + "/")
        for c in node["children"]:
            show(c, depth + 1)
    else:
        print("  " * depth + f"{node['name']}  ({len(node['tracks'])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write SC / Sax and SC / Aura into Rekordbox")
    args = ap.parse_args()
    sax, aura_tree, tracks, lost, _ = build()
    show(sax)
    show(aura_tree)
    for v in getattr(build, "vibes", []):
        show(v)
    if lost:
        print(f"note: {len(lost)} filed songs are no longer in the library data (excluded or removed), skipped")
    if not args.apply:
        print("Preview only. Add --apply to write both folders (Rekordbox closed).")
        return
    from rekordbox_write import write_folder

    def to_tracks(node):
        if "children" in node:
            return {"name": node["name"], "children": [to_tracks(c) for c in node["children"]]}
        return {"name": node["name"], "tracks": [tracks[i] for i in node["tracks"]]}

    for tree in [sax, aura_tree] + getattr(build, "vibes", []):
        result = write_folder(to_tracks(tree), parent="SC", replace=True)
        print(f"Created SC / {tree['name']}: {result['folders']} folders, {result['playlists']} playlists, "
              f"{result['added']} tracks. Backup: {result['backup']}")


if __name__ == "__main__":
    main()
