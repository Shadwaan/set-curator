"""
Set roles, library-wide: SC / Set roles / 1 Openers ... 7 Closers, each split by tempo.

Every track is scored on evidence relative to the tracks at a similar tempo
(half-time folded), so each tempo lane gets its own opener-to-closer arc:
    energy    loudness of the peak section (hot cue C) + Essentia's party mood
    contrast  how much bigger the peak is than the 8 bars before it (drop size)
    melody    pitch clarity of the peak section: how much of it is clear notes and
              chords rather than drums and noise (Fatima Yamaha - What's a Girl to Do
              and its remixes score at the top)
    busy      hits per second from the peak on: how busy the rhythm is
    intro     bars before the kick comes in (cue A)
    steady    share of bars with the kick in
    vocal, relaxed, happy, sad    Essentia models on the stored analysis
and goes to the role it fits best. Needs cues.json (python cues.py).

    python roles.py              # features (cached in roles_features.json) + preview
    python roles.py --apply      # write SC / Set roles into Rekordbox
"""

import argparse
import json

import numpy as np

import cluster
import cues
from cluster import HERE, label_of
from crosspollinate import tempo_folder

FEATURES = HERE / "roles_features.json"
ROLES = ["1 Openers", "2 Warmers", "3 Momentum builders", "4 Crowd attractors", "5 Peak time", "6 Sustainers",
         "7 Moments", "8 Closers"]


def pitch_clarity(path: str, start: float, end: float) -> float:
    import essentia.standard as es
    audio = es.EqualLoudness()(es.EasyLoader(filename=path, sampleRate=44100, startTime=start, endTime=end)())
    window, spectrum = es.Windowing(type="hann"), es.Spectrum()
    peaks = es.SpectralPeaks(minFrequency=300, maxFrequency=5000, magnitudeThreshold=1e-5, maxPeaks=60)
    hpcp = es.HPCP(size=12, minFrequency=300, maxFrequency=5000)
    clarity = []
    for frame in es.FrameGenerator(audio, frameSize=4096, hopSize=2048):
        h = hpcp(*peaks(spectrum(window(frame))))
        if h.sum() > 0:
            clarity.append(h.max() / h.sum())
    return float(np.mean(clarity)) if clarity else 0.0


def busyness(path: str, start: float) -> float:
    """Hits per second over 30 s from the peak: how busy the rhythm is,
    whatever the master level (the Ineffekt Glow Mix scores high, the calm
    Fatima Yamaha original lowest)."""
    import essentia.standard as es
    audio = es.EasyLoader(filename=path, sampleRate=44100, startTime=start, endTime=start + 30)()
    _, rate = es.OnsetRate()(audio)
    return float(rate)


def audio_features(tracks, ids, cue_data) -> dict[str, dict]:
    feats = json.loads(FEATURES.read_text()) if FEATURES.exists() else {}
    # measurements taken at a track's peak section (cue C) are redone when C has moved
    previous = HERE / "cues_v1.json"
    legacy_c = {k: v["bars"]["C"] for k, v in json.loads(previous.read_text()).items()} if previous.exists() else {}
    for i in ids:
        if i in feats and i in cue_data:
            if feats[i].get("c", legacy_c.get(i)) != cue_data[i]["bars"]["C"]:
                del feats[i]
            else:
                feats[i]["c"] = cue_data[i]["bars"]["C"]
                feats[i]["intro"] = cue_data[i]["bars"]["A"]
    todo = [i for i in ids if i in cue_data and i not in feats]
    paths = cues.analysis_paths() if todo else {}
    for n, i in enumerate(todo, 1):
        bars = cues.downbeats(paths[i])
        energy = cues.bar_energy(tracks[i]["path"], bars)
        level = 10 * np.log10(sum(10 ** (energy[b] / 10) for b in cues.BANDS))
        low = energy["low"]
        kick = low >= np.percentile(low, 90) - 8
        a, c = cue_data[i]["bars"]["A"], cue_data[i]["bars"]["C"]
        peak_end = bars[min(c + 16, len(bars) - 1)] if c + 16 < len(bars) else tracks[i]["length"]
        feats[i] = {"c": int(c), "peak_level": float(level[c:c + 8].mean()),
                    "contrast": float(level[c:c + 8].mean() - level[max(0, c - 8):c].mean()) if c else 0.0,
                    "steady": float(kick.mean()), "intro": int(a),
                    "melody": pitch_clarity(tracks[i]["path"], float(bars[c]), float(peak_end)),
                    "busy": busyness(tracks[i]["path"], float(bars[c]))}
        if n % 25 == 0:
            print(f"features {n}/{len(todo)}", flush=True)
            FEATURES.write_text(json.dumps(feats))
    need = [i for i in ids if i in feats and "bass" not in feats[i]]
    if need:
        found = cues.analysis_paths_from_files({i: tracks[i] for i in need})
        for n, i in enumerate(need, 1):
            if i not in found:
                feats[i]["bass"] = None          # no analysis file to read the beatgrid from; filled with the median later
                continue
            bars = cues.downbeats(found[i])
            e = cues.bar_energy(tracks[i]["path"], bars)
            c = cue_data[i]["bars"]["C"]; sl = slice(c, c + 8)
            full = 10 * np.log10(sum(10 ** (e[b][sl] / 10) for b in cues.BANDS).mean())
            low = 10 * np.log10((10 ** (e["low"][sl] / 10)).mean())
            feats[i]["bass"] = [float(low), float(low - full)]   # bass level, bass share of the full mix (dB)
            if n % 50 == 0:
                print(f"bass {n}/{len(need)}", flush=True)
                FEATURES.write_text(json.dumps(feats))
    for n, i in enumerate([i for i in ids if i in feats and "busy" not in feats[i]], 1):
        feats[i]["busy"] = busyness(tracks[i]["path"], cue_data[i]["C"])
        if n % 50 == 0:
            print(f"busyness {n}", flush=True)
            FEATURES.write_text(json.dumps(feats))
    FEATURES.write_text(json.dumps(feats))
    return feats


def model_scores(ids) -> dict[str, np.ndarray]:
    from essentia.standard import TensorflowPredict2D
    data = np.load(HERE / "embeddings.npz")
    raw = {str(i): v for i, v in zip(data["ids"], data["vectors"])}
    x = np.stack([raw[i] for i in ids]).astype(np.float32)
    out = {}
    for model, positive in (("voice_instrumental", "voice"), ("mood_party", "party"), ("mood_relaxed", "relaxed"),
                            ("mood_happy", "happy"), ("mood_sad", "sad"), ("mood_aggressive", "aggressive")):
        meta = json.loads((HERE / "models" / f"{model}-discogs-effnet-1.json").read_text())
        head = TensorflowPredict2D(graphFilename=str(HERE / "models" / f"{model}-discogs-effnet-1.pb"),
                                   input="model/Placeholder", output="model/Softmax")
        out[positive] = np.array(head(x))[:, meta["classes"].index(positive)]
    return out


def lane_bpm(bpm: float) -> float:
    """Fold half/double time into one range, so 69 and 138 share a lane."""
    while bpm and bpm < 100:
        bpm *= 2
    while bpm > 200:
        bpm /= 2
    return bpm


def z_in_lane(x, bpms, width=6.0) -> np.ndarray:
    """How a track compares with tracks at a similar tempo (within +-width BPM,
    half-time folded): a hyped 138 track reads as hyped among 138 tracks rather
    than as average against the whole library."""
    x, lanes = np.asarray(x, dtype=float), np.array([lane_bpm(b) for b in bpms])
    out = np.zeros(len(x))
    for n, b in enumerate(lanes):
        near = np.abs(lanes - b) <= width
        if near.sum() < 15:  # too few neighbours: widen to the 15 nearest tempos
            near = np.argsort(np.abs(lanes - b))[:15]
        out[n] = (x[n] - x[near].mean()) / (x[near].std() or 1)
    return out


# Optional nudges from your own knowledge of your playlists, in playlist_leans.json:
#   {"My Warm-up Crate": [["1 Openers", 0.3], ["2 Warmers", 0.4]], ...}
# Each entry pushes songs in that playlist towards those roles. Scores are in units of one
# standard deviation, so these only nudge: the audio evidence still decides when it disagrees.
LEANS_FILE = HERE / "playlist_leans.json"
LEANS = {k: [tuple(x) for x in v] for k, v in json.loads(LEANS_FILE.read_text(encoding="utf-8")).items()} \
    if LEANS_FILE.exists() else {}


def assign(ids, feats, models, bpms, playlists=None) -> dict[str, list[str]]:
    z = lambda v: z_in_lane(v, bpms)
    f = {k: z([feats[i][k] for i in ids]) for k in ("peak_level", "contrast", "steady", "intro", "melody", "busy")}
    m = {k: z(v) for k, v in models.items()}
    # bass strength: level and share of the full mix at the peak (a strong bassline is energy
    # even when the rhythm is sparse); tracks with no reading get the library median
    got = [feats[i].get("bass") for i in ids]
    med = np.nanmedian(np.array([b for b in got if b], dtype=float), axis=0) if any(got) else np.zeros(2)
    bass = np.array([b if b else med for b in got], dtype=float)
    f["bass"] = (z(bass[:, 0]) + z(bass[:, 1])) / 2
    energy = (0.5 * f["peak_level"] + m["party"] + f["busy"] + 0.8 * f["bass"]) / 3.3
    scores = np.stack([
        -energy + m["relaxed"] + f["intro"] - f["contrast"],                    # openers: gentle, long intro
        f["steady"] - 0.8 * energy + 0.3 * m["party"] - 0.5 * f["contrast"] - 0.3 * f["busy"],    # warmers: gentle groove, below average energy
        1.5 * f["busy"] + 0.5 * f["steady"] + 0.3 * m["relaxed"] + 0.8 * energy - f["contrast"] - 1.2 * np.maximum(0, -energy),
                                                                                 # momentum builders: busy, laid back, no peak
        m["happy"] + m["voice"] + m["party"],                                   # crowd attractors: bright hooks
        energy + 0.7 * f["contrast"] + 0.5 * m["aggressive"] + 0.5 * f["busy"] - 0.5 * m["relaxed"],
                                                                                 # peak time: top energy, hard drop, driving
        f["steady"] + 1.2 * energy + 0.5 * f["busy"] - 0.5 * f["melody"] - 0.3 * m["voice"] - f["contrast"],        # sustainers: functional, driving
        2.5 * f["melody"] + f["contrast"] + 0.5 * energy,                       # moments: powerful melody first
        m["sad"] + 0.25 * f["melody"] - energy + 0.5 * m["happy"],              # closers: emotional, winding down
    ], axis=1)
    scores = (scores - scores.mean(axis=0)) / scores.std(axis=0)
    for n, crates in enumerate(playlists or []):
        for crate in crates:
            for role, push in LEANS.get(crate, []):
                scores[n, ROLES.index(role)] += push
    # every song goes to its closest role, however clear the fit, so nothing has to
    # be hunted for outside the folders
    roles = {r: [] for r in ROLES}
    for i, row in zip(ids, scores):
        roles[ROLES[row.argmax()]].append(i)
    return roles


CORRECTIONS = HERE / "corrections" / "corrections"  # <doc id>.json files fetched from the Crate Notes page


def apply_corrections(roles: dict[str, list[str]], ids: list[str]) -> int:
    """Move songs to the role you chose on the Crate Notes page ("Song corrections").
    A song is matched by a hash of its normalised artist - title, so it survives a
    re-import. Returns how many songs were moved."""
    import hashlib
    from curator import Library, track_key
    chosen = {}
    for f in CORRECTIONS.glob("*.json") if CORRECTIONS.exists() else []:
        data = json.loads(f.read_text(encoding="utf-8"))
        data = data.get("data", data)
        if data.get("role"):
            chosen[f.stem] = data["role"]
    if not chosen:
        return 0
    lib = Library()
    by_name = {r[2:]: r for r in ROLES}
    moved = 0
    for i in ids:
        if i not in lib.row:
            continue
        role = by_name.get(chosen.get("s-" + hashlib.sha1(track_key(*lib.label(i)).encode()).hexdigest()[:16], ""))
        if role and i not in roles[role]:
            for members in roles.values():
                if i in members:
                    members.remove(i)
            roles[role].append(i)
            moved += 1
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bpm-width", type=float, default=8)
    ap.add_argument("--apply", action="store_true", help="write SC / Set roles into Rekordbox")
    ap.add_argument("--replace", action="store_true", help="rebuild the folder if it already exists")
    ap.add_argument("--arcs", action="store_true",
                    help="SC / Set arcs instead: within each Palettes sound family, one playlist per role, "
                         "so moving between roles keeps the palette as well as the tempo")
    args = ap.parse_args()

    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, _ = cluster.merge_copies(tracks, ids, vecs)
    cue_data = json.loads(cues.OUT.read_text())
    ids = [i for i in songs if i in cue_data]
    feats = audio_features(tracks, ids, cue_data)
    roles = assign(ids, feats, model_scores(ids), [tracks[i]["bpm"] for i in ids],
                   [tracks[i]["playlists"] for i in ids])
    moved = apply_corrections(roles, ids)
    if moved:
        print(f"{moved} songs placed by your corrections")

    if args.arcs:
        tree = arcs(tracks, songs, roles)
    else:
        tree = role_folders(tracks, ids, roles, args.bpm_width)

    if not args.apply:
        print(f"Preview only. Add --apply to write SC / {tree['name']} into Rekordbox.")
        return
    from rekordbox_write import write_folder

    def to_tracks(node):
        if "children" in node:
            return {"name": node["name"], "children": [to_tracks(c) for c in node["children"]]}
        return {"name": node["name"], "tracks": [tracks[i] for i in node["tracks"]]}

    result = write_folder(to_tracks(tree), parent="SC", replace=args.replace)
    print(f"Created SC / {tree['name']}: {result['folders']} folders, {result['playlists']} playlists, "
          f"{result['added']} tracks. Backup: {result['backup']}")


def arcs(tracks, songs, roles, width=8, min_tracks=2):
    """Set arcs: Mood > your playlist > (sound style) > [tempo lane] > All, then the roles in
    set order. A family spanning several tempo lanes gets a folder per lane (half-time
    folded towards the family's tempo), so moving from role to role keeps both the palette
    and the tempo. "All" holds every song of the lane and "Other tempos" the songs that fit
    no lane, so no song is left out. This is the old Mood palettes folder with the roles added."""
    from crosspollinate import mood_scores, mood_tree, palette_families, palette_matrix
    role_of = {i: role for role, members in roles.items() for i in members}

    def role_playlists(members, eff):
        out = []
        for role in ROLES:
            picked = sorted((i for i in members if role_of.get(i) == role), key=eff.get)
            if len(picked) >= min_tracks:
                out.append({"name": role, "tracks": picked, "eff": eff})
        return out

    def leaf(name, family):
        members = family["members"]                      # every song in the family, with or without a role
        eff = cluster.mix_bpm(members, tracks)
        lanes = [b for b in cluster.tempo_bands(members, eff, width) if len(b) >= 2 * min_tracks]

        def lane_children(lane_members):
            out = [{"name": "All", "tracks": sorted(lane_members, key=eff.get), "eff": eff}] if len(lane_members) >= 2 else []
            return out + role_playlists(lane_members, eff)

        if len(lanes) <= 1:
            children = lane_children(members)
        else:
            children = [{"name": cluster.bpm_range(lane, eff), "children": lane_children(lane)} for lane in lanes]
            placed = {i for lane in lanes for i in lane}
            rest = sorted((i for i in members if i not in placed), key=lambda i: tracks[i]["bpm"])
            if rest:
                children.append({"name": "Other tempos", "tracks": rest})
        return {"name": name, "children": children} if children else None

    families = palette_families(tracks, songs, palette_matrix(songs), mood_scores(songs))
    tree = {"name": "Set arcs", "children": mood_tree(families, leaf)}

    def show(node, depth=0):
        pad = "    " * depth
        if "children" in node:
            print(f"{pad}{node['name']}/")
            for c in node["children"]:
                show(c, depth + 1)
        else:
            print(f"{pad}{node['name']}  ({len(node['tracks'])})")
    show(tree)
    return tree


def role_folders(tracks, ids, roles, width):
    tree = {"name": "Set roles", "children": []}
    for role, members in roles.items():
        folder = tempo_folder(role, members, tracks, width, 5) or {"name": role, "children": []}
        placed = {i for p in folder["children"] for i in p["tracks"]}
        rest = sorted((i for i in members if i not in placed), key=lambda i: tracks[i]["bpm"])
        if rest:
            folder["children"].append({"name": "Other tempos", "tracks": rest})
        tree["children"].append(folder)
        print(f"\n{role}  ({len(members)} tracks)")
        for p in folder["children"]:
            print(f"    {p['name']}  ({len(p['tracks'])})  " + ", ".join(label_of(tracks[i])[:32] for i in p["tracks"][:2]))
    unassigned = len(ids) - sum(len(v) for v in roles.values())
    print(f"\n{unassigned} tracks fit no role strongly and are left out")
    return tree


if __name__ == "__main__":
    main()
