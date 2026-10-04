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
ROLES = ["1 Openers", "2 Warmers", "3 Momentum builders", "4 Crowd attractors", "5 Sustainers", "6 Moments",
         "7 Closers"]


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
        feats[i] = {"peak_level": float(level[c:c + 8].mean()),
                    "contrast": float(level[c:c + 8].mean() - level[max(0, c - 8):c].mean()) if c else 0.0,
                    "steady": float(kick.mean()), "intro": int(a),
                    "melody": pitch_clarity(tracks[i]["path"], float(bars[c]), float(peak_end)),
                    "busy": busyness(tracks[i]["path"], float(bars[c]))}
        if n % 25 == 0:
            print(f"features {n}/{len(todo)}", flush=True)
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
                            ("mood_happy", "happy"), ("mood_sad", "sad")):
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


def assign(ids, feats, models, bpms) -> dict[str, list[str]]:
    z = lambda v: z_in_lane(v, bpms)
    f = {k: z([feats[i][k] for i in ids]) for k in ("peak_level", "contrast", "steady", "intro", "melody", "busy")}
    m = {k: z(v) for k, v in models.items()}
    energy = (0.5 * f["peak_level"] + m["party"] + f["busy"]) / 2.5
    scores = np.stack([
        -energy + m["relaxed"] + f["intro"] - f["contrast"],                    # openers: gentle, long intro
        f["steady"] - np.abs(energy) + 0.5 * m["party"] - 0.5 * f["contrast"] - 0.5 * f["busy"],  # warmers: gentle groove
        1.5 * f["busy"] + 0.5 * f["steady"] + 0.5 * m["relaxed"] - f["contrast"] - 0.5 * f["peak_level"],
                                                                                 # momentum builders: busy, laid back, no peak
        m["happy"] + m["voice"] + m["party"],                                   # crowd attractors: bright hooks
        f["steady"] + energy - f["melody"] - m["voice"] - f["contrast"],        # sustainers: functional, driving
        2.5 * f["melody"] + f["contrast"] + 0.5 * energy,                       # moments: powerful melody first
        m["sad"] + 0.25 * f["melody"] - energy + 0.5 * m["happy"],              # closers: emotional, winding down
    ], axis=1)
    scores = (scores - scores.mean(axis=0)) / scores.std(axis=0)
    roles = {r: [] for r in ROLES}
    for i, row in zip(ids, scores):
        if row.max() >= 0.5:
            roles[ROLES[row.argmax()]].append(i)
    return roles


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
    roles = assign(ids, feats, model_scores(ids), [tracks[i]["bpm"] for i in ids])

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
    """Per Palettes sound family, its tracks split by role in set order. A family
    spanning several tempo lanes gets a folder per lane (half-time folded towards
    the family's tempo), with the roles inside, so moving from role to role keeps
    both the palette and the tempo."""
    from crosspollinate import mood_scores, palette_groups, palette_matrix
    role_of = {i: role for role, members in roles.items() for i in members}

    def role_playlists(members, eff):
        out = []
        for role in ROLES:
            picked = sorted((i for i in members if role_of.get(i) == role), key=eff.get)
            if len(picked) >= min_tracks:
                out.append({"name": role, "tracks": picked, "eff": eff})
        return out

    tree = {"name": "Set arcs", "children": []}
    for name, members in palette_groups(tracks, songs, palette_matrix(songs), mood_scores(songs)):
        members = [i for i in members if i in role_of]
        eff = cluster.mix_bpm(members, tracks)
        lanes = [b for b in cluster.tempo_bands(members, eff, width) if len(b) >= 2 * min_tracks]
        if len(lanes) <= 1:
            children = role_playlists(members, eff)
        else:
            children = []
            for lane in lanes:
                playlists = role_playlists(lane, eff)
                if len(playlists) >= 2:
                    children.append({"name": cluster.bpm_range(lane, eff), "children": playlists})
            if len(children) == 1:  # one lane left: no need for a lane folder
                children = children[0]["children"]
        if len(children) >= 2 or (children and "children" in children[0]):
            tree["children"].append({"name": name, "children": children})
            print(f"\n{name}")
            for c in children:
                for pl in (c["children"] if "children" in c else [c]):
                    lane = f"{c['name']:>8} " if "children" in c else ""
                    bpms = [eff[i] for i in pl["tracks"]]
                    print(f"    {lane}{pl['name']:22} {len(pl['tracks']):3} tracks  {min(bpms):.0f}-{max(bpms):.0f}")
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
