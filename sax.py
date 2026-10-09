"""
Saxophone detection. Essentia's MTG-Jamendo instrument model scores every ~3 s slice of a song for
"saxophone"; the song's score is the mean of its best 5 slices (a sax hook may only play for a few
seconds, which an average over the whole song would wash out).

    python sax.py scan          # score every song -> sax_scan.json (~12 min, new songs only)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "sax_scan.json"
MODEL = HERE / "models" / "mtg_jamendo_instrument-discogs-effnet-1"
KEEP = ("saxophone", "brass", "horn", "trumpet", "trombone", "clarinet", "flute", "voice")


def scan():
    from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictEffnetDiscogs
    import cluster
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, _ = cluster.merge_copies(tracks, ids, vecs)
    classes = json.loads(Path(str(MODEL) + ".json").read_text())["classes"]
    cols = {c: classes.index(c) for c in KEEP}
    embed = TensorflowPredictEffnetDiscogs(graphFilename=str(HERE / "models" / "discogs-effnet-bs64-1.pb"),
                                           output="PartitionedCall:1")
    head = TensorflowPredict2D(graphFilename=str(MODEL) + ".pb", input="model/Placeholder", output="model/Sigmoid")
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [i for i in songs if i not in out]
    for n, i in enumerate(todo, 1):
        try:
            audio = MonoLoader(filename=tracks[i]["path"], sampleRate=16000, resampleQuality=4)()
            p = np.array(head(np.array(embed(audio), dtype=np.float32)))      # (slices, tags)
            top = lambda k: float(np.sort(p[:, cols[k]])[-5:].mean())
            out[i] = {**{k: top(k) for k in KEEP}, "sax_max": float(p[:, cols["saxophone"]].max()),
                      "sax_share": float((p[:, cols["saxophone"]] > 0.3).mean())}
        except Exception as e:
            out[i] = {"error": str(e)[:80]}
        if n % 50 == 0:
            print(f"{n}/{len(todo)}", flush=True)
            OUT.write_text(json.dumps(out))
    OUT.write_text(json.dumps(out))
    print(f"{len(out)} songs scanned")


if __name__ == "__main__" and sys.argv[1:] == ["scan"]:
    scan()


ROOM = HERE / "sax_room.json"


def room():
    """How much of the range a sax plays in (300 Hz - 2.5 kHz) is already taken, over ~40 s from the peak
    section: the share of the full mix's energy (dB, lower = more room) and how much of that sits in
    a few strong spectral peaks (a busy chord/vocal stack) rather than being open."""
    from essentia.standard import EasyLoader
    import cluster
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, _ = cluster.merge_copies(tracks, ids, vecs)
    cues = json.loads((HERE / "cues.json").read_text())
    out = json.loads(ROOM.read_text()) if ROOM.exists() else {}
    for n, i in enumerate([i for i in songs if i in cues and i not in out], 1):
        audio = EasyLoader(filename=tracks[i]["path"], sampleRate=44100, startTime=float(cues[i]["C"]),
                           endTime=float(cues[i]["C"]) + 40)()
        k = len(audio) // 4096
        if k < 4:
            continue
        power = (np.abs(np.fft.rfft(audio[: k * 4096].reshape(k, 4096) * np.hanning(4096), axis=1)) ** 2)
        f = np.fft.rfftfreq(4096, 1 / 44100)
        band = (f >= 300) & (f < 2500)
        share = float(10 * np.log10(power[:, band].sum() / power.sum()))
        out[i] = {"mid_share": share}
        if n % 100 == 0:
            ROOM.write_text(json.dumps(out))
    ROOM.write_text(json.dumps(out))
    print(f"{len(out)} songs measured")


if __name__ == "__main__" and sys.argv[1:] == ["room"]:
    room()


GROUPS = HERE / "sax_groups.json"
HAS_SAX, TOP_MELODY, TOP_ROOM = 0.25, 25, 25


def groups() -> dict[str, list[str]]:
    """The three sax lists: songs where a sax is heard (score >= HAS_SAX), the TOP_MELODY songs with the
    clearest melody that is not vocal-led, and the TOP_ROOM songs with the most space in the sax's
    range, no vocal and a steady groove. A song can be in more than one; "has sax" songs are left out
    of the other two (a sax already plays there). Reggae-world songs are included like any other."""
    import cluster
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, _ = cluster.merge_copies(tracks, ids, vecs)
    sx = json.loads(OUT.read_text())
    rm = json.loads(ROOM.read_text())
    fe = json.loads((HERE / "roles_features.json").read_text())
    rows = [i for i in songs if i in sx and "saxophone" in sx[i]]
    z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
    s = np.array([sx[i]["saxophone"] for i in rows])
    voice = z(np.array([sx[i]["voice"] for i in rows]))
    mel = np.array([fe[i]["melody"] if i in fe else np.nan for i in rows])
    mid = np.array([rm[i]["mid_share"] if i in rm else np.nan for i in rows])
    steady = np.array([fe[i]["steady"] if i in fe else np.nan for i in rows])
    melody = np.where(~np.isnan(mel), z(mel) - 0.7 * voice, -9)
    room = np.where(~np.isnan(mid), -z(mid) - 0.7 * voice + 0.3 * np.nan_to_num(z(steady)), -9)
    has = [rows[n] for n in np.argsort(-s) if s[n] >= HAS_SAX]
    left = [n for n in range(len(rows)) if rows[n] not in has]
    out = {"Has sax": has,
           "Melody a sax could play": [rows[n] for n in sorted(left, key=lambda n: -melody[n])[:TOP_MELODY]],
           "Room for a sax": [rows[n] for n in sorted(left, key=lambda n: -room[n])[:TOP_ROOM]]}
    GROUPS.write_text(json.dumps(out))
    return out


if __name__ == "__main__" and sys.argv[1:] == ["groups"]:
    import cluster
    tracks = cluster.load(["Deep Tech FLAC"])[0]
    for name, members in groups().items():
        print(f"\n{name}: {len(members)}")
        for i in members:
            print("  ", cluster.label_of(tracks[i])[:60])


LISTEN = HERE / "sax_listen.json"


def listen_times() -> dict[str, float]:
    """Where to listen: the moment the sax is strongest for 'Has sax' songs, cue C for the others."""
    from essentia.standard import MonoLoader, TensorflowPredict2D, TensorflowPredictEffnetDiscogs
    groups_ = json.loads(GROUPS.read_text())
    cues = json.loads((HERE / "cues.json").read_text())
    import cluster
    tracks = cluster.load(["Deep Tech FLAC"])[0]
    col = json.loads(Path(str(MODEL) + ".json").read_text())["classes"].index("saxophone")
    embed = TensorflowPredictEffnetDiscogs(graphFilename=str(HERE / "models" / "discogs-effnet-bs64-1.pb"),
                                           output="PartitionedCall:1")
    head = TensorflowPredict2D(graphFilename=str(MODEL) + ".pb", input="model/Placeholder", output="model/Sigmoid")
    out = {}
    for name, members in groups_.items():
        for i in members:
            if name == "Has sax":
                audio = MonoLoader(filename=tracks[i]["path"], sampleRate=16000, resampleQuality=4)()
                p = np.array(head(np.array(embed(audio), dtype=np.float32)))[:, col]
                out[i] = max(float(np.argmax(p)) * 62 * 256 / 16000 - 8, 0.0)
            else:
                out[i] = float(cues.get(i, {}).get("C", 30.0))
    LISTEN.write_text(json.dumps(out))
    return out


if __name__ == "__main__" and sys.argv[1:] == ["listen"]:
    print(len(listen_times()), "listening points saved")


MORE = HERE / "sax_more.json"


def more(per_group=15, artist_min=3) -> dict[str, list[str]]:
    """The wider search: songs you haven't judged yet that may belong. For each of your four groups (Has sax,
    Melody, Room, aura) the closest by sound to the songs you filed there, with a small bonus for artists you
    have kept several times, plus every unjudged song by an artist you have kept `artist_min` times or more.
    Dub reggae (your Dub Reggae Bass Addict crate, flags cached in reggae_all.json) is left out.
    -> sax_more.json {id: [reasons]}; songs already filed or answered are not offered again."""
    import cluster
    import palette
    from collections import Counter
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, merged = cluster.merge_copies(tracks, ids, vecs)
    filed = json.loads((HERE / "sax_filed.json").read_text())
    aura = json.loads((HERE / "sax_aura.json").read_text())
    reggae = json.loads((HERE / "reggae_all.json").read_text())
    sec = palette.load()
    seen = set(filed) | set(aura)
    pool = [i for i in songs if reggae.get(i, 0) < 0.3 and i in sec and i not in seen]
    first = lambda i: tracks[i]["artist"].split(",")[0].strip().lower()
    kept = {i for i, v in filed.items() if v} | {i for i, v in aura.items() if v == ["Has the aura"]}
    kept_by_artist = Counter(first(i) for i in kept if i in tracks and first(i))
    bonus = np.array([0.02 * min(kept_by_artist.get(first(i), 0), 3) for i in pool])
    targets = {"sounds like your Has sax songs": [i for i, v in filed.items() if "Has sax" in v],
               "sounds like your Melody songs": [i for i, v in filed.items() if "Melody a sax could play" in v],
               "sounds like your Room songs": [i for i, v in filed.items() if "Room for a sax" in v],
               "sounds like your Has the aura songs": [i for i, v in aura.items() if v == ["Has the aura"]]}
    out: dict[str, list[str]] = {}
    for reason, members in targets.items():
        seeds = [i for i in members if i in sec and reggae.get(i, 0) < 0.3]
        score = sum(palette.similarity(sec, s, pool) for s in seeds) / len(seeds) + bonus
        for k in np.argsort(-score)[:per_group]:
            out.setdefault(pool[k], []).append(reason)
    for i in pool:
        n = kept_by_artist.get(first(i), 0)
        if first(i) and n >= artist_min:
            out.setdefault(i, []).append(f"by {tracks[i]['artist'].split(',')[0]}, whom you have kept {n} times")
    MORE.write_text(json.dumps(out))
    return out


if __name__ == "__main__" and sys.argv[1:] == ["more"]:
    print(len(more()), "songs to review")
