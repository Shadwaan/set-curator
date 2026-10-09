"""
High-end openness: how much of a song's sound sits in the top of the spectrum compared with
its mids, measured over ~40 s from the peak section (cue C).

    presence   5-10 kHz relative to 200 Hz-2 kHz (dB): hats, snares, vocal sibilance
    air        10-16 kHz relative to 200 Hz-2 kHz (dB): the open, airy top
    top        16-20 kHz relative to 10-16 kHz (dB): a very low value means the file is cut
               off around 16 kHz (typical of a lossy source), so "air" may say more about
               the file than about the production

Higher presence and air = a more open high end; low = muted, closed.

    python highend.py            # measure every song -> highend.json
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "highend.json"
SR = 44100
SIZE = 4096


def measure(path: str, start: float, seconds: float = 40.0) -> dict:
    from essentia.standard import EasyLoader
    audio = EasyLoader(filename=path, sampleRate=SR, startTime=float(start), endTime=float(start) + seconds)()
    n = len(audio) // SIZE
    frames = audio[: n * SIZE].reshape(n, SIZE) * np.hanning(SIZE)
    power = (np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(axis=0)
    freqs = np.fft.rfftfreq(SIZE, 1 / SR)
    band = lambda lo, hi: max(float(power[(freqs >= lo) & (freqs < hi)].sum()), 1e-12)
    mid = band(200, 2000)
    db = lambda a, b: float(10 * np.log10(a / b))
    return {"presence": db(band(5000, 10000), mid), "air": db(band(10000, 16000), mid),
            "top": db(band(16000, 20000), band(10000, 16000))}


def main():
    import cluster
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, _ = cluster.merge_copies(tracks, ids, vecs)
    cue_data = json.loads((HERE / "cues.json").read_text())
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [i for i in songs if i in cue_data and i not in out]
    for n, i in enumerate(todo, 1):
        out[i] = measure(tracks[i]["path"], cue_data[i]["C"])
        if n % 100 == 0:
            print(f"{n}/{len(todo)}", flush=True)
            OUT.write_text(json.dumps(out))
    OUT.write_text(json.dumps(out))
    print(f"{len(out)} songs measured")


if __name__ == "__main__":
    main()
