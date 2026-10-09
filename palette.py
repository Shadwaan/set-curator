"""
Sound-palette analysis: keep each track's section-by-section discogs-effnet
embeddings (every 2nd patch, ~2 s apart) instead of one average, PCA-reduced to
128 dims, in patches.npz. Two tracks with a shared palette have many sections
that sound alike, even at different points in the track.

    python palette.py build        # analysis pass over the library (~15 min)
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "patches.npz"
EVERY, DIMS = 2, 128


def build():
    """Analyse songs that have no sections yet (all of them on the first run). Later runs reuse the
    saved PCA, so earlier results stay valid and only new songs are processed."""
    from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
    import cluster
    tracks, ids, _ = cluster.load(["Deep Tech FLAC"])
    model = TensorflowPredictEffnetDiscogs(graphFilename=str(HERE / "models" / "discogs-effnet-bs64-1.pb"),
                                           output="PartitionedCall:1")
    have, mean, comps = {}, None, None
    if OUT.exists():
        d = np.load(OUT)
        off = d["offsets"]
        have = {str(i): d["data"][off[n]:off[n + 1]] for n, i in enumerate(d["ids"])}
        mean, comps = d["mean"], d["components"]
    todo = [i for i in ids if i not in have]
    raw = {}
    for n, i in enumerate(todo, 1):
        audio = MonoLoader(filename=tracks[i]["path"], sampleRate=16000, resampleQuality=4)()
        raw[i] = np.array(model(audio))[::EVERY].astype(np.float16)
        if n % 25 == 0:
            print(f"{n}/{len(todo)}", flush=True)
    if not raw:
        print("nothing new to analyse")
        return
    if mean is None:
        sample = np.concatenate([p for p in raw.values()]).astype(np.float32)
        rng = np.random.default_rng(0)
        sample = sample[rng.choice(len(sample), min(60000, len(sample)), replace=False)]
        mean = sample.mean(axis=0)
        _, _, vt = np.linalg.svd(sample - mean, full_matrices=False)
        comps = vt[:DIMS]
    for i, p in raw.items():
        z = (p.astype(np.float32) - mean) @ comps.T
        z /= np.linalg.norm(z, axis=1, keepdims=True)
        have[i] = z.astype(np.float16)
    out_ids, offsets = list(have), [0]
    for i in out_ids:
        offsets.append(offsets[-1] + len(have[i]))
    np.savez(OUT, ids=np.array(out_ids), offsets=np.array(offsets), data=np.concatenate([have[i] for i in out_ids]),
             mean=mean.astype(np.float32), components=comps.astype(np.float32))
    print(f"Done: {len(raw)} new, {len(out_ids)} songs in {OUT.name}")


def load() -> dict[str, np.ndarray]:
    if not OUT.exists():
        return {}
    d = np.load(OUT)
    off = d["offsets"]
    return {str(i): d["data"][off[n]:off[n + 1]].astype(np.float32) for n, i in enumerate(d["ids"])}


def similarity(sections: dict[str, np.ndarray], query: str, ids: list[str]) -> np.ndarray:
    """Palette match of `query` against each of `ids`: how close every section of
    one track is to its best counterpart in the other, averaged both ways.
    Tracks without section analysis score 0."""
    a = sections[query]
    out = np.zeros(len(ids))
    for n, j in enumerate(ids):
        if j in sections:
            s = a @ sections[j].T
            out[n] = (s.max(axis=1).mean() + s.max(axis=0).mean()) / 2
    return out


if __name__ == "__main__" and sys.argv[1:] == ["build"]:
    build()
