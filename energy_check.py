"""
Calibrate "energy" on your ear.

    python energy_check.py sample   # features for every song -> energy_features.json, and a spread of songs to rate -> energy_sample.json
    python energy_check.py fit      # after rating them in the sax page (Energy check tab): fit the energy score to your ratings

Club energy is not brightness or busyness, so the rating decides which measurements count.
"""

import json
import sys

import numpy as np

import cluster
import cues
import organise
import roles
from cluster import HERE

FEATURES = HERE / "energy_features.json"
SAMPLE = HERE / "energy_sample.json"
RATINGS = HERE / "sax_energy.json"
MODEL = HERE / "energy_model.json"
NAMES = ["peak_level", "busy", "contrast", "steady", "melody", "bass_level", "bass_share", "openness", "presence", "air",
         "voice", "party", "relaxed", "happy", "sad", "aggressive", "tempo"]
MUST = ["Horny", "Space Cowboy", "Orbits of Dust", "House in the Hills", "Summer Breeze", "Amicalement", "Underground Jam"]


def collect():
    tracks, ids, merged, energy, reggae = organise.load_state()
    cd = json.loads(cues.OUT.read_text())
    feats = roles.audio_features(tracks, ids, cd)
    ms = roles.model_scores(ids)
    he = json.loads((HERE / "highend.json").read_text())
    op = cluster.openness()
    out = {}
    for n, i in enumerate(ids):
        f = feats[i]
        b = f.get("bass") or [np.nan, np.nan]
        out[i] = dict(peak_level=f["peak_level"], busy=f["busy"], contrast=f["contrast"], steady=f["steady"], melody=f["melody"],
                      bass_level=b[0], bass_share=b[1], openness=op.get(i, np.nan), presence=he.get(i, {}).get("presence", np.nan),
                      air=he.get(i, {}).get("air", np.nan), voice=float(ms["voice"][n]), party=float(ms["party"][n]),
                      relaxed=float(ms["relaxed"][n]), happy=float(ms["happy"][n]), sad=float(ms["sad"][n]),
                      aggressive=float(ms["aggressive"][n]), tempo=roles.lane_bpm(tracks[i]["bpm"]), my_energy=float(energy[i]))
    return tracks, ids, merged, reggae, out


def sample():
    tracks, ids, merged, reggae, f = collect()
    FEATURES.write_text(json.dumps(f))
    filed = json.loads((HERE / "sax_filed.json").read_text())
    have = set(ids)
    sax = sorted({merged.get(i, i) for i, v in filed.items() if v and merged.get(i, i) in have and not reggae[merged.get(i, i)]})
    rest = [i for i in ids if i not in set(sax) and not reggae[i]]
    pick = []
    for name in MUST:
        hit = next((i for i in ids if name.lower() in cluster.label_of(tracks[i]).lower()), None)
        if hit and hit not in pick:
            pick.append(hit)
    for pool, k in ((sax, 22), (rest, 22)):
        pool = sorted([i for i in pool if i not in pick], key=lambda i: f[i]["my_energy"])
        step = len(pool) / k
        pick += [pool[int(step * j + step / 2)] for j in range(k)]
    rng = np.random.default_rng(1)
    rng.shuffle(pick)
    if SAMPLE.exists():                   # keep the songs you were given to rate; only the measurements are refreshed
        print(f"{len(f)} songs measured; kept the existing sample of {len(json.loads(SAMPLE.read_text()))} songs to rate")
        return
    SAMPLE.write_text(json.dumps(pick))
    print(f"{len(f)} songs measured; {len(pick)} songs to rate")


def fit():
    f = json.loads(FEATURES.read_text())
    rated = json.loads(RATINGS.read_text())
    level = {"Low energy": 0.0, "Mid energy": 1.0, "High energy": 2.0}
    ids = [i for i, v in rated.items() if v and v[0] in level and i in f]
    X = np.array([[f[i][k] for k in NAMES] for i in ids], float)
    allX = np.array([[f[i][k] for k in NAMES] for i in f], float)
    mu, sd = np.nanmean(allX, axis=0), np.nanstd(allX, axis=0)
    fill = lambda M: np.where(np.isnan(M), mu, M)
    Z = (fill(X) - mu) / sd
    y = np.array([level[rated[i][0]] for i in ids])
    lam = 3.0
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(len(NAMES)), Z.T @ (y - y.mean()))
    pred = Z @ w + y.mean()
    # leave-one-out check
    loo = []
    for n in range(len(ids)):
        keep = np.arange(len(ids)) != n
        wn = np.linalg.solve(Z[keep].T @ Z[keep] + lam * np.eye(len(NAMES)), Z[keep].T @ (y[keep] - y[keep].mean()))
        loo.append(Z[n] @ wn + y[keep].mean())
    mine = np.array([f[i]["my_energy"] for i in ids])
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
    print(f"{len(ids)} songs rated | my old energy vs your ratings: r={corr(mine, y):.2f} | new fit (leave-one-out): r={corr(np.array(loo), y):.2f}")
    print("what counts for you (+ more energy, - less energy), in standard deviations:")
    for k, v in sorted(zip(NAMES, w), key=lambda x: -abs(x[1])):
        print(f"  {k:12} {v:+.2f}")
    MODEL.write_text(json.dumps({"names": NAMES, "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "intercept": float(y.mean()), "n": len(ids)}))
    print("saved energy_model.json")


if __name__ == "__main__":
    {"sample": sample, "fit": fit}[sys.argv[1]]()
