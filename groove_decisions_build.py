"""My proposals about the groove resets, for you to check in the page ("My decisions"). -> groove_decisions.json"""
import json, numpy as np
import cluster, groove, leftfield
f = json.load(open("energy_features.json")); m = json.load(open("energy_model.json")); ans = json.load(open("sax_roles.json"))
tracks, ids, vecs = cluster.load(["Deep Tech FLAC"]); songs, _, merged = cluster.merge_copies(tracks, ids, vecs)
names, mu, sd, w, b = m["names"], np.array(m["mu"]), np.array(m["sd"]), np.array(m["w"]), m["intercept"]
sc = lambda i: float((np.where(np.isnan(v := np.array([f[i][k] for k in names], float)), mu, v) - mu) / sd @ w + b)
lo, hi = np.percentile([sc(i) for i in f], [100/3, 200/3])
yes = sorted({merged.get(i, i) for i, a in ans.items() if a} | set(groove.seeds(merged)))
out = {}
for i in yes:
    if i not in f: continue
    side = leftfield.side(i); z = leftfield.axis().get(i, 0)
    energy = sc(i); high = energy > hi
    if high:
        out[i] = {"decision": "drop", "why": f"DECISION: drop as a groove reset. It is High energy on your own scale ({energy:.2f}); a reset should calm the floor down. It stays in its other roles. ({side}, axis {z:+.2f})"}
    else:
        out[i] = {"decision": "keep", "why": f"DECISION: keep as a {side} reset. {'Mid' if energy >= lo else 'Low'} energy ({energy:.2f}), steadiness {f[i]['steady']:.2f}, {tracks[i]['bpm']:.0f} BPM; axis {z:+.2f} ({'left of' if z >= leftfield.SPLIT else 'right of'} the split at {leftfield.SPLIT})."}
json.dump(out, open("groove_decisions.json", "w"))
for i, d in out.items(): print(f"  {tracks[i]['title'][:34]:34} {d['decision']}")
