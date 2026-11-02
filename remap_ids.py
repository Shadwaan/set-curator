"""
Rekordbox song IDs changed when the entries were rebuilt (rebuild_entries.py). Every Set Curator data file keyed by the old ID gets the new one,
matched by the file's path (tracks.json holds old id -> path, master.db holds path -> new id). Backs the files up first.

    python remap_ids.py            # report only
    python remap_ids.py --apply
"""

import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

import rekordbox_write as w
from pyrekordbox.db6 import tables as T

HERE = Path(__file__).parent
SKIP = {"tracks.json", "tracks.before-chillin.json", "tracks.before-ffimport.json"}     # tracks.json is re-exported from Rekordbox


def mapping():
    old = {t["id"]: t["path"] for t in json.loads((HERE / "tracks.json").read_text())}
    db = w.open_db()
    new = {c.FolderPath: c.ID for c in db.session.query(T.DjmdContent).all()}
    exists = {c.ID for c in db.session.query(T.DjmdContent).all()}
    db.session.close()
    db.engine.dispose()
    m = {i: new[p] for i, p in old.items() if p in new and new[p] != i}
    same = sum(1 for i, p in old.items() if new.get(p) == i)
    gone = [i for i, p in old.items() if p not in new]
    return m, same, gone, exists


def remap(x, m, counter):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if isinstance(k, str) and k in m:
                counter[0] += 1
                k = m[k]
            out[k] = remap(v, m, counter)
        return out
    if isinstance(x, list):
        return [remap(v, m, counter) for v in x]
    if isinstance(x, str) and x in m:
        counter[0] += 1
        return m[x]
    return x


def main(apply):
    m, same, gone, exists = mapping()
    print(f"old->new ids: {len(m)} changed | {same} unchanged | {len(gone)} songs no longer in the collection")
    files = sorted(p for p in HERE.glob("*.json") if p.name not in SKIP)
    npz = [HERE / "embeddings.npz", HERE / "patches.npz"]
    if not apply:
        for p in files:
            c = [0]
            remap(json.loads(p.read_text()), m, c)
            if c[0]:
                print(f"  {p.name}: {c[0]} ids")
        return
    if w.is_rekordbox_running():
        raise SystemExit("Close Rekordbox first")
    snap = HERE / "snapshots" / f"before-idremap-{datetime.now():%Y%m%d_%H%M}"
    snap.mkdir(parents=True)
    for p in files + npz + [HERE / "tracks.json"]:
        shutil.copy2(p, snap / p.name)
    print("backed up the data files to", snap)
    total = 0
    for p in files:
        c = [0]
        data = remap(json.loads(p.read_text()), m, c)
        if c[0]:
            p.write_text(json.dumps(data, ensure_ascii=False))
            total += c[0]
            print(f"  {p.name}: {c[0]} ids")
    for p in npz:
        z = dict(np.load(p, allow_pickle=False))
        ids = [str(i) for i in z["ids"]]
        n = sum(1 for i in ids if i in m)
        z["ids"] = np.array([m.get(i, i) for i in ids])
        np.savez_compressed(p, **z) if p.name != "embeddings.npz" else np.savez(p, **z)
        print(f"  {p.name}: {n} ids")
        total += n
    print("done:", total, "ids remapped; tracks.json is next re-exported from Rekordbox")


if __name__ == "__main__":
    main("--apply" in sys.argv)
