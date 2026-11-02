"""Read-only: find collection entries that share a file path and describe each pair (who references them, cues, differences)."""
import collections, json, unicodedata, sys
import rekordbox_write as w
from pyrekordbox.db6 import tables as T

nfc = lambda s: unicodedata.normalize("NFC", s or "")
db = w.open_db()
rows = db.session.query(T.DjmdContent).all()
by_path = collections.defaultdict(list)
for c in rows:
    by_path[c.FolderPath].append(c)
pls = {p.ID: p for p in db.session.query(T.DjmdPlaylist).all()}
def ppath(p):
    o = []
    while p: o.append(nfc(p.Name)); p = pls.get(p.ParentID)
    return "/".join(reversed(o))
sp = db.session.query(T.DjmdSongPlaylist).all()
refs = collections.defaultdict(list)
for s in sp:
    p = pls.get(s.PlaylistID)
    refs[s.ContentID].append((ppath(p) if p else "?", s.PlaylistID, s.TrackNo, s.ID))
cues = collections.defaultdict(list)
for r in db.session.query(T.DjmdCue): cues[r.ContentID].append(r)
hist = collections.Counter(h.ContentID for h in db.session.query(T.DjmdSongHistory))
dups = {p: v for p, v in by_path.items() if len(v) > 1}
print("collection rows:", len(rows), "| distinct paths:", len(by_path), "| paths with 2+ rows:", len(dups), "| paths with 3+:", sum(1 for v in dups.values() if len(v) > 2))
def age(c): return c.created_at.strftime("%m-%d")
print("created-date buckets (all rows):", sorted(collections.Counter(age(c) for c in rows).items()))
print("created-date buckets (duplicate paths):", sorted(collections.Counter(age(c) for v in dups.values() for c in v).items()))
# which of each pair is 'new' (rebuilt: usn None, has DeviceID) and 'old' (usn 0)
kinds = collections.Counter()
detail = []
for p, v in dups.items():
    v = sorted(v, key=lambda c: c.created_at)
    old, new = v[0], v[-1]
    k = ("old usn=%s" % old.usn, "new usn=%s" % new.usn, "old created %s" % age(old), "new created %s" % age(new))
    kinds[k] += 1
    ro, rn = refs[old.ID], refs[new.ID]
    co, cn = sorted((r.Kind, r.InMsec) for r in cues[old.ID]), sorted((r.Kind, r.InMsec) for r in cues[new.ID])
    diffs = [f for f in ("Title", "Commnt", "ArtistID", "AlbumID", "KeyID", "DJPlayCount", "Rating") if getattr(old, f) != getattr(new, f)]
    detail.append({"path": p, "old": old.ID, "new": new.ID, "old_refs": len(ro), "new_refs": len(rn), "old_top": sorted({r[0] for r in ro if not r[0].startswith("SC")}),
                   "old_sc": sum(1 for r in ro if r[0].startswith("SC")), "new_sc": sum(1 for r in rn if r[0].startswith("SC")),
                   "cues_old": len(co), "cues_new": len(cn), "cues_match": co == cn, "diffs": diffs, "hist_old": hist[old.ID], "hist_new": hist[new.ID], "n_rows": len(v)})
for k, n in kinds.most_common(): print(" ", n, k)
json.dump(detail, open("/private/tmp/claude-501/-Users-user-Documents-Claude-Set-Curator/a1dbddd5-c1fb-4191-bf05-5d1a575a5e8b/scratchpad/dup_detail.json", "w"))
print("\npairs where OLD row is still referenced by a playlist:", sum(1 for d in detail if d["old_refs"]))
print("   of those, referenced by SC playlists:", sum(1 for d in detail if d["old_sc"]), "| by top-level playlists only:", sum(1 for d in detail if d["old_refs"] and not d["old_sc"]))
print("pairs where NEW row is referenced by SC playlists:", sum(1 for d in detail if d["new_sc"]), "| new row with no refs at all:", sum(1 for d in detail if not d["new_refs"]))
print("pairs with cues on both and matching:", sum(1 for d in detail if d["cues_old"] and d["cues_new"] and d["cues_match"]), "| cues differ:", sum(1 for d in detail if d["cues_old"] != d["cues_new"] and (d["cues_old"] or d["cues_new"]) or (d["cues_old"] and d["cues_new"] and not d["cues_match"])), "| neither has cues:", sum(1 for d in detail if not d["cues_old"] and not d["cues_new"]))
print("pairs with field differences:", sum(1 for d in detail if d["diffs"]), collections.Counter(f for d in detail for f in d["diffs"]))
print("history rows on old:", sum(d["hist_old"] for d in detail), "| on new:", sum(d["hist_new"] for d in detail))
db.session.close(); db.engine.dispose()
