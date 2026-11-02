"""
One collection entry per file path. For each path with two entries (an old File Fetcher one and a rebuilt one -- see rebuild_entries.py) keep the
rebuilt entry: merge what only the old one holds (a longer comment, a higher play count, DJ history, cues the new one lacks), take the old one out of
every playlist (an entry the new one is already in is removed, one it is not in is repointed), renumber the playlists, then delete the old entry
and the rows that hang off it. Audio files and analysis files on disk are never touched.

    python dedupe_entries.py            # preview with counts
    python dedupe_entries.py --apply    # Rekordbox must be closed; backs up master.db first
"""

import collections
import inspect
import sys

import rekordbox_write as w
from pyrekordbox.db6 import tables as T


def find_pairs(db):
    by_path = collections.defaultdict(list)
    for c in db.session.query(T.DjmdContent).all():
        by_path[c.FolderPath].append(c)
    pairs = []
    for path, rows in by_path.items():
        if len(rows) < 2:
            continue
        assert len(rows) == 2, f"{len(rows)} entries for {path}"
        rows.sort(key=lambda c: c.created_at)
        old, new = rows
        assert old.created_at < new.created_at and new.usn is None, f"unexpected pair for {path}"
        pairs.append((old, new))
    return pairs


def plan(db, pairs):
    in_pl = collections.defaultdict(set)
    for s in db.session.query(T.DjmdSongPlaylist):
        in_pl[s.PlaylistID].add(s.ContentID)
    remove = repoint = 0
    for old, new in pairs:
        for s in db.session.query(T.DjmdSongPlaylist).filter_by(ContentID=old.ID):
            if new.ID in in_pl[s.PlaylistID]:
                remove += 1
            else:
                repoint += 1
    return remove, repoint


def merge_and_delete(db, pairs):
    """Move everything of value from each old entry to its new one, then delete the old entries. Returns counts."""
    out = collections.Counter()
    touched = set()
    for old, new in pairs:
        if old.Commnt and old.Commnt != new.Commnt and old.Commnt.startswith(new.Commnt or ""):
            new.Commnt = old.Commnt
            out["comments merged"] += 1
        if (old.DJPlayCount or 0) > (new.DJPlayCount or 0):
            new.DJPlayCount = old.DJPlayCount
            out["play counts merged"] += 1
        for h in db.session.query(T.DjmdSongHistory).filter_by(ContentID=old.ID).all():
            h.ContentID = new.ID
            out["history rows moved"] += 1
        old_cues = db.session.query(T.DjmdCue).filter_by(ContentID=old.ID).all()
        if old_cues and not db.session.query(T.DjmdCue).filter_by(ContentID=new.ID).count():
            for r in old_cues:
                r.ContentID, r.ContentUUID = new.ID, new.UUID
            for cc in db.session.query(T.ContentCue).filter_by(ContentID=old.ID).all():
                cc.ContentID = new.ID
            out["cue sets moved"] += 1
        here = {s.PlaylistID for s in db.session.query(T.DjmdSongPlaylist).filter_by(ContentID=new.ID)}
        for s in db.session.query(T.DjmdSongPlaylist).filter_by(ContentID=old.ID).all():
            touched.add(s.PlaylistID)
            if s.PlaylistID in here:
                db.session.delete(s)
                out["playlist entries removed"] += 1
            else:
                s.ContentID = new.ID
                here.add(s.PlaylistID)
                out["playlist entries repointed"] += 1
    db.session.flush()
    for pid in touched:                                        # keep the order, close the gaps
        rows = db.session.query(T.DjmdSongPlaylist).filter_by(PlaylistID=pid).order_by(T.DjmdSongPlaylist.TrackNo, T.DjmdSongPlaylist.ID).all()
        for n, s in enumerate(rows, 1):
            if s.TrackNo != n:
                s.TrackNo = n
    ids = [old.ID for old, _ in pairs]
    tables = [cls for _, cls in inspect.getmembers(T, inspect.isclass) if hasattr(cls, "__tablename__") and hasattr(cls, "ContentID")]
    for i in range(0, len(ids), 400):
        chunk = ids[i:i + 400]
        for cls in tables:
            n = db.session.query(cls).filter(cls.ContentID.in_(chunk)).delete(synchronize_session=False)
            if n:
                out[f"{cls.__tablename__} rows deleted"] += n
        out["entries deleted"] += db.session.query(T.DjmdContent).filter(T.DjmdContent.ID.in_(chunk)).delete(synchronize_session=False)
    return out


def main(apply):
    db = w.open_db()
    pairs = find_pairs(db)
    remove, repoint = plan(db, pairs)
    print(f"{len(pairs)} pairs | playlist entries to remove: {remove} | to repoint: {repoint}")
    if not apply:
        return
    if w.is_rekordbox_running():
        raise SystemExit("Close Rekordbox first")
    print("backup:", w.backup_master_db().name)
    try:
        out = merge_and_delete(db, pairs)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    finally:
        db.session.close()
        db.engine.dispose()
    w.flush_wal()
    for k, v in sorted(out.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main("--apply" in sys.argv)
