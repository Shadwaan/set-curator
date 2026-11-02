"""
Rebuild File Fetcher-made Rekordbox entries as fresh entries (what Rekordbox itself accepts for playing over PRO DJ LINK).

For every collection entry with usn = 0 that sits in a playlist, make a new entry for the same file, copy the song's own data (title, artist, album,
key, comment, play count ...) across, move its playlist places, cues and DJ history onto it, and leave the old entry in the collection with no
playlists (so it can be undone). The new entries go in a playlist "Analyse these (rebuilt)": select all there and Analyze Track in Rekordbox.

    python rebuild_entries.py            # preview
    python rebuild_entries.py --apply    # Rekordbox must be closed; backs up master.db first
    python rebuild_entries.py --apply --list "Name"   # put the new entries in a playlist of that name

The old entries stay in the collection with no playlists; python dedupe_entries.py --apply then removes them.
"""

import json
import os
import sys
from datetime import datetime, timezone

import rekordbox_write as w
from pyrekordbox.db6 import tables as T

COPY = ["Title", "ArtistID", "AlbumID", "GenreID", "LabelID", "KeyID", "Commnt", "Rating", "ColorID", "TrackNo", "DiscNo", "ReleaseYear",
        "ReleaseDate", "Subtitle", "DJPlayCount", "ComposerID", "RemixerID", "OrgArtistID"]
LIST = sys.argv[sys.argv.index("--list") + 1] if "--list" in sys.argv else "Analyse these (rebuilt)"


def candidates(db):
    in_playlist = {s.ContentID for s in db.session.query(T.DjmdSongPlaylist)}
    return [c for c in db.session.query(T.DjmdContent).all() if c.usn == 0 and c.ID in in_playlist]


def standard_cues(rows):
    return sorted((r.Kind, r.Color, r.OutMsec, bool(r.Comment)) for r in rows) == [(1, -1, -1, False), (2, -1, -1, False), (3, -1, -1, False)]


def main(apply: bool):
    db = w.open_db()
    todo = candidates(db)
    print(f"{len(todo)} entries to rebuild")
    if not apply:
        return
    if w.is_rekordbox_running():
        raise SystemExit("Close Rekordbox first")
    assert all(os.path.exists(c.FolderPath) for c in todo), "a file is missing: is the SSD connected?"
    print("backup:", w.backup_master_db().name)
    now = datetime.now(timezone.utc)
    hot, moved_cues, pairs = {}, 0, []
    listing = db.create_playlist(LIST)
    db.session.flush()
    for n, old in enumerate(todo, 1):
        path = old.FolderPath
        rows = db.session.query(T.DjmdCue).filter_by(ContentID=old.ID).all()
        standard = standard_cues(rows) and old.FileType != 1
        pts = {k: next(r.InMsec for r in rows if r.Kind == i) / 1000 for i, k in ((1, "A"), (2, "B"), (3, "C"))} if standard else None
        old.FolderPath = path + ".old-entry"
        db.session.flush()
        new = db.add_content(path)
        for col in COPY:
            setattr(new, col, getattr(old, col))
        new.updated_at = now
        new.rb_local_usn = w._next_usn(db, T.DjmdContent)
        db.session.flush()                          # new entries carry a numeric ID, old ones a text ID: never flush both at once
        old.FolderPath = path
        db.session.flush()
        for s in db.session.query(T.DjmdSongPlaylist).filter_by(ContentID=old.ID).all():
            s.ContentID, s.updated_at = new.ID, now
        for h in db.session.query(T.DjmdSongHistory).filter_by(ContentID=old.ID).all():
            h.ContentID = new.ID
        if pts:
            hot[new.ID] = pts
        elif rows:                                  # cues you made yourself (or an MP3's): move the rows as they are
            for r in rows:
                r.ContentID, r.ContentUUID = new.ID, new.UUID
            for cc in db.session.query(T.ContentCue).filter_by(ContentID=old.ID).all():
                cc.ContentID = new.ID
            moved_cues += 1
        db.session.flush()
        db.add_to_playlist(listing, new)
        db.session.flush()
        pairs.append({"old": old.ID, "new": new.ID, "title": old.Title})
        if n % 100 == 0:
            print(f"  {n}/{len(todo)}")
    db.session.commit()
    db.session.close()
    db.engine.dispose()
    w.flush_wal()
    print("hot cues written:", w.write_hot_cues(hot)["written"], "| entries whose cue rows were moved instead:", moved_cues)
    json.dump(pairs, open("saved_lists/library_rebuild_pairs.json", "w"))
    print(f"done: {len(pairs)} entries rebuilt; pairs in saved_lists/library_rebuild_pairs.json")


if __name__ == "__main__":
    main("--apply" in sys.argv)
