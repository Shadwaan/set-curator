"""
Import a folder of music into Rekordbox: one playlist per sub-folder, named like the sub-folder, all inside one Rekordbox folder.

Entries are made the way that plays on the XDJs (pyrekordbox add_content, not File Fetcher's usn = 0 rows) and carry an artist and an
album, so Rekordbox can analyse them in a batch. MP3s that are low bitrate, or labelled 320 but with a lossy cutoff (File Fetcher's
authenticity check), get a trailing label on the title the way sff labels versions: "Title [MP3 128k]" or "Title [MP3 ~128k]".

    python import_folder.py <tags.json> <auth.json> <folder name in Rekordbox>           # preview
    python import_folder.py <tags.json> <auth.json> <folder name in Rekordbox> --apply   # Rekordbox must be closed

tags.json is a list of {"path", "sub", "artist", "title", "album", "kbps", "ok"}; auth.json maps path -> File Fetcher's check_authenticity result.
"""

import collections
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import rekordbox_write as w
from curator import norm
from pyrekordbox.db6 import tables as T

SKIP_NAMES = {"Hatch b2b bhai/Alhassan (160 Flip).mp3", "Hatch b2b bhai/Baianá.mp3"}            # (the unreadable ones are skipped by "ok" too)


def label_for(rec, auth):
    """The trailing [label] for a low-quality MP3, or "" for a good file / a WAV."""
    if not rec["path"].lower().endswith(".mp3"):
        return ""
    kbps = rec["kbps"]
    if 0 < kbps < 250:                                   # VBR files average about 245-260 and are fine; CBR 256 and up is fine
        return f"MP3 {kbps}k"
    a = auth.get(rec["path"], {})
    if a.get("suspect"):
        return "MP3 ~128k" if (a.get("cliff_khz") or 19) <= 17 else "MP3 ~192-256k"
    return ""


def names(rec):
    stem = Path(rec["path"]).stem
    title = rec["title"] or stem
    artist = rec["artist"]
    if not artist and " - " in stem:
        artist, _, rest = stem.partition(" - ")
        title = rec["title"] or rest
    return title.strip(), (artist or "Unknown Artist").strip(), (rec["album"] or rec["sub"]).strip()


def main(tags_path, auth_path, folder_name, apply):
    tags = json.loads(Path(tags_path).read_text())
    auth = json.loads(Path(auth_path).read_text())
    db = w.open_db()
    have = {c.FolderPath for c in db.session.query(T.DjmdContent).all()}
    lib = {}
    for c in db.session.query(T.DjmdContent).all():
        a = db.session.query(T.DjmdArtist).filter_by(ID=c.ArtistID).first()
        lib.setdefault((norm(a.Name if a else ""), norm(c.Title)), c)
    todo, skipped = [], collections.defaultdict(list)
    for r in sorted(tags, key=lambda r: (r["sub"].lower(), Path(r["path"]).name.lower())):
        if r["path"] in have:
            skipped["already in Rekordbox"].append(r["path"])
        elif not r["ok"]:
            skipped["cannot be read (broken file)"].append(r["path"])
        elif r["title"] and (norm(r["artist"]), norm(r["title"])) in lib:
            skipped["same song as one already in your library"].append(r["path"])
        else:
            todo.append(r)
    labelled = [r for r in todo if label_for(r, auth)]
    subs = collections.Counter(r["sub"].strip() for r in todo)
    print(f"{len(todo)} files to import into {len(subs)} playlists | labelled as low bitrate: {len(labelled)} | skipped: " + ", ".join(f"{len(v)} {k}" for k, v in skipped.items()))
    print("label examples:", sorted({label_for(r, auth) for r in labelled})[:8])
    if not apply:
        return skipped
    if w.is_rekordbox_running():
        raise SystemExit("Close Rekordbox first")
    print("backup:", w.backup_master_db().name)
    now = datetime.now(timezone.utc)
    artists, albums = {}, {}

    def artist_id(name):
        if name not in artists:
            row = db.session.query(T.DjmdArtist).filter_by(Name=name).first() or db.add_artist(name=name)
            artists[name] = str(row.ID)
        return artists[name]

    def album_id(name):
        if name not in albums:
            row = db.session.query(T.DjmdAlbum).filter_by(Name=name).first() or db.add_album(name=name)
            albums[name] = str(row.ID)
        return albums[name]

    folder = db.create_playlist_folder(folder_name)
    db.session.flush()
    w._register_playlist_in_xml(str(folder.ID), "root", 1)
    lists, made = {}, 0
    for r in todo:
        sub = r["sub"].strip()
        if sub not in lists:
            lists[sub] = db.create_playlist(sub, parent=folder)
            db.session.flush()
            w._register_playlist_in_xml(str(lists[sub].ID), str(folder.ID), 0)
        title, artist, album = names(r)
        lab = label_for(r, auth)
        c = db.add_content(r["path"], Title=f"{title} [{lab}]" if lab else title)
        c.ArtistID, c.AlbumID = artist_id(artist), album_id(album)
        c.updated_at = now
        db.session.flush()
        db.add_to_playlist(lists[sub], c)
        db.session.flush()
        made += 1
    db.session.commit()
    db.session.close()
    db.engine.dispose()
    w.flush_wal()
    print(f"imported {made} songs into {len(lists)} playlists inside the Rekordbox folder '{folder_name}'")
    return skipped


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], "--apply" in sys.argv)
