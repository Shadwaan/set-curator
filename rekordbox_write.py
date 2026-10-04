"""
Write playlists into Rekordbox's master.db.

Ported from Standalone File Fetcher (sff, services/rekordbox.py), which worked
these rules out against Rekordbox 6/7 the hard way. A playlist that misses any
of them shows up EMPTY in Rekordbox even though the rows exist:

  DjmdPlaylist      explicit 32-bit ID (Rekordbox keeps playlist IDs as 32-bit
                    hex in masterPlaylists6.xml), a UUID, rb_data_status=0,
                    next rb_local_usn, usn=None; placed at the top (Seq=0).
  masterPlaylists6.xml
                    a <NODE Id="{HEX}" .../> entry, or Rekordbox doesn't know
                    the playlist exists.
  DjmdSongPlaylist  explicit ID, a UUID (rows without one are filtered out),
                    rb_data_status=0, next rb_local_usn, 1-based TrackNo.
  WAL               checkpoint TWICE after writing, and check master.db-wal is
                    0 bytes -- Rekordbox reads master.db and never the WAL.

Safety (also from sff): refuse while Rekordbox is running, back up master.db
first, only ever ADD -- nothing here deletes tracks or playlists.

Only playlists are created; tracks must already be in the Rekordbox library.
"""

import os
import random
import shutil
import sys
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


def rekordbox_dir() -> Path:
    override = os.environ.get("SET_CURATOR_RB_DIR")  # e.g. a copy, for testing
    if override:
        return Path(override)
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "Pioneer" / "rekordbox"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Pioneer" / "rekordbox"
    raise RuntimeError("Rekordbox only runs on Windows and macOS")


def open_db():
    from pyrekordbox import Rekordbox6Database
    return Rekordbox6Database(path=str(rekordbox_dir() / "master.db"))


def is_rekordbox_running() -> bool:
    import psutil
    for proc in psutil.process_iter(["name"]):
        try:
            if (proc.info.get("name") or "").lower() in {"rekordbox.exe", "rekordbox"}:
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def backup_master_db() -> Path:
    src = rekordbox_dir() / "master.db"
    dst = src.with_name(f"master.db.bak.setcurator.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(src, dst)
    return dst


def _next_usn(db, table) -> int:
    from sqlalchemy import func
    return (db.session.query(func.max(table.rb_local_usn)).scalar() or 0) + 1


def _new_id(db, table, low: int, high: int) -> str:
    while True:
        new_id = str(random.randint(low, high))
        if not db.session.query(table).filter_by(ID=new_id).first():
            return new_id


def _sync_fields(row, usn: int):
    now = datetime.now(timezone.utc)
    row.UUID = str(uuid.uuid4())
    row.rb_data_status = 0
    row.rb_local_data_status = 0
    row.rb_local_deleted = 0
    row.rb_local_synced = 0
    row.usn = None
    row.rb_local_usn = usn
    row.created_at = now
    row.updated_at = now


def _register_playlist_in_xml(playlist_id: str, parent_id: str = "root", attribute: int = 0):
    """Nodes are a flat list; nesting is expressed by ParentId (hex, "0" = root)."""
    xml_path = rekordbox_dir() / "masterPlaylists6.xml"
    if not xml_path.exists():
        print(f"warning: {xml_path} not found -- playlist may appear empty in Rekordbox")
        return
    id_hex = format(int(playlist_id), "X")
    tree = ET.parse(xml_path)
    playlists = tree.getroot().find("PLAYLISTS")
    if playlists is None:
        print("warning: no PLAYLISTS node in masterPlaylists6.xml")
        return
    if any(n.get("Id") == id_hex for n in playlists.findall("NODE")):
        return
    ET.SubElement(playlists, "NODE", {
        "Id": id_hex, "ParentId": "0" if parent_id == "root" else format(int(parent_id), "X"),
        "Attribute": str(attribute),
        "Timestamp": str(int(datetime.now().timestamp() * 1000)),
        "Lib_Type": "0", "CheckType": "0",
    })
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)


def _delete_subtree(db, root_id: str) -> list[str]:
    """Delete a playlist/folder and everything inside it (playlist rows and their
    track entries; the tracks themselves stay in the collection)."""
    from pyrekordbox.db6 import tables
    gone, stack = [], [root_id]
    while stack:
        pid = stack.pop()
        stack += [str(p.ID) for p in db.session.query(tables.DjmdPlaylist).filter_by(ParentID=pid)]
        db.session.query(tables.DjmdSongPlaylist).filter_by(PlaylistID=pid).delete(synchronize_session=False)
        gone.append(pid)
    db.session.query(tables.DjmdPlaylist).filter(tables.DjmdPlaylist.ID.in_(gone)).delete(synchronize_session=False)
    db.session.flush()
    return gone


def _unregister_playlists_in_xml(playlist_ids: list[str]):
    xml_path = rekordbox_dir() / "masterPlaylists6.xml"
    if not playlist_ids or not xml_path.exists():
        return
    tree = ET.parse(xml_path)
    playlists = tree.getroot().find("PLAYLISTS")
    hexes = {format(int(i), "X") for i in playlist_ids}
    for node in list(playlists.findall("NODE")):
        if node.get("Id") in hexes:
            playlists.remove(node)
    tree.write(xml_path, encoding="UTF-8", xml_declaration=True)


def flush_wal():
    from sqlalchemy import text
    for _ in range(2):
        db = open_db()
        db.session.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
        db.session.commit()
        db.session.close()
        db.engine.dispose()
    wal = rekordbox_dir() / "master.db-wal"
    size = wal.stat().st_size if wal.exists() else 0
    if size:
        print(f"warning: master.db-wal is {size} bytes after checkpoint -- Rekordbox may not see the changes")


def resolve_content_ids(db, tracks: list[dict]) -> tuple[list[str], list[dict]]:
    """Map tracks.json entries to DjmdContent IDs in THIS Rekordbox database.
    The exported ID is used when it still points at the same file name; otherwise
    (e.g. tracks.json came from another machine's library) fall back to a unique
    file-name match. Returns (content_ids, unresolved)."""
    from pyrekordbox.db6 import tables
    by_name: dict[str, list[str]] = {}
    for c in db.session.query(tables.DjmdContent).all():
        by_name.setdefault((c.FileNameL or "").lower(), []).append(str(c.ID))
    ids, missing = [], []
    for t in tracks:
        fname = Path(t["path"]).name.lower()
        c = db.session.query(tables.DjmdContent).filter_by(ID=t["id"]).first()
        if c and (c.FileNameL or "").lower() == fname:
            ids.append(str(c.ID))
        elif len(by_name.get(fname, [])) == 1:
            ids.append(by_name[fname][0])
        else:
            missing.append(t)
    return ids, missing


def write_playlist(name: str, tracks: list[dict], append: bool = False) -> dict:
    """Create playlist `name` holding `tracks` (tracks.json entries) in order.
    If it already exists: refuse, or with append=True add the tracks it lacks."""
    return write_playlists([(name, tracks)], append=append)[0]


def write_playlists(playlists: list[tuple[str, list[dict]]], append: bool = False) -> list[dict]:
    """Write several playlists in one transaction with one backup: either all
    are written or none. They end up at the top of the tree in the given order."""
    from pyrekordbox.db6 import tables

    # The check guards the live library; a SET_CURATOR_RB_DIR copy is safe to write.
    if not os.environ.get("SET_CURATOR_RB_DIR") and is_rekordbox_running():
        raise RuntimeError("Close Rekordbox first -- it overwrites master.db with its in-memory copy")

    backup = backup_master_db()
    db = open_db()
    results = []
    try:
        playlist_usn = _next_usn(db, tables.DjmdPlaylist)
        usn = _next_usn(db, tables.DjmdSongPlaylist)
        # each new playlist is inserted at Seq=0, so create them last-first
        for name, tracks in reversed(playlists):
            content_ids, missing = resolve_content_ids(db, tracks)

            playlist = db.session.query(tables.DjmdPlaylist).filter_by(Name=name).first()
            created = playlist is None
            if playlist and not append:
                raise RuntimeError(f"Playlist '{name}' already exists (use --append to add to it)")
            if created:
                for pl in db.session.query(tables.DjmdPlaylist).filter_by(ParentID="root").all():
                    if pl.Seq is not None:
                        pl.Seq += 1
                playlist = tables.DjmdPlaylist()
                playlist.ID = _new_id(db, tables.DjmdPlaylist, 1, 2**32 - 1)
                playlist.Name = name
                playlist.Seq = 0
                playlist.Attribute = 0
                playlist.ParentID = "root"
                _sync_fields(playlist, playlist_usn)
                playlist_usn += 1
                db.session.add(playlist)
                db.session.flush()

            existing = {str(s.ContentID): s.TrackNo or 0
                        for s in db.session.query(tables.DjmdSongPlaylist).filter_by(PlaylistID=playlist.ID)}
            track_no = max(existing.values(), default=0)
            added = 0
            for cid in content_ids:
                if cid in existing:
                    continue
                track_no += 1
                song = tables.DjmdSongPlaylist()
                song.ID = _new_id(db, tables.DjmdSongPlaylist, 10**9, 10**10 - 1)
                song.PlaylistID = playlist.ID
                song.ContentID = cid
                song.TrackNo = track_no
                _sync_fields(song, usn)
                usn += 1
                db.session.add(song)
                existing[cid] = track_no
                added += 1
            db.session.flush()
            results.append({"name": name, "playlist_id": str(playlist.ID), "created": created,
                            "added": added, "missing": missing, "backup": str(backup)})

        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    finally:
        db.session.close()
        db.engine.dispose()

    for r in results:
        if r["created"]:
            _register_playlist_in_xml(r["playlist_id"])
    flush_wal()
    return results[::-1]


def write_folder(folder: dict, parent: str | None = None, replace: bool = False) -> dict:
    """Create a folder tree at the top of the playlist tree -- or, with `parent`
    ("SC" or a path like "SC/Cross-pollination"), as the last item inside that
    existing folder -- in one transaction
    with one backup. `folder` is {"name", "children"}; each child is either a
    folder of the same shape or a playlist {"name", "tracks"} (tracks.json
    entries). Folders are DjmdPlaylist rows with Attribute=1; children point to
    them via ParentID and are numbered Seq 1..n within their folder. If the
    destination already has an item with the folder's name: refuse, or with
    replace=True delete it (and everything inside) and put the new tree in its place."""
    from pyrekordbox.db6 import tables

    if not os.environ.get("SET_CURATOR_RB_DIR") and is_rekordbox_running():
        raise RuntimeError("Close Rekordbox first -- it overwrites master.db with its in-memory copy")

    backup = backup_master_db()
    db = open_db()
    created, missing, added, removed = [], [], 0, []
    try:
        parent_id = "root"
        for part in (parent.split("/") if parent else []):  # e.g. "SC/Cross-pollination"
            host = db.session.query(tables.DjmdPlaylist).filter_by(ParentID=parent_id, Name=part, Attribute=1).first()
            if host is None:
                raise RuntimeError(f"No folder '{part}' in '{parent}'")
            parent_id = str(host.ID)
        seq = None
        existing = db.session.query(tables.DjmdPlaylist).filter_by(ParentID=parent_id, Name=folder["name"]).first()
        if existing:
            if not replace:
                raise RuntimeError(f"'{folder['name']}' already exists there -- use replace to rebuild it")
            seq = existing.Seq
            removed = _delete_subtree(db, str(existing.ID))
        usn = {"playlist": _next_usn(db, tables.DjmdPlaylist), "song": _next_usn(db, tables.DjmdSongPlaylist)}

        def make(node, parent_id, seq):
            nonlocal added
            row = tables.DjmdPlaylist()
            row.ID = _new_id(db, tables.DjmdPlaylist, 1, 2**32 - 1)
            row.Name = node["name"]
            row.Seq = seq
            row.Attribute = 1 if "children" in node else 0
            row.ParentID = parent_id
            _sync_fields(row, usn["playlist"])
            usn["playlist"] += 1
            db.session.add(row)
            db.session.flush()
            created.append((str(row.ID), parent_id, row.Attribute))
            if "children" in node:
                for n, child in enumerate(node["children"], 1):
                    make(child, str(row.ID), n)
                return
            content_ids, unresolved = resolve_content_ids(db, node["tracks"])
            missing.extend(unresolved)
            for track_no, cid in enumerate(dict.fromkeys(content_ids), 1):
                song = tables.DjmdSongPlaylist()
                song.ID = _new_id(db, tables.DjmdSongPlaylist, 10**9, 10**10 - 1)
                song.PlaylistID = row.ID
                song.ContentID = cid
                song.TrackNo = track_no
                _sync_fields(song, usn["song"])
                usn["song"] += 1
                db.session.add(song)
                added += 1

        if seq is not None:
            make(folder, parent_id, seq)  # replacing: same position
        elif parent_id == "root":
            for pl in db.session.query(tables.DjmdPlaylist).filter_by(ParentID="root").all():
                if pl.Seq is not None:
                    pl.Seq += 1
            make(folder, "root", 0)
        else:
            make(folder, parent_id, db.session.query(tables.DjmdPlaylist).filter_by(ParentID=parent_id).count() + 1)
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    finally:
        db.session.close()
        db.engine.dispose()

    _unregister_playlists_in_xml(removed)
    for playlist_id, parent_id, attribute in created:
        _register_playlist_in_xml(playlist_id, parent_id, attribute)
    flush_wal()
    return {"folders": sum(1 for c in created if c[2] == 1),
            "playlists": sum(1 for c in created if c[2] == 0),
            "added": added, "missing": missing, "removed": len(removed), "backup": str(backup)}


def write_hot_cues(cues: dict[str, dict[str, float]]) -> dict:
    """Hot cues A/B/C for tracks that have no cues yet, in one transaction with
    one backup. `cues` maps a track ID to {"A": seconds, "B": ..., "C": ...}.

    Stored the way Rekordbox 7 stores them (copied from cues set in Rekordbox):
    one djmdCue row per cue (Kind 1/2/3 = A/B/C, InFrame in 1/150 s, no colour,
    rb_local_usn None) plus one contentCue row per track, keyed by the track's
    UUID, whose Cues column repeats the rows as JSON. The analysis files' cue
    tags stay empty, as Rekordbox leaves them locally. Tracks with any existing
    cue are never touched; MP3s are skipped (their cues also carry MPEG frame
    offsets we have no example of)."""
    import json
    from pyrekordbox.db6 import tables

    if not os.environ.get("SET_CURATOR_RB_DIR") and is_rekordbox_running():
        raise RuntimeError("Close Rekordbox first -- it overwrites master.db with its in-memory copy")

    backup = backup_master_db()
    db = open_db()
    written, skipped = 0, {"already has cues": 0, "mp3": 0, "not in library": 0}
    stamp = lambda t: t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}+00:00"
    try:
        has = {str(r[0]) for r in db.session.query(tables.DjmdCue.ContentID)} | \
              {str(r[0]) for r in db.session.query(tables.ContentCue.ContentID)}
        usn = _next_usn(db, tables.ContentCue)
        for cid, points in cues.items():
            content = db.session.query(tables.DjmdContent).filter_by(ID=cid).first()
            if content is None:
                skipped["not in library"] += 1
                continue
            if cid in has:
                skipped["already has cues"] += 1
                continue
            if content.FileType == 1:
                skipped["mp3"] += 1
                continue
            now = datetime.now(timezone.utc)
            entries = []
            for kind, key in ((1, "A"), (2, "B"), (3, "C")):
                ms = int(round(points[key] * 1000))
                row = tables.DjmdCue(
                    ID=_new_id(db, tables.DjmdCue, 1, 2**32 - 1), ContentID=cid,
                    InMsec=ms, InFrame=ms * 150 // 1000, InMpegFrame=0, InMpegAbs=0,
                    OutMsec=-1, OutFrame=0, OutMpegFrame=0, OutMpegAbs=0,
                    Kind=kind, Color=-1, ContentUUID=content.UUID, UUID=str(uuid.uuid4()),
                    rb_data_status=0, rb_local_data_status=0, rb_local_deleted=0, rb_local_synced=0,
                    usn=None, rb_local_usn=None, created_at=now, updated_at=now)
                db.session.add(row)
                entries.append({"ID": row.ID, "ContentID": cid, "ContentUUID": content.UUID,
                                "InMsec": row.InMsec, "InFrame": row.InFrame, "InMpegFrame": 0, "InMpegAbs": 0,
                                "OutMsec": -1, "OutFrame": 0, "OutMpegFrame": 0, "OutMpegAbs": 0,
                                "Kind": kind, "Color": -1, "UUID": row.UUID,
                                "created_at": stamp(now), "updated_at": stamp(now)})
            db.session.add(tables.ContentCue(
                ID=content.UUID, ContentID=cid, Cues=json.dumps(entries, ensure_ascii=False, separators=(",", ":")),
                rb_cue_count=len(entries), UUID=str(uuid.uuid4()),
                rb_data_status=0, rb_local_data_status=0, rb_local_deleted=0, rb_local_synced=0,
                usn=None, rb_local_usn=usn, created_at=now, updated_at=now))
            usn += 1
            db.session.flush()
            written += 1
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise
    finally:
        db.session.close()
        db.engine.dispose()
    flush_wal()
    return {"written": written, "skipped": skipped, "backup": str(backup)}
