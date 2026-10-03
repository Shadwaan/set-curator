"""
Export the Rekordbox track list to tracks.json for embed.py.

Read-only: master.db is copied to a temp folder first and only the copy is
opened, so Rekordbox's own files are never touched. Runs wherever pyrekordbox
is installed (on this PC: File Fetcher's venv).

    python export_tracks.py [--out tracks.json]
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from pyrekordbox import Rekordbox6Database
from pyrekordbox.db6 import tables


def rekordbox_dir() -> Path:
    if sys.platform == "win32":
        return Path(os.environ["APPDATA"]) / "Pioneer" / "rekordbox"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Pioneer" / "rekordbox"
    raise SystemExit("Rekordbox only runs on Windows and macOS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("tracks.json")))
    args = ap.parse_args()

    src = rekordbox_dir() / "master.db"
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "master.db"
        shutil.copy2(src, copy)
        db = Rekordbox6Database(path=str(copy))

        playlists = {}
        for sp in db.session.query(tables.DjmdSongPlaylist).all():
            playlists.setdefault(str(sp.ContentID), []).append(str(sp.PlaylistID))
        playlist_names = {str(p.ID): p.Name for p in db.session.query(tables.DjmdPlaylist).all()}

        tracks = []
        for c in db.session.query(tables.DjmdContent).all():
            tracks.append({
                "id": str(c.ID),
                "title": c.Title or "",
                "artist": c.ArtistName or "",
                "bpm": (c.BPM or 0) / 100,
                "key": c.KeyName or "",
                "length": c.Length or 0,
                "path": c.FolderPath or "",
                "playlists": [playlist_names.get(p, p) for p in playlists.get(str(c.ID), [])],
            })
        db.session.close()
        db.engine.dispose()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tracks, f, ensure_ascii=False, indent=1)
    print(f"Exported {len(tracks)} tracks to {args.out}")


if __name__ == "__main__":
    main()
