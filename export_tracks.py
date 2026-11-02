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


def playlist_rules() -> tuple[set[str], dict[str, str]]:
    """Optional playlist_rules.json beside this file: {"ignore": [playlist names that are only histories],
    "same_as": {"a copy's name": "the playlist it counts as"}}. Ignored playlists never become a crate; a same_as playlist's songs
    are filed under the other name, so the curator sees one crate."""
    path = Path(__file__).with_name("playlist_rules.json")
    if not path.exists():
        return set(), {}
    rules = json.loads(path.read_text(encoding="utf-8"))
    return set(rules.get("ignore", [])), dict(rules.get("same_as", {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("tracks.json")))
    ap.add_argument("--ignore-folder", action="append", default=["SC"], metavar="NAME",
                    help="playlists inside a folder with this name are generated, not yours (default: SC)")
    args = ap.parse_args()

    src = rekordbox_dir() / "master.db"
    with tempfile.TemporaryDirectory() as tmp:
        copy = Path(tmp) / "master.db"
        shutil.copy2(src, copy)
        db = Rekordbox6Database(path=str(copy))

        playlists = {}
        for sp in db.session.query(tables.DjmdSongPlaylist).all():
            playlists.setdefault(str(sp.ContentID), []).append(str(sp.PlaylistID))
        all_playlists = {str(p.ID): p for p in db.session.query(tables.DjmdPlaylist).all()}

        def generated(p) -> bool:
            """Playlists this tool built (inside a folder named in --ignore-folder), Rekordbox's own
            "CUE Analysis Playlist", and folders are not your playlists."""
            if p.Attribute != 0 or p.Name == "CUE Analysis Playlist":
                return True
            while p is not None:
                if p.Name in args.ignore_folder and p.Attribute == 1:
                    return True
                p = all_playlists.get(str(p.ParentID))
            return False

        ignore, same_as = playlist_rules()
        rules_file = Path(__file__).with_name("playlist_rules.json")
        args.ignore_folder = args.ignore_folder + (json.loads(rules_file.read_text()).get("ignore_folders", []) if rules_file.exists() else [])
        playlist_names = {pid: same_as.get(p.Name, p.Name).strip() for pid, p in all_playlists.items() if not generated(p) and p.Name not in ignore}

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
                "playlists": list(dict.fromkeys(playlist_names[p] for p in playlists.get(str(c.ID), []) if p in playlist_names)),
            })
        db.session.close()
        db.engine.dispose()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(tracks, f, ensure_ascii=False, indent=1)
    print(f"Exported {len(tracks)} tracks to {args.out}")


if __name__ == "__main__":
    main()
