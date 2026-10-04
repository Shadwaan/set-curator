"""
Every track you have kept in a curated set, across all sets (ledger.json),
plus to_download.csv: the kept tracks that aren't in your Rekordbox library yet.
A track already in the library is marked in_library with its Rekordbox ID, so
later sets reference the same track instead of listing it to download again.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).parent
LEDGER = HERE / "ledger.json"
TO_DOWNLOAD = HERE / "to_download.csv"
FIELDS = ("artist", "title", "cosine_id", "video_id", "link", "spotify_uri")


def load() -> dict:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"tracks": {}, "sets": []}


def save(data: dict):
    tmp = LEDGER.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(LEDGER)
    with open(TO_DOWNLOAD, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["artist", "title", "youtube", "link", "spotify", "sets"])
        for t in data["tracks"].values():
            if not t["in_library"]:
                w.writerow([t["artist"], t["title"],
                            f"https://www.youtube.com/watch?v={t['video_id']}" if t.get("video_id") else "",
                            t.get("link") or "", t.get("spotify_uri") or "", " | ".join(t["sets"])])


def refresh(data: dict, lib):
    """Re-check which ledger tracks are now in the Rekordbox library."""
    for t in data["tracks"].values():
        mine = t.get("disk_id") if t.get("disk_id") in lib.tracks else lib.find(t["artist"], t["title"])
        t["in_library"] = mine is not None
        t["disk_id"] = mine
        t["path"] = lib.tracks[mine]["path"] if mine else None


def add_set(name: str, results: list[dict], lib, spotify_url: str | None = None) -> dict:
    data = load()
    for r in results:
        t = data["tracks"].setdefault(r["key"], {"sets": [], "added": datetime.now().isoformat(timespec="seconds")})
        for f in FIELDS:
            if r.get(f):
                t[f] = r[f]
        t.setdefault("artist", r["artist"])
        t.setdefault("title", r["title"])
        if r.get("disk_id"):
            t["disk_id"] = r["disk_id"]
        if name not in t["sets"]:
            t["sets"].append(name)
    data["sets"].append({"name": name, "created": datetime.now().isoformat(timespec="seconds"),
                         "keys": [r["key"] for r in results], "spotify_url": spotify_url})
    refresh(data, lib)
    save(data)
    return data


def picked_before(data: dict) -> dict[str, list[str]]:
    """key -> names of sets it was kept in, for badges on new results."""
    return {k: t["sets"] for k, t in data["tracks"].items()}
