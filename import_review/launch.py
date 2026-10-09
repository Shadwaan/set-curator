"""Runs File Fetcher's reviewer as a separate copy for the songs the import added.

Only your keep / remove marks and notes are recorded (import_review_marks.json here). The reviewer's actions that
move files or write to Rekordbox are switched off in this copy.

    <File Fetcher>/app/.venv/bin/python launch.py      # http://127.0.0.1:8767/review  (SFF_APP_DIR if File Fetcher is not next to this repo)
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_pages  # noqa: E402

FF = build_pages.SFF_APP
sys.path.insert(0, str(FF))

rows, tracks = build_pages.build_csv()
build_pages.build_page()

import review as R  # noqa: E402  (File Fetcher's reviewer)

R.AUDIT_FILE = HERE / "import_audit.csv"
R.MARKS_FILE = HERE / "import_review_marks.json"
R.HISTORY_FILE = HERE / "import_review_history.jsonl"
R.CACHE_DIR = HERE / ".cache"
R.PAGE = HERE / "review.html"
R._reject_file = lambda *a, **k: None              # never move a file
R._move_to_playlist = lambda *a, **k: None

_known = R._tracks_by_file


def _tracks_by_file():
    out = _known()
    for t in tracks.values():                       # artist for the YouTube search, when sff has no record of the file
        out.setdefault(t["path"].replace("\\", "/").lower(), {"artist": t["artist"], "title": t["title"], "spotify_id": ""})
    return out


R._tracks_by_file = _tracks_by_file

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import RedirectResponse, Response  # noqa: E402

app = FastAPI()
BLOCKED = ("/api/review/apply-labels", "/api/review/restore")


@app.middleware("http")
async def no_writes(request, call_next):
    if request.url.path in BLOCKED:
        return Response("Switched off in the import review: it only records your marks.", status_code=403)
    return await call_next(request)


import csv  # noqa: E402
import os  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
from datetime import datetime  # noqa: E402

from pydantic import BaseModel  # noqa: E402

SC = HERE.parent
sys.path.insert(0, str(SC))
EDITS = HERE / "import_edits.json"
def _music_root() -> Path:
    """Your music folder: SC_MUSIC_ROOT, or the part of a library path before "Music/Incoming" (where sff keeps its files)."""
    if os.environ.get("SC_MUSIC_ROOT"):
        return Path(os.environ["SC_MUSIC_ROOT"])
    for t in tracks.values():
        head, sep, _ = t["path"].partition("/Music/Incoming/")
        if sep:
            return Path(head + "/Music")
    return Path("/")


MUSIC = _music_root()
BY_PATH = {t["path"]: t["id"] for t in tracks.values()}


def _rb_info(path: str) -> dict | None:
    """How Rekordbox has the file at `path` (read-only)."""
    cid = BY_PATH.get(path)
    if not cid:
        return None
    import rekordbox_write as w
    from pyrekordbox.db6 import tables as T
    db = w.open_db()
    try:
        c = db.session.query(T.DjmdContent).filter_by(ID=cid).first()
        if c is None:
            return None
        lists = [p.Name for s in db.session.query(T.DjmdSongPlaylist).filter_by(ContentID=cid)
                 for p in db.session.query(T.DjmdPlaylist).filter_by(ID=s.PlaylistID)]
        mine = [n for n in lists if n.endswith(("AIFF", "FLAC"))]
        out = {"id": cid, "title": c.Title or "", "artist": getattr(c, "ArtistName", "") or tracks[cid]["artist"],
               "file": Path(path).name, "where": str(Path(path).relative_to(MUSIC)) if path.startswith(str(MUSIC)) else path, "playlists": mine, "sc_folders": len(lists) - len(mine), "comment": c.Commnt or "",
               "analysed": bool(c.AnalysisDataPath), "cues": db.session.query(T.DjmdCue).filter_by(ContentID=cid).count(),
               "added": str(c.created_at)[:16]}
    finally:
        db.session.close()
        db.engine.dispose()
    size = Path(path).stat().st_size if Path(path).exists() else 0
    dur = ""
    try:
        dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=20).stdout.strip()
        dur = f"{int(float(dur)) // 60}:{int(float(dur)) % 60:02d}"
    except Exception:
        pass
    out.update(format=Path(path).suffix.lstrip(".").upper(),
               length=dur or "?", size=f"{size / 1e6:.1f} MB")
    edits = json.loads(EDITS.read_text()) if EDITS.exists() else {}
    out["edit"] = edits.get(cid)
    return out


@app.get("/api/import/info/{item_id}")
def import_info(item_id: str):
    row = next((r for r in csv.DictReader(R.AUDIT_FILE.open(encoding="utf-8-sig")) if R._item_id(r["new_file"]) == item_id), None)
    if row is None:
        return Response("Unknown item", status_code=404)
    return {"new": _rb_info(row["new_file"]), "old": _rb_info(row["reference"]) if row["reference"] else None}


class Edit(BaseModel):
    id: str
    title: str = ""
    artist: str = ""
    clear: bool = False


@app.post("/api/import/edit")
def import_edit(body: Edit):
    """Saved as a pending edit only; nothing in Rekordbox changes until it is applied with Rekordbox closed."""
    if body.id not in tracks:
        return Response("Unknown song", status_code=404)
    edits = json.loads(EDITS.read_text()) if EDITS.exists() else {}
    if body.clear:
        edits.pop(body.id, None)
    else:
        edits[body.id] = {"title": body.title, "artist": body.artist, "at": datetime.now().isoformat(timespec="seconds")}
    EDITS.write_text(json.dumps(edits, indent=1, ensure_ascii=False))
    return {"ok": True}


import unicodedata  # noqa: E402
from fastapi import HTTPException  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

COMPARE_MARKS = HERE / "compare_marks.json"
_emb = None


def _vec(cid):
    global _emb
    if _emb is None:
        import numpy as np
        d = np.load(SC / "embeddings.npz")
        _emb = {str(i): v / np.linalg.norm(v) for i, v in zip(d["ids"], d["vectors"])}
    return _emb.get(cid)


def _norm(s):
    return unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()


def _probe(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,bit_rate", "-of", "csv=p=0", path],
                             capture_output=True, text=True, timeout=30).stdout.strip().split(",")
        dur, br = float(out[0]), (int(out[1]) // 1000 if len(out) > 1 and out[1].isdigit() else 0)
        return f"{int(dur) // 60}:{int(dur) % 60:02d}", (f"{br} kbps" if path.lower().endswith(".mp3") and br else "")
    except Exception:
        return "?", ""


@app.get("/api/compare/find")
def compare_find(q: str = ""):
    words = _norm(q).split()
    if not words:
        return []
    marks = json.loads(COMPARE_MARKS.read_text()) if COMPARE_MARKS.exists() else {}
    hits = [t for t in tracks.values() if t["path"].startswith(str(MUSIC) + "/")
            and all(w_ in _norm(t["artist"] + " " + t["title"] + " " + Path(t["path"]).stem) for w_ in words)]
    hits.sort(key=lambda t: (_norm(t["title"]), t["path"]))
    out, first = [], None
    for t in hits:
        info = _rb_info(t["path"]) or {}
        dur, br = _probe(t["path"])
        v = _vec(t["id"])
        if first is None:
            first = v
        sim = float(v @ first) if v is not None and first is not None else 0.0
        out.append({"id": t["id"], "title": info.get("title") or t["title"], "artist": info.get("artist") or t["artist"],
                    "where": info.get("where") or t["path"], "format": info.get("format", ""), "length": dur, "size": info.get("size", ""),
                    "bitrate": br, "bpm": round(t["bpm"] or 0), "key": t["key"] or "", "playlists": info.get("playlists", []),
                    "comment": info.get("comment", ""), "analysed": info.get("analysed", False), "cues": info.get("cues", 0),
                    "sc_folders": info.get("sc_folders", 0), "added": info.get("added", ""), "match_first": sim,
                    "mark": (marks.get(t["id"]) or {}).get("mark"), "note": (marks.get(t["id"]) or {}).get("note", "")})
    return out


@app.get("/api/compare/audio/{cid}")
def compare_audio(cid: str):
    t = tracks.get(cid)
    if not t or not t["path"].startswith(str(MUSIC) + "/") or not Path(t["path"]).is_file():
        raise HTTPException(status_code=404, detail="Unknown or missing file")
    p = R._playable(Path(t["path"]))
    return FileResponse(str(p), media_type="audio/mpeg" if p.suffix == ".mp3" else None)


class CMark(BaseModel):
    id: str
    mark: str | None = None
    note: str = ""


@app.post("/api/compare/mark")
def compare_mark(body: CMark):
    if body.id not in tracks:
        return Response("Unknown song", status_code=404)
    marks = json.loads(COMPARE_MARKS.read_text()) if COMPARE_MARKS.exists() else {}
    if body.mark or body.note.strip():
        marks[body.id] = {"mark": body.mark, "note": body.note.strip(), "title": tracks[body.id]["title"],
                          "file": Path(tracks[body.id]["path"]).name, "at": datetime.now().isoformat(timespec="seconds")}
    else:
        marks.pop(body.id, None)
    COMPARE_MARKS.write_text(json.dumps(marks, indent=1, ensure_ascii=False))
    return {"ok": True}


@app.get("/compare")
def compare_page():
    return FileResponse(str(HERE / "compare.html"))


app.include_router(R.router)


@app.get("/")
def home():
    return RedirectResponse("/review")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8767, log_level="warning")
