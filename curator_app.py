"""
Set Curator GUI: pick seed tracks, curate from your library and/or the web,
preview snippets, keep tracks, save the set (ledger + to-download list) and
optionally create it as a Spotify playlist.

    python curator_app.py        # then open http://127.0.0.1:8765

Read-only towards Rekordbox: it reads tracks.json/embeddings.npz, never master.db.
"""

import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, redirect, request, send_file, send_from_directory

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

import ledger  # noqa: E402
import spotify_api  # noqa: E402
from cosine_api import Cosine  # noqa: E402
from curator import Library, curate, web_result  # noqa: E402

app = Flask(__name__, static_folder=None)

# Reached through the Cloudflare tunnel (requests carry Cloudflare's headers) you need the password in .env
# (GUI_PASSWORD); on this computer, directly at 127.0.0.1, there is no login.
import hmac  # noqa: E402
import secrets  # noqa: E402
import time  # noqa: E402

SESSIONS: set[str] = set()
FAILS: list[float] = []
LOGIN_PAGE = (
    "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
    "<title>Set Curator</title><style>body{font:15px ui-monospace,Menlo,monospace;background:#ecebe6;color:#1d2b24;"
    "display:grid;place-items:center;min-height:100vh;margin:0}form{display:grid;gap:12px;width:min(320px,86vw)}"
    "input,button{font:inherit;padding:10px;border:1px solid #bbb;border-radius:3px}button{background:#2f5d4a;color:#f7f6f2;"
    "cursor:pointer}@media(prefers-color-scheme:dark){body{background:#141816;color:#e3e6e1}input{background:#1c211e;"
    "color:#e3e6e1;border-color:#2d3430}}</style><form method=post action=/login><b>Set Curator</b>"
    "<input type=password name=password placeholder=Password autofocus autocomplete=current-password>"
    "<button>Sign in</button>{msg}</form>")


def _public() -> bool:
    return bool(request.headers.get("Cf-Connecting-Ip") or request.headers.get("Cf-Ray"))


@app.before_request
def gate():
    if not _public():
        return None
    password = os.getenv("GUI_PASSWORD", "")
    if not password:
        return Response("The password is not set on the server, so the public link is closed.", 503)
    if request.path == "/login" and request.method == "POST":
        now = time.time()
        FAILS[:] = [x for x in FAILS if now - x < 300]
        if len(FAILS) >= 8:
            return Response("Too many tries, wait a few minutes.", 429)
        if hmac.compare_digest(request.form.get("password", ""), password):
            token = secrets.token_urlsafe(32)
            SESSIONS.add(token)
            resp = redirect("/sax")
            resp.set_cookie("sc_session", token, max_age=60 * 60 * 24 * 30, httponly=True, secure=True, samesite="Lax")
            return resp
        FAILS.append(now)
        return Response(LOGIN_PAGE.replace("{msg}", "<span>Wrong password.</span>"), 401, mimetype="text/html")
    if request.cookies.get("sc_session") in SESSIONS:
        return None
    if request.path.startswith("/api/"):
        return Response("Sign in first.", 401)
    return Response(LOGIN_PAGE.replace("{msg}", ""), 401, mimetype="text/html")
lib = Library()
cosine = Cosine()


def with_history(results: list[dict]) -> list[dict]:
    before = ledger.picked_before(ledger.load())
    for r in results:
        r["picked_in"] = before.get(r["key"], [])
    return results


@app.get("/")
def index():
    return send_from_directory(HERE / "static", "curator.html")


@app.get("/api/status")
def status():
    return jsonify(library=len(lib.ids), cosine=cosine.enabled, spotify=spotify_api.configured())


@app.get("/api/search")
def search():
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify(disk=[], web=[])
    if q.startswith(("http://", "https://")):
        web = [web_result(t, t.get("score") or 0) for t in cosine.lookup(q)] if cosine.enabled else []
        return jsonify(disk=[], web=with_history(web))
    web = [web_result(t, 0) for t in cosine.search(q, limit=8)] if cosine.enabled else []
    return jsonify(disk=with_history(lib.search(q, 8)), web=with_history(web))


@app.post("/api/curate")
def run_curate():
    body = request.get_json()
    out = curate(lib, cosine, body["seeds"], body.get("mode", "both"), int(body.get("size", 30)))
    out["results"] = with_history(out["results"])
    return jsonify(out)


FULL_CACHE = HERE / ".audio_cache"


@app.get("/api/full/<disk_id>")
def full_track(disk_id: str):
    """The whole song as MP3 (cached after the first request) so the player can jump anywhere in it:
    send_file answers range requests, which a streamed snippet can't. Only library IDs are accepted."""
    t = lib.tracks.get(disk_id)
    if not t or disk_id not in lib.row:
        abort(404)
    FULL_CACHE.mkdir(exist_ok=True)
    out = FULL_CACHE / f"{disk_id}.mp3"
    if not out.exists():
        tmp = out.with_suffix(".part.mp3")
        proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", t["path"], "-vn", "-ac", "2", "-b:a", "160k",
                               "-f", "mp3", str(tmp)], capture_output=True)
        if proc.returncode or not tmp.exists():
            abort(500)
        tmp.rename(out)
    return send_file(out, mimetype="audio/mpeg", conditional=True)


@app.get("/api/snippet/<disk_id>")
def snippet(disk_id: str):
    """A 30 s MP3 clip of a library file (browsers can't play AIFF), cached and sent with a length so
    the player can scrub inside it. Only library IDs are accepted."""
    t = lib.tracks.get(disk_id)
    if not t or disk_id not in lib.row:
        abort(404)
    start = request.args.get("start", type=float)
    if start is None:
        start = max(0.0, (t["length"] or 0) * 0.4)
    start = max(0.0, min(start, max((t["length"] or 30) - 30, 0)))
    clips = FULL_CACHE / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    out = clips / f"{disk_id}_{int(start)}.mp3"
    if not out.exists():
        tmp = out.with_suffix(".part.mp3")
        proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{int(start)}", "-t", "30", "-i", t["path"],
                               "-vn", "-ac", "2", "-b:a", "128k", "-f", "mp3", str(tmp)], capture_output=True)
        if proc.returncode or not tmp.exists() or not tmp.stat().st_size:
            abort(500)
        tmp.rename(out)
    return send_file(out, mimetype="audio/mpeg", conditional=True)


SAX_FILED = HERE / "sax_filed.json"
SAX_NOTES = HERE / "sax_notes.json"
SAX_AURA = HERE / "sax_aura.json"
AURA_OPTIONS = ["Has the aura", "No aura"]
SAX_IMPORT = HERE / "sax_import.json"
ENERGY_OPTIONS = ["Low energy", "Mid energy", "High energy"]
SAX_ENERGY = HERE / "sax_energy.json"
SAX_GROUPS = ["Has sax", "Melody a sax could play", "Room for a sax"]


def _read(path):
    import json
    return json.loads(path.read_text()) if path.exists() else {}


@app.get("/sax")
def sax_page():
    resp = send_from_directory(HERE / "static", "sax.html")
    resp.headers["Cache-Control"] = "no-store"     # always the newest version of the page
    return resp


def _peak(i: str) -> float:
    """Cue C (the peak section) of a song, or 30 s in."""
    import json
    cues = HERE / "cues.json"
    return float(json.loads(cues.read_text()).get(i, {}).get("C", 30.0)) if cues.exists() else 30.0


@app.get("/api/sax")
def sax_lists():
    """The three candidate lists with where to listen, and how each song has been filed so far:
    `filed` is null until you decide, then the list of groups it belongs in (empty = none)."""
    import json
    groups = json.loads((HERE / "sax_groups.json").read_text())
    listen = json.loads((HERE / "sax_listen.json").read_text())
    filed, notes = _read(SAX_FILED), _read(SAX_NOTES)
    out = []
    for name, ids in groups.items():
        songs = []
        for i in ids:
            if i not in lib.row:
                continue
            artist, title = lib.label(i)
            songs.append({"id": i, "artist": artist, "title": title, "start": round(listen.get(i, 30.0), 1),
                          "filed": filed.get(i), "note": notes.get(i, ""),
                          "where": "sax" if name == "Has sax" else "C"})
        out.append({"name": name, "songs": songs})
    # the aura check: every melody-list pick except dub and reggae, to say whether it has the aura
    aura = _read(SAX_AURA)
    groups_ = json.loads((HERE / "sax_groups.json").read_text())
    reggae = _read(HERE / "sax_reggae.json")
    ids = [i for i in groups_["Melody a sax could play"] if i in lib.row and reggae.get(i, 0) < 0.3]
    ids.sort(key=lambda i: __import__("hashlib").md5(i.encode()).hexdigest())      # fixed shuffle, so no order hints
    songs = []
    for i in ids:
        artist, title = lib.label(i)
        songs.append({"id": i, "artist": artist, "title": title, "start": round(listen.get(i, 30.0), 1),
                      "filed": aura.get(i), "note": notes.get(i, ""), "where": "C"})
    if songs:
        out.append({"name": "Aura check", "options": AURA_OPTIONS, "endpoint": "/api/aura/answer", "songs": songs,
                    "question": "The melody picks (dub and reggae left out). Does this song have the aura? You decide what that means."})
    # the energy check: songs to rate Low / Mid / High by club feel (energy_check.py fits the score to your ratings)
    sample = _read(HERE / "energy_sample.json") or []
    rated = _read(SAX_ENERGY)
    es = []
    for i in sample:
        if i not in lib.row:
            continue
        artist, title = lib.label(i)
        es.append({"id": i, "artist": artist, "title": title, "start": round(listen.get(i, _peak(i)), 1),
                   "filed": rated.get(i), "note": notes.get(i, ""), "where": "C"})
    if es:
        out.append({"name": "Energy check", "options": ENERGY_OPTIONS, "endpoint": "/api/energy/answer", "songs": es,
                    "question": "How much club energy does it have? Judge it as you would on a dance floor: drive, weight and tension, "
                                "not brightness or busyness."})
    # imported songs to keep or remove (import_check.json: {id: why}); your answers are in sax_import.json
    check = _read(HERE / "import_check.json")
    answered = _read(SAX_IMPORT)
    cs = []
    for i, why in check.items():
        if i not in lib.row:
            continue
        artist, title = lib.label(i)
        cs.append({"id": i, "artist": artist, "title": title, "start": round(listen.get(i, _peak(i)), 1),
                   "filed": answered.get(i), "note": notes.get(i, ""), "where": "C", "why": why})
    if cs:
        out.append({"name": "Imports to check", "options": ["Keep", "Remove"], "endpoint": "/api/import/answer", "songs": cs,
                    "question": "Songs the File Fetcher import added that may be unwanted or doubled. Listen, then Keep or Remove. "
                                "Remove only marks it for me; you remove it in Rekordbox."})
    # songs found by the later searches, to file under the sax groups and/or the aura
    more = _read(HERE / "sax_more.json")
    songs = []
    for i, why in more.items():
        if i not in lib.row:
            continue
        artist, title = lib.label(i)
        f, a = filed.get(i), aura.get(i)
        songs.append({"id": i, "artist": artist, "title": title, "start": round(listen.get(i, _peak(i)), 1),
                      "filed": None if f is None and a is None else (f or []) + (a or []), "note": notes.get(i, ""),
                      "where": "C", "why": " + ".join(why)})
    songs.sort(key=lambda x: (-len(x["why"]), x["title"]))
    if songs:
        out.append({"name": "More to check", "more": True, "options": SAX_GROUPS + AURA_OPTIONS, "songs": songs,
                    "question": "Songs I found by sound that you haven't judged. File under any sax group, the aura, both or neither."})
    return jsonify(groups=out, names=SAX_GROUPS)


@app.post("/api/sax/file")
def sax_file():
    """Set which of the three groups a song belongs in (a list, possibly empty)."""
    import json
    body = request.get_json()
    filed = _read(SAX_FILED)
    filed[body["id"]] = [g for g in SAX_GROUPS if g in body["groups"]]
    SAX_FILED.write_text(json.dumps(filed, indent=1))
    return jsonify(ok=True)


@app.get("/api/sax/lookup")
def sax_lookup():
    """Any library song by words from its name, with how you filed it, for listening and filing."""
    q = request.args.get("q", "").strip()
    if len(q) < 2:
        return jsonify([])
    filed, aura = _read(SAX_FILED), _read(SAX_AURA)
    out = []
    for r in lib.search(q, 12):
        i = r["disk_id"]
        t = lib.tracks[i]
        out.append({"id": i, "artist": r["artist"], "title": r["title"], "length": t["length"] or 0, "bpm": t["bpm"],
                    "file": Path(t["path"]).name, "crates": [c for c in t["playlists"] if c != "Deep Tech FLAC"],
                    "filed": filed.get(i), "aura": aura.get(i)})
    return jsonify(out)


@app.post("/api/aura/answer")
def aura_answer():
    import json
    body = request.get_json()
    aura = _read(SAX_AURA)
    aura[body["id"]] = [g for g in AURA_OPTIONS if g in body["groups"]]
    SAX_AURA.write_text(json.dumps(aura, indent=1))
    return jsonify(ok=True)


@app.post("/api/energy/answer")
def energy_answer():
    import json
    body = request.get_json()
    rated = _read(SAX_ENERGY)
    rated[body["id"]] = [g for g in ENERGY_OPTIONS if g in body["groups"]]
    SAX_ENERGY.write_text(json.dumps(rated, indent=1))
    return jsonify(ok=True)


@app.post("/api/import/answer")
def import_answer():
    import json
    body = request.get_json()
    ans = _read(SAX_IMPORT)
    ans[body["id"]] = [g for g in ("Keep", "Remove") if g in body["groups"]]
    SAX_IMPORT.write_text(json.dumps(ans, indent=1))
    return jsonify(ok=True)


@app.post("/api/sax/note")
def sax_note():
    """A free-text note on a song (one per song, whichever group it is shown in)."""
    import json
    body = request.get_json()
    notes = json.loads(SAX_NOTES.read_text()) if SAX_NOTES.exists() else {}
    text = (body.get("note") or "").strip()
    if text:
        notes[body["id"]] = text
    else:
        notes.pop(body["id"], None)
    SAX_NOTES.write_text(json.dumps(notes, indent=1, ensure_ascii=False))
    return jsonify(ok=True)


@app.post("/api/commit")
def commit():
    body = request.get_json()
    name, tracks = body["name"].strip(), body["tracks"]
    if not name or not tracks:
        return jsonify(error="Give the set a name and keep at least one track"), 400
    out = {"spotify": None}
    if body.get("spotify"):
        if not spotify_api.configured():
            return jsonify(error="SPOTIFY_CLIENT_ID is not set in .env"), 400
        try:
            out["spotify"] = spotify_api.create_playlist(name, tracks, public=bool(body.get("public")))
        except Exception as e:  # auth cancelled, port 8888 busy, API errors
            return jsonify(error=f"Spotify: {e}"), 502
    data = ledger.add_set(name, tracks, lib, out["spotify"]["url"] if out["spotify"] else None)
    kept = [data["tracks"][t["key"]] for t in tracks]
    out["to_download"] = [{"artist": t["artist"], "title": t["title"]} for t in kept if not t["in_library"]]
    out["to_download_file"] = str(ledger.TO_DOWNLOAD)
    return jsonify(out)


@app.get("/api/ledger")
def get_ledger():
    data = ledger.load()
    ledger.refresh(data, lib)
    todo = [t for t in data["tracks"].values() if not t["in_library"]]
    return jsonify(sets=data["sets"][::-1], to_download=todo, total=len(data["tracks"]))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False)
