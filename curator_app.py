"""
Set Curator GUI: pick seed tracks, curate from your library and/or the web,
preview snippets, keep tracks, save the set (ledger + to-download list) and
optionally create it as a Spotify playlist.

    python curator_app.py        # then open http://127.0.0.1:8765

Read-only towards Rekordbox: it reads tracks.json/embeddings.npz, never master.db.
"""

import subprocess
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, Response, abort, jsonify, request, send_from_directory

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

import ledger  # noqa: E402
import spotify_api  # noqa: E402
from cosine_api import Cosine  # noqa: E402
from curator import Library, curate, web_result  # noqa: E402

app = Flask(__name__, static_folder=None)
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


@app.get("/api/snippet/<disk_id>")
def snippet(disk_id: str):
    """30 s MP3 from a library file (browsers can't play AIFF). Only library IDs are accepted."""
    t = lib.tracks.get(disk_id)
    if not t or disk_id not in lib.row:
        abort(404)
    start = request.args.get("start", type=float)
    if start is None:
        start = max(0.0, (t["length"] or 0) * 0.4)
    proc = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.1f}", "-t", "30", "-i", t["path"],
                           "-vn", "-ac", "2", "-b:a", "192k", "-f", "mp3", "pipe:1"], capture_output=True)
    if proc.returncode or not proc.stdout:
        abort(500)
    return Response(proc.stdout, mimetype="audio/mpeg")


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
