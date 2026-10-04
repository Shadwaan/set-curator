"""
Create a Spotify playlist from curated tracks.

Uses a Spotify dev app via PKCE (no client secret): SPOTIFY_CLIENT_ID in .env,
redirect URI http://127.0.0.1:8888/callback (must be listed in the app's
settings). The first run opens the browser to approve; the token is cached in
.spotify_cache. Dev-mode apps need the app owner to have Spotify Premium.
"""

import os
from pathlib import Path

import spotipy
from spotipy.oauth2 import SpotifyPKCE

from curator import norm

HERE = Path(__file__).parent
SCOPE = "playlist-modify-private playlist-modify-public"


def configured() -> bool:
    return bool(os.getenv("SPOTIFY_CLIENT_ID"))


def client() -> spotipy.Spotify:
    auth = SpotifyPKCE(client_id=os.environ["SPOTIFY_CLIENT_ID"],
                       redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8888/callback"),
                       scope=SCOPE, cache_path=str(HERE / ".spotify_cache"), open_browser=True)
    return spotipy.Spotify(auth_manager=auth)


def find_uri(sp: spotipy.Spotify, artist: str, title: str) -> str | None:
    """Spotify URI for a track, only when title and artist actually agree."""
    for q in (f'track:"{title}" artist:"{artist}"', f"{artist} {title}"):
        for item in sp.search(q=q, type="track", limit=5)["tracks"]["items"]:
            names = norm(" ".join(a["name"] for a in item["artists"]))
            title_ok = norm(title) in norm(item["name"]) or norm(item["name"]) in norm(title)
            artist_ok = not artist or set(norm(artist).split()) & set(names.split())
            if title_ok and artist_ok:
                return item["uri"]
    return None


def create_playlist(name: str, tracks: list[dict], public: bool = False) -> dict:
    """tracks: curated results. Returns {url, found, missing}; sets t["spotify_uri"]."""
    sp = client()
    uris, missing = [], []
    for t in tracks:
        uri = t.get("spotify_uri") or find_uri(sp, t["artist"], t["title"])
        if uri:
            t["spotify_uri"] = uri
            uris.append(uri)
        else:
            missing.append(t)
    playlist = sp.current_user_playlist_create(name, public=public, description="Curated with Set Curator")
    for i in range(0, len(uris), 100):
        sp.playlist_add_items(playlist["id"], uris[i:i + 100])
    return {"url": playlist["external_urls"]["spotify"], "found": len(uris), "missing": missing}
