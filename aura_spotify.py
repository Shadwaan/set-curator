"""
The Aura songs as a text list, and optionally as a private Spotify playlist.

    python aura_spotify.py            # write aura_songs.txt ("Artist - Title", in energy then tempo order)
    python aura_spotify.py --spotify  # also create the playlist "Aura" (first run: approve in the browser ON THIS MAC)

Songs come from your aura answers (sax_aura.json), dub reggae left out, in the order of SC / Aura / All (by tempo).
Spotify only gets songs whose title and artist match a search result; the rest are listed as missing. The playlist
is private. Sign-in uses SPOTIFY_CLIENT_ID in .env (see spotify_api.py; the app owner needs Premium).
"""

import argparse

from dotenv import load_dotenv

import organise
from cluster import HERE
from curator import Library

load_dotenv(HERE / ".env")


def aura_songs() -> list[dict]:
    _, aura_tree, tracks, _, _ = organise.build()
    everything = next(c for c in aura_tree["children"] if c["name"].startswith("All"))
    lib = Library()
    out = []
    for i in everything["tracks"]:
        artist, title = lib.label(i) if i in lib.row else (tracks[i]["artist"], tracks[i]["title"])
        out.append({"id": i, "artist": artist, "title": title})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spotify", action="store_true", help="also create the private Spotify playlist")
    ap.add_argument("--name", default="Aura")
    args = ap.parse_args()
    songs = aura_songs()
    path = HERE / "aura_songs.txt"
    path.write_text("\n".join(f"{s['artist']} - {s['title']}".strip(" -") for s in songs) + "\n", encoding="utf-8")
    print(f"{len(songs)} songs written to {path.name}")
    if args.spotify:
        import spotify_api
        result = spotify_api.create_playlist(args.name, songs, public=False)
        print(f"Playlist: {result['url']}  ({result['found']} found)")
        for m in result["missing"]:
            print("  not on Spotify:", m["artist"], "-", m["title"])


if __name__ == "__main__":
    main()
