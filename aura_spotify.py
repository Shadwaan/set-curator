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


def acid_songs() -> list[dict]:
    """The songs in SC / Acid / Acid lines (not the 'room for a 303' folder), in that folder's order."""
    tracks = organise.build()[2]
    lib = Library()
    acid = next(c for c in organise.build.vibes if c["name"] == "Acid")
    lines = next(c for c in acid["children"] if c["name"].startswith("Acid lines"))
    ids = []
    def walk(node):
        ids.extend(i for i in node.get("tracks", []) if i not in ids)
        for c in node.get("children", []):
            walk(c)
    walk(lines)
    return [dict(id=i, **dict(zip(("artist", "title"), lib.label(i) if i in lib.row else (tracks[i]["artist"], tracks[i]["title"])))) for i in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spotify", action="store_true", help="also create the private Spotify playlist")
    ap.add_argument("--name", default="Aura")
    ap.add_argument("--update", metavar="PLAYLIST_ID", help="replace the contents of this existing Spotify playlist")
    ap.add_argument("--acid", action="store_true", help="use the Acid lines songs instead of Aura (text list acid_songs.txt)")
    args = ap.parse_args()
    songs = acid_songs() if args.acid else aura_songs()
    path = HERE / ("acid_songs.txt" if args.acid else "aura_songs.txt")
    path.write_text("\n".join(f"{s['artist']} - {s['title']}".strip(" -") for s in songs) + "\n", encoding="utf-8")
    print(f"{len(songs)} songs written to {path.name}")
    if args.spotify:
        import spotify_api
        result = (spotify_api.update_playlist(args.update, songs) if args.update
                  else spotify_api.create_playlist(args.name, songs, public=False))
        print(f"Playlist: {result['url']}  ({result['found']} found)")
        for m in result["missing"]:
            print("  not on Spotify:", m["artist"], "-", m["title"])


if __name__ == "__main__":
    main()
