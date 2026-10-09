"""
Release year of each song from Discogs (needs DISCOGS_TOKEN in .env), cached in discogs_years.json.

The year is the EARLIEST year among Discogs releases that carry the track and whose title or
artist contains the song's first artist, a proxy for when the song first came out. It is a guess
for songs with odd tags, remixes of obscure tracks and your own productions; those get None
rather than a wrong year. Discogs allows 60 requests a minute, so ~1 s per song.

    python discogs_years.py           # every song not yet looked up
"""

import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).parent
OUT = HERE / "discogs_years.json"
load_dotenv(HERE / ".env")
HEADERS = {"Authorization": f"Discogs token={os.getenv('DISCOGS_TOKEN', '')}", "User-Agent": "set-curator/1.0"}


def clean(text: str) -> str:
    text = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", text)                        # (Original Mix) [Extended]
    text = re.sub(r"\s-\s.*(mix|edit|remix|version|dub|cut|recut).*$", " ", text, flags=re.I)
    return " ".join(text.split())


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def year_for(artist: str, title: str) -> int | None:
    first = re.split(r",|&| feat\.| ft\.| x ", artist)[0].strip()
    if not first or not title:
        return None
    for _ in range(3):
        r = requests.get("https://api.discogs.com/database/search", headers=HEADERS, timeout=30,
                         params={"artist": first, "track": clean(title), "type": "release", "per_page": 25})
        if r.status_code == 429:
            time.sleep(10)
            continue
        r.raise_for_status()
        break
    else:
        return None
    years = [int(x["year"]) for x in r.json().get("results", [])
             if x.get("year") and norm(first) in norm(x.get("title", "")) and 1950 < int(x["year"]) <= 2026]
    return min(years) if years else None


def main():
    import cluster
    from curator import Library
    lib = Library()
    tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
    songs, _, _ = cluster.merge_copies(tracks, ids, vecs)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [i for i in songs if i in lib.row and i not in out]
    for n, i in enumerate(todo, 1):
        artist, title = lib.label(i)
        try:
            out[i] = {"year": year_for(artist, title), "label": f"{artist} - {title}"}
        except Exception as e:
            out[i] = {"year": None, "label": f"{artist} - {title}", "error": str(e)[:80]}
        if n % 50 == 0:
            print(f"{n}/{len(todo)}", flush=True)
            OUT.write_text(json.dumps(out, ensure_ascii=False))
        time.sleep(1.05)
    OUT.write_text(json.dumps(out, ensure_ascii=False))
    print(f"{sum(v['year'] is not None for v in out.values())} of {len(out)} songs have a year")


if __name__ == "__main__":
    main()
