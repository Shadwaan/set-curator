"""
Discogs genres and styles of each song (needs DISCOGS_TOKEN in .env), cached in discogs_styles.json.

Same lookup as discogs_years.py, but keeps the styles ("Acid House", "Deep House", ...) of the releases whose title or
artist contains the song's first artist. Discogs allows 60 requests a minute, so ~1 s per song.

    python discogs_styles.py           # every song not yet looked up
"""

import json
import os
import re
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

HERE = Path(__file__).parent
OUT = HERE / "discogs_styles.json"
load_dotenv(HERE / ".env")
HEADERS = {"Authorization": f"Discogs token={os.getenv('DISCOGS_TOKEN', '')}", "User-Agent": "set-curator/1.0"}


def clean(text: str) -> str:
    text = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", text)
    text = re.sub(r"\s-\s.*(mix|edit|remix|version|dub|cut|recut).*$", " ", text, flags=re.I)
    return " ".join(text.split())


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def styles_for(artist: str, title: str) -> dict | None:
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
    styles, genres, n = {}, {}, 0
    for x in r.json().get("results", []):
        if norm(first) in norm(x.get("title", "")):
            n += 1
            for s in x.get("style") or []:
                styles[s] = styles.get(s, 0) + 1
            for g in x.get("genre") or []:
                genres[g] = genres.get(g, 0) + 1
    return {"releases": n, "styles": styles, "genres": genres}


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
            out[i] = {"label": f"{artist} - {title}", **(styles_for(artist, title) or {"releases": 0, "styles": {}, "genres": {}})}
        except Exception as e:
            out[i] = {"label": f"{artist} - {title}", "releases": 0, "styles": {}, "genres": {}, "error": str(e)[:80]}
        if n % 50 == 0:
            print(f"{n}/{len(todo)}", flush=True)
            OUT.write_text(json.dumps(out, ensure_ascii=False))
        time.sleep(1.05)
    OUT.write_text(json.dumps(out, ensure_ascii=False))
    print(f"{sum(1 for v in out.values() if v['styles'])} of {len(out)} songs have styles")


if __name__ == "__main__":
    main()
