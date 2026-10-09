"""
Seed-based curation over your library (local embeddings) and/or the web
(cosine.club candidates, re-ranked here across all seeds).

Every result is a dict with one shape, whichever side it came from:
    key, source ("disk" | "web"), artist, title, score, seed_hits,
    disk_id, bpm, musical_key, cosine_id, video_id, link, in_library
"""

import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

import palette

HERE = Path(__file__).parent
MIX_ONLY = re.compile(r"^(original|extended|radio|club|dub|instrumental|vocal|edit|12\"|7\")?\s*(mix|edit|version|remix)$", re.I)
STAND_INS = 3        # released library tracks used for a seed cosine.club doesn't know
WEB_PER_SEED = 100   # candidates fetched per web seed (API max per page)


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[\(\[]?\b(original|extended|radio)\s+(mix|edit|version)\b[\)\]]?", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def track_key(artist: str, title: str) -> str:
    return f"{norm(artist)} - {norm(title)}"


class Library:
    """Your Rekordbox tracks that exist on disk and have an embedding."""

    def __init__(self):
        tracks = {t["id"]: t for t in json.loads((HERE / "tracks.json").read_text(encoding="utf-8"))}
        data = np.load(HERE / "embeddings.npz")
        ids = [str(i) for i in data["ids"]]
        skip = set(json.loads((HERE / "excluded.json").read_text(encoding="utf-8"))) \
            if (HERE / "excluded.json").exists() else set()
        keep = [n for n, i in enumerate(ids) if i in tracks and i not in skip and os.path.exists(tracks[i]["path"])
                and "/rekordbox/Sampler/" not in tracks[i]["path"]]
        self.tracks = tracks
        self.ids = [ids[n] for n in keep]
        vecs = data["vectors"][keep]
        self.vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        self.row = {i: n for n, i in enumerate(self.ids)}
        self.sections = palette.load()
        self.by_title: dict[str, list[str]] = {}
        for i in self.ids:
            self.by_title.setdefault(norm(self.label(i)[1]), []).append(i)

    def label(self, i: str) -> tuple[str, str]:
        t = self.tracks[i]
        if t["artist"]:
            return t["artist"], t["title"]
        # untagged tracks often carry "Artist - Title" in the title or the file name
        if " - " in t["title"]:
            left, right = (x.strip() for x in t["title"].split(" - ", 1))
            if MIX_ONLY.match(right):
                # "Song - Original Mix": the right part is a version, so the left part is the title;
                # the artist, if any, is in a file name like "202-modjo-chillin.aiff"
                m = re.match(r"^\d*[-_. ]*([^-_]+)[-_]", Path(t["path"]).stem)
                return (m.group(1).strip().title() if m else ""), t["title"]
            return left, right
        stem = Path(t["path"]).stem
        stem = re.sub(r"^\d+[\s.\-]+", "", stem)
        if " - " in stem and norm(t["title"]) in norm(stem):
            artist = stem.split(" - ")[0]
            if norm(artist) != norm(t["title"]):
                return artist, t["title"]
        return "", t["title"] or stem

    def as_result(self, i: str, score: float = 0.0) -> dict:
        artist, title = self.label(i)
        t = self.tracks[i]
        return {"key": track_key(artist, title), "source": "disk", "artist": artist, "title": title,
                "score": round(float(score), 4), "seed_hits": 0, "disk_id": i, "bpm": t["bpm"],
                "musical_key": t["key"], "cosine_id": None, "video_id": None, "link": None, "in_library": True}

    def search(self, q: str, limit: int = 10) -> list[dict]:
        words = norm(q).split()
        hits = [i for i in self.ids if all(w in norm(" ".join(self.label(i)) + " " + Path(self.tracks[i]["path"]).stem)
                                           for w in words)]
        return [self.as_result(i) for i in hits[:limit]]

    def find(self, artist: str, title: str) -> str | None:
        """The library track matching a web track, if you already have it."""
        for i in self.by_title.get(norm(title), []):
            mine = norm(self.label(i)[0])
            theirs = norm(artist)
            if not theirs or not mine or theirs in mine or mine in theirs \
                    or set(theirs.split()) & set(norm(self.tracks[i]["path"]).split()):
                return i
        return None

    def blend(self, seeds: list[tuple[str, float]]) -> np.ndarray:
        target = sum(w * self.vecs[self.row[i]] for i, w in seeds)
        return target / np.linalg.norm(target)

    def similar(self, seeds: list[tuple[str, float]], n: int) -> list[dict]:
        if self.sections and all(i in self.sections for i, _ in seeds):
            # palette match per seed (shared sounds, section by section), weighted
            scores = sum(w * palette.similarity(self.sections, i, self.ids) for i, w in seeds) / sum(w for _, w in seeds)
        else:
            scores = self.vecs @ self.blend(seeds)
        taken = [self.row[i] for i, _ in seeds]
        order = []
        for r in np.argsort(-scores):
            # skip other copies of the same audio (an mp3 and a wav of one track)
            if (self.vecs[taken] @ self.vecs[r]).max() >= 0.999:
                continue
            order.append(r)
            taken.append(r)
            if len(order) == n:
                break
        return [self.as_result(self.ids[r], scores[r]) for r in order]

    def neighbours(self, i: str, n: int) -> list[tuple[str, float]]:
        scores = self.vecs @ self.vecs[self.row[i]]
        return [(self.ids[r], float(scores[r])) for r in np.argsort(-scores) if self.ids[r] != i][:n]


def web_result(t: dict, score: float) -> dict:
    return {"key": track_key(t["artist"], t["track"]), "source": "web", "artist": t["artist"], "title": t["track"],
            "score": round(float(score), 4), "seed_hits": 0, "disk_id": None, "bpm": None, "musical_key": None,
            "cosine_id": str(t["id"]), "video_id": t.get("video_id"), "link": t.get("external_link"),
            "in_library": False}


def resolve_on_cosine(cosine, artist: str, title: str) -> dict | None:
    """cosine.club's copy of a track, if its catalogue has it."""
    for hit in cosine.search(f"{artist} {title}".strip(), limit=5):
        same_title = norm(hit["track"]) == norm(title) or norm(title) in norm(hit["name"])
        same_artist = not artist or set(norm(artist).split()) & set(norm(hit["artist"]).split())
        if same_title and same_artist:
            return hit
    return None


def curate(lib: Library, cosine, seeds: list[dict], mode: str, size: int = 30) -> dict:
    """seeds: [{"source": "disk", "disk_id", "weight"} | {"source": "web", "cosine_id", "artist", "title", "weight"}]
    Returns {"results": [...], "notes": [per-seed explanation]}."""
    notes, disk_seeds, web_seeds = [], [], []

    for s in seeds:
        w = float(s.get("weight", 1))
        if s["source"] == "disk":
            artist, title = lib.label(s["disk_id"])
            note = {"seed": f"{artist} - {title}".strip(" -"), "disk": True}
            disk_seeds.append((s["disk_id"], w))
            if mode != "disk" and cosine.enabled:
                hit = resolve_on_cosine(cosine, artist, title)
                if hit:
                    web_seeds.append((str(hit["id"]), w))
                    note["web"] = "matched on cosine.club"
                else:
                    stand = []
                    for j, sim in lib.neighbours(s["disk_id"], 12):
                        h = resolve_on_cosine(cosine, *lib.label(j))
                        if h:
                            stand.append((str(h["id"]), sim, " - ".join(lib.label(j)).strip(" -")))
                        if len(stand) == STAND_INS:
                            break
                    total = sum(sim for _, sim, _ in stand) or 1
                    web_seeds += [(cid, w * sim / total) for cid, sim, _ in stand]
                    note["web"] = ("not on cosine.club; stand-ins: " + ", ".join(n for _, _, n in stand)) if stand \
                        else "not on cosine.club and no stand-ins found"
        else:
            note = {"seed": f"{s['artist']} - {s['title']}", "web": "cosine.club track"}
            web_seeds.append((str(s["cosine_id"]), w))
            mine = lib.find(s["artist"], s["title"])
            if mine:
                disk_seeds.append((mine, w))
                note["disk"] = True
        notes.append(note)

    seed_keys = {track_key(*lib.label(i)) for i, _ in disk_seeds} | \
                {track_key(s["artist"], s["title"]) for s in seeds if s["source"] == "web"}
    seed_cosine = {cid for cid, _ in web_seeds}

    disk = lib.similar(disk_seeds, size * 2) if disk_seeds and mode != "web" else []

    web = []
    if web_seeds and mode != "disk" and cosine.enabled:
        per_seed, found = [], {}
        for cid, w in web_seeds:
            hits = {str(t["id"]): t for t in cosine.similar(cid, limit=WEB_PER_SEED)}
            for tid, t in hits.items():
                found.setdefault(tid, t)
            per_seed.append((w, hits, min((t.get("score") or 0 for t in hits.values()), default=0)))
        total_w = sum(w for w, _, _ in per_seed) or 1
        for tid, t in found.items():
            if tid in seed_cosine:
                continue
            # a candidate missing from a seed's list scores that seed's floor:
            # tracks close to several seeds rise, one-seed matches sink
            score = sum(w * ((hits[tid].get("score") or 0) if tid in hits else floor)
                        for w, hits, floor in per_seed) / total_w
            r = web_result(t, score)
            r["seed_hits"] = sum(tid in hits for _, hits, _ in per_seed)
            if r["key"] in seed_keys:
                continue
            mine = lib.find(r["artist"], r["title"])
            if mine:
                r.update(in_library=True, disk_id=mine, bpm=lib.tracks[mine]["bpm"],
                         musical_key=lib.tracks[mine]["key"])
            web.append(r)
        web.sort(key=lambda r: -r["score"])

    if mode == "both":
        # scores from the two sides aren't on one scale, so merge by rank
        merged, seen = [], set()
        for pair in zip(disk + [None] * len(web), web + [None] * len(disk)):
            for r in pair:
                if r and r["key"] not in seen and (r["disk_id"] is None or r["disk_id"] not in seen):
                    seen.update({r["key"], r["disk_id"]} - {None})
                    merged.append(r)
        results = merged
    else:
        results = disk if mode == "disk" else web
    return {"results": results[:size], "notes": notes}
