"""
Minimal client for the cosine.club API (https://cosine.club/api/v1).
Key from COSINE_API_KEY (get one at https://cosine.club/account/api).
Limit: 120 requests/minute per key; 429s are retried after Retry-After.
"""

import os
import time

import requests

BASE = "https://cosine.club/api/v1"


class Cosine:
    def __init__(self, key: str | None = None):
        self.key = key or os.getenv("COSINE_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.key}", "User-Agent": "set-curator/1.0"})

    @property
    def enabled(self) -> bool:
        return bool(self.key)

    def _get(self, path: str, **params):
        params = {k: v for k, v in params.items() if v is not None}
        for _ in range(4):
            r = self.session.get(BASE + path, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(float(r.headers.get("Retry-After", 2)))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()["data"]
        r.raise_for_status()

    def search(self, q: str, limit: int = 10) -> list[dict]:
        return self._get("/search", q=q[:200], limit=limit) or []

    def lookup(self, url: str) -> list[dict]:
        return self._get("/tracks/lookup", url=url) or []

    def similar(self, track_id: str, limit: int = 100, page: int = 1, **filters) -> list[dict]:
        data = self._get(f"/tracks/{track_id}/similar", limit=limit, page=page, **filters)
        return (data or {}).get("similar_tracks", [])
