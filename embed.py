"""
Compute a discogs-effnet embedding (1280 floats, averaged over the track) for
every track in tracks.json -- the same model and method cosine.club uses.
Runs on Linux (WSL on Windows) or macOS, wherever essentia-tensorflow installs.

Incremental: tracks already in embeddings.npz are skipped, so re-running only
processes new files.

    python embed.py [--limit 10] [--playlist "Dub"]
    python embed.py --path-map "D:/Music=/Volumes/Music"   # files moved to another machine
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs

HERE = Path(__file__).parent
MODEL = HERE / "models" / "discogs-effnet-bs64-1.pb"
OUT = HERE / "embeddings.npz"


def local_path(path: str, path_map: list[tuple[str, str]]) -> str:
    """Rekordbox stores Windows paths ("D:/Music/x.mp3"). --path-map rewrites a
    prefix (for a library copied to another machine); otherwise inside WSL the
    same file is at /mnt/d/Music/x.mp3, and on macOS paths are already native."""
    for old, new in path_map:
        if path.lower().startswith(old.lower()):
            return new + path[len(old):]
    m = re.match(r"^([A-Za-z]):/(.*)$", path)
    if m and sys.platform.startswith("linux"):
        return f"/mnt/{m.group(1).lower()}/{m.group(2)}"
    return path


def load_existing() -> dict[str, np.ndarray]:
    if not OUT.exists():
        return {}
    data = np.load(OUT)
    return {str(i): v for i, v in zip(data["ids"], data["vectors"])}


def save(vectors: dict[str, np.ndarray]):
    ids = list(vectors)
    tmp = OUT.with_name("embeddings.tmp.npz")
    np.savez(tmp, ids=np.array(ids), vectors=np.stack([vectors[i] for i in ids]))
    tmp.replace(OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only process this many new tracks")
    ap.add_argument("--playlist", help="only tracks in this Rekordbox playlist")
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite a path prefix, e.g. D:/Music=/Volumes/Music (repeatable)")
    args = ap.parse_args()
    path_map = [tuple(m.split("=", 1)) for m in args.path_map]

    tracks = json.loads((HERE / "tracks.json").read_text(encoding="utf-8"))
    if args.playlist:
        tracks = [t for t in tracks if args.playlist in t["playlists"]]

    vectors = load_existing()
    todo = [t for t in tracks if t["id"] not in vectors]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(vectors)} already embedded, {len(todo)} to do")

    model = TensorflowPredictEffnetDiscogs(graphFilename=str(MODEL), output="PartitionedCall:1")
    failed = []
    for n, t in enumerate(todo, 1):
        path = local_path(t["path"], path_map)
        start = time.time()
        try:
            audio = MonoLoader(filename=path, sampleRate=16000, resampleQuality=4)()
            frames = model(audio)  # (n_patches, 1280)
            vectors[t["id"]] = frames.mean(axis=0).astype(np.float32)
            print(f"[{n}/{len(todo)}] {time.time() - start:5.1f}s  {t['artist']} - {t['title']}")
        except Exception as e:
            failed.append((t["path"], str(e)))
            print(f"[{n}/{len(todo)}] FAILED  {t['path']}: {e}")
        if n % 25 == 0:
            save(vectors)

    save(vectors)
    print(f"Done. {len(vectors)} embeddings in {OUT.name}, {len(failed)} failed")


if __name__ == "__main__":
    main()
