"""
Auto cue points from Rekordbox's own beatgrid plus the track's energy, bar by bar.

Same meaning on every track (hot cues A, B, C):
    A  mix-in    first bar where kick and bass are in
    B  build     8 bars before the first drop (the kick returning after a 4+ bar break)
    C  peak      the first drop where the track reaches its full energy
                 (no break at all: the biggest sustained energy rise, B 8 bars before it)
All cues sit on downbeats, on a 4-bar boundary (an 8-bar phrase when one is within 2 bars).

    python cues.py [--limit 10] [--match "Human Voice"]   # detect -> cues.json
    python cues.py --apply                                 # write cues.json into Rekordbox
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
OUT = HERE / "cues.json"
RB_SHARE = Path.home() / "Library" / "Pioneer" / "rekordbox" / "share"
SR = 22050
BANDS = {"low": (30, 150), "mid": (150, 2000), "high": (2000, 10000)}


def analysis_paths() -> dict[str, str]:
    """Track ID -> its .DAT analysis file, read from a temp copy of master.db."""
    from pyrekordbox import Rekordbox6Database
    from pyrekordbox.db6 import tables
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy2(RB_SHARE.parent / "master.db", Path(tmp) / "master.db")
        db = Rekordbox6Database(path=str(Path(tmp) / "master.db"))
        out = {str(c.ID): c.AnalysisDataPath for c in db.session.query(tables.DjmdContent) if c.AnalysisDataPath}
        db.session.close()
        db.engine.dispose()
    return out


def analysis_paths_from_files(tracks: dict) -> dict[str, str]:
    """Track ID -> .DAT analysis file, found by the file name stored inside each
    analysis file (it keeps only "?/<file name>"), so no database access is needed
    and it is safe while Rekordbox is open. File names shared by several tracks or
    analysis files are skipped rather than guessed."""
    import unicodedata
    from collections import Counter
    from pyrekordbox import anlz
    norm = lambda x: unicodedata.normalize("NFC", str(x)).strip("\x00 ").split("/")[-1].lower()
    found = []
    for p in (RB_SHARE / "PIONEER" / "USBANLZ").rglob("ANLZ0000.DAT"):
        try:
            name = norm(anlz.AnlzFile.parse_file(p).get_tag("PPTH").get())
        except Exception:
            continue
        if name:
            found.append((name, "/" + str(p.relative_to(RB_SHARE))))
    counts = Counter(name for name, _ in found)
    by_name = {name: path for name, path in found if counts[name] == 1}
    track_names = Counter(norm(t["path"]) for t in tracks.values())
    return {i: by_name[norm(t["path"])] for i, t in tracks.items()
            if track_names[norm(t["path"])] == 1 and norm(t["path"]) in by_name}


def downbeats(dat_path: str) -> np.ndarray:
    from pyrekordbox import anlz
    beats, _, times = anlz.AnlzFile.parse_file(RB_SHARE / dat_path.lstrip("/")).get_tag("PQTZ").get()
    return np.array([t for b, t in zip(beats, times) if b == 1], dtype=float)


def bar_energy(path: str, bars: np.ndarray) -> dict[str, np.ndarray]:
    """Mean power per bar in each band, in dB, plus "low_body": the bar's median
    low-end level, which stays low when the low end is only a few hits (a thump
    without bass) and rises when kick and bass sustain it."""
    from essentia.standard import MonoLoader
    audio = MonoLoader(filename=path, sampleRate=SR)()
    hop, size = 512, 2048
    n = 1 + (len(audio) - size) // hop
    frames = np.lib.stride_tricks.sliding_window_view(audio, size)[::hop][:n] * np.hanning(size)
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(size, 1 / SR)
    frame_t = (np.arange(n) * hop + size / 2) / SR
    edges = np.append(bars, len(audio) / SR)
    idx = np.searchsorted(frame_t, edges)
    out = {}
    for name, (lo, hi) in BANDS.items():
        band = power[:, (freqs >= lo) & (freqs < hi)].sum(axis=1)
        per_bar = [band[a:b].mean() if b > a else 0.0 for a, b in zip(idx[:-1], idx[1:])]
        out[name] = 10 * np.log10(np.maximum(per_bar, 1e-10))
        if name == "low":
            db = 10 * np.log10(np.maximum(band, 1e-10))
            out["low_body"] = np.array([np.median(db[a:b]) if b > a else -100.0 for a, b in zip(idx[:-1], idx[1:])])
    return out


def snap(bar: int, first: int, n_bars: int) -> int:
    """Nearest 8-bar phrase (counted from the first downbeat) if within 2 bars, else nearest 4-bar line."""
    phrase = first + round((bar - first) / 8) * 8
    line = phrase if abs(phrase - bar) <= 2 else first + round((bar - first) / 4) * 4
    return min(max(line, 0), n_bars - 1)


def detect(bars: np.ndarray, energy: dict[str, np.ndarray]) -> dict:
    """Bar numbers and times for hot cues A, B, C."""
    low = energy["low"]
    n = len(bars)
    kick = low >= np.percentile(low, 90) - 8           # kick/bass present in this bar
    kick = np.array([np.mean(kick[max(0, i - 1):i + 2]) >= 0.5 for i in range(n)])  # smooth single bars
    level = 10 * np.log10(sum(10 ** (energy[b] / 10) for b in BANDS))
    after = lambda k: level[k:k + 8].mean()
    late = int(n * 0.9)

    # A: the bar the kick actually comes in (it must stay in for the next 8 bars), not the
    # first bar whose following window happens to contain it
    entry = next((i for i in range(n - 8) if kick[i] and kick[i:i + 8].sum() >= 6), 0)
    near = round(entry / 4) * 4
    a = near if 0 <= near - entry <= 1 or entry - near == 1 else entry   # tidy to a 4-bar line only when 1 bar off
    a = min(max(a, 0), n - 1)

    # drops: the kick coming back after a break of 4+ bars
    drops, i = [], a + 4
    while i < late:
        if not kick[i]:
            j = i
            while j < n and not kick[j]:
                j += 1
            if j - i >= 4 and j < n:
                drops.append(snap(j, 0, n))
            i = j
        else:
            i += 1
    fallback = not drops
    if drops:
        peak = max(after(d) for d in drops)
        c = next(d for d in drops if after(d) >= peak - 0.5)   # first drop at full energy
        b = drops[0] - 8                                        # 8 bars before the first drop
    else:
        rise = [(after(k) - level[max(0, k - 8):k].mean(), k) for k in range(a + 8, late)]
        c = snap(max(rise)[1], 0, n) if rise else min(a + 32, n - 1)
        b = c - 8
    b = snap(b, 0, n)
    if not a < b < c:
        b = max(a + 4, c - 8)
        if not a < b < c:
            b = (a + c) // 2

    t = lambda k: round(float(bars[k]), 3)
    return {"A": t(a), "B": t(b), "C": t(c), "bars": {"A": a, "B": b, "C": c}, "fallback": fallback}


def main():
    import cluster
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--match", help="only tracks whose artist/title contains this")
    ap.add_argument("--apply", action="store_true", help="write the cues in cues.json into Rekordbox")
    ap.add_argument("--update", metavar="FILE", help="replace cues this tool wrote earlier (FILE holds the positions "
                    "it wrote, e.g. cues_v1.json) with the ones in cues.json; cues you changed are left alone")
    args = ap.parse_args()

    if args.update:
        from rekordbox_write import update_hot_cues
        previous = HERE / args.update
        result = update_hot_cues(json.loads(OUT.read_text()), json.loads(previous.read_text()))
        print(f"Cues updated on {result['updated']} tracks; left alone: {result['skipped']}. Backup: {result['backup']}")
        return
    if args.apply:
        from rekordbox_write import write_hot_cues
        result = write_hot_cues(json.loads(OUT.read_text()))
        print(f"Hot cues written for {result['written']} tracks; skipped {result['skipped']}. Backup: {result['backup']}")
        return

    tracks, ids, _ = cluster.load(["Deep Tech FLAC"])
    if args.match:
        ids = [i for i in ids if args.match.lower() in cluster.label_of(tracks[i]).lower()]
    ids = ids[: args.limit] if args.limit else ids
    paths = analysis_paths_from_files({i: tracks[i] for i in ids})
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for n, i in enumerate(ids, 1):
        if i not in paths:
            print(f"[{n}/{len(ids)}] no Rekordbox analysis: {cluster.label_of(tracks[i])}")
            continue
        bars = downbeats(paths[i])
        cues = detect(bars, bar_energy(tracks[i]["path"], bars))
        out[i] = cues
        bb = cues["bars"]
        print(f"[{n}/{len(ids)}] A bar {bb['A']}, B {bb['B']}, C {bb['C']} of {len(bars)}"
              f"{'  (no break: fallback)' if cues['fallback'] else ''}  {cluster.label_of(tracks[i])[:50]}")
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
