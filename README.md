# Set Curator

Sound-based similarity for your own Rekordbox library, using the same method as
[cosine.club](https://cosine.club): Essentia's `discogs-effnet` model turns each
track into a 1280-dimension embedding, and tracks are compared by cosine
similarity. Everything runs locally on your files.

Rekordbox is only ever read, never written: `export_tracks.py` copies
`master.db` to a temp folder and reads the copy.

## Pipeline

| Step | Script | Needs |
|---|---|---|
| 1. Export the Rekordbox track list to `tracks.json` | `export_tracks.py` | `pyrekordbox` (Windows or macOS, wherever Rekordbox is) |
| 2. Compute embeddings into `embeddings.npz` (incremental) | `embed.py` | Essentia (macOS or Linux; on Windows use WSL) |
| 3. Show nearest neighbours | `similar.py` | `numpy` |

## Setup (macOS)

Essentia's pip wheels support Python 3.9–3.14. The newest ones need macOS 15;
on older macOS, pip falls back to an older wheel automatically.

```bash
./setup_essentia.sh          # venv at ~/.venvs/set-curator + model download
PY=~/.venvs/set-curator/bin/python
$PY -m pip install pyrekordbox   # only if exporting from a Mac Rekordbox library
```

## Usage

```bash
$PY export_tracks.py                         # -> tracks.json
$PY embed.py --playlist "Progressive"        # test on one playlist first
$PY embed.py                                 # then the whole library (skips done tracks)
$PY similar.py "Instant Death" --top 10
```

If `tracks.json` was exported on Windows and the files now live on the Mac,
rewrite the path prefix:

```bash
$PY embed.py --path-map "D:/Music=/Users/me/Music"
```

On Windows/WSL, `D:/...` paths are mapped to `/mnt/d/...` automatically.

## Licensing

Essentia is AGPL-3.0; the `discogs-effnet` model is CC BY-NC-SA 4.0
(non-commercial use; MTG offers a commercial license on request).
