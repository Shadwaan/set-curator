# Set Curator

Sound-based similarity for your own Rekordbox library, using the same method as
[cosine.club](https://cosine.club): Essentia's `discogs-effnet` model turns each
track into a 1280-dimension embedding, and tracks are compared by cosine
similarity. Everything runs locally on your files.

`export_tracks.py` reads Rekordbox from a temp copy of `master.db`. The only
thing that writes to Rekordbox is `make_playlist.py --apply`, which adds a
playlist and never deletes anything.

## Pipeline

| Step | Script | Needs |
|---|---|---|
| 1. Export the Rekordbox track list to `tracks.json` | `export_tracks.py` | `pyrekordbox` (Windows or macOS, wherever Rekordbox is) |
| 2. Compute embeddings into `embeddings.npz` (incremental) | `embed.py` | Essentia (macOS or Linux; on Windows use WSL) |
| 3. Show nearest neighbours | `similar.py` | `numpy` |
| 4. Build a playlist from weighted seeds, write it to Rekordbox | `make_playlist.py` + `rekordbox_write.py` | `pyrekordbox`, `psutil` |

## Setup (macOS)

Essentia's pip wheels support Python 3.9–3.14. The newest ones need macOS 15;
on older macOS, pip falls back to an older wheel automatically.

```bash
./setup_essentia.sh          # venv at ~/.venvs/set-curator + model download
PY=~/.venvs/set-curator/bin/python
$PY -m pip install pyrekordbox psutil   # for export + writing playlists
```

## Usage

```bash
$PY export_tracks.py                         # -> tracks.json
$PY embed.py --playlist "Progressive"        # test on one playlist first
$PY embed.py                                 # then the whole library (skips done tracks)
$PY similar.py "Instant Death" --top 10
```

### Writing playlists into Rekordbox

```bash
$PY make_playlist.py "Instant Death:2" "Clockwork Dub" --size 20 --bpm 125-140   # preview
$PY make_playlist.py "Instant Death:2" "Clockwork Dub" --size 20 --name "SC Dub" --apply
```

Seeds are `text[:weight]` matched against "Artist - Title"; their embeddings are
blended by weight and the closest tracks fill the playlist. `--apply`:

- refuses while Rekordbox is running (it would overwrite the change)
- backs up `master.db` to `master.db.bak.setcurator.<timestamp>` first
- refuses an existing playlist name unless `--append` (which only adds missing tracks)
- follows the rules from sff for playlists Rekordbox actually shows (32-bit
  playlist ID + UUIDs, `rb_data_status=0`, registration in
  `masterPlaylists6.xml`, double WAL checkpoint) -- see `rekordbox_write.py`

Only tracks already in that Rekordbox library can be added. Tracks are matched
by exported ID when the file name agrees, else by unique file name, so a
`tracks.json` from another machine still resolves; anything unmatched is listed.

To try it against a copy: `SET_CURATOR_RB_DIR=/path/to/copy` (a folder holding
`master.db` and `masterPlaylists6.xml`).

### Files on another machine

If `tracks.json` was exported on Windows and the files now live on the Mac,
rewrite the path prefix:

```bash
$PY embed.py --path-map "D:/Music=/Users/me/Music"
```

On Windows/WSL, `D:/...` paths are mapped to `/mnt/d/...` automatically.

## Licensing

Essentia is AGPL-3.0; the `discogs-effnet` model is CC BY-NC-SA 4.0
(non-commercial use; MTG offers a commercial license on request).
