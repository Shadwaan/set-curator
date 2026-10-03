# Set Curator

Sound-based similarity for your own Rekordbox library, using the same method as
[cosine.club](https://cosine.club): Essentia's `discogs-effnet` model turns each
track into a 1280-dimension embedding, and tracks are compared by cosine
similarity. Everything runs locally on your files.

`export_tracks.py` reads Rekordbox from a temp copy of `master.db`. The only
things that write to Rekordbox are `make_playlist.py --apply` and
`cluster.py --apply`, which add playlists and never delete anything.

## Pipeline

| Step | Script | Needs |
|---|---|---|
| 1. Export the Rekordbox track list to `tracks.json` | `export_tracks.py` | `pyrekordbox` (Windows or macOS, wherever Rekordbox is) |
| 2. Compute embeddings into `embeddings.npz` (incremental) | `embed.py` | Essentia (macOS or Linux; on Windows use WSL) |
| 3. Show nearest neighbours | `similar.py` | `numpy` |
| 4. Build a playlist from weighted seeds, write it to Rekordbox | `make_playlist.py` + `rekordbox_write.py` | `pyrekordbox`, `psutil` |
| 5. Auto-group the library into sound + tempo playlists, write them to Rekordbox | `cluster.py` + `rekordbox_write.py` | `scikit-learn`, `pyrekordbox`, `psutil` |

## Setup (macOS)

Essentia's pip wheels support Python 3.9–3.14. The newest ones need macOS 15;
on older macOS, pip falls back to an older wheel automatically.

```bash
./setup_essentia.sh          # venv at ~/.venvs/set-curator + model download
PY=~/.venvs/set-curator/bin/python
$PY -m pip install pyrekordbox psutil scikit-learn   # export, writing playlists, auto-grouping
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

Works on Windows and macOS (Rekordbox folder and process name are detected per
platform); used on Windows and on macOS with Rekordbox 7.2.18. On a new
machine, try it against a copy first -- `SET_CURATOR_RB_DIR` points everything
at a folder holding `master.db` and `masterPlaylists6.xml`:

```bash
mkdir -p ~/rb-test && cp ~/Library/Pioneer/rekordbox/{master.db,masterPlaylists6.xml} ~/rb-test/
SET_CURATOR_RB_DIR=~/rb-test $PY make_playlist.py "some track" --size 10 --name "SC Test" --apply
```

If the copy looks right, run without `SET_CURATOR_RB_DIR` (Rekordbox closed).

### Auto-grouping the whole library

```bash
$PY cluster.py            # preview: every playlist with its tracks' BPMs
$PY cluster.py --apply    # write them all into Rekordbox ("SC 01 ...", "SC 02 ...")
```

Tracks are first grouped by sound (k-means on the embeddings, `--size` tracks
per group), then each group is split into tempo bands no wider than
`--bpm-width` (default 8). Half/double time counts: within a group, a track's
BPM is read as half, as-is or double, whichever is nearest the group's median,
so an 80 in a 160 dub group sits with the 160s (marked `*` in the preview).
Bands smaller than `--min` go to the closest-sounding band they fit tempo-wise;
anything that fits nowhere ends up in "Other tempos".

Each playlist is named after the existing Rekordbox playlist most specific to
it plus its BPM range. Missing files, Rekordbox's built-in sampler sounds and
duplicate copies of the same audio (the lossless copy is kept) are left out.
All playlists are written in one transaction with one backup, so either all
are created or none.

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
