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
| 6. Section-by-section "palette" analysis into `patches.npz` | `palette.py build` | Essentia |
| 7. Cross-pollination: bridges, moods, palettes, relatives of your productions | `crosspollinate.py` | as 5 |
| 8. Auto hot cues A/B/C on Rekordbox's beatgrid | `cues.py` | as 5 |
| 9. Set roles (openers ... closers) and palette-coherent set arcs | `roles.py` (`--arcs`) | as 5, needs `cues.json` |
| 10. Seed-based curation GUI: library + cosine.club, snippets, Spotify playlists | `curator_app.py` | `flask`, `requests`, `spotipy`, `python-dotenv`, `ffmpeg` |

## Setup (macOS)

Essentia's pip wheels support Python 3.9–3.14. The newest ones need macOS 15;
on older macOS, pip falls back to an older wheel automatically.

```bash
./setup_essentia.sh          # venv at ~/.venvs/set-curator + model download
PY=~/.venvs/set-curator/bin/python
$PY -m pip install pyrekordbox psutil scikit-learn   # export, writing playlists, auto-grouping
$PY -m pip install flask requests spotipy python-dotenv   # curator app
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
$PY cluster.py            # preview the folder tree with every track's BPM
$PY cluster.py --apply    # write it into Rekordbox
```

The result is a folder tree, which also carries over to USB exports:

```
SC/
    Deep House/
        110-118
        119-127
    Reggae Dub/
        125-133
        ...
    Other tempos
```

Tracks are first grouped by sound (k-means on the embeddings, `--size` tracks
per group). Each group is named by a Discogs style from Essentia's
`genre_discogs400` classifier: the style that sets the group apart from the
rest of the library, distinct per group (approximate, since it runs on each
track's averaged embedding). Each group is then split into tempo bands no wider than
`--bpm-width` (default 8). Half/double time counts: within a group, a track's
BPM is read as half, as-is or double, whichever is nearest the group's median,
so an 80 in a 160 dub group sits with the 160s (marked `*` in the preview).
Bands smaller than `--min` go to the closest-sounding band they fit tempo-wise;
anything that fits nowhere ends up in "Other tempos".

Missing files, Rekordbox's built-in sampler sounds and duplicate copies of the
same audio (the lossless copy is kept) are left out, as are tracks in any
playlist passed with `--exclude-playlist "Name"` (repeatable). The whole tree is written
in one transaction with one backup, so either all of it is created or none.
`--apply` refuses if a top-level item named `SC` (`--folder`) already exists;
add `--replace` to rebuild it in the same place. `--per-playlist` builds one
folder per Rekordbox playlist instead, each split by sound and tempo.

All Rekordbox writes (here and below) refuse while Rekordbox is running, back up
`master.db` first, and are a single transaction.

### Cross-pollination

```bash
$PY palette.py build                       # section-by-section analysis (~15 min)
$PY crosspollinate.py --apply --replace    # SC / Cross-pollination
```

- **Bridges**: for each pair of your playlists that overlap, the tracks from
  either that sound closest to the other; **Bridges (palette)** judges the same
  by shared sounds.
- **Moods**: the library by Essentia mood (Dark & Driving, Peak Time, ...).
- **Palettes**: the library grouped by shared sound palette, named by mood and
  the crates each group mostly comes from.
- **Relatives**: each of your productions (files under `~/Documents/Samples`),
  followed by the tracks sharing most of its palette.

Every group is a folder of tempo playlists ("120-128", half-time folded), and
copies of one song (separate files with the same audio) count once.

### Hot cues

```bash
$PY cues.py            # detect -> cues.json
$PY cues.py --apply    # write hot cues A/B/C into Rekordbox
```

A: the kick comes in. B: 8 bars before the first drop. C: the first drop at full
energy. Positions come from Rekordbox's beatgrid and snap to 4/8-bar phrases.
Tracks that already have cues are never touched; MP3s are skipped.

### Set roles and set arcs

```bash
$PY roles.py --apply --replace          # SC / Set roles: 1 Openers ... 7 Closers
$PY roles.py --arcs --apply --replace   # SC / Set arcs: roles inside each palette family
```

Roles are judged against tracks at a similar tempo, from peak loudness, drop
size, melody (pitch clarity), rhythmic busyness, intro length and Essentia moods.
Set arcs keep a role sequence inside each sound family (and tempo lane), so
moving from role to role keeps the palette as well as the tempo.

### Curator app

```bash
$PY curator_app.py     # http://127.0.0.1:8765
```

Seeds from your library or cosine.club, curate from your library (palette
matching), the web (cosine.club candidates re-ranked across seeds) or both,
preview snippets, keep tracks, save the set (`ledger.json`, `to_download.csv`)
and optionally create it as a Spotify playlist. Needs a `.env` with
`COSINE_API_KEY` (https://cosine.club/account/api) and `SPOTIFY_CLIENT_ID` (a
Spotify dev app with redirect URI `http://127.0.0.1:8888/callback`).

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
