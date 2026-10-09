#!/bin/bash
# Bring songs that Rekordbox has just analysed into everything: cues, measurements, every SC folder, comments.
#
#   ./update_new.sh preview    # computes everything, writes NOTHING to Rekordbox
#   ./update_new.sh apply      # the same, then writes to Rekordbox (it must be CLOSED; every write makes a backup)
#
# Order matters: the measurements need the cues, the folders need the measurements, the comments need the folders' tags.
cd "$(dirname "$0")"
PY=~/.venvs/set-curator/bin/python
MODE=${1:-preview}
A=""; [ "$MODE" = "apply" ] && A="--apply"
F='^\[\|WARNING\|^I0000\|^W0000\|pyrekordbox\|mlir\|absl\|INFO'
step() { echo; echo "=== $(date +%H:%M:%S) $*"; "$PY" "$@" 2>&1 | grep -v "$F" | tail -${TAIL:-6}; echo "exit: ${PIPESTATUS[0]}"; }

step export_tracks.py                                  # the new analysis, beat grids and all
step cues.py                                           # detect hot cues for songs that have none yet
TAIL=3 step highend.py                                 # open / closed high end (needs the cues)
TAIL=3 step sax.py room                                # room for a sax (needs the cues)
step sax.py scan                                       # sax model for anything not scanned yet
"$PY" - <<'PYEOF' 2>&1 | grep -v "$F" | tail -3
import json, cluster, roles
tracks, ids, vecs = cluster.load(["Deep Tech FLAC"])
songs, _, _ = cluster.merge_copies(tracks, ids, vecs)
sc = roles.reggae_scores(list(songs))
rg = {i: (1.0 if any("Dub Reggae Bass Addict" in p for p in tracks[i]["playlists"]) else round(v, 3)) for i, v in sc.items()}
json.dump(rg, open("reggae_all.json", "w")); print("reggae flags:", len(rg), "| reggae-world:", sum(v >= 0.3 for v in rg.values()))
PYEOF
step energy_check.py sample                            # energy features for the new songs

if [ "$MODE" = "apply" ]; then
  step cues.py --apply                                 # hot cues A/B/C into Rekordbox (songs that already have cues are skipped)
  # one folder per playlist of yours; the four new playlists get their folder, the others are rebuilt with the new songs
  for P in $("$PY" -c "
import json
t = json.load(open('tracks.json'))
print('\n'.join(sorted({p for x in t for p in x['playlists'] if p.endswith(('AIFF',))})).replace(' ', '@@'))"); do
    NAME=${P//@@/ }
    step cluster.py --per-playlist --only "$NAME" --exclude-playlist "Deep Tech FLAC" --apply
  done
else
  step cluster.py --per-playlist --exclude-playlist "Deep Tech FLAC"      # preview of the per-playlist folders
fi
step roles.py $A --replace                             # SC / Set roles
step roles.py --reggae $A --replace                    # SC / Dub & Reggae
step roles.py --arcs $A --replace                      # SC / Set arcs
step crosspollinate.py $A --replace                    # SC / Cross-pollination (bridges, moods, relatives)
step organise.py $A                                    # SC / Sax and SC / Aura
step comments.py $A                                    # the comment tags on every song
echo; echo "=== DONE ($MODE) $(date +%H:%M:%S)"
