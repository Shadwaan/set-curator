#!/usr/bin/env bash
# One-time setup of Essentia + the discogs-effnet model.
# Run inside WSL (Ubuntu) on Windows, or in Terminal on macOS.
set -e
cd "$(dirname "$0")"

if command -v apt-get >/dev/null; then
  sudo apt-get update && sudo apt-get install -y python3-venv python3-pip curl
fi

# On WSL, keep the venv on the Linux filesystem: a venv under /mnt/c is very slow.
VENV="${SET_CURATOR_VENV:-$HOME/.venvs/set-curator}"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install essentia-tensorflow numpy

mkdir -p models
curl -fL -o models/discogs-effnet-bs64-1.pb \
  https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb

"$VENV/bin/python" -c "import essentia; print('essentia', essentia.__version__, 'OK')"
echo "Venv: $VENV   (run scripts with $VENV/bin/python)"
