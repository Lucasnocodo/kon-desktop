#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo 'This setup targets Apple Silicon macOS.' >&2
    exit 1
fi
if ! python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
    echo 'Python 3.12+ is required.' >&2
    exit 1
fi
WHISPER_COMMIT=0c7be696371069e0baa2515ed7aa08d5a5473aa6
mkdir -p vendor
if [ ! -e vendor/whisper.cpp ]; then
    git clone https://github.com/ggml-org/whisper.cpp.git vendor/whisper.cpp
    git -C vendor/whisper.cpp checkout --detach "$WHISPER_COMMIT"
fi
if [ ! -d vendor/whisper.cpp/.git ] ||
   [ "$(git -C vendor/whisper.cpp rev-parse HEAD)" != "$WHISPER_COMMIT" ] ||
   [ -n "$(git -C vendor/whisper.cpp status --porcelain --untracked-files=no)" ]; then
    echo "vendor/whisper.cpp must be a clean checkout of $WHISPER_COMMIT. Preserve your changes elsewhere and restore that revision before setup." >&2
    exit 1
fi
if [ ! -x .venv/bin/python ]; then
    python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/download-models.py
sh scripts/build-whisper.sh
echo 'KON is ready. Run .venv/bin/kon or double-click 啟動 KON.command.'
