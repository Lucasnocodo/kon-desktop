#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$TASK_ROOT"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
    echo 'This setup targets Apple Silicon macOS.' >&2
    exit 1
fi
PYTHON_BIN=${KON_PYTHON:-python3}
SUPPORTED_PYTHON='import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)'
if [ -x .venv/bin/python ]; then
    if ! .venv/bin/python -c "$SUPPORTED_PYTHON"; then
        echo 'The existing .venv must use Python 3.12 or 3.13. Remove or rename .venv, then rerun setup with a supported interpreter, for example: KON_PYTHON=python3.13 sh scripts/setup.sh' >&2
        exit 1
    fi
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1 || ! "$PYTHON_BIN" -c "$SUPPORTED_PYTHON"; then
    echo 'Python 3.12 or 3.13 is required; Python 3.14+ is not supported by the pinned Qt dependencies. Install a supported interpreter and run: KON_PYTHON=python3.13 sh scripts/setup.sh' >&2
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
    "$PYTHON_BIN" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.lock.txt
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python scripts/download-models.py
sh scripts/build-whisper.sh
echo 'KON is ready. Run .venv/bin/kon or double-click 啟動 KON.command.'
