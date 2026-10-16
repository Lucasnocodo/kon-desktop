#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TASK_ROOT"
if [ ! -x .venv/bin/python ]; then
    echo '請先執行 sh scripts/setup.sh 完成安裝。'
    exit 1
fi
if ! .venv/bin/python -c 'import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)'; then
    echo '需要使用 Python 3.12 或 3.13 的 .venv，請移除或改名舊環境，再執行 KON_PYTHON=python3.13 sh scripts/setup.sh。' >&2
    echo 'The .venv must use Python 3.12 or 3.13. Remove or rename it, then rerun setup with a supported interpreter.' >&2
    exit 1
fi
exec .venv/bin/python -m kon.app "$@"
