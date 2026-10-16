#!/bin/sh
set -eu
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TASK_ROOT"
if [ ! -x .venv/bin/python ]; then
    echo '請先執行 sh scripts/setup.sh 完成安裝。'
    exit 1
fi
exec .venv/bin/python -m kon.app "$@"
