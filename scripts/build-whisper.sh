#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASK_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
CMAKE="$TASK_ROOT/.venv/bin/cmake"
NINJA="$TASK_ROOT/.venv/bin/ninja"
WHISPER_SOURCE="$TASK_ROOT/vendor/whisper.cpp"
WHISPER_BUILD="$WHISPER_SOURCE/build"
BUILD_JOBS=${KON_BUILD_JOBS:-4}
WHISPER_COMMIT=0c7be696371069e0baa2515ed7aa08d5a5473aa6

if [ ! -x "$CMAKE" ] || [ ! -x "$NINJA" ]; then
    echo "Missing .venv cmake/ninja. Run the project setup first." >&2
    exit 1
fi
if [ ! -f "$WHISPER_SOURCE/CMakeLists.txt" ]; then
    echo "Missing vendor/whisper.cpp checkout." >&2
    exit 1
fi
if [ ! -d "$WHISPER_SOURCE/.git" ] ||
   [ "$(git -C "$WHISPER_SOURCE" rev-parse HEAD)" != "$WHISPER_COMMIT" ] ||
   [ -n "$(git -C "$WHISPER_SOURCE" status --porcelain --untracked-files=no)" ]; then
    echo "whisper.cpp must be a clean checkout of $WHISPER_COMMIT. Run setup or restore that revision." >&2
    exit 1
fi
case "$BUILD_JOBS" in
    ''|*[!0-9]*|0) echo "KON_BUILD_JOBS must be a positive integer." >&2; exit 1 ;;
esac

"$CMAKE" -S "$WHISPER_SOURCE" -B "$WHISPER_BUILD" -G Ninja \
    -DCMAKE_MAKE_PROGRAM="$NINJA" -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_BUILD_EXAMPLES=ON -DWHISPER_BUILD_TESTS=OFF \
    -DGGML_METAL=OFF -DGGML_OPENMP=OFF
"$CMAKE" --build "$WHISPER_BUILD" --target whisper-cli --parallel "$BUILD_JOBS"
"$WHISPER_BUILD/bin/whisper-cli" --help
