"""The setup must not silently build a different or modified whisper checkout."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = "0c7be696371069e0baa2515ed7aa08d5a5473aa6"


@pytest.mark.parametrize("script", ["setup.sh", "build-whisper.sh"])
@pytest.mark.parametrize("actual,dirty", [("0" * 40, ""), (PIN, " M CMakeLists.txt")])
def test_existing_wrong_or_modified_whisper_is_rejected_before_install(tmp_path, script, actual, dirty):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(ROOT / "scripts" / script, scripts / script)
    checkout = tmp_path / "vendor/whisper.cpp"
    (checkout / ".git").mkdir(parents=True)
    (checkout / "CMakeLists.txt").touch()
    bins = tmp_path / "stubs"
    bins.mkdir()
    (bins / "uname").write_text('#!/bin/sh\ncase "$1" in -s) echo Darwin;; -m) echo arm64;; esac\n')
    (bins / "python3").write_text('#!/bin/sh\nexit 0\n')
    (bins / "git").write_text(f'#!/bin/sh\ncase "$*" in *rev-parse*) echo "{actual}";; *status*) echo "{dirty}";; *) exit 97;; esac\n')
    venv = tmp_path / ".venv/bin"
    venv.mkdir(parents=True)
    for tool in ["python", "cmake", "ninja"]:
        (venv / tool).write_text('#!/bin/sh\necho "unexpected-tool-execution" >&2\nexit 98\n')
    for file in [*bins.iterdir(), *venv.iterdir()]:
        file.chmod(0o755)
    result = subprocess.run(["/bin/sh", str(scripts / script)], cwd=tmp_path,
                            env={**os.environ, "PATH": f"{bins}:/usr/bin:/bin"},
                            text=True, capture_output=True)
    assert result.returncode == 1
    assert "clean checkout" in result.stderr
    assert "unexpected-tool-execution" not in result.stderr
