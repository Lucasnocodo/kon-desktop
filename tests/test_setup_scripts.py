"""The setup must not silently build a different or modified whisper checkout."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import json
import tomllib

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
        (venv / tool).write_text('#!/bin/sh\nif [ "$1" = -c ]; then exit 0; fi\necho "unexpected-tool-execution" >&2\nexit 98\n')
    for file in [*bins.iterdir(), *venv.iterdir()]:
        file.chmod(0o755)
    result = subprocess.run(["/bin/sh", str(scripts / script)], cwd=tmp_path,
                            env={**os.environ, "PATH": f"{bins}:/usr/bin:/bin"},
                            text=True, capture_output=True)
    assert result.returncode == 1
    assert "clean checkout" in result.stderr
    assert "unexpected-tool-execution" not in result.stderr


def _fake_python(path, version, invocation_log):
    """Evaluate the installer's actual preflight against an interpreter version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        f"sys.version_info = {version!r}\n"
        "if len(sys.argv) >= 3 and sys.argv[1] == '-c':\n"
        "    exec(sys.argv[2])\n"
        "else:\n"
        f"    Path({str(invocation_log)!r}).write_text(json.dumps(sys.argv[1:]))\n"
    )
    path.chmod(0o755)


def _setup_with_wrong_checkout(tmp_path, version):
    """A wrong existing pin stops setup before any install or network action."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(ROOT / "scripts/setup.sh", scripts / "setup.sh")
    (tmp_path / "vendor/whisper.cpp/.git").mkdir(parents=True)
    bins = tmp_path / "stubs"
    bins.mkdir()
    (bins / "uname").write_text('#!/bin/sh\ncase "$1" in -s) echo Darwin;; -m) echo arm64;; esac\n')
    git_log = tmp_path / "git-invoked"
    (bins / "git").write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\n"
        f"Path({str(git_log)!r}).touch()\nprint('{'0' * 40}')\n"
    )
    for file in bins.iterdir():
        file.chmod(0o755)
    _fake_python(bins / "python3", version, tmp_path / "unexpected-python-operation")
    environment = {**os.environ, "PATH": f"{bins}:/usr/bin:/bin"}
    environment.pop("KON_PYTHON", None)
    return scripts / "setup.sh", bins, git_log, environment


@pytest.mark.parametrize("version, accepted", [
    ((3, 11, 9), False), ((3, 12, 0), True), ((3, 13, 9), True),
    ((3, 14, 0), False), ((4, 0, 0), False),
])
def test_fresh_setup_version_preflight_accepts_only_supported_python(tmp_path, version, accepted):
    script, _, git_log, environment = _setup_with_wrong_checkout(tmp_path, version)
    result = subprocess.run(["/bin/sh", str(script)], cwd=tmp_path, env=environment,
                            text=True, capture_output=True)
    assert result.returncode == 1
    if accepted:
        assert "clean checkout" in result.stderr
        assert git_log.exists()
    else:
        assert "Python 3.12 or 3.13 is required" in result.stderr
        assert "KON_PYTHON=python3.13" in result.stderr
        assert not git_log.exists()
    assert not (tmp_path / "unexpected-python-operation").exists()


def test_explicit_supported_python_overrides_unsupported_default(tmp_path):
    script, bins, git_log, environment = _setup_with_wrong_checkout(tmp_path, (3, 14, 0))
    _fake_python(bins / "python3.13", (3, 13, 0), tmp_path / "unexpected-python-operation")
    environment["KON_PYTHON"] = "python3.13"
    result = subprocess.run(["/bin/sh", str(script)], cwd=tmp_path, env=environment,
                            text=True, capture_output=True)
    assert result.returncode == 1 and "clean checkout" in result.stderr
    assert git_log.exists()


@pytest.mark.parametrize("venv_version, accepted", [((3, 13, 0), True), ((3, 11, 9), False), ((3, 14, 0), False)])
def test_setup_checks_existing_environment_instead_of_unused_default_python(tmp_path, venv_version, accepted):
    script, _, git_log, environment = _setup_with_wrong_checkout(tmp_path, (3, 14, 0))
    _fake_python(tmp_path / ".venv/bin/python", venv_version, tmp_path / "unexpected-venv-operation")
    result = subprocess.run(["/bin/sh", str(script)], cwd=tmp_path, env=environment,
                            text=True, capture_output=True)
    assert result.returncode == 1
    if accepted:
        assert "clean checkout" in result.stderr and git_log.exists()
    else:
        assert "existing .venv must use Python 3.12 or 3.13" in result.stderr
        assert "Remove or rename .venv" in result.stderr
        assert not git_log.exists()
    assert not (tmp_path / "unexpected-venv-operation").exists()


@pytest.mark.parametrize("version, accepted", [
    ((3, 11, 9), False), ((3, 12, 0), True), ((3, 13, 0), True), ((3, 14, 0), False),
])
def test_source_launcher_checks_venv_python_before_opening_app(tmp_path, version, accepted):
    launcher = tmp_path / "啟動 KON.command"
    shutil.copyfile(ROOT / launcher.name, launcher)
    invocation_log = tmp_path / "launch-arguments.json"
    _fake_python(tmp_path / ".venv/bin/python", version, invocation_log)
    result = subprocess.run(["/bin/sh", str(launcher), "--language", "en"], cwd=tmp_path,
                            text=True, capture_output=True)
    if accepted:
        assert result.returncode == 0
        assert json.loads(invocation_log.read_text()) == ["-m", "kon.app", "--language", "en"]
    else:
        assert result.returncode == 1
        assert "Python 3.12 or 3.13" in result.stderr
        assert not invocation_log.exists()


def test_project_metadata_matches_supported_installer_versions():
    from packaging.specifiers import SpecifierSet
    python_range = SpecifierSet(tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["requires-python"])
    assert all(version in python_range for version in ("3.12", "3.13"))
    assert all(version not in python_range for version in ("3.11", "3.14", "4.0"))
