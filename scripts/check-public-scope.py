#!/usr/bin/env python3
"""Check the Git publication inventory without opening sensors or private data.

This is a bounded release check, not a general-purpose security audit. Findings
report paths and rule names only, never the matching potentially secret value.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODULES = frozenset('''
__init__ aim_dwell app audio_devices calibration calibration_diagnostics
calibration_history calibration_report calibration_store cursor_motion desktop
effects geometry gesture_calibration gesture_calibration_dialog gesture_stability
grid_selection hand_aim hand_aim_dialog overlay paths personal_calibration
pronunciation pronunciation_dialog selection_settings speech stereo stereo_capture
stereo_dialog stereo_live_diagnostics stereo_samples stereo_settings stereo_vision
summon_sound vision
'''.split())
ROOT_FILES = frozenset({
    '.gitignore', 'README.md', 'RULES.md', 'LICENSE', 'NOTICE', 'UPSTREAMS.json',
    'pyproject.toml', 'requirements.txt', 'requirements.lock.txt',
    'models.expected.json', '啟動 KON.command', 'KON-checkerboard-25mm.png',
})
SCRIPTS = frozenset({
    'setup.sh', 'build-whisper.sh', 'download-models.py', 'prepare-fox-sprites.py',
    'prepare-smoke-sprites.py', 'preview-fox-animation.py', 'check-public-scope.py',
})
POLICY_FILES = {'scripts/check-public-scope.py', 'tests/test_public_scope.py'}
EXCLUDED_NAME = re.compile(
    r'portal|(?:^|_)pet(?:_|\.)|cat_pet|native_pet|photo|phone_import|calibration_voice|voxcpm|hayakawa',
    re.IGNORECASE,
)
EXCLUDED_CONTENT = re.compile(
    r'早川秋|hayakawa|voxcpm|calibration_voice|CalibrationVoice|PortalController|'
    r'portal_world_points|portal_gesture|傳送門|PetHuntController|PetClient|'
    r'PhoneImportServer|photo_import|--native-pets|--pet-service|assets/cat-pet',
    re.IGNORECASE,
)
PRIVATE_PATH = re.compile(
    r'(?:/' r'Users/[^/\s]+/|/' r'home/[^/\s]+/|[A-Z]:\\Users\\)', re.IGNORECASE,
)
SECRET_PATTERNS = (
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    re.compile(r'\bgithub_pat_[A-Za-z0-9_]{40,}\b'),
    re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
    re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{35,}\b'),
)


def allowed_path(name: str) -> bool:
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        return False
    if name in ROOT_FILES or name == '.github/workflows/ci.yml':
        return True
    if len(path.parts) == 2 and path.parts[0] == 'kon':
        return path.suffix == '.py' and path.stem in MODULES
    if len(path.parts) == 2 and path.parts[0] == 'scripts':
        return path.name in SCRIPTS
    if len(path.parts) == 2 and path.parts[0] == 'tests':
        return path.name.startswith('test_') and path.suffix == '.py' and not EXCLUDED_NAME.search(path.name)
    if len(path.parts) == 2 and path.parts[0] == 'THIRD_PARTY_LICENSES':
        return path.suffix.lower() in {'.txt', '.md', '.json'}
    if path.parts and path.parts[0] == 'assets':
        if len(path.parts) == 2:
            return path.name in {'fox-heads.png', 'NOTICE.md', 'provenance.json', 'IMAGEGEN.md'}
        return path.parts[1] in {'fox-dash', 'fox-smoke', 'sfx'} and path.suffix in {'.png', '.wav', '.json', '.txt', '.md'}
    return False


def repository_files(root: Path) -> list[str]:
    result = subprocess.run(
        ['git', 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
        cwd=root, check=True, capture_output=True,
    )
    return sorted(set(result.stdout.decode('utf-8').rstrip('\0').split('\0')) - {''})


def check_files(root: Path, names: list[str]) -> list[str]:
    findings = []
    for name in names:
        if not allowed_path(name):
            findings.append(f'{name}: outside public allowlist')
            continue
        path = root / name
        if path.is_symlink():
            findings.append(f'{name}: symlinks are not public source files')
            continue
        if not path.is_file():
            findings.append(f'{name}: missing publication file')
            continue
        if path.stat().st_size > 20 * 1024 * 1024:
            findings.append(f'{name}: unexpectedly large source asset')
        if path.suffix.lower() in {'.png', '.wav'}:
            continue
        try:
            content = path.read_text(encoding='utf-8')
        except (UnicodeError, OSError):
            findings.append(f'{name}: unreadable source text')
            continue
        if PRIVATE_PATH.search(content):
            findings.append(f'{name}: private machine path')
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            findings.append(f'{name}: possible credential or private key')
        if name not in POLICY_FILES and EXCLUDED_CONTENT.search(content):
            findings.append(f'{name}: excluded feature or narration reference')
    return findings


def check_media_inventory(root: Path, names: list[str]) -> list[str]:
    """Require explicit provenance and exact bytes for every published asset."""
    findings = []
    try:
        provenance = json.loads((root / 'assets/provenance.json').read_text(encoding='utf-8'))
        declared = {}
        for group in provenance['assets']:
            for item in group['files']:
                relative = PurePosixPath(item['path'])
                digest = item['sha256']
                if (relative.is_absolute() or '..' in relative.parts
                        or not re.fullmatch('[0-9a-f]{64}', digest)):
                    raise ValueError('Invalid provenance entry')
                name = 'assets/' + str(relative)
                if name in declared:
                    raise ValueError('Duplicate provenance entry')
                declared[name] = digest
    except (OSError, ValueError, KeyError, TypeError):
        return ['assets/provenance.json: missing or invalid publication inventory']
    metadata = {'assets/NOTICE.md', 'assets/provenance.json', 'assets/IMAGEGEN.md'}
    published_assets = {name for name in names if name.startswith('assets/')} - metadata
    for name in sorted(published_assets - declared.keys()):
        findings.append(f'{name}: asset is absent from reviewed provenance')
    for name in sorted(declared.keys() - published_assets):
        findings.append(f'{name}: provenance asset is absent from publication')
    for name in sorted(published_assets & declared.keys()):
        path = root / name
        if not allowed_path(name) or path.is_symlink() or not path.is_file():
            continue  # Already rejected by the ordinary file boundary checks.
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest != declared[name]:
            findings.append(f'{name}: asset differs from reviewed provenance')
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    names = repository_files(root)
    findings = check_files(root, names) + check_media_inventory(root, names)
    if findings:
        print('\n'.join(findings), file=sys.stderr)
        return 1
    print(f'Public scope verified: {len(names)} source files; no excluded content or detected credentials.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
