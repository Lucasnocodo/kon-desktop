"""Publication regressions use synthetic content and never print secret values."""
import importlib.util
import hashlib
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'check-public-scope.py'
spec = importlib.util.spec_from_file_location('public_scope', SCRIPT)
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)


def test_ignored_private_data_cannot_be_force_added():
    for path in ('.kon/preferences.json', 'models/task.bin', 'vendor/source.py',
                 'output/result.json', '.env', '../README.md', '/tmp/README.md'):
        assert not scope.allowed_path(path)
    assert scope.allowed_path('kon/app.py')
    assert scope.allowed_path('assets/fox-dash/frames/01.png')


def test_sensitive_text_is_flagged_without_revealing_value(tmp_path):
    sample = 'ghp_' + 'x' * 36
    path = tmp_path / 'README.md'
    path.write_text(sample)
    findings = scope.check_files(tmp_path, ['README.md'])
    assert len(findings) == 1 and 'credential' in findings[0]
    assert sample not in str(findings)


def test_machine_paths_and_excluded_features_are_rejected(tmp_path):
    path = tmp_path / 'README.md'
    path.write_text('/' + 'Users/' + 'example/project')
    assert 'private machine path' in scope.check_files(tmp_path, ['README.md'])[0]
    path.write_text('from kon.' + 'portal_controller import PortalController')
    assert 'excluded feature' in scope.check_files(tmp_path, ['README.md'])[0]


def test_publication_does_not_follow_symlinks(tmp_path):
    outside = tmp_path / 'outside.txt'
    outside.write_text('private')
    (tmp_path / 'README.md').symlink_to(outside)
    assert 'symlinks' in scope.check_files(tmp_path, ['README.md'])[0]


def test_extra_or_modified_media_cannot_hide_in_public_asset_folder(tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    approved = assets / 'fox-heads.png'
    approved.write_bytes(b'approved sample')
    manifest = {'assets': [{'files': [{
        'path': approved.name, 'sha256': hashlib.sha256(approved.read_bytes()).hexdigest(),
    }]}]}
    (assets / 'provenance.json').write_text(json.dumps(manifest))
    names = ['assets/fox-heads.png', 'assets/provenance.json']
    assert not scope.check_media_inventory(tmp_path, names)
    assert 'absent from reviewed' in scope.check_media_inventory(
        tmp_path, names + ['assets/fox-dash/source/unreviewed.png'])[0]
    approved.write_bytes(b'changed sample')
    assert 'differs from reviewed' in scope.check_media_inventory(tmp_path, names)[0]


def test_provenance_cannot_reference_outside_assets(tmp_path):
    assets = tmp_path / 'assets'
    assets.mkdir()
    (assets / 'provenance.json').write_text(json.dumps({'assets': [{'files': [{
        'path': '../private.txt', 'sha256': '0' * 64,
    }]}]}))
    assert 'invalid publication inventory' in scope.check_media_inventory(tmp_path, [])[0]
