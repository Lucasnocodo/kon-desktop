from __future__ import annotations

import json

import numpy as np
import pytest

from kon import stereo_samples as module
from kon.stereo_capture import CameraSource
from kon.stereo_samples import MAX_SAMPLES, StereoSampleSession


def sample(*, offset=0):
    points = np.mgrid[0:9, 0:6].T.reshape(-1, 2).astype(float) * 15 + [40, 30]
    return {
        'primary': points + offset,
        'secondary': points * 2 + [100, 60] + offset,
        'primary_size': (640, 480), 'secondary_size': (1920, 1080),
        'primary_id': 'usb-camera', 'secondary_id': 'camo-camera',
        'primary_name': 'eMeet Nova', 'secondary_name': 'Camo Camera',
    }


def session(*, samples=None, **kwargs):
    values = dict(square_mm=23, primary_source=0, secondary_source=1,
                  samples=tuple(samples) if samples is not None else (sample(), sample(offset=8)))
    values.update(kwargs)
    return StereoSampleSession(**values)


def test_session_round_trip_preserves_metric_square_and_actual_unequal_frames(tmp_path):
    original = session()
    path = tmp_path / 'nested' / 'stereo-samples.json'
    original.save(path)
    restored = StereoSampleSession.load(path)
    assert restored.square_mm == 23.0
    assert restored.session_id == original.session_id
    assert restored.created_at_utc == original.created_at_utc
    assert restored.primary_source == 0 and restored.secondary_source == 1
    assert len(restored.samples) == 2
    for actual, expected in zip(restored.samples, original.samples):
        for side in ('primary', 'secondary'):
            assert actual[side] == pytest.approx(expected[side])
            for key in ('size', 'id', 'name'):
                assert actual[f'{side}_{key}'] == expected[f'{side}_{key}']
    payload = json.loads(path.read_text())
    assert payload['pattern'] == {'columns': 9, 'rows': 6}
    assert payload['units'] == {'square_size': 'millimeters', 'corners': 'pixels'}
    assert payload['sources']['primary']['image_size'] == [640, 480]
    assert payload['sources']['secondary']['image_size'] == [1920, 1080]
    assert payload['sources']['secondary']['name'] == 'Camo Camera'
    assert payload['sample_count'] == 2
    assert not list(path.parent.glob('*.tmp'))


def test_session_copies_input_arrays_and_defaults_optional_names():
    original = sample()
    del original['primary_name'], original['secondary_name']
    stored = session(samples=[original])
    original['primary'][0] = [300, 300]
    assert stored.samples[0]['primary'][0] == pytest.approx([40, 30])
    assert stored.samples[0]['primary_name'] == ''
    assert stored.samples[0]['secondary_name'] == ''


@pytest.mark.parametrize('value', [0, 4.9, 100.1, True, False, float('nan'), float('inf'), '23', None])
def test_session_rejects_invalid_square_edge(value):
    with pytest.raises(ValueError, match='棋盤格邊長'):
        session(square_mm=value)


@pytest.mark.parametrize('primary,secondary', [(0, 0), (True, 1), (0, False), (-1, 1), (0, 65536)])
def test_session_rejects_duplicate_or_invalid_indices(primary, secondary):
    with pytest.raises(ValueError):
        session(primary_source=primary, secondary_source=secondary)


def test_session_rejects_duplicate_physical_device_identity():
    frame = sample()
    frame['secondary_id'] = frame['primary_id']
    with pytest.raises(ValueError, match='相同裝置識別'):
        session(samples=[frame])


@pytest.mark.parametrize('key,value', [('primary_size', (800, 600)), ('secondary_size', (1280, 720)),
                                      ('primary_id', 'another-usb'), ('secondary_id', 'another-camo')])
def test_session_rejects_mixed_frame_sizes_or_physical_sources(key, value):
    changed = sample(offset=8)
    changed[key] = value
    with pytest.raises(ValueError, match='混用'):
        session(samples=[sample(), changed])


@pytest.mark.parametrize('size', [(31, 480), (640, 16385), (True, 480), (640.0, 480), (640,), '640x480'])
def test_session_rejects_invalid_image_dimensions(size):
    frame = sample()
    frame['primary_size'] = size
    with pytest.raises(ValueError, match='影像尺寸'):
        session(samples=[frame])


@pytest.mark.parametrize('invalid', ['missing', 'count', 'dimensions', 'boolean', 'string',
                                   'nan', 'infinity', 'negative', 'right-edge', 'bottom-edge'])
def test_session_rejects_incomplete_non_numeric_or_out_of_frame_corners(invalid):
    frame = sample()
    if invalid == 'missing':
        del frame['primary']
    elif invalid == 'count':
        frame['primary'] = frame['primary'][:53]
    elif invalid == 'dimensions':
        frame['primary'] = np.ones((54, 3))
    elif invalid == 'boolean':
        frame['primary'] = np.ones((54, 2), dtype=bool)
    elif invalid == 'string':
        frame['primary'] = np.full((54, 2), '40')
    else:
        frame['primary'][0] = {
            'nan': [float('nan'), 30], 'infinity': [40, float('inf')],
            'negative': [-.01, 30], 'right-edge': [640, 30], 'bottom-edge': [40, 480],
        }[invalid]
    with pytest.raises(ValueError):
        session(samples=[frame])


@pytest.mark.parametrize('count', [0, MAX_SAMPLES + 1])
def test_session_rejects_unsupported_sample_count(count):
    with pytest.raises(ValueError, match='組棋盤角點'):
        session(samples=[sample()] * count)


@pytest.mark.parametrize('key,value', [('schema_version', 2), ('schema_version', True),
                                      ('kind', 'other'), ('units', {'corners': 'meters'}),
                                      ('pattern', {'columns': 8, 'rows': 6}),
                                      ('pattern', {'columns': 9, 'rows': 7}),
                                      ('sample_count', 3), ('sample_count', True),
                                      ('samples', {}), ('session_id', 'not-a-uuid'),
                                      ('created_at_utc', '2026-10-15T12:00:00')])
def test_saved_schema_count_grid_and_session_metadata_are_validated(key, value):
    payload = session().to_dict()
    payload[key] = value
    with pytest.raises(ValueError):
        StereoSampleSession.from_dict(payload)


@pytest.mark.parametrize('side', ['primary', 'secondary'])
def test_loading_revalidates_corner_finiteness_and_frame_bounds(side):
    payload = session().to_dict()
    payload['samples'][0][side][0][0] = float('nan')
    with pytest.raises(ValueError, match='有限數值'):
        StereoSampleSession.from_dict(payload)
    payload['samples'][0][side][0][0] = payload['sources'][side]['image_size'][0]
    with pytest.raises(ValueError, match='影像範圍'):
        StereoSampleSession.from_dict(payload)


@pytest.mark.parametrize('contents', [b'{broken', b'null', b'[]', b'{}', b'\xff',
                                     ('[' * 2000 + '0' + ']' * 2000).encode()])
def test_invalid_saved_file_is_rejected(tmp_path, contents):
    path = tmp_path / 'invalid.json'
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        StereoSampleSession.load(path)


def test_oversized_saved_file_is_rejected_before_json_loading(tmp_path):
    path = tmp_path / 'oversized.json'
    with path.open('wb') as stream:
        stream.truncate(8 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match='過大'):
        StereoSampleSession.load(path)


def test_matching_uses_physical_identity_when_names_or_indices_change():
    stored = session()
    stored.require_matching_sources(CameraSource(7, 'Renamed USB', 'usb-camera'),
                                    CameraSource(2, 'Renamed Camo', 'camo-camera'))


@pytest.mark.parametrize('saved_empty,current_primary,current_secondary', [
    (False, CameraSource(0, 'eMeet Nova', ''), CameraSource(1, 'Camo Camera', 'camo-camera')),
    (False, CameraSource(0, 'eMeet Nova', 'wrong-usb'), CameraSource(1, 'Camo Camera', 'camo-camera')),
    (False, CameraSource(0, 'eMeet Nova', 'usb-camera'), CameraSource(1, 'Camo Camera', 'wrong-camo')),
    (False, CameraSource(0, 'eMeet Nova', 'camo-camera'), CameraSource(1, 'Camo Camera', 'usb-camera')),
    (True, CameraSource(0, 'eMeet Nova', 'usb-camera'), CameraSource(1, 'Camo Camera', 'camo-camera')),
])
def test_matching_refuses_unverified_or_different_ids_even_with_same_indices(
        saved_empty, current_primary, current_secondary):
    frames = [sample()]
    if saved_empty:
        frames[0]['secondary_id'] = ''
    stored = session(samples=frames)
    with pytest.raises(ValueError, match='不可只按編號沿用'):
        stored.require_matching_sources(current_primary, current_secondary)


def test_atomic_replace_failure_preserves_previous_checkpoint_and_cleans_temporary(tmp_path, monkeypatch):
    path = tmp_path / 'samples.json'
    session().save(path)
    previous = path.read_bytes()

    def fail_replace(source, target):
        assert target == path
        assert source.parent == path.parent
        raise OSError('simulated replacement failure')

    monkeypatch.setattr(module.os, 'replace', fail_replace)
    with pytest.raises(OSError, match='replacement failure'):
        session(square_mm=24).save(path)
    assert path.read_bytes() == previous
    assert not list(tmp_path.glob('*.tmp'))


def test_checkpoints_cannot_serialize_injected_camera_images_or_unknown_extras(tmp_path):
    frame = sample()
    frame['pixels'] = np.zeros((480, 640, 3), dtype=np.uint8)
    frame['raw_image'] = object()
    stored = session(samples=[frame])
    # Only known corner/source fields are copied; even later injected extras are
    # omitted by the explicit on-disk schema.
    assert 'pixels' not in stored.samples[0]
    stored.samples[0]['injected_image'] = object()
    payload = stored.to_dict()
    assert set(payload['samples'][0]) == {'primary', 'secondary'}
    path = tmp_path / 'corners-only.json'
    stored.save(path)
    saved = json.loads(path.read_text())
    assert set(saved['samples'][0]) == {'primary', 'secondary'}
    assert 'raw_image' not in path.read_text()
    assert 'injected_image' not in path.read_text()
