"""Download integrity and cache behavior without network or GPU dependencies."""
import hashlib
import importlib
import importlib.util

import httpx
import pytest


def weights_module():
    assert importlib.util.find_spec('percept_harness.weights'), 'weight downloader is missing'
    return importlib.import_module('percept_harness.weights')


@pytest.mark.parametrize('failure', [None, 'hash', 'http', 'disconnect'])
def test_atomic_download_preserves_old_file_on_failure(tmp_path, failure):
    weights = weights_module()
    asset = weights.Weight('model/file.pth', 'https://weights.test/model', hashlib.sha256(b'good').hexdigest())
    target = tmp_path / 'model/file.pth'
    target.parent.mkdir()
    target.write_bytes(b'old')

    class Interrupted(httpx.SyncByteStream):
        def __iter__(self):
            yield b'partial'
            raise httpx.ReadError('disconnected')

    def serve(request):
        if request.url.path == '/model':
            return httpx.Response(302, headers={'location': '/data'})
        if failure == 'disconnect':
            return httpx.Response(200, stream=Interrupted())
        return httpx.Response(503 if failure == 'http' else 200,
                              content=b'bad' if failure == 'hash' else b'good')

    with httpx.Client(transport=httpx.MockTransport(serve), follow_redirects=True) as client:
        if failure:
            with pytest.raises((ValueError, httpx.HTTPError)):
                weights.download_weight(asset, tmp_path, client)
            assert target.read_bytes() == b'old'
        else:
            assert weights.download_weight(asset, tmp_path, client) == target
            assert target.read_bytes() == b'good'
    assert list(target.parent.iterdir()) == [target]


def test_verified_cache_is_reused_without_network(tmp_path):
    weights = weights_module()
    (tmp_path / 'model.pth').write_bytes(b'good')
    asset = weights.Weight('model.pth', 'https://weights.test/model', hashlib.sha256(b'good').hexdigest())
    def offline(request):
        raise AssertionError('verified file must not be downloaded again')
    with httpx.Client(transport=httpx.MockTransport(offline)) as client:
        assert weights.download_weight(asset, tmp_path, client).read_bytes() == b'good'


def test_download_cli_reuses_selected_metric_cache(tmp_path, monkeypatch, capsys):
    weights = weights_module()
    from percept_harness.cli import main
    asset = weights.Weight('model.pth', 'https://weights.test/model', hashlib.sha256(b'good').hexdigest())
    monkeypatch.setitem(weights.WEIGHTS, 'motion', (asset,))
    (tmp_path / 'model.pth').write_bytes(b'good')
    assert main(['download', 'motion', '--cache-dir', str(tmp_path)]) == 0
    assert str(tmp_path / 'model.pth') in capsys.readouterr().out


def test_verify_rejects_missing_or_corrupt_weights(tmp_path):
    weights = weights_module()
    with pytest.raises((FileNotFoundError, ValueError)):
        weights.verify_weights('motion', tmp_path)
    target = tmp_path / 'amt_model/amt-s.pth'
    target.parent.mkdir()
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA-256'):
        weights.verify_weights('motion', tmp_path)


def test_clipiqa_custom_cache_is_verified_before_loading_runtime(tmp_path):
    from percept_harness.video_metrics.clipiqa import score_video
    video = tmp_path / 'video.mp4'
    video.touch()
    with pytest.raises(ValueError, match='percept download clipiqa'):
        score_video(video, cache_dir=tmp_path / 'weights')


def test_pyiqa_cache_context_restores_upstream_settings(tmp_path, monkeypatch):
    import sys
    import types
    from percept_harness.video_metrics import clipiqa
    assert hasattr(clipiqa, '_weight_cache'), 'custom cache integration is missing'
    # Stand-ins for the two upstream cache globals; no neural model is needed.
    hub = types.SimpleNamespace(_hub_dir='/original')
    hub.get_dir = lambda: hub._hub_dir
    hub.set_dir = lambda value: setattr(hub, '_hub_dir', value)
    download = types.ModuleType('pyiqa.utils.download_util')
    download.DEFAULT_CACHE_DIR = '/original/pyiqa'
    utils = types.ModuleType('pyiqa.utils')
    utils.download_util = download
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(hub=hub))
    monkeypatch.setitem(sys.modules, 'pyiqa', types.ModuleType('pyiqa'))
    monkeypatch.setitem(sys.modules, 'pyiqa.utils', utils)
    monkeypatch.setitem(sys.modules, 'pyiqa.utils.download_util', download)
    with pytest.raises(RuntimeError):
        with clipiqa._weight_cache(tmp_path):
            assert hub._hub_dir == str(tmp_path)
            assert download.DEFAULT_CACHE_DIR == str(tmp_path / 'pyiqa')
            raise RuntimeError('model load failure')
    assert hub._hub_dir == '/original'
    assert download.DEFAULT_CACHE_DIR == '/original/pyiqa'
