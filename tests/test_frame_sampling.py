"""Real decoder checks for source timestamps and bounded process startup."""
import subprocess
from unittest.mock import patch

import pytest

from percept_harness.media import FrameExtractionError, TimeSpan, extract_frames
from percept_harness.cv.timeline import probe_frame_timeline


def make_video(path, rate=10, duration=1, vf=None, offset=None):
    args = ['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
            f'testsrc2=size=64x64:rate={rate}:duration={duration}']
    if vf:
        args += ['-vf', vf]
    args += ['-fps_mode', 'passthrough', '-c:v', 'mpeg4', '-q:v', '2']
    if offset is not None:
        args += ['-output_ts_offset', str(offset)]
    subprocess.run(args + [str(path)], check=True)
    return path


def test_one_decoder_returns_actual_source_pts_between_requested_ticks(tmp_path):
    video = make_video(tmp_path / 'clip.mp4')
    run = subprocess.run
    with patch('percept_harness.media.subprocess.run', wraps=run) as calls:
        frames = extract_frames(video, TimeSpan(.05, .86), 4, tmp_path / 'frames')
    assert [f.timestamp for f in frames] == pytest.approx([.1, .3, .6, .8])
    assert len([c for c in calls.call_args_list if c.args[0][0] == 'ffmpeg']) == 1
    assert all(f.path.is_file() for f in frames)


def test_rate_above_source_does_not_duplicate_frames(tmp_path):
    video = make_video(tmp_path / 'slow.mp4', rate=2)
    frames = extract_frames(video, TimeSpan(0, 1), 10, tmp_path / 'frames')
    assert [f.timestamp for f in frames] == pytest.approx([0, .5])


def test_variable_rate_pts_and_jpegs_are_the_same_source_frames(tmp_path):
    video = make_video(tmp_path / 'vfr.mp4', rate=10, duration=2,
                       vf="select='eq(n,0)+eq(n,1)+eq(n,4)+eq(n,9)+eq(n,12)+eq(n,18)'")
    frames = extract_frames(video, TimeSpan(.05, 1.81), 4, tmp_path / 'frames')
    expected = [.1, .4, .9, 1.2, 1.8]
    assert [f.timestamp for f in frames] == pytest.approx(expected)
    timeline = probe_frame_timeline(video)
    source_times = [f.timestamp_seconds for f in timeline.frames]
    for frame, timestamp in zip(frames, expected):
        index = next(i for i,t in enumerate(source_times) if abs(t-timestamp) < 1e-6)
        reference = tmp_path / f'reference-{index}.jpg'
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-vf',
                        f'select=eq(n\\,{index})', '-frames:v', '1', '-q:v', '2',
                        str(reference)], check=True)
        assert frame.path.read_bytes() == reference.read_bytes()


def test_nonzero_container_offset_keeps_playback_timestamps(tmp_path):
    video = make_video(tmp_path / 'offset.mp4', offset=5)
    frames = extract_frames(video, TimeSpan(.15, .85), 4, tmp_path / 'frames')
    assert [f.timestamp for f in frames] == pytest.approx([.2, .4, .7])


def test_invalid_input_does_not_return_stale_output(tmp_path):
    video = tmp_path / 'bad.mp4'; video.write_bytes(b'not video')
    output = tmp_path / 'frames'; output.mkdir()
    sentinel = output / 'frame_000001.jpg'; sentinel.write_bytes(b'existing')
    with pytest.raises(FrameExtractionError):
        extract_frames(video, TimeSpan(0, 1), 2, output)
    assert sentinel.read_bytes() == b'existing'


def test_fractional_pts_are_not_rounded_to_log_display_precision(tmp_path):
    video = make_video(tmp_path / 'thirds.mp4', rate=3)
    frames = extract_frames(video, TimeSpan(0, 1), 6, tmp_path / 'frames')
    assert [f.timestamp for f in frames] == [0, 1 / 3, 2 / 3]


def test_empty_span_and_repeated_output_directory_are_isolated(tmp_path):
    video = make_video(tmp_path / 'slow.mp4', rate=2)
    output = tmp_path / 'frames'
    first = extract_frames(video, TimeSpan(0, 1), 2, output)
    original = [f.path.read_bytes() for f in first]
    with pytest.raises(FrameExtractionError):
        extract_frames(video, TimeSpan(.1, .2), 2, output)
    second = extract_frames(video, TimeSpan(0, 1), 2, output)
    assert {f.path for f in first}.isdisjoint(f.path for f in second)
    assert [f.path.read_bytes() for f in first] == original


def test_old_sampling_contract_cannot_hit_model_request_cache(tmp_path):
    from percept_harness.models.base import ModelRequest
    from percept_harness.models.openai_compat import OpenAICompatVideoModel
    from percept_harness.runner import SemanticResultCache
    video = tmp_path / 'video'; video.write_bytes(b'cache identity input')
    request = ModelRequest(stage='grounded_draft', video_path=video, span=TimeSpan(0, 1),
                           fps=2, prompt='describe', schema_name='GroundedEvents',
                           model_name='test', video_session_id=None)
    model = OpenAICompatVideoModel(base_url='https://example.test/v1', api_key='test',
                                  model_registry={'test': 'test'})
    try:
        cache = SemanticResultCache(tmp_path / 'cache')
        current = model.semantic_cache_identity(request)
        old = {**current, 'frame_extraction_contract_version': 'extract-frames-jpeg-timestamps-v1'}
        old_key = cache.key(request, old, None)
        assert cache.publish(old_key, {'duration': 1, 'events': []})
        new_key = cache.key(request, current, None)
        assert old_key != new_key
        assert cache.load(new_key) is None
    finally:
        model.close()


def test_resolution_change_keeps_sampler_state_and_frame_budget(tmp_path):
    elementary = []
    for index, size in enumerate(('64x64', '128x128')):
        segment = tmp_path / f'{index}.ts'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                        f'testsrc2=size={size}:rate=5:duration=1', '-c:v',
                        'mpeg2video', '-output_ts_offset', str(index),
                        '-f', 'mpegts', str(segment)], check=True)
        elementary.append(segment.read_bytes())
    joined = tmp_path / 'joined.ts'; joined.write_bytes(b''.join(elementary))
    video = tmp_path / 'changing.mkv'
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(joined),
                    '-c:v', 'copy', str(video)], check=True)
    frames = extract_frames(video, TimeSpan(0, 2), 2, tmp_path / 'frames')
    assert [f.timestamp for f in frames] == pytest.approx([0, .6, 1, 1.6])
    assert all(f.path.is_file() for f in frames)


def test_real_sampling_respects_vlm_frame_budget(tmp_path):
    import httpx
    from percept_harness.models.base import ModelRequest
    from percept_harness.models.openai_compat import OpenAICompatVideoModel
    video = make_video(tmp_path / 'budget.mp4', rate=30, duration=2)
    sent = []
    def respond(request):
        import json
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={'choices': [{'message': {
            'role': 'assistant', 'content': '{"duration":2,"events":[]}'}}]})
    model = OpenAICompatVideoModel(base_url='https://example.test/v1', api_key='test',
                                  model_registry={'test': 'test'}, max_frames=4,
                                  transport=httpx.MockTransport(respond))
    try:
        request = ModelRequest(stage='grounded_draft', video_path=video, span=TimeSpan(0, 2),
                               fps=100, prompt='describe', schema_name='GroundedEvents',
                               model_name='test', video_session_id=None)
        assert model.generate(request)['events'] == []
        content = sent[0]['messages'][0]['content']
        assert sum(item['type'] == 'image_url' for item in content) == 4
        assert any('Frame timestamp: 0.600000 seconds' == item.get('text') for item in content)
    finally:
        model.close()
