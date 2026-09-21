"""Annotation resume must identify the experiment, not just a filename."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from percept_harness.config import Settings
from percept_harness.models.fake import FakeVideoModel
from percept_harness.runner import SyncRunner, run_batch

TEMPLATE = 'general_video_captioning'


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'video.mp4'
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'color=c=blue:s=32x32:r=4', '-t', '1', '-c:v', 'libx264',
                    '-pix_fmt', 'yuv420p', str(path)], check=True)
    return path


def runner(tmp_path, **settings):
    return SyncRunner(FakeVideoModel(), Settings(work_root=tmp_path/'work', **settings),
                      model_alias='qwen3-vl-8b-instruct')


def batch(r, videos, output, **kwargs):
    return run_batch(r, videos, TEMPLATE, output, log=lambda _: None, **kwargs)


def records(output):
    return [json.loads(p.read_text()) for p in output.glob('*.json')]


def test_same_stem_videos_are_distinct_and_order_independent(tmp_path, video):
    other = tmp_path/'other'/video.name
    other.parent.mkdir(); shutil.copyfile(video, other)
    output = tmp_path/'out'; r = runner(tmp_path)
    assert [o.status for o in batch(r, [video, other], output)] == ['completed', 'completed']
    assert len(records(output)) == 2
    assert {v['video_path'] for v in records(output)} == {str(video.resolve()), str(other.resolve())}
    assert [o.status for o in batch(r, [other, video], output)] == ['skipped', 'skipped']


def test_separate_batches_do_not_reuse_or_overwrite_same_stem(tmp_path, video):
    other=tmp_path/'other'/video.name; other.parent.mkdir(); shutil.copyfile(video,other)
    output=tmp_path/'out'; r=runner(tmp_path)
    batch(r,[video],output)
    original=(output/'video.json').read_bytes()
    assert batch(r,[other],output)[0].status=='completed'
    assert (output/'video.json').read_bytes()==original
    assert len(records(output))==2
    assert batch(r,[video,other],output)[1].status=='skipped'


@pytest.mark.parametrize('change', ['bytes','config','alias','context','query'])
def test_changed_experiment_reruns_and_archives(tmp_path,video,change):
    output=tmp_path/'out'; r=runner(tmp_path)
    batch(r,[video],output)
    original=(output/'video.json').read_bytes()
    kwargs={}
    if change=='bytes': video.write_bytes(video.read_bytes()+b'changed')
    elif change=='config': r=runner(tmp_path, segment_seconds=15)
    elif change=='alias': r.model_alias='other-model'
    elif change=='context': kwargs['prompt_context']='red object'
    elif change=='query': kwargs['query']='describe motion'
    # Alternate alias is a valid identifier even if the fake backend's registry differs.
    assert batch(r,[video],output,**kwargs)[0].status!='skipped'
    history=list((output/'.percept/history').glob('*.json'))
    assert any(p.read_bytes()==original for p in history)


def test_legacy_record_is_not_reused(tmp_path,video):
    output=tmp_path/'out'; output.mkdir()
    legacy={'status':'completed','template':TEMPLATE,'data':{'legacy':True},'video_path':str(video)}
    (output/'video.json').write_text(json.dumps(legacy))
    assert batch(runner(tmp_path),[video],output)[0].status=='completed'
    assert 'legacy' not in json.loads((output/'video.json').read_text())['data']


def test_force_reruns_matching_result(tmp_path,video):
    r=runner(tmp_path); output=tmp_path/'out'
    batch(r,[video],output)
    assert batch(r,[video],output)[0].status=='skipped'
    assert batch(r,[video],output,force=True)[0].status=='completed'


def test_result_payload_tampering_prevents_reuse(tmp_path,video):
    r=runner(tmp_path); output=tmp_path/'out'; batch(r,[video],output)
    p=output/'video.json'; data=json.loads(p.read_text());data['data']={'tampered':True};p.write_text(json.dumps(data))
    assert batch(r,[video],output)[0].status=='completed'


def test_run_manifest_and_aggregate_point_to_provenance(tmp_path,video):
    output=tmp_path/'out'; batch(runner(tmp_path),[video],output)
    record=json.loads((output/'video.json').read_text())
    assert record['provenance']['input']['sha256']
    assert record['provenance']['run_fingerprint']
    assert record['provenance']['run']['code_sha256']
    manifest=list((output/'.percept/runs').glob('*.json'))
    assert len(manifest)==1
    aggregate=json.loads((output/'results.jsonl').read_text())
    assert aggregate['result_file']=='video.json'
    assert aggregate['provenance']==record['provenance']


def test_changed_packaged_prompts_invalidates_resume(tmp_path,video,monkeypatch):
    from percept_harness import result_store
    output=tmp_path/'out';r=runner(tmp_path)
    batch(r,[video],output)
    # The package digest covers Python source and packaged prompt bytes.
    monkeypatch.setattr(result_store,'package_digest',lambda: 'changed-prompt-or-code')
    assert batch(r,[video],output)[0].status=='completed'


def test_effective_backend_identity_and_secrets(tmp_path,video,monkeypatch):
    from percept_harness.models.openai_compat import OpenAICompatVideoModel
    from percept_harness.runner import EvalOutcome
    secret='private-credential-marker'
    model=OpenAICompatVideoModel(base_url=f'https://user:{secret}@example.test/v1?token={secret}',
                                api_key=secret, model_registry={'judge':'model-a'},
                                extra_headers={'Authorization':secret},extra_body={'vendor_token':secret})
    try:
        settings=Settings(openai_api_key=secret,openai_extra_headers={'Authorization':secret},
                          openai_extra_body={'token':secret},openai_proxy=f'http://user:{secret}@proxy.test')
        r=SyncRunner(model,settings,model_alias='judge')
        monkeypatch.setattr(r,'evaluate',lambda path,template,**kwargs:
                            EvalOutcome(path,template,'completed',data={'annotation':'fixture'}))
        output=tmp_path/'out'
        assert batch(r,[video],output)[0].status=='completed'
        assert batch(r,[video],output)[0].status=='skipped'
        model._registry['judge']='model-b'
        assert batch(r,[video],output)[0].status=='completed'
        model.max_frames=64
        assert batch(r,[video],output)[0].status=='completed'
        for path in output.rglob('*'):
            if path.is_file(): assert secret not in path.read_text()
    finally:
        model.close()


def test_custom_backend_never_reuses_unverified_state(tmp_path,video):
    class CustomFake(FakeVideoModel):
        pass
    r=SyncRunner(CustomFake(),Settings(work_root=tmp_path/'work'),model_alias='qwen3-vl-8b-instruct')
    output=tmp_path/'out'
    assert batch(r,[video],output)[0].status=='completed'
    assert batch(r,[video],output)[0].status=='completed'
    assert not json.loads((output/'video.json').read_text())['provenance']['run']['reusable']


def test_mutated_input_is_not_saved_as_success(tmp_path,video,monkeypatch):
    r=runner(tmp_path);real=r.evaluate
    def changing(*args,**kwargs):
        outcome=real(*args,**kwargs)
        video.write_bytes(video.read_bytes()+b'changed-during-evaluation')
        return outcome
    monkeypatch.setattr(r,'evaluate',changing)
    output=tmp_path/'out'
    assert batch(r,[video],output)[0].status=='failed'
    assert json.loads((output/'video.json').read_text())['data'] is None


def test_mutated_configuration_is_not_saved_as_success(tmp_path,video,monkeypatch):
    r=runner(tmp_path);real=r.evaluate
    def changing(*args,**kwargs):
        outcome=real(*args,**kwargs)
        r.settings.segment_seconds=15
        return outcome
    monkeypatch.setattr(r,'evaluate',changing)
    assert batch(r,[video],tmp_path/'out')[0].status=='failed'


def test_corrupt_current_file_is_preserved_and_recomputed(tmp_path,video):
    output=tmp_path/'out';output.mkdir()
    (output/'video.json').write_text('{truncated')
    assert batch(runner(tmp_path),[video],output)[0].status=='completed'
    assert (output/'video.json').read_text()=='{truncated'
    assert len(list(output.glob('*.json')))==2


def test_cli_force_option_is_accepted(tmp_path,video,monkeypatch):
    from percept_harness.cli import main
    monkeypatch.setenv('PERCEPT_WORK_ROOT',str(tmp_path/'work'))
    args=['annotate','--videos',str(video),'--template',TEMPLATE,
          '--backend','fake','--output',str(tmp_path/'out')]
    assert main(args)==0
    assert main([*args,'--force'])==0
    assert list((tmp_path/'out/.percept/history').glob('*.json'))


def test_same_output_cannot_be_written_by_overlapping_batches(tmp_path,video,monkeypatch):
    r=runner(tmp_path);real=r.evaluate;output=tmp_path/'out'
    def overlapping(*args,**kwargs):
        with pytest.raises(RuntimeError,match='already in use'):
            batch(runner(tmp_path),[video],output)
        return real(*args,**kwargs)
    monkeypatch.setattr(r,'evaluate',overlapping)
    assert batch(r,[video],output)[0].status=='completed'
    # The lock is released even after an evaluation failure.
    assert batch(r,[video],output)[0].status=='skipped'


def test_cli_device_override_is_recorded(tmp_path,video,monkeypatch):
    from percept_harness.cli import main
    monkeypatch.setenv('PERCEPT_WORK_ROOT',str(tmp_path/'work'))
    output=tmp_path/'out'
    assert main(['annotate','--videos',str(video),'--template',TEMPLATE,'--backend','fake',
                 '--cv-device','7','--output',str(output)])==0
    record=json.loads((output/'video.json').read_text())
    assert record['provenance']['run']['settings']['cv_device']==7


def test_recorded_proxy_tracks_actual_backend_configuration():
    from percept_harness.models.openai_compat import OpenAICompatVideoModel
    options=dict(base_url='https://example.test/v1',api_key='private',model_registry={'judge':'m'})
    a=OpenAICompatVideoModel(**options)
    b=OpenAICompatVideoModel(**options,proxy='http://localhost:1234')
    try:
        assert a.run_identity('judge') != b.run_identity('judge')
    finally:
        a.close();b.close()


@pytest.mark.parametrize('value',[float('nan'),float('inf'),'\ud800'])
def test_noncanonical_cached_payload_is_recomputed(tmp_path,video,value):
    output=tmp_path/'out';r=runner(tmp_path);batch(r,[video],output)
    path=output/'video.json';record=json.loads(path.read_text());record['data']={'invalid':value}
    path.write_text(json.dumps(record))
    assert batch(r,[video],output)[0].status=='completed'


def test_archive_publish_failure_keeps_current_result_and_is_retryable(tmp_path,video,monkeypatch):
    from percept_harness import result_store
    output=tmp_path/'out';r=runner(tmp_path);batch(r,[video],output)
    before=(output/'video.json').read_bytes()
    replace=result_store.os.replace
    def fail_archive(source,destination):
        if Path(destination).parent.name=='history':
            raise OSError('archive publication interrupted')
        return replace(source,destination)
    with monkeypatch.context() as patch:
        patch.setattr(result_store.os,'replace',fail_archive)
        with pytest.raises(OSError,match='archive publication interrupted'):
            batch(r,[video],output,force=True)
    assert (output/'video.json').read_bytes()==before
    assert not list((output/'.percept/history').glob('*'))
    assert batch(r,[video],output,force=True)[0].status=='completed'
    assert any(p.read_bytes()==before for p in (output/'.percept/history').glob('*.json'))


def test_backend_account_change_invalidates_identity():
    from percept_harness.models.openai_compat import OpenAICompatVideoModel
    options=dict(base_url='https://example.test/v1',model_registry={'judge':'m'})
    a=OpenAICompatVideoModel(**options,api_key='account-one')
    b=OpenAICompatVideoModel(**options,api_key='account-two')
    try:
        assert a.run_identity('judge')!=b.run_identity('judge')
    finally:
        a.close();b.close()


def test_same_size_and_mtime_do_not_hide_changed_video_bytes(tmp_path,video):
    import os
    video.write_bytes(video.read_bytes()+b'ab')
    initial=video.stat();r=runner(tmp_path);output=tmp_path/'out'
    batch(r,[video],output)
    video.write_bytes(video.read_bytes()[:-2]+b'cd')
    os.utime(video,ns=(initial.st_atime_ns,initial.st_mtime_ns))
    assert video.stat().st_size==initial.st_size
    assert batch(r,[video],output)[0].status=='completed'


def test_custom_pipeline_registry_disables_resume(tmp_path,video):
    from percept_harness.runner import default_pipeline_registry
    r=SyncRunner(FakeVideoModel(),Settings(work_root=tmp_path/'work'),
                 model_alias='qwen3-vl-8b-instruct',registry=default_pipeline_registry())
    output=tmp_path/'out'
    assert batch(r,[video],output)[0].status=='completed'
    assert batch(r,[video],output)[0].status=='completed'


def test_cv_execution_is_recorded_but_not_reused(tmp_path,video):
    # The general pipeline does not call CV, but the run identity must still
    # conservatively distinguish an unverified externally supplied executor.
    r=runner(tmp_path);r.cv_executor=object();output=tmp_path/'out'
    assert batch(r,[video],output)[0].status=='completed'
    assert batch(r,[video],output)[0].status=='completed'
    assert not json.loads((output/'video.json').read_text())['provenance']['run']['reusable']
