import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_wbench_references.py"
spec = importlib.util.spec_from_file_location("build_wbench_references", SCRIPT)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

# Same signature and navigation table shape as WBench's src/models/text/prompt_builder.py.
PROMPT_BUILDER = '''
THIRD_PERSON_ACTIONS = {"W": "The subject moves forward."}
def build_turn_prompt(case, interaction, perspective="first_person", is_first_turn=False):
    parts = [case["environment_prompt"]] if is_first_turn else []
    if interaction["type"] == "navigation":
        parts.append(THIRD_PERSON_ACTIONS[interaction["action"]] if perspective == "third_person" else "camera")
    else:
        parts.append(interaction["action"])
    return " ".join(parts)
'''


@pytest.fixture
def layout(tmp_path, monkeypatch):
    wbench = tmp_path / "WBench"
    (wbench / "src/models/text").mkdir(parents=True)
    (wbench / "src/models/text/prompt_builder.py").write_text(PROMPT_BUILDER)
    git = ["git", "-C", str(wbench), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run(git[:3] + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "."], check=True)
    subprocess.run(git + ["commit", "-qm", "pin"], check=True)
    head = subprocess.run(git[:3] + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(builder, "WBENCH_REVISION", head)
    cases, videos = tmp_path / "cases", tmp_path / "videos"
    cases.mkdir()
    (cases / "case_7.json").write_text(json.dumps({
        "id": "7", "environment_prompt": "A street.", "settings": {"perspective": "third_person"},
        "interactions": [{"turn": 2, "type": "event_edit", "action": "It rains."},
                         {"turn": 1, "type": "navigation", "action": "W"}]}))
    (videos / "7").mkdir(parents=True)
    (videos / "7" / "combined.json").write_text(json.dumps({"video": {"fps": 24.0, "duration": 10.0},
                                                             "turn_frames": [0, 120, 240]}))
    return wbench, cases, videos, tmp_path / "references"


def test_one_reference_event_per_turn_on_the_video_clock(layout):
    wbench, cases, videos, output = layout
    assert builder.build(wbench, cases, videos, output) == {"samples": 1, "output": str(output)}
    assert json.loads((output / "7" / "reference.json").read_text()) == {"duration": 10.0, "events": [
        {"start": 0.0, "end": 5.0, "description": "The subject moves forward."},
        {"start": 5.0, "end": 10.0, "description": "It rains."}]}
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["wbench_revision"] == builder.WBENCH_REVISION
    assert provenance["samples"][0]["turn_types"] == ["navigation", "event_edit"]
    with pytest.raises(FileExistsError):
        builder.build(wbench, cases, videos, output)


def test_rejects_unpinned_checkout_missing_case_and_bad_turn_frames(layout, monkeypatch):
    wbench, cases, videos, output = layout
    (videos / "8").mkdir()
    (videos / "8" / "combined.json").write_text(json.dumps({"video": {"fps": 24.0}, "turn_frames": [0, 120]}))
    with pytest.raises(ValueError, match=r"no WBench case for samples \['8'\]"):
        builder.build(wbench, cases, videos, output)
    (cases / "case_8.json").write_text(json.dumps({"id": "8", "interactions": [
        {"turn": 1, "type": "subject_action", "action": "Wave."}, {"turn": 2, "type": "subject_action", "action": "Sit."}]}))
    with pytest.raises(ValueError, match="turn_frames must start at 0"):
        builder.build(wbench, cases, videos, tmp_output := output.with_name("other"))
    assert not tmp_output.exists()
    monkeypatch.setattr(builder, "WBENCH_REVISION", "0" * 40)
    with pytest.raises(ValueError, match="revision mismatch"):
        builder.build(wbench, cases, videos, output)
