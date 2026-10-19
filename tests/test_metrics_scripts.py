import importlib.util
import json
from pathlib import Path

import pytest

from test_metrics_vbench import dumps

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(f"scripts_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest = load("validate_metric_manifest")
adapters = load("official_input_adapters")
reference = load("build_reference_manifest")
verify_install = load("verify_install")


# --- validate_metric_manifest ---------------------------------------------

KNOWN = {
    "clip_score", "overall_consistency", "instruction_following", "action_binding", "motion_binding",
    "motion_order_understanding", "motion_rationality", "object_interactions", "physical_adherence", "mechanics",
    "thermotics", "material", "video_phy_sa", "phygen_eval_pca", "psnr", "ssim", "lpips", "clipiqa+",
    "clipiqa_plus", "imaging_quality", "aesthetic_quality", "dover", "motion_smoothness", "temporal_flickering",
    "dynamic_degree", "video_phy_pc", "subject_consistency", "background_consistency", "long_subject_consistency",
    "long_background_consistency", "fid", "fvd", "object_class", "multiple_objects", "color",
    "spatial_relationship", "scene", "temporal_style", "human_action", "appearance_style",
}


def test_known_dimensions_pinned():
    assert manifest.KNOWN_DIMENSIONS == KNOWN
    assert manifest.AUXILIARY_DIMENSIONS == {"motion_order_understanding", "motion_rationality", "mechanics", "thermotics", "material"}


def test_known_dimensions_not_derivable_from_external_specs():
    from percept_harness.video_metrics.external import SPECS
    from percept_harness.video_metrics.vbench_backend import DIRECT_DIMENSIONS
    assert set(SPECS) | DIRECT_DIMENSIONS != KNOWN


def test_validate_manifest_good(tmp_path):
    (tmp_path / "v.mp4").write_bytes(b"v")
    (tmp_path / "ref").mkdir()
    payload = [{"video_id": "a", "video": "v.mp4", "prompt": "p", "reference_frames": "ref",
                "dimensions": ["psnr", "clip_score"]},
               {"video_id": "b", "video": "w.mp4", "prompt": "p", "dimensions": ["motion_order_understanding"],
                "auxiliary_info": ["first", "second"]}]
    assert manifest.validate_manifest(payload, root=tmp_path) == {
        "records": 2, "video_ids": ["a", "b"], "dimensions_checked": ["clip_score", "motion_order_understanding", "psnr"],
        "validation_scope": "Preparation schema only; target-specific metadata, reference alignment and official input compatibility still require review."}


def test_validate_manifest_errors(tmp_path):
    payload = [
        {"video_id": "a", "video": "a.mp4", "dimensions": ["clip_score"]},
        {"video_id": "a", "video": "a.mp4", "dimensions": ["bogus", "psnr", "psnr"], "reference_frames": "a.mp4"},
        {"video_id": "c", "video": "c.mp4", "dimensions": ["mechanics"], "auxiliary_info": []},
        {"video_id": "d", "video": "d.mp4", "dimensions": ["motion_order_understanding"], "auxiliary_info": ["x"]},
        {"video_id": "e", "video": "e.mp4", "auxiliary_info": [" "]},
        {"video_id": " ", "video": "", "dimensions": "x", "prompt": ""},
    ]
    with pytest.raises(ValueError) as error:
        manifest.validate_manifest(payload, root=tmp_path, check_files=True)
    assert str(error.value).split("\n") == [
        "records[0].video does not exist: " + str((tmp_path / "a.mp4").resolve()),
        "records[0].prompt is required for semantic dimensions",
        "duplicate video_id: a",
        "duplicate video path: a.mp4",
        "records[1].video does not exist: " + str((tmp_path / "a.mp4").resolve()),
        "records[1].dimensions contains duplicate dimensions",
        "records[1].dimensions has unknown names: ['bogus']",
        "records[1].reference_frames cannot be the generated video path",
        "records[1].reference_frames does not exist: " + str((tmp_path / "a.mp4").resolve()),
        "records[2].video does not exist: " + str((tmp_path / "c.mp4").resolve()),
        "records[2].prompt is required for semantic dimensions",
        "records[2].auxiliary_info must be a non-empty list of strings for VBench-2.0 question dimensions",
        "records[3].video does not exist: " + str((tmp_path / "d.mp4").resolve()),
        "records[3].prompt is required for semantic dimensions",
        "records[3].auxiliary_info must contain exactly two ordered actions for Motion Order Understanding",
        "records[4].video does not exist: " + str((tmp_path / "e.mp4").resolve()),
        "records[4].auxiliary_info must be a non-empty list of strings for VBench-2.0 question dimensions",
        "records[5].video_id must be a non-empty string",
        "records[5].video must be a non-empty path",
        "records[5].dimensions must be a list of non-empty strings",
        "records[5].prompt must be a non-empty string when supplied",
    ]


def test_validate_manifest_argument_errors():
    with pytest.raises(ValueError, match="Unknown dimensions"):
        manifest.validate_manifest([{}], dimensions={"nope"})
    with pytest.raises(ValueError, match="collection of non-empty strings"):
        manifest.validate_manifest([{}], dimensions="psnr")
    with pytest.raises(ValueError, match="non-empty JSON array"):
        manifest.validate_manifest([])
    with pytest.raises(ValueError, match="must be a JSON object"):
        manifest.validate_manifest([1])
    payload = [{"video_id": "a", "video": "a.mp4", "auxiliary_info": ["x"]}]
    with pytest.raises(ValueError, match="exactly two ordered actions"):
        manifest.validate_manifest(payload, dimensions={"motion_order_understanding"})


def test_validate_manifest_main(tmp_path, capsys):
    path = tmp_path / "m.json"
    path.write_text(json.dumps([{"video_id": "a", "video": "a.mp4"}]))
    assert manifest.main(["--manifest", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["video_ids"] == ["a"]
    assert manifest.main(["--manifest", str(path), "--dimension", "psnr"]) == 1
    assert capsys.readouterr().err == "Manifest validation failed: records[0].reference_frames is required for PSNR/SSIM/LPIPS\n"


# --- official_input_adapters ----------------------------------------------

def vb2(dimension, aux, **extra):
    return {"video_id": "a", "video": "a.mp4", "prompt_en": " p ", "dimension": dimension, "auxiliary_info": aux, **extra}


def test_vbench2_dimension_names_match_between_scripts():
    assert adapters._VB2_NAMES == {"Motion_Order_Understanding", "Motion_Rationality", "Mechanics", "Thermotics", "Material"}
    assert {name.lower() for name in adapters._VB2_NAMES} == manifest.AUXILIARY_DIMENSIONS


def test_adapt_vbench2():
    result = adapters.adapt_vbench2([vb2("Mechanics", [" q "]), {**vb2(["Motion_Order_Understanding"], ["a", "b"]), "video_id": "b", "video": "b.mp4"}])
    assert result == {"official": [{"prompt_en": "p", "dimension": "Mechanics", "auxiliary_info": ["q"]},
                                   {"prompt_en": "p", "dimension": ["Motion_Order_Understanding"], "auxiliary_info": ["a", "b"]}],
                      "video_map": [{"video_id": "a", "video": "a.mp4"}, {"video_id": "b", "video": "b.mp4"}],
                      "format": "VBench-2.0/VBench2_full_info.json"}


@pytest.mark.parametrize("row, message", [
    (vb2("mechanics", ["q"]), r"records\[0\].dimension must be one VBench-2.0 retained dimension"),
    (vb2(["Mechanics", "Material"], ["q"]), "must be one VBench-2.0 retained dimension"),
    (vb2("Mechanics", []), r"records\[0\].auxiliary_info must be a non-empty list of strings"),
    (vb2("Mechanics", ["q", " "]), "auxiliary_info must be a non-empty list of strings"),
    (vb2("Mechanics", "q"), "auxiliary_info must be a non-empty list of strings"),
    (vb2("Motion_Order_Understanding", ["a"]), "^Motion_Order_Understanding auxiliary_info must contain exactly two actions$"),
])
def test_adapt_vbench2_errors(row, message):
    with pytest.raises(adapters.InputSchemaError, match=message):
        adapters.adapt_vbench2([row])


def test_adapters_other_formats():
    rows = [{"video_id": "a", "video": "a.mp4", "first_frame": "f.png", "text_first_frame": "t", "text_instruction": "i", "domain": "d"}]
    assert adapters.adapt_worldmodelbench(rows)["official"] == [{"text_first_frame": "t", "text_instruction": "i", "first_frame": "f.png", "domain": "d"}]
    motion_rows = [{"video_id": "a", "video": "a.mp4", "prompt": "p", "object_1": "dog", "d_1": "Left", "object_2": "", "d_2": ""}]
    assert adapters.adapt_motion_binding(motion_rows)["official"] == [{"prompt": "p", "object_1": "dog", "d_1": "left", "object_2": "", "d_2": ""}]
    with pytest.raises(adapters.InputSchemaError, match="duplicate video_id: a"):
        adapters.adapt_object_interactions([{"video_id": "a", "video": "a", "prompt": "p"}, {"video_id": "a", "video": "b", "prompt": "p"}])
    assert adapters.adapt_videophy_csv([{"videopath": "a", "caption": "c"}], task="sa") == [{"caption": "c", "videopath": "a"}]
    assert adapters.validate_phygen_eval([{"prompt": "p"}], stage="prompts") == {"official": [{"prompt": "p"}], "format": "PhyGenBench/PhyGenEval/prompts"}


def test_adapters_main_writes_and_refuses_overwrite(tmp_path, capsys):
    source = tmp_path / "in.json"
    source.write_text(json.dumps([{"video_id": "a", "video": "a.mp4", "prompt": "猫", "phrase_0": ["x", "y"], "phrase_1": ["z", "w"]}]))
    out, video_map = tmp_path / "o" / "official.json", tmp_path / "o" / "map.json"
    argv = ["--adapter", "action_binding", "--input", str(source), "--output", str(out), "--video-map-output", str(video_map)]
    assert adapters.main(argv) == 0
    assert out.read_text() == dumps([{"prompt": "猫", "phrase_0": ["x", "y"], "phrase_1": ["z", "w"]}])
    assert video_map.read_text() == dumps([{"video_id": "a", "video": "a.mp4"}])
    assert adapters.main(argv) == 1
    assert capsys.readouterr().out == f"Input adaptation failed: refusing to overwrite existing output: {out}\n"


def test_adapters_main_rejects_non_finite_json(tmp_path, capsys):
    source = tmp_path / "in.json"
    source.write_text('[{"prompt": "p", "score": NaN}]')
    out = tmp_path / "official.json"
    assert adapters.main(["--adapter", "phygen_prompts", "--input", str(source), "--output", str(out)]) == 1
    assert capsys.readouterr().out == f"Input adaptation failed: cannot write {out}: Out of range float values are not JSON compliant: nan\n"
    assert not out.exists()


def test_adapters_main_videophy_csv(tmp_path, capsys):
    source = tmp_path / "in.json"
    source.write_text(json.dumps([{"videopath": "a.mp4"}]))
    out = tmp_path / "pc.csv"
    assert adapters.main(["--adapter", "videophy_pc", "--input", str(source), "--output", str(out)]) == 0
    assert out.read_bytes() == b"videopath\r\na.mp4\r\n"
    assert adapters.main(["--adapter", "videophy_pc", "--input", str(source), "--output", str(out)]) == 1
    assert capsys.readouterr().out == f"Input adaptation failed: refusing to overwrite existing output: {out}\n"


# --- build_reference_manifest ---------------------------------------------

def frames(tmp_path):
    generated, ref = tmp_path / "gen", tmp_path / "ref"
    for root, names in ((generated, ("0001.png", "0002.PNG", "notes.txt")), (ref, ("0001.png", "0003.png"))):
        root.mkdir()
        for name in names:
            (root / name).write_bytes(b"x")
    return generated.resolve(), ref.resolve()


def test_build_pairs(tmp_path):
    generated, ref = frames(tmp_path)
    assert reference.build_pairs(generated, ref) == {
        "protocol": "identical relative frame filenames; no alignment or synthesis",
        "alignment_verified": False,
        "warning": "Matching filenames do not establish temporal or geometric alignment or prove real future GT; verify these separately before scoring.",
        "generated_root": str(generated), "reference_root": str(ref),
        "matched": [{"relative_path": "0001.png", "generated": str(generated / "0001.png"), "reference": str(ref / "0001.png")}],
        "unmatched_generated": ["0002.PNG"], "unmatched_reference": ["0003.png"],
        "counts": {"generated": 2, "reference": 2, "matched": 1}}


def test_build_pairs_errors(tmp_path):
    generated, ref = frames(tmp_path)
    with pytest.raises(ValueError, match="must be different"):
        reference.build_pairs(generated, generated)
    (generated / "inner").mkdir()
    with pytest.raises(ValueError, match="must not contain one another"):
        reference.build_pairs(generated, generated / "inner")
    with pytest.raises(ValueError, match="Both frame inputs must be directories"):
        reference.build_pairs(generated / "0001.png", ref)
    with pytest.raises(ValueError, match="At least one non-empty frame extension"):
        reference.build_pairs(generated, ref, {"."})
    with pytest.raises(ValueError, match="at least one supported image"):
        reference.build_pairs(generated, ref, {"jpg"})
    with pytest.raises(FileNotFoundError):
        reference.build_pairs(tmp_path / "none", ref)


def test_build_reference_main(tmp_path, capsys):
    generated, ref = frames(tmp_path)
    out = tmp_path / "o" / "pairs.json"
    argv = ["--generated-frames", str(generated), "--reference-frames", str(ref), "--output", str(out)]
    assert reference.main(argv) == 0
    expected = reference.build_pairs(generated, ref)
    assert out.read_text() == dumps(expected)
    assert capsys.readouterr().out == dumps(expected)
    assert reference.main(argv) == 1
    assert capsys.readouterr().err == f"Reference manifest failed: Output already exists: {out}\n"
    assert reference.main(argv[:-1] + [str(tmp_path / "x.json"), "--require-complete"]) == 1
    assert "not complete matches" in capsys.readouterr().err
    assert reference.main(["--generated-frames", str(generated / "0001.png"), "--reference-frames", str(ref), "--output", str(tmp_path / "y.json")]) == 1
    assert capsys.readouterr().err == "Reference manifest failed: Both frame inputs must be directories\n"
    missing = tmp_path / "none"
    assert reference.main(["--generated-frames", str(missing), "--reference-frames", str(ref), "--output", str(tmp_path / "z.json")]) == 1
    assert capsys.readouterr().err.startswith("Reference manifest failed: [Errno 2] No such file or directory")


# --- verify_install -------------------------------------------------------

def test_verify_install_smoke_and_weights(tmp_path):
    assert verify_install.check_smoke() == "JSON parsing, temporal matching and grounded event scoring passed"
    with pytest.raises(ValueError, match="Missing checkpoint"):
        verify_install.check_weights("motion", tmp_path)
    (tmp_path / "amt_model").mkdir()
    (tmp_path / "amt_model" / "amt-s.pth").write_bytes(b"x")
    assert verify_install.check_weights("motion", tmp_path) == f"Checkpoint files present: {tmp_path}"
