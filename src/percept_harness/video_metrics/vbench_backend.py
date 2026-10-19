#!/usr/bin/env python3
"""Shared backend for individual metrics using the pinned VBench implementation.

No video preprocessing or score formulas are implemented here. Run inference
inside tmux. --check-only checks inputs, source and pre-provisioned model files
without importing GPU packages, downloading anything, or creating an output.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from percept_harness.video_metrics._runtime import (
    VBENCH_REVISION, finite_number, package_versions, run_cli, run_vbench, single_row, subprocess_env, verify_source,
    write_json_exclusive,
)

SOURCE_FILES = ("evaluate.py", "vbench/__init__.py")
DIRECT_DIMENSIONS = frozenset({
    "imaging_quality", "aesthetic_quality", "temporal_flickering", "dynamic_degree",
    "subject_consistency", "background_consistency", "overall_consistency",
})
REQUIRED_FILES = {
    "imaging_quality": ("pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",),
    "aesthetic_quality": ("clip_model/ViT-L-14.pt", "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth"),
    "background_consistency": ("clip_model/ViT-B-32.pt",),
    "dynamic_degree": ("raft_model/models/raft-things.pth",),
    "overall_consistency": ("ViCLIP/ViClip-InternVid-10M-FLT.pth", "ViCLIP/bpe_simple_vocab_16e6.txt.gz"),
    "subject_consistency": (
        "dino_model/dino_vitbase16_pretrain.pth",
        # The official VBench call is torch.hub.load(..., source="local")
        # with this repository and checkpoint.
    ),
    "temporal_flickering": (),
}

DINO_REPOSITORY_PREFIX = "dino_model/facebookresearch_dino_main"
DINO_TORCH_CHECKPOINT_NAME = "dino_vitbase16_pretrain.pth"


def _object_without_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=_object_without_duplicates)


def resolve_prompt(video: Path, prompt: str | None, prompt_file: Path | None) -> str | None:
    if prompt is not None and prompt_file is not None:
        raise ValueError("--prompt and --prompt-file are mutually exclusive")
    if prompt_file is not None:
        prompt_file = prompt_file.expanduser().resolve(strict=True)
        mapping = read_json(prompt_file)
        if not isinstance(mapping, dict) or len(mapping) != 1:
            raise ValueError("Single-video --prompt-file must contain exactly one video path -> prompt entry")
        key, prompt = next(iter(mapping.items()))
        key_path = Path(key).expanduser()
        if not key_path.is_absolute():
            key_path = prompt_file.parent / key_path
        if key_path.resolve() != video:
            raise ValueError("Prompt file key does not match the requested video")
    if prompt is not None and (not isinstance(prompt, str) or not prompt.strip() or prompt.strip() == "None"):
        raise ValueError("Prompt must be a non-empty original task instruction, not the reserved literal 'None'")
    return prompt


def verify_cache(cache: Path, dimensions: list[str], manifest: Path | None,
                 torch_home: Path | None = None) -> dict:
    required = set().union(*(set(REQUIRED_FILES[dimension]) for dimension in dimensions))
    if manifest is not None:
        entries = read_json(manifest)
        if not isinstance(entries, (list, dict)) or any(not isinstance(item, str) for item in entries):
            raise ValueError("Weights manifest must be a JSON list of cache-relative paths")
        for relative in entries:
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Weights manifest path must be cache-relative: {relative}")
            required.add(relative)
    records = {}
    for relative in sorted(required):
        path = cache / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Pre-provision model/cache file; automatic downloads are not started: {path}")
        records[relative] = str(path)
    if "subject_consistency" in dimensions:
        if torch_home is None:
            raise ValueError("subject_consistency requires an explicit Torch hub cache path")
        repo = cache / DINO_REPOSITORY_PREFIX
        if not (repo / "hubconf.py").is_file():
            raise FileNotFoundError(f"Pre-provision official DINO source checkout is missing: {repo}")
        records[DINO_REPOSITORY_PREFIX] = str(repo)
        checkpoint = torch_home / "hub" / "checkpoints" / DINO_TORCH_CHECKPOINT_NAME
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise FileNotFoundError(f"Pre-provision official DINO Torch hub checkpoint is missing: {checkpoint}")
        records[f"torch_home/hub/checkpoints/{DINO_TORCH_CHECKPOINT_NAME}"] = str(checkpoint)
    return records


def validate_result(path: Path, video: Path, dimensions: list[str]) -> dict:
    data = read_json(path)
    if not isinstance(data, dict) or set(data) != set(dimensions):
        raise ValueError("Official result dimensions do not exactly match the request")
    scores = {}
    for dimension in dimensions:
        item = data[dimension]
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError(f"Invalid official result structure: {dimension}")
        aggregate, rows = item
        row = single_row(rows, f"Expected exactly one video result: {dimension}")
        if not isinstance(row.get("video_path"), str) or Path(row["video_path"]).resolve() != video:
            raise ValueError(f"Official result refers to a different video: {dimension}")
        raw = row.get("video_results")
        aggregate = finite_number(aggregate, f"Expected a finite numeric {dimension} aggregate")
        if dimension == "dynamic_degree":
            # Official RAFT path emits a JSON Boolean, aggregate is 0.0/1.0.
            if not isinstance(raw, bool):
                raise ValueError("Official single-video Dynamic Degree must be Boolean")
            numeric = float(raw)
        else:
            numeric = finite_number(raw, f"Expected a finite numeric {dimension} video score")
        expected = numeric / 100.0 if dimension == "imaging_quality" else numeric
        if not math.isclose(aggregate, expected, rel_tol=1e-7, abs_tol=1e-9):
            raise ValueError(f"Official aggregate/video ratio mismatch: {dimension}")
        # MUSIQ and the aesthetic linear regressor have unbounded raw outputs;
        # do not silently clip these scores or impose a fabricated [0, 1] range.
        if dimension in ("subject_consistency", "background_consistency", "temporal_flickering", "dynamic_degree"):
            if not -1e-6 <= aggregate <= 1.000001:
                raise ValueError(f"Invalid normalized score: {dimension}")
        if dimension == "overall_consistency" and not -1.000001 <= aggregate <= 1.000001:
            raise ValueError("ViCLIP cosine score must lie in [-1, 1]")
        scores[dimension] = {"score": aggregate, "official_video_score": raw,
                             "higher_is_better": None if dimension == "dynamic_degree" else True}
    return scores


def run_official(*, vbench_root: Path, videos_path: Path, dimensions: list[str], output_dir: Path,
                 mode: str = "custom_input", prompt: str | None = None, prompt_file: Path | None = None,
                 cache_dir: Path | None = None, torch_home: Path | None = None, gpu: str | None = None,
                 imaging_quality_preprocessing_mode: str = "longer", load_ckpt_from_local: bool = True,
                 static_subset_ack: bool = False, weights_manifest: Path | None = None,
                 allow_trusted_pickle: bool = False, check_only: bool = False) -> dict:
    root = vbench_root.expanduser().resolve(strict=True)
    video = videos_path.expanduser().resolve(strict=True)
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    if output == root or root in output.parents:
        raise ValueError("Keep evaluation output outside the pinned source tree")
    if not video.is_file() or video.suffix != ".mp4":
        raise ValueError("This wrapper requires one lowercase .mp4 file, not a directory")
    dimensions = list(dict.fromkeys(dimensions))
    if not dimensions or set(dimensions) - DIRECT_DIMENSIONS:
        raise ValueError(f"Supported dimensions: {', '.join(sorted(DIRECT_DIMENSIONS))}; motion uses its already-submitted wrapper")
    if mode != "custom_input" or not load_ckpt_from_local:
        raise ValueError("This single-video wrapper requires custom_input and local checkpoints")
    if imaging_quality_preprocessing_mode not in ("shorter", "longer", "shorter_centercrop", "None"):
        raise ValueError("Invalid imaging-quality preprocessing mode")
    actual_prompt = resolve_prompt(video, prompt, prompt_file)
    if "overall_consistency" in dimensions and actual_prompt is None:
        raise ValueError("overall_consistency requires --prompt or a one-video --prompt-file")
    if "temporal_flickering" in dimensions and not static_subset_ack:
        raise ValueError("temporal_flickering requires --static-subset-ack after checking the static/near-static input")
    if gpu is not None and (not gpu.strip() or "," in gpu):
        raise ValueError("Select one GPU index or UUID for this single-process wrapper")
    if cache_dir is None:
        raise ValueError("Specify an explicit --cache-dir")
    cache = cache_dir.expanduser().resolve(strict=True)
    if torch_home is None:
        torch_home = cache / "torch"
    else:
        torch_home = torch_home.expanduser().resolve()
    if torch_home == root or root in torch_home.parents:
        raise ValueError("Keep the Torch hub cache outside the pinned source tree")
    if weights_manifest is not None:
        weights_manifest = weights_manifest.expanduser().resolve(strict=True)
    source = verify_source(root, SOURCE_FILES, {"expected_revision": VBENCH_REVISION})
    weights = verify_cache(cache, dimensions, weights_manifest, torch_home=torch_home)
    if allow_trusted_pickle and not weights:
        raise ValueError("--allow-trusted-pickle requires local model files")
    env = subprocess_env(
        venv_path=True, alloc_conf=True, gpu=gpu,
        # Never inherit a request to enable unrestricted pickle loading.
        unset=("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",) + (("TORCH_FORCE_WEIGHTS_ONLY_LOAD",) if allow_trusted_pickle else ()),
        values={"VBENCH_CACHE_DIR": str(cache), "TORCH_HOME": str(torch_home), "PYTHONPATH": str(root),
                "PYTHONNOUSERSITE": "1",
                **({"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"} if allow_trusted_pickle else {}),
                # Do not inherit a torchrun allocation and accidentally wait for other ranks.
                "RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "MASTER_ADDR": "127.0.0.1"})
    command = [sys.executable, str(root / "evaluate.py"), "--videos_path", str(video),
               "--dimension", *dimensions, "--mode", "custom_input", "--output_path", str(output),
               "--load_ckpt_from_local", "True", "--imaging_quality_preprocessing_mode", imaging_quality_preprocessing_mode]
    # Official single-file handling indexes prompt_list[0], so pass the resolved
    # prompt as a string, never the official --prompt_file dictionary path.
    if actual_prompt is not None:
        command += ["--prompt", actual_prompt]
    metadata = {"wrapper": "score_vbench_official.py", "source": source, "video": str(video),
                "dimensions": dimensions, "mode": "custom_input",
                "prompt": actual_prompt, "cache_dir": str(cache), "weights": weights,
                "torch_home": env["TORCH_HOME"], "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
                "static_subset_ack": static_subset_ack, "allow_trusted_pickle": allow_trusted_pickle,
                "imaging_quality_preprocessing_mode": imaging_quality_preprocessing_mode,
                "command": command, "status": "preflight_passed"}
    if check_only:
        return metadata
    output.mkdir(parents=True, exist_ok=False)
    log_path = output / "vbench.log"
    print(f"Official VBench output and log: {output}", file=sys.stderr)
    write_json_exclusive(output / "wrapper_request.json", metadata, newline=False)
    result_path = run_vbench(command, log=log_path, output=output, label="Official VBench", cwd=root, env=env,
                             missing=ValueError(f"Expected exactly one official evaluation JSON; see {output}"))
    scores = validate_result(result_path, video, dimensions)
    versions = package_versions(("torch", "torchvision", "pyiqa", "openai-clip", "decord", "numpy"))
    metadata.update(status="complete", scores=scores, versions=versions,
                    official_result=str(result_path), log=str(log_path))
    write_json_exclusive(output / "wrapper_metadata.json", metadata)
    return metadata


def metric_main(dimension: str, argv: list[str] | None = None) -> int:
    if dimension not in DIRECT_DIMENSIONS:
        raise ValueError(f"Unsupported metric: {dimension}")
    name = dimension.replace("_", "-")
    parser = argparse.ArgumentParser(
        prog=f"percept {name}",
        description=f"Score {name} using the pinned official VBench implementation.",
    )
    parser.set_defaults(dimensions=[dimension])
    parser.add_argument("--vbench-root", type=Path, required=True)
    parser.add_argument("--videos-path", type=Path, required=True, help="One lowercase .mp4 file")
    parser.add_argument("--mode", choices=("custom_input",), default="custom_input")
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--torch-home", type=Path, help="Writable Torch hub cache for official loaders such as DINO; defaults to <cache-dir>/torch")
    parser.add_argument("--weights-manifest", type=Path, help="Optional JSON list of cache-relative model/code paths")
    parser.add_argument("--gpu", help="One CUDA_VISIBLE_DEVICES index or UUID")
    parser.add_argument("--output-dir", type=Path, required=True, help="New result directory outside source checkout")
    parser.add_argument("--imaging-quality-preprocessing-mode", choices=("shorter", "longer", "shorter_centercrop", "None"), default="longer")
    parser.add_argument("--load-ckpt-from-local", action="store_true", default=True, help="Always enabled by this wrapper")
    parser.add_argument("--static-subset-ack", action="store_true")
    parser.add_argument("--allow-trusted-pickle", action="store_true", help="Only for trusted checkpoints requiring legacy torch.load")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    return run_cli(name, lambda: run_official(**vars(args)))
