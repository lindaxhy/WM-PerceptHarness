# Standalone metric source packages

These five directories contain the reviewed source packages and experiment
adaptations for metrics that already have entries in the harness. Each directory
is independently usable with its documented runtime and external assets. They
are source directories, not ZIP files or a shared Python environment, and are
not installed by `pip install -e .`.

Every evaluator is also exposed as a `percept score` subcommand; the command
locates the runner below, starts it with the evaluator environment's interpreter
and keeps the package output protocol. Keep the existing `percept` commands for
their documented workflows, and use these packages when you need the additional
runners or explicitly selected modified protocols below. Their scores must
retain the protocol name and configuration.

| Package | Official code and existing harness support | Additional code in this directory | Main limits |
| --- | --- | --- | --- |
| [01_semantic_adherence](01_semantic_adherence/README.md) | Pinned VideoPhy-2 source; main has CSV preparation and an external launcher | Independent SA inference; official 32-frame mode or `min(max(ceil(N/2),32),256)` sampling; raw responses and sampling records | Requires a frozen target caption; added frames have no established accuracy benefit; official answer parser can return 0 |
| [02_physical_commonsense](02_physical_commonsense/README.md) | Same VideoPhy-2 source and fixed physical-commonsense question | Independent PC inference with the same selectable sampling protocols | Physical plausibility does not establish task completion; judge reliability and added-frame benefit remain unvalidated |
| [06_psnr](06_psnr/README.md) | Unchanged IQA-PyTorch RGB PSNR kernel; main only prepares reference pairs and launches external scoring | Source-PTS matching, real conditioning-image identity checks, native Wan/Dense geometry and exclusion of generated frame 0 | Requires matched source videos; protocol is limited to documented Wan/Dense geometry and 24 fps; valid timing/trajectory changes can lower scores |
| [07_dynamic_degree](07_dynamic_degree/README.md) | Unchanged VBench 0.1.5 wheel baseline; main already provides official scoring | Explicit `official` or `cpu-staging` modes; modified mode keeps decoded inputs on CPU and transfers the current frame pair | Camera motion affects the Boolean motion threshold; input staging is not streaming decoding and does not reduce model memory |
| [08_motion_smoothness](08_motion_smoothness/README.md) | Unchanged VBench 0.1.5 / AMT-S baseline; main already provides official scoring | Explicit `official` or `cpu-staging` modes; modified mode keeps input tensors on CPU | Smoothness is not action correctness; CPU resize can differ numerically; long videos still need CPU memory |

Package numbers retain the delivered layout. The missing `03`, `04` and `05`
are intentional: Motion Rationality, Object Interactions and CLIPScore are
excluded. CLIPScore overlaps the Semantic Adherence measurement, and Object
Interactions requires a target interaction that only some samples define. The
SA/PC sampler's unused MR branch and context-budget helper have also been
removed; VideoPhy sampling and scoring are unchanged by that cleanup. Action
Binding, historical candidates, old comparison directories and already adequate
MUSIQ/Subject Consistency packages are outside this addition.

## Run a package

Each evaluator has a matching `percept score` command:

| Evaluator | Command |
| --- | --- |
| Semantic Adherence | `percept score semantic-adherence` |
| Physical Commonsense | `percept score physical-commonsense` |
| PSNR | `percept score psnr` |
| Dynamic Degree | `percept score dynamic-degree` |
| Motion Smoothness | `percept score motion-smoothness` |

The command locates the runner in this directory, starts it with `--python`
(default: the current interpreter) and forwards the evaluator options. For
example:

```bash
percept score semantic-adherence --python /path/to/videophy-env/bin/python \
  --input-csv /data/input.csv --checkpoint /path/to/videophy_2_auto \
  --output /data/results/sa
```

Use `percept score <evaluator> --help` for the options and `--dry-run` to print
the exact runner command. `--metric-packages-root` (or `PERCEPT_METRIC_PACKAGES`)
selects another checkout.

The directories remain directly runnable: read the package README, prepare a
separate compatible environment and the exact local weights/checkouts, then run
from that package directory. Input files under `examples/` are synthetic
placeholders. No videos, checkpoints or private evaluation outputs are included.
Several `official/` directories are selected source subsets, not complete
installable upstream repositories.

Before remote installation, downloads or inference, enter and verify a
persistent session:

```bash
tmux new-session -s metric-run
test -n "$TMUX"
tmux display-message -p '#S'
```

Detach with `Ctrl-b`, then `d`; reattach with `tmux attach -t metric-run`.
Preserve existing caches, partial files, destination paths, authentication
environment and download backend when resuming.

## Provenance and validation

Each package has a `MANIFEST.json` covering its files, plus upstream provenance
and retained licenses. `.gitattributes` preserves exact bytes because some
runners validate source hashes. Official files are kept separate from project
adaptations. SA and PC, and the two VBench wheel-based packages, deliberately
repeat necessary files so each directory can be copied independently.

Repository tests check file hashes, complete manifest coverage, command help
without model imports, SA/PC sampling boundaries, and the `percept score`
dispatch and argument forwarding. These checks do not install the metric
environments, load models or demonstrate score accuracy. This source submission
has not repeated real CPU/GPU scoring, fresh metric-environment installation or
full-dataset validation. Recorded earlier experiments in the individual READMEs
are not validation of a new installation.

Third-party source and model licenses remain separate from this repository's
MIT license. In particular, the PSNR package retains IQA-PyTorch's PolyForm
Noncommercial and S-Lab terms. See [third-party notices](../docs/third-party-notices.md#vendored-standalone-metric-packages).
