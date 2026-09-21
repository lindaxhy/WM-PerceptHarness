> Historical server-specific operating record (2026-09-20). For current public status, see the [metric catalog](../../metrics/README.md). Relative file references in this record refer to the original `docs/metrics/` directory.

# DOVER 官方单视频融合分接入

本次只准备包装脚本和本地契约测试，**尚未安装 DOVER 环境、下载权重或完成真实视频推理**。`--check-only` 成功也只表示文件与依赖版本检查通过，不能作为评分完成的证据。

## 固定的评分协议

- 官方仓库：[VQAssessment/DOVER](https://github.com/VQAssessment/DOVER/tree/f1ddc96215bc7fbcf8f315c65d47905f339c3419)，固定提交 `f1ddc96215bc7fbcf8f315c65d47905f339c3419`。
- 包装调用原样的 `evaluate_one_video.py -o dover.yml -v VIDEO -d DEVICE -f`。`-f` 是官方融合分开关；不带它时是与仓库参考数据库比较的相对排名，不能混用。
- 使用默认 `dover.yml` 的 `val-l1080p` 采样配置。包装不改采样、分辨率、模型、融合公式或随机数设置。
- DOVER 是整段视频观感质量。融合分范围 `[0,1]`、越高越好；它不是动作成功率或物理正确率。
- 源码验证使用固定提交的 17 个文件（全部运行时 Python 文件、入口、默认配置、requirements、setup）的 SHA-256，并只归一化 CRLF/LF。因此 Windows 检出目录与不带 `.git` 的官方 tar 都可核验；不允许混入额外 `dover/**/*.py`。

## 两个本地权重都必须提前具备

| 资源 | 官方默认加载位置 | 来源 |
|---|---|---|
| 完整 DOVER 模型 | `DOVER/pretrained_weights/DOVER.pth` | [作者 README 指定权重](https://huggingface.co/teowu/DOVER/resolve/main/DOVER.pth) |
| ConvNeXt 初始化模型 | `$TORCH_HOME/hub/checkpoints/convnext_tiny_1k_224_ema.pth` | [官方代码指定的 Meta 权重](https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth) |

第二项容易遗漏：`dover/models/evaluator.py` 构造 aesthetic 分支时固定调用 `convnext_3d_tiny(pretrained=True)`；其内部先执行 `torch.hub.load_state_dict_from_url`，随后才加载完整 DOVER checkpoint。即使已经有 `DOVER.pth`，缺少 ConvNeXt 缓存仍会触发官方代码下载。

包装在启动官方程序前要求两份文件都已存在且非空，设置显式 `TORCH_HOME`，避免这个已知自动下载路径。它本身不下载任何文件，也不是操作系统级网络隔离器。不要在运行期间移动或删除缓存。

两份文件均计算并保存完整 SHA-256；可传 `--checkpoint-sha256` 与 `--convnext-sha256` 对照可信获取记录。未提供预期哈希时，只能证明本次用了哪些字节，不能声称哈希已与作者公布值核对。本次没有下载权重，因此没有编造“官方正确 SHA-256”。保存下载 URL、后端、目标路径和获取日期；续传沿用同一后端与 partial/cache。

## Ali02 的环境选择

Ali02 现有 RTX 5090 属于 `sm_120`。原仓库 `requirements.txt` 的 `torch~=1.13` 对应旧 PyTorch 系列，旧 CUDA wheel 不支持这张显卡。**不要在现有 Torch 2.10 评测环境中执行会降级环境的 DOVER 安装命令，也不要认为安装好 Torch 1.13 就能在 RTX 5090 上运行。**

可选路线：

1. 建独立 Python 3.10 / Torch 1.13.1 / torchvision 0.14.1 **CPU** 环境，先做文件检查和单视频 CPU 冒烟。选择 Python 3.10 是因为官方分发提供这组 wheel，且本包装要求 Python 3.10+；不是声称 DOVER 作者只支持 3.10。[PyTorch 对应版本安装说明](https://docs.pytorch.org/get-started/previous-versions/)、[官方 CPU torchvision wheel 清单](https://download.pytorch.org/whl/cpu/torchvision/)。
2. 另建支持 RTX 5090 的现代 Torch 环境，显式使用 `--allow-unverified-runtime` 做兼容实验。该环境不满足原 DOVER 的 Torch 版本约束，需要单独解决依赖并和基线对照；结果标记 `scored_unverified_runtime`。本包没有提供已经验证成功的现代 GPU 安装组合。

官方入口确实提供 `-d cpu`。源码中 Swin 的 `global_position_index` 实际被传入 `x.device`；另一默认 CUDA helper 的调用在 `if False` 分支。未发现默认 DOVER 推理路径必须使用 CUDA，但这只是源码检查，**不代表 CPU 冒烟已经成功**。

官方 `requirements.txt` 原样包含：`torch~=1.13`、`torchvision`、`opencv-python`、`decord`、`matplotlib`、`scipy`、`numpy`、`tqdm`、`timm`、`einops`、`wandb`、`scikit-video`、`thop==0.0.31-2005241907`、`onnx`。入口还直接 `import yaml`，应显式安装 `PyYAML`。大部分依赖没有锁版本，因此不能把今天一次 `pip install -e .` 视作作者环境的完整复现。安装成功后保存 `pip freeze` 与 `pip check`；遇到不兼容停止并记录，不修改既有评测环境。

下面是独立 CPU 环境的**待验证安装起点**，不是已通过的 lockfile。`numpy<2`、旧 timm 范围是兼容旧代码的约束建议，需通过冒烟后再冻结确切版本。

```bash
# 必须先进入持久会话，确认当前 shell 在 tmux 中。
ssh Ali02
tmux -L official_metrics has-session -t official-metrics-20260918
tmux -L official_metrics attach -t official-metrics-20260918
test -n "$TMUX" || exit 1

# 在 tmux 内运行。源码需事先由现有 D 盘归档上传并解压到此处。
export DOVER_ROOT=/root/official_video_metrics_20260918/sources/DOVER
export DOVER_ENV=/root/official_video_metrics_20260918/venvs/dover-cpu
python3.10 -m venv "$DOVER_ENV"
source "$DOVER_ENV/bin/activate"
python -m pip install 'torch==1.13.1+cpu' 'torchvision==0.14.1+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu
cat > /root/official_video_metrics_20260918/dover-cpu-constraints.txt <<'EOF'
torch==1.13.1+cpu
torchvision==0.14.1+cpu
numpy<2
timm<0.7
EOF
python -m pip install -c /root/official_video_metrics_20260918/dover-cpu-constraints.txt \
  -r "$DOVER_ROOT/requirements.txt" PyYAML
python -m pip install --no-deps --no-build-isolation -e "$DOVER_ROOT"
python -m pip check
python -m pip freeze > /root/official_video_metrics_20260918/dover-cpu-freeze.txt
```

若持久会话不存在，应先创建并验证，再安装或下载；若 `tmux` 未安装，先准备持久机制。分离按 `Ctrl+B`、再按 `D`；重新连接使用同一个 `tmux -L official_metrics attach -t official-metrics-20260918`。本次没有远程执行上述安装。

本地已有源码归档：`D:\北大相关\ai\WM评测\评测指标官方代码包_20260918\archives\DOVER-f1ddc96215bc.tar.gz`，SHA-256 为 `a2548cb38ea957f990f5a260643afd0348684810913ed5a9ac847993c64f2d50`。归档内顶层目录是 `DOVER/`。上传后先核对哈希，再解压到单独 `sources` 目录，保留 LICENSE；本任务未将整个官方仓库复制进 PR。

如完整权重实际在模型盘，可在确认目标尚不存在后创建符号链接：

```bash
mkdir -p "$DOVER_ROOT/pretrained_weights"
ln -s /root/official_video_metrics_20260918/weights/DOVER/DOVER.pth \
  "$DOVER_ROOT/pretrained_weights/DOVER.pth"
```

这个命令不会下载权重；若目标已存在，不覆盖。ConvNeXt 文件应放在下例 `TORCH_HOME/hub/checkpoints/`。

## 先检查，再做一条真实视频的冒烟

以下 `/data/example.mp4` 和仓库路径需要替换为已核验的实际路径；使用 DOVER 专用环境的 Python。从本 PR 仓库根目录执行：

```bash
python scripts/score_dover_official.py \
  --dover-root /root/official_video_metrics_20260918/sources/DOVER \
  --video /data/example.mp4 \
  --torch-home /root/official_video_metrics_20260918/cache/dover-torch \
  --device cpu \
  --check-only \
  --output-dir /root/official_video_metrics_20260918/results/dover_preflight_001
```

缺权重、源码不一致或依赖缺失都会写入 `result.json` 并以非零返回码退出。准备齐全后去掉 `--check-only`，同时换成全新的 `--output-dir .../dover_smoke_001`；不要复用检查目录。真实推理仍应在 tmux 内运行。

输出目录包含：

- `result.json`：状态、输入视频 SHA-256、两份权重 SHA-256、源码版本、依赖版本、完整命令、设备及日志路径。
- `dover.log`：官方 stdout/stderr 原文；仅实际启动官方推理后才存在。

仅当官方进程返回 0，日志恰有一行 `Normalized fused overall score (scale in [0,1]): <number>`，数值有限且在 `[0,1]`，且运行前后视频/权重/源码未变时，包装才接受分数。缺失或无法唯一解析时保留 `score: null`、状态 `official_finished_score_unparsed` 和原日志；不会猜数值或自行代入融合公式。

`preflight_passed_not_scored` 没有分数。`scored` 表示官方入口本次正常产出数字，不额外保证裁判在我们的分布上已经验证。`scored_unverified_runtime` 还表示使用了超出 Torch 1.13 基线的环境。统计表只纳入明确的有效结果，并保留失败数量。

现有官方脚本无批量结构化结果接口。本包装一条视频一个进程与输出目录，尚未做吞吐量或 CPU 用时测试。先保留一条真实成功日志、依赖冻结文件和加载耗时，再考虑批量；不要提前宣称 DOVER 已完成服务器部署或全部视频评分。

## 本地验证范围

```bash
python -m unittest discover -s tests -p 'test_dover_official_wrapper.py' -v
python scripts/score_dover_official.py --help
```

测试使用模拟官方进程，覆盖精确解析、旧输出拒绝、缺权重阻断、哈希不一致、非零返回码、仅检查模式和源码变更。它们不验证模型权重可加载、解码正确性、GPU/CPU 数值、评分质量或官方依赖安装成功。
