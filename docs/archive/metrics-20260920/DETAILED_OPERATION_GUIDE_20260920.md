> Historical server-specific operating record (2026-09-20). For current public status, see the [metric catalog](../../metrics/README.md). Relative file references in this record refer to the original `docs/metrics/` directory.

# 指标接入、Smoke Test 与 PR 操作指南

更新时间：2026-09-20

这份指南解决三个容易混淆的问题：

1. **代码 PR** 与 **指标已经可以稳定评分** 是两个不同的完成阶段。
2. 模型权重、输入资料和运行环境不一定都要提交到 Git，但必须明确由谁准备、放在哪里、如何核验。
3. Smoke test 不要求先跑完整 75 条；每个指标先用一条符合官方输入的真实视频完成一次官方入口试跑即可。

最终指标范围只以同目录的 `FINAL_METRICS.md` 和 `FINAL_METRICS.json` 为准。

## 一、先区分两种“完成”

### A. 可以提交代码 PR

满足以下条件，就可以提交“接入准备”PR：

- 使用固定版本的官方源码；
- 没有改写官方评分公式、采样方式、模型或裁判；
- 输入字段名称和官方一致；
- 缺少权重或环境时会明确报错，而不是自动换模型；
- 有 `--check-only`、输入校验或契约测试；
- 文档明确说明哪些资源尚未准备。

当前分支已经达到这个标准。当前 PR 可以包含：

- VBench 基础指标官方调用包装器；
- VBench 官方 CLIPScore 包装器；
- DOVER 官方命令预检包装器；
- WorldModelBench、T2V-CompBench、VBench-2.0、PhyGenBench、VideoPhy-2 输入校验器；
- 参考帧配对清单和 VideoPhy-2 Joint 分析工具；
- 最终冻结的 21 项主榜 + 5 项固定附表清单。

这种 PR 应写成“official integration/preflight”或“官方接入准备”，不能写成“全部指标已完成”。

### B. 可以声称某个指标已接入并可评分

单个指标要进入“score-ready”状态，至少需要：

1. 固定官方源码版本和 SHA-256；
2. 确认官方权重或官方缓存，记录 SHA-256；
3. 使用与官方兼容的运行环境，记录 `pip freeze` 和 `pip check`；
4. 准备官方要求的输入文件；
5. 用一条真实、可解码的视频运行官方入口；
6. 保存官方原始输出和完整日志；
7. 包装器能够解析输出，且不猜分、不补缺失字段；
8. 把成功命令、环境、权重和结果写入验证记录。

因此，**不是每个指标都必须在代码 PR 之前跑 smoke test**。代码准备 PR 可以先提交；但在 PR 中把某项标记为“已完成/可用”，必须先通过该项 smoke test。

## 二、模型权重、输入文件、环境分别由谁准备

### 项目方必须提供或确认的内容

项目方必须把以下内容写清楚并固定下来：

- 官方源码仓库、commit 和许可证；
- 官方权重名称、下载来源、文件布局和哈希核验方式；
- 官方 Python/CUDA/Torch 等版本要求；
- 官方输入字段、文件命名和目录结构；
- 哪些字段来自原始任务，而不是从生成视频反推；
- 结果文件的字段含义、分数方向和聚合方法；
- 缺少参考数据时是否输出 `N/A`。

项目方**不应把大模型权重、模型缓存和虚拟环境提交进 Git**。项目方需要提供的是可复现的来源、版本、路径约定和命令。

### 用户或运行者准备的内容

运行者通常负责：

- 下载或挂载官方模型权重；
- 创建独立 Python 环境；
- 把视频、prompt、首帧、题目元数据放到指定位置；
- 根据官方格式生成自己的输入清单；
- 运行 smoke test 和正式批量评测；
- 保存结果、日志和环境快照。

这不是把工作推给用户，而是评测项目的正常分层：权重和环境通常很大、需要 GPU/许可证/网络条件，不能随代码一起提交。项目方必须提供可执行指令和失败检查，用户才能按同一官方协议运行。

### 哪些输入不是项目方统一准备的

以下输入必须来自每个使用者自己的实验，不能由项目方凭空生成：

- 生成时实际使用的原始 prompt；
- I2V 任务实际使用的首帧；
- 自定义题目的动作顺序、对象和方向；
- VBench-2.0 的 `auxiliary_info` 问题；
- PSNR/SSIM/LPIPS 的真实未来参考帧；
- FVD/FID 的匹配真实集合。

项目方可以提供官方示例和校验器，但不能用自动 caption、生成结果或复制出的首帧冒充这些输入。

## 三、统一准备目录和持久会话

大文件下载、环境安装和推理必须在持久会话里开始：

```bash
ssh Ali02

tmux -L codexdebug has-session -t official-metrics \
  || tmux -L codexdebug new-session -d -s official-metrics

tmux -L codexdebug attach -t official-metrics
```

建议目录：

```text
/root/official_video_metrics_20260918/
  sources/       # 固定版本的官方源码
  weights/       # 官方权重，不提交 Git
  envs/          # 各指标独立环境
  manifests/     # prompt、首帧、auxiliary_info、视频映射
  results/       # 每次运行新的输出目录
  logs/          # 下载、安装和推理日志
```

分离 tmux：按 `Ctrl+B`，再按 `D`。重新进入必须使用同一个 socket 和会话名。

## 四、第一步：冻结任务输入

准备一个本地映射清单。它是项目内部清单，不等于某个上游官方 metadata 文件：

```json
[
  {
    "video_id": "sample-0001",
    "video": "/data/videos/sample-0001.mp4",
    "prompt": "原始生成 prompt",
    "first_frame": "/data/first_frames/sample-0001.png",
    "dimensions": ["imaging_quality", "clip_score", "overall_consistency"],
    "auxiliary_info": "/data/manifests/sample-0001_vbench2.json"
  }
]
```

运行校验：

```bash
python scripts/validate_metric_manifest.py \
  --manifest /root/official_video_metrics_20260918/manifests/videos.json \
  --dimension overall_consistency \
  --check-files
```

这个工具只检查字段、重复 ID、路径和问题列表，不会生成 prompt、问题或参考帧。

对于 T2V-CompBench、VBench-2.0 等官方文件，先用适配器生成官方字段文件，并把本地视频路径保存在独立 sidecar：

```bash
python scripts/official_input_adapters.py \
  --adapter action_binding \
  --input /root/official_video_metrics_20260918/manifests/action_binding.local.json \
  --output /root/official_video_metrics_20260918/manifests/action_binding.official.json \
  --video-map-output /root/official_video_metrics_20260918/manifests/action_binding.video_map.json
```

不要把 `video`、`video_path` 等非官方字段强行添加到上游 metadata。

## 五、VBench 基础指标：项目方现在可以先做的 smoke test

固定源码：

```text
VBench fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490
```

### 1. Imaging Quality、Temporal Flickering、Dynamic Degree、Subject Consistency、Background Consistency

先做不加载 GPU 的预检：

```bash
python scripts/score_vbench_official.py \
  --vbench-root /root/official_video_metrics_20260918/sources/VBench \
  --videos-path /data/videos/sample-0001.mp4 \
  --dimension imaging_quality \
  --cache-dir /mnt/models/vbench-cache \
  --weights-manifest /root/official_video_metrics_20260918/manifests/imaging_quality.weights.json \
  --output-dir /root/official_video_metrics_20260918/results/imaging_quality_check \
  --check-only
```

预检通过后，去掉 `--check-only` 做一条真实视频运行。每个 dimension 应使用新的输出目录。

对应官方缓存必须与官方代码匹配：

- Imaging Quality：MUSIQ-SPAQ；
- Dynamic Degree：RAFT `raft-things.pth`；
- Subject Consistency：官方 DINO；
- Background Consistency：官方 OpenAI CLIP `ViT-B/32`；
- Temporal Flickering：必须记录静态/近静态子集规则和排除数。

不要用另一个数据集的 MUSIQ 权重、RN50、Qwen2-VL 或其他模型替代这些文件。

### 2. CLIPScore

官方入口固定使用 VBench `competitions/clip_score.py` 和 OpenAI CLIP `ViT-B/32`：

```bash
python scripts/score_vbench_clip_score.py \
  --vbench-root /root/official_video_metrics_20260918/sources/VBench \
  --videos-path /data/videos/sample-0001.mp4 \
  --prompt "原始生成 prompt" \
  --clip-home /root/official_video_metrics_20260918 \
  --output-dir /root/official_video_metrics_20260918/results/clip_score_check \
  --check-only
```

权重必须位于：

```text
/root/official_video_metrics_20260918/.cache/clip/ViT-B-32.pt
```

CLIPScore 和 ViCLIP 都需要原始 prompt，但两者模型和评分公式不同，不能共用结果或分数缩放。

### 3. ViCLIP / Overall Consistency

单视频官方入口的调用形态是：

```bash
cd /root/official_video_metrics_20260918/sources/VBench
python evaluate.py \
  --videos_path /data/videos/sample-0001.mp4 \
  --dimension overall_consistency \
  --mode custom_input \
  --prompt "原始生成 prompt" \
  --output_path /root/official_video_metrics_20260918/results/overall_consistency
```

批量运行前先生成“视频路径 → 原始 prompt”的 JSON，并确认路径和视频 ID 一一对应。不能使用生成视频的自动字幕。

## 六、DOVER：必须独立环境，但不必提交环境本身

固定源码：

```text
DOVER f1ddc96215bc7fbcf8f315c65d47905f339c3419
```

DOVER 的官方 `requirements.txt` 使用 Torch 1.13 系列，而现有评测环境是 Torch 2.10。正确做法是创建独立环境，不要在现有环境里降级 Torch。

项目方需要提供：

- 官方源码版本；
- 官方 `requirements.txt`；
- DOVER.pth 的官方来源和文件位置；
- ConvNeXt 初始化权重的官方来源和缓存位置；
- 下面的预检和 smoke test 命令。

运行者在 Ali02 创建环境：

```bash
python3.10 -m venv /root/official_video_metrics_20260918/envs/dover-cpu
source /root/official_video_metrics_20260918/envs/dover-cpu/bin/activate

python -m pip install \
  'torch==1.13.1+cpu' 'torchvision==0.14.1+cpu' \
  --extra-index-url https://download.pytorch.org/whl/cpu

python -m pip install \
  -r /root/official_video_metrics_20260918/sources/DOVER/requirements.txt \
  PyYAML

python -m pip check
python -m pip freeze \
  > /root/official_video_metrics_20260918/logs/dover-cpu-freeze.txt
```

先预检：

```bash
python scripts/score_dover_official.py \
  --dover-root /root/official_video_metrics_20260918/sources/DOVER \
  --video /data/videos/sample-0001.mp4 \
  --torch-home /root/official_video_metrics_20260918/cache/dover-torch \
  --device cpu \
  --output-dir /root/official_video_metrics_20260918/results/dover_check \
  --check-only
```

准备齐全后，去掉 `--check-only`。只有官方进程返回 0、日志有官方融合分数、结果可解析，DOVER 才能标记为 `scored`。

独立环境的原则是：**环境由运行者创建，环境版本和安装命令由项目方固定并记录**。环境不需要提交进 PR，但必须能按文档重建。

## 七、需要外部裁判模型的指标

这些指标不应由项目方在没有权重时写一个“假入口”并声称完成。项目方提交官方来源、输入适配器和命令；运行者准备裁判模型和独立环境。

本节中的外部命令是依据官方代码包中记录的固定仓库和入口整理的“调用形态”，不是本项目包装器已经在 Ali02 上完成的真实推理。首次运行必须先进入对应官方仓库执行 `python <entry>.py --help`，核对该固定 commit 的参数，再执行命令。若 `--help` 或 README 与下面示例不一致，以固定 commit 的源码为准，不自行改参数名后继续宣称严格复现。

### Instruction Following / WorldModelBench

官方字段必须包含：

- `domain`
- `subdomain`
- `text_first_frame`
- `text_instruction`
- `first_frame`

官方入口形态：

```bash
python /path/to/WorldModelBench/evaluation.py \
  --judge /root/official_video_metrics_20260918/weights/worldmodelbench \
  --video_dir /data/generated_videos \
  --model_name MODEL \
  --save_name /root/official_video_metrics_20260918/results/worldmodelbench
```

`MODEL`、judge 路径和数据工作目录必须按官方仓库 README 设置，不能自行换成小模型后仍称为严格复现。

### Action Binding / Object Interactions

先生成 T2V-CompBench 官方 metadata：

- Action Binding：`prompt`、`phrase_0`、`phrase_1`；
- Object Interactions：官方 `object_interactions.json`；
- 视频文件名必须能映射到官方题目 ID。

Action Binding 使用官方脚本：

```bash
python LLaVA/llava/eval/compbench_eval_action_binding.py \
  --video-path /data/compbench_videos \
  --read-prompt-file /data/manifests/action_binding.official.json \
  --output-path /root/official_video_metrics_20260918/results/action_binding \
  --t2v-model /path/to/official/t2v/model
```

Object Interactions 使用官方脚本：

```bash
python LLaVA/llava/eval/compbench_eval_interaction.py \
  --video-path /data/compbench_videos \
  --read-prompt-file /data/manifests/object_interactions.official.json \
  --output-path /root/official_video_metrics_20260918/results/object_interactions \
  --t2v-model /path/to/official/t2v/model
```

这些命令中的参数名称以固定 T2V-CompBench V2 源码为准；如果上游 README 或固定源码有差异，应以源码 `--help` 和该版本 README 为准，不自行改写成另一套接口。

### Motion Binding

官方流程是两阶段：

```bash
python Grounded-Segment-Anything/compbench_motion_binding_seg.py ...
python dot/compbench_eval_motion_binding.py ...
```

第一阶段产生官方分割结果，第二阶段使用 DOT 轨迹。不能用总 RAFT 光流替代，也不能跳过数字 ID 和官方 metadata 映射。

### Motion Order / Motion Rationality / Mechanics / Thermotics / Material

这些使用 VBench-2.0 标准 metadata 和 `auxiliary_info`。其中：

- Motion Order 的 `auxiliary_info` 必须是两个有序动作；
- Motion Rationality 必须是从原始任务得到的可观察后果问题；
- Mechanics、Thermotics、Material 必须有对应的事件问题和物理表现问题；
- 不支持把任意 `custom_input` 视频直接塞给这些维度。

现阶段可以提交格式适配器，但必须等官方 LLaVA-Video/Qwen 环境和裁判权重齐备后再做 smoke test。

## 八、PhyGenEval 与 VideoPhy-2

### PhyGenEval PCA

PhyGenEval 需要官方题目资产，而不是只需要视频：

- 原始 prompt 和题目 ID；
- 适用物理规律；
- 关键现象问题；
- 事件顺序问题；
- 视频自然性问题或 rubric。

这些内容应该在生成任务设计阶段冻结，不能看完生成视频后再修改成“视频表现出来的答案”。官方流程按仓库原始阶段运行：

```text
问题生成 → single/vqascore.py → multi/multiimage_clip.py
→ 多图裁判 → 视频裁判 → overall.py
```

本项目不自行重写这条流程。闭源路径还需要官方 GPT-4o API；开源路径需要 VQAScore、LLaVA-Interleave 和 InternVideo2 环境。

### VideoPhy-2 PC / SA

官方输入是 CSV：

- PC：至少 `videopath`；
- SA：`videopath` 和原始 `caption`。

PC：

```bash
python inference.py \
  --input_csv examples/sa_pc.csv \
  --checkpoint /root/official_video_metrics_20260918/weights/videophy_2_auto \
  --output_csv /root/official_video_metrics_20260918/results/output_pc.csv \
  --task pc
```

SA：

```bash
python inference.py \
  --input_csv examples/sa_pc.csv \
  --checkpoint /root/official_video_metrics_20260918/weights/videophy_2_auto \
  --output_csv /root/official_video_metrics_20260918/results/output_sa.csv \
  --task sa
```

完成两次官方推理后再合并：

```bash
python scripts/aggregate_videophy_joint.py \
  --pc /root/official_video_metrics_20260918/results/output_pc.csv \
  --sa /root/official_video_metrics_20260918/results/output_sa.csv \
  --output /root/official_video_metrics_20260918/results/videophy_joint.csv
```

Joint 只能按同一 `video_id` 应用 `SA >= 4 且 PC >= 4`，不能先求集合平均再判断。

## 九、PSNR、SSIM、LPIPS、FVD、FID

### PSNR / SSIM / LPIPS

先配对生成帧与真实未来帧：

```bash
python scripts/build_reference_manifest.py \
  --generated-frames /data/generated_frames \
  --reference-frames /data/reference_frames \
  --output /root/official_video_metrics_20260918/manifests/reference_pairs.json \
  --require-complete
```

如果没有真实、时间对齐的未来帧，就输出 `N/A`，不能把生成视频自身复制成参考答案。

LPIPS 官方目录 CLI 的调用形态是：

```bash
python lpips_2dirs.py \
  -d0 /data/reference_frames \
  -d1 /data/generated_frames \
  -o /root/official_video_metrics_20260918/results/lpips.txt \
  --use_gpu
```

必须在运行前固定 LPIPS backbone（AlexNet、VGG 或 SqueezeNet）和版本。

### FVD

FVD 需要同分布真实视频集合和生成视频集合，不能拿一条视频得到有意义的集合分布结论。原版 Google Research 使用 TensorFlow/TF-Hub；不要用其他视频编码器或任意 PyTorch 实现替代后继续称作同一官方 FVD。

### FID

FID 需要同分布真实图像集合、生成图像集合或严格匹配的统计文件。官方 TTUR 入口形态是：

```bash
python fid.py /data/real_images /data/generated_images
```

不能直接使用其他数据集的均值、协方差或 Inception 统计量。

## 十、什么时候提交哪一种 PR

### 现在可以提交

提交“官方接入准备 PR”，包含：

- `FINAL_METRICS.md/json`；
- 输入适配器；
- VBench/CLIPScore/DOVER 包装器；
- 预检和契约测试；
- `REMAINING_METRICS.md`；
- 本指南。

### 完成 smoke test 后提交

以下指标各自需要官方单视频 smoke test 后，才能在 PR 中标记为 score-ready：

- VBench Imaging Quality；
- Temporal Flickering；
- Dynamic Degree；
- CLIPScore；
- ViCLIP / Overall Consistency；
- Subject Consistency；
- Background Consistency；
- DOVER。

### 外部环境和题目资料齐备后提交

- Instruction Following；
- Action Binding；
- Motion Binding；
- Motion Order；
- Motion Rationality；
- Object Interactions；
- PhyGenEval PCA；
- VideoPhy-2 PC/SA；
- Mechanics；
- Thermotics；
- Material；
- FVD/FID；
- PSNR/SSIM/LPIPS。

## 十一、每个指标的最小验收记录

为每个指标保存一个目录，例如：

```text
results/imaging_quality_smoke_001/
  command.txt
  source_revision.txt
  weights_manifest.json
  environment.txt
  official_stdout.log
  official_result.json
  wrapper_result.json
```

其中：

- `command.txt`：完整命令；
- `source_revision.txt`：官方 commit 和 SHA；
- `weights_manifest.json`：每个权重路径和 SHA-256；
- `environment.txt`：Python、Torch、CUDA、`pip freeze`；
- `official_result.json`：官方原始输出；
- `wrapper_result.json`：包装器解析结果。

运行成功后再把 `STATUS_REMAINING.json` 的状态改成 `scored`。没有分数、只有预检通过时，状态应保持 `conditional` 或 `preflight_passed_not_scored`。

## 十二、最终执行顺序

1. 提交当前官方接入准备 PR。
2. 固定 75 条及未来数据的原始 prompt、首帧和 ID 映射。
3. 先完成 VBench 基础指标和 CLIPScore 的单视频 smoke test。
4. 单独创建 DOVER 环境，完成 DOVER smoke test。
5. 再准备 WorldModelBench、T2V-CompBench、VBench-2.0、PhyGenBench、VideoPhy-2 的官方裁判环境。
6. 整理参考帧和真实分布集合，决定哪些样本输出 `N/A`。
7. 每完成一组，提交一组“score-ready”更新 PR。
8. 最后才对全部视频批量运行。

完整 75 条不是第一步。第一步是确保每个指标的官方入口、输入格式、权重、环境和一条 smoke test 都能被复核。
