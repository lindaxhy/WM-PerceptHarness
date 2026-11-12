# Motion Smoothness / AMT-S

用 AMT-S 对相邻保留帧插帧，再与原视频的中间帧比较，衡量局部运动平滑性。输出在 `[0,1]`，越高表示插帧误差越小；它不判断动作语义或物理正确性。

**官方基线**：PyPI `vbench==0.1.5` 的原始 wheel。`official/pypi/vbench/motion_smoothness.py` 和 `AMT-S.yaml` 从此 wheel 原样提取，完整依赖库、上游许可证和依赖声明均随包提供。wheel SHA256：`13266ed4715d91a7669ab58484b9e010bc91bb96e129d414fb512d4937890023`。[发布文件](https://files.pythonhosted.org/packages/06/82/13e401ec97d2686397201dfe74e2dda88b3695192de1b25a5ab267ab8960/vbench-0.1.5-py3-none-any.whl)

**当前修改**：`modified/vbench/motion_smoothness.py` 将输入张量列表保留在 CPU，沿用官方内循环将当前帧对送到 GPU。采帧、插帧模型、尺度公式和分数公式保持不变。入口要求显式选择 `official` 或 `cpu-staging`；两种模式都传入 `torch.device('cuda')`，与官方 CLI 一致，不使用可能触发不同缩放分支的字符串 `'cuda'`。

## 通过 `percept score` 调用

```bash
percept score motion-smoothness --mode cpu-staging \
  --video /data/example.mp4 --cache-dir /data/vbench-cache \
  --gpu 0 --output /data/results/motion_smoothness.json
```

`percept` 会在 `metric_packages/` 中定位 `run_motion_smoothness.py`，并沿用版本、源码、AMT 配置与权重哈希检查；`--mode official` 使用官方方法，`--dry-run` 只打印实际命令。已有的 `percept score motion` 是已安装 VBench CLI 路径，与本包的 hash 固定 wheel 适配器不是同一协议。

## 环境与运行

需要 Linux、一个可见 CUDA GPU、Python 3.12 和兼容的 PyTorch/TorchVision。可用版本参考为 torch 2.13.0、torchvision 0.28.0、numpy 2.5.2、Pillow 11.3.0；这不是完整环境锁文件。上游完整依赖声明在 `official/pypi/vbench-0.1.5.dist-info/METADATA`，包含其他 VBench 维度的旧版本约束。准备好对应运行依赖后，安装本包 wheel，避免自动改写已有 GPU 环境：

```bash
python -m pip install --no-deps official/distributions/vbench-0.1.5-py3-none-any.whl
```

权重需预先放置于 `<cache-dir>/amt_model/amt-s.pth`，SHA256 必须为 `07e7e03405c213fe4405678db9b42d05671e48c56faf6a441b99bb0047d3cf77`。[上游 AMT-S 权重](https://huggingface.co/lalala125/AMT/resolve/main/amt-s.pth)。本包没有权重或自动下载步骤，哈希通过后才使用官方 checkpoint 加载方式。

远程安装、下载或推理前先进入持久会话并验证：

```bash
tmux new -s motion-smoothness
test -n "$TMUX"
tmux display-message -p '#S'
```

分离用 `Ctrl-b d`，恢复用 `tmux attach -t motion-smoothness`。复用下载时保留原缓存、partial 文件、目标路径、认证环境和下载后端。

在本包目录运行：

```bash
python run_motion_smoothness.py --mode cpu-staging \
  --video /data/example.mp4 --cache-dir /data/vbench-cache \
  --gpu 0 --output /data/results/motion_smoothness.json
```

`--mode official` 使用未修改的官方方法。只接受 MP4，输出路径必须不存在；入口检查安装版本、核心源码、AMT配置和权重哈希，并记录视频及运行版本。`--help` 不加载模型。

## 局限

输入 padding 也随张量在 CPU 执行；普通等尺寸 MP4 的 replicate padding 不改像素值，异尺寸帧触发的 bilinear resize 可能有 CPU/GPU 数值差异。CPU staging 不减少模型本身显存，也不是流式解码；长视频仍会消耗大量 CPU 内存。

指标受帧率、速度、静止画面和插帧难度影响，接近上限时区分度有限。平滑但错误的动作仍可得高分。跨依赖环境不保证逐位相同；本次拆包仅验证入口和源码完整性，没有重新进行 GPU 推理。
