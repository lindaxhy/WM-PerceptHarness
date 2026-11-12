# Dynamic Degree

衡量视频是否具有达到官方光流阈值的运动。单视频输出为 `0.0` 或 `1.0`；摄像机运动也会贡献光流，因此不能把该值直接解释为动作正确性或视频质量。

**官方基线**：PyPI `vbench==0.1.5` 的原始 wheel。`official/pypi/vbench/dynamic_degree.py` 从此 wheel 原样提取，完整依赖库、上游许可证和依赖声明均随包提供。wheel SHA256：`13266ed4715d91a7669ab58484b9e010bc91bb96e129d414fb512d4937890023`。[发布文件](https://files.pythonhosted.org/packages/06/82/13e401ec97d2686397201dfe74e2dda88b3695192de1b25a5ab267ab8960/vbench-0.1.5-py3-none-any.whl)

**当前修改**：`modified/vbench/dynamic_degree.py` 将解码帧张量保留在 CPU，推理时只把当前帧对送到 GPU。官方采帧间隔、RAFT 的20次迭代、阈值和评分公式保持不变。入口默认要求显式选择 `official` 或 `cpu-staging`；两种模式均采用官方 CLI 的 `torch.device('cuda')`。

## 通过 `percept score` 调用

```bash
percept score dynamic-degree --mode cpu-staging \
  --video /data/example.mp4 --cache-dir /data/vbench-cache \
  --gpu 0 --output /data/results/dynamic_degree.json
```

`percept` 会在 `metric_packages/` 中定位 `run_dynamic_degree.py`，并沿用版本、源码与权重哈希检查；`--mode official` 使用官方方法，`--dry-run` 只打印实际命令。已有的 `percept dynamic-degree` 是官方 VBench checkout 路径，与本包的 hash 固定 wheel 适配器不是同一协议。

## 环境与运行

需要 Linux、一个可见 CUDA GPU、Python 3.12 和兼容的 PyTorch/TorchVision。可用版本参考为 torch 2.13.0、torchvision 0.28.0、numpy 2.5.2、Pillow 11.3.0；这不是完整环境锁文件。上游完整依赖声明在 `official/pypi/vbench-0.1.5.dist-info/METADATA`，包含其他 VBench 维度的旧版本约束。准备好对应运行依赖后，安装本包 wheel，避免自动改写已有 GPU 环境：

```bash
python -m pip install --no-deps official/distributions/vbench-0.1.5-py3-none-any.whl
```

权重需预先放置于 `<cache-dir>/raft_model/models/raft-things.pth`，SHA256 必须为 `fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1`。上游提供的 [RAFT 权重压缩包](https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip) 包含该文件；本包没有权重或自动下载步骤。

远程安装、下载或推理前先进入持久会话并验证：

```bash
tmux new -s dynamic-degree
test -n "$TMUX"
tmux display-message -p '#S'
```

分离用 `Ctrl-b d`，恢复用 `tmux attach -t dynamic-degree`。复用下载时保留原缓存、partial 文件、目标路径、认证环境和下载后端。

在本包目录运行：

```bash
python run_dynamic_degree.py --mode cpu-staging \
  --video /data/example.mp4 --cache-dir /data/vbench-cache \
  --gpu 0 --output /data/results/dynamic_degree.json
```

`--mode official` 使用未修改的官方方法。只接受 MP4，输出路径必须不存在；入口检查安装版本、核心源码和权重哈希，并记录输入视频、代码、权重及运行版本。`--help` 不加载模型。

## 局限

CPU staging 减少 GPU 上保留的输入帧，不减少模型本身的显存需求，也不是流式解码；长视频仍会消耗大量 CPU 内存。静态任务不一定应该得到高动态分，光流也不区分物体运动与相机运动。跨环境不保证逐位相同；本次拆包仅验证入口和源码完整性，没有重新进行 GPU 推理。
