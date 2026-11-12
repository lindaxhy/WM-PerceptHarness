# PSNR

将生成视频与原始源视频配对，计算RGB源参考PSNR诊断值。评分核来自固定 IQA-PyTorch；项目适配负责时间与原生输入几何对齐。本协议不是人工真值评测，也不是通用感知质量指标。

## 方案与改动

- 原官方 PSNR/psnr：RGB、crop_border=0、data_range=1、eps=1e-8。逐帧先转dB，再在视频内平均；完全相同图像在此eps下为80 dB。
- 源视频和生成视频哈希固定；FFmpeg提取源首个显示帧的RGB PNG，必须匹配真实conditioning PNG的SHA-256。
- 两条时间轴各减首个显示PTS。生成视频须符合记录的24 fps；逐生成时刻找最近源PTS，同距取更早帧。只接受源时间范围内且误差不超过1/48秒的匹配，不外推、不变速、不做光流对齐。
- 源帧按生成模型原生几何变换：Wan采用官方 best_output_size、32步长/1280×704面积预设、PIL LANCZOS cover和floor中心裁剪；Dense采用官方uint8 Torch bilinear和round中心裁剪到832×480。
- 主分数排除生成第0帧，首帧分、未匹配帧及逐帧映射另存。模型汇总应对成功视频等权；跨模型比较须对齐共同成功样本。本入口输出逐视频值及数量状态。

当前入口 modified/run_psnr.py 只在CPU评分，无学习权重。评分和几何函数原样加载；仅在加载PSNR类时移除注册装饰器，避免导入整个PyIQA，不执行Wan/LingBot生成模型。

official/ 是选取源码，不是完整可安装仓库。IQA-PyTorch固定于18dd7a19694e94aac21019170e3f5e63d6b4e19e，LingBot固定于f348dca454c8f1e5707721614dbac10fa5290fed，Wan固定于42bf4cfaa384bc21833865abc2f9e6c0e67233dc。逐文件来源及哈希见 OFFICIAL_MANIFEST.json。

## 通过 `percept score` 调用

```bash
percept score psnr --manifest /data/manifests/psnr.jsonl \
  --output-dir /data/results/psnr_source_reference_v1 --workers 3
```

`percept` 会在 `metric_packages/` 中定位 `modified/run_psnr.py`，并沿用相同的哈希、PTS 对齐和输出协议；`--python` 可指定本包的 CPU Torch 环境，`--dry-run` 只打印实际命令。直接运行 `python modified/run_psnr.py ...` 仍可用于独立复制本包。

## 环境与运行

Linux、Python3.12、CPU可用的 torch==2.10.0，其余依赖见 requirements.txt。FFmpeg与ffprobe须在PATH；conditioning PNG哈希对版本/解码差异敏感，既定配置为FFmpeg4.4.2。这里不是新安装验收后的完整lockfile，不需要下载生成模型。

远程安装或评分前先进入并验证持久会话，再开始长任务：

    tmux new-session -A -s psnr
    test -n "$TMUX"

在该会话内准备CPU Torch，进入本包目录，再运行：

    python -m pip install -r requirements.txt
    python modified/run_psnr.py --manifest /data/manifests/psnr.jsonl --output-dir /data/results/psnr_source_reference_v1 --workers 3

分离：Ctrl-b 后按 d；重新进入：tmux attach -t psnr。保留已有下载缓存、目标和续传后端。

examples/psnr_manifest.jsonl 仅为格式示例，必须替换所有路径、帧数及全零哈希。相对路径按清单目录解析；model 限 wan / lingbot-dense。相同 sample_id 的两模型必须绑定同一源视频、源哈希和真实conditioning PNG哈希；也可只提供单模型。不能拿生成第0帧冒充原始conditioning PNG。

输出 results/ 逐视频评分，evidence/ PTS映射及逐帧值，progress.json 和 batch_finished.json 数量状态。失败独立记录，不替代分数。清单、协议或代码变化时使用新输出目录；续跑核对实际源视频和生成视频哈希。

## 局限与许可

协议只覆盖上述Wan/Dense几何及24 fps输出。合理动作时刻或轨迹变化也可能降低PSNR，原生分辨率和裁剪差异限制跨模型比较；该值不能替代物理合理性或任务成功率。输入身份、首帧或几何条件不成立时，该条停止评分。

本轮仅检查独立包源码路径、命令行帮助和评分函数一致性；未运行Torch评分、重评分真实视频或验收新安装环境。

IQA-PyTorch的 LICENSE（PolyForm Noncommercial）和 LICENSE-S-Lab 均原样保留，PSNR文件中的PIQ/PIQA等归属保留。Wan、LingBot各自Apache-2.0许可证保留在对应目录。项目适配不是官方发布版本，不统一覆盖第三方许可。
