# Semantic Adherence（SA，语义遵循）

用 VideoPhy-2 判断视频与输入 caption 的整体语义匹配，输出官方 1—5 评分（解析未命中时保留官方 0）。

## 当前方案与修改

- 官方 VideoPhy-2 模型、提示模板、生成参数及数字解析保持原样。
- 当前默认抽帧为 `min(max(ceil(N/2),32),256)`。令 `K=ceil(N/2)`，其中 `N` 是源视频帧数：`K<32` 时使用官方 32 帧索引；`32≤K≤256` 时取源帧 `0,2,4,...`；`K>256` 时对全片均匀取 256 帧并覆盖首尾。显式 `--sampling official` 使用官方固定 32 帧。
- 修改入口支持本地权重、CSV 输入、每视频原始回答/得分与抽帧记录。视觉 token hook 仅记录。新增薄入口固定 `sa`；共用运行模块仅调整参数绑定，抽帧和评分没有因拆包而改变。
- 示例 `examples/input.csv` 是合成占位，运行前替换路径及 caption；caption 必须预先冻结并按原文输入。

## 来源、环境与范围

官方代码：[Hritikbansal/videophy](https://github.com/Hritikbansal/videophy/tree/01d366c979e4a82875852adbf427ccc036e7373a)，commit `01d366c979e4a82875852adbf427ccc036e7373a`。
权重：[videophysics/videophy_2_auto](https://huggingface.co/videophysics/videophy_2_auto/tree/ca38370e1dcc79c99c7017dd8755d0a087299afc)，revision `ca38370e1dcc79c99c7017dd8755d0a087299afc`。

本包包含所需选取官方源码及许可证，不是完整上游仓库。`official_sources.json` 逐文件列出来源和哈希；`MANIFEST.json` 列出本包文件。Python 3.10、Torch 2.10/CUDA 12.8、Transformers 4.28.1、Accelerate 0.28.0、PEFT 0.3.0、NumPy 1.26.4、Decord 0.6.0；详见 `environment/`。这些是已知运行版本，本次没有安装复验。运行须另备完整固定 checkpoint、依赖及视频；入口使用离线模式，不自动下载。

## 通过 `percept score` 调用

在可运行 `percept` 的环境中，可直接调用同一入口：

```bash
percept score semantic-adherence --python /path/to/videophy-env/bin/python \
  --input-csv /absolute/path/input.csv \
  --checkpoint /absolute/path/videophy_2_auto \
  --output /absolute/path/new_sa_results
```

`percept` 会在 `metric_packages/` 中定位本包 runner，并沿用相同的抽帧、输出与 tmux 检查；`--python` 指向本包环境（Python 3.10），`--dry-run` 只打印实际命令。直接运行 `python modified/sapc/run_sa.py ...` 仍可用于独立复制本包。

## 运行

以下命令在 Linux GPU 主机、本指标包根目录中执行。先进入并验证持久会话，再开始安装、下载或推理：

```bash
tmux new-session -s metric-run
test -n "$TMUX"
tmux display-message -p '#S'
```

暂离：`Ctrl-b` 后按 `d`；重连：`tmux attach -t metric-run`。没有 tmux 时先准备持久会话。使用现有下载缓存和原下载后端，保留 partial 文件。下面假定环境和完整本地权重已准备好；输出使用新的独立目录。

```bash
python modified/sapc/run_sa.py --help
python modified/sapc/run_sa.py \
  --input-csv /absolute/path/input.csv \
  --checkpoint /absolute/path/videophy_2_auto \
  --output /absolute/path/new_sa_results
# 官方32帧：在同一命令中增加 --sampling official
```

`official/videophy/VIDEOPHY2/inference.py` 是原样官方参考入口；日常使用上面的单指标入口。`--preprocess-only` 仍需模型配置/tokenizer/processor 和解码依赖。输出包括 `protocol.json`、`rows.json`、`complete.json`。没有覆盖保护或断点恢复；失败会记录并抛出停止。

## 局限与验证边界

- SA 衡量整体语义匹配，不能直接替代具体任务成功率。输入 caption 的准确性影响结果；机器生成 caption 不等同于人工真值，未进行人工标注准确率认证。

- 抽帧按源帧索引，变帧率视频不保证时间等距；短片可能重复帧。视觉 abstractor 仍是 64 latent 加 1 结束 token，加帧没有已确认的准确率收益。
- 数字解析保留官方子串匹配及未匹配时返回 0；额外记录解析状态，不擅自改分。
- 仅支持固定 checkpoint 内完整 `text_config` 的加载方式。上游 `text_config=None` 分支引用未发布的 `..llama.configuration_llama`，不支持空配置构模；官方代码未补改。
- 本次只检查包内文件/哈希、语法及独立 CLI 帮助和固定指标参数；未安装模型依赖、未做 GPU 推理或语义准确率验证。
