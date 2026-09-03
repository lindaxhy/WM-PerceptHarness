# 本地 LAS 兼容实现与 LAS/本地标注对比报告

日期：2026-09-03
评估代码快照：冻结的本地评估实现（发布前清理）

## 结论摘要

当前交付是自托管的视频理解模型服务，不是把请求转发给方舟、远程 LAS 或其他模型 API。服务复现 LAS 的异步 Submit/Poll 使用方式，由本地 `Qwen/Qwen3-VL-8B-Instruct` 完成视觉推理；运行观察未发现模型 API 出站连接，也没有提取音轨、调用 ASR 或把转写作为证据。

这次诊断性对比预先冻结了 5 个样本。4 个得到严格可验证的本地结果，1 个（`full_0021`）在 Pass B 初次输出和内置修复后仍违反边界结构约束，因而保留为失败且不进入评分。4 个已评分结果覆盖 25 个 LAS 事件：每个事件都与至少一个本地时间片有正时间重叠，但动作兼容覆盖均值只有 `0.12441008674259292`；可比语义中，actor 为 `11/12`，action family 为 `2/15`，target 为 `3/8`。LAS 的 11 个遮挡事件没有一个本地显式遮挡事件匹配，主要是输出 schema 能力差异，不应全部归为基础模型错误。

现有证据不支持立即进行广泛微调。更合理的顺序是：先补齐遮挡、状态、结果和人手 actor schema，调整提示词和事件聚合，再用更大的盲测、人工复核集复测。只有视觉语义遗漏或时间边界偏差在这些改动后仍稳定复现，才应针对性 LoRA/SFT。

## 1. 已实现能力

### 1.1 API、算子与模板

- [API 契约](../../src/las_repro/contracts.py)和[控制面](../../src/las_repro/api.py)提供 `POST /api/v1/submit` 与 `POST /api/v1/poll`，接受 `las_long_video_understand`、`las_video_understanding` 两个算子 ID，版本严格为 `v1`。Poll 必须使用提交时的算子身份。
- 支持 `general_video_captioning`、`embodied_active_object_detection`、`embodied_action_captioning` 三条管线；只提供非空 `query` 时，本地有效模板为 general captioning。入口、Fake 模式和独立进程角色见 [README](../../README.md)与[命令行实现](../../src/las_repro/cli.py)。
- 五个云端兼容字段可被接收，但在日志和持久化前丢弃并给出 ignored warning；Ark key、endpoint、响应缓存 ID 均不参与本地推理。模型名经[别名 allowlist](../../src/las_repro/model_alias.py)路由，不能作为任意模型路径。

### 1.2 本地模型、管线和严格输出

- [Qwen3-VL 后端](../../src/las_repro/models/qwen3_vl.py)实现统一视觉模型接口；生产验收使用固定本地快照、离线加载和 `qwen3-vl-8b-instruct` 单一别名。API 与 coordinator 不加载 GPU 依赖，4 个 GPU worker 各固定到一张卡。
- [general 管线](../../src/las_repro/pipelines/general.py)执行长视频切段、时间线排序与聚合；[embodied 管线](../../src/las_repro/pipelines/embodied.py)实现 active objects，以及 Pass A 粗动作 → Pass B 可见状态边界/不超过 1 秒的 fine segments → 六字段 enrichment。
- [时间与枚举验证器](../../src/las_repro/pipelines/validators.py)拥有最终边界：索引连续、相邻边界相等、完整覆盖视频、片段为正且不超过上限。模型输出先经严格 JSON/schema 校验；失败只允许一次修复。修复后的 enrichment 若仅含受控 enum 错误，可把对应字段保守归一为 `unknown`，随后再次严格验证，并写入不含原始非法值的封闭审计 envelope；首次 enrichment 不能使用该 fallback。实现见[输出验证](../../src/las_repro/pipelines/output_validation.py)。
- 具身提示词明确要求仅使用可见证据、严格 JSON、固定枚举和完整 skeleton，见 [Pass A](../../src/las_repro/prompts/embodied_pass_a.txt)、[Pass B](../../src/las_repro/prompts/embodied_pass_b.txt)与[enrichment](../../src/las_repro/prompts/embodied_enrichment.txt)。

### 1.3 调度、持久化和安全

- [SQLite/WAL 存储](../../src/las_repro/store.py)保存 tasks 与 inference jobs。`BEGIN IMMEDIATE` 原子领取、租约/心跳、过期恢复、终态不可逆、worker affinity/fallback 和不可变 model routing 支持进程重启后的安全恢复。
- [worker 实现](../../src/las_repro/workers.py)由一个 coordinator 与四个单卡 GPU worker 组成；同一任务的后续视觉阶段优先复用 task-scoped `VideoSession`，终态释放资源。
- [媒体层](../../src/las_repro/media.py)限制本地 allowlist、解析 symlink 后复核、限制 HTTP(S) 协议/地址/跳转/超时/字节数，并以临时文件加原子发布完成下载。视频时长优先采用精确的 video-stream timestamp rational，避免音轨延长的 container duration 使末段越过最后视频帧。
- 整条生产管线没有音频或 ASR 依赖。此次 5 个输入各有 1 条音轨，但仅探测了流元数据，未解码、抽取或使用音轨。
- [JSONL 导出](../../src/las_repro/export.py)生成连续 frame range 的 `action_captions.jsonl`，采用 owner-only、fsync 和原子替换。日志与错误经过[安全过滤](../../src/las_repro/security.py)，数据库、临时目录和运行状态使用严格权限。

### 1.4 测试与真实 GPU 证据

- 当前精确时长修订在对比运行前通过 `693` 项测试；此前分支级覆盖验收为 `85.99%`。测试覆盖 API、契约、安全、媒体、SQLite 租约、worker、三条管线、严格 schema、本地客户端和 JSONL 导出。
- [GPU 验收记录](2026-09-02-gpu-acceptance.md)显示四张 GPU 均完成设备隔离 smoke，单卡峰值为 `17,581,941,760` bytes；真实 Qwen 具身流程曾完成 12 个连续片段和 JSONL 导出。此次比较又在同一四-worker 架构上完成 4 个样本并保留 1 个严格失败。
- 模型推理使用经过清单和 hash 验证的 16 文件固定快照；服务运行时采用离线 hub 控制。易失运行环境丢失后的重建也验证了 wheel/安装树一致性、SQLite 完整性和加密持久化；敏感状态没有明文写入权限语义不足的持久存储。

## 2. 数据与方法

### 2.1 冻结样本

在查看任何本地推理结果之前，对 `1,869` 份标注和 `1,869` 个唯一视频做了 bounded parsing、SHA-256 与时长核验，得到 `1,869` 个有效候选。冻结选择文件的 SHA-256 是：

`8e1b39b4f9a77d85e182132aaf3ba254f1588e7d30c6914522ee12149e3fb918`

| Slot | Sample | Video SHA-256 | Stratum | Duration (s) | LAS events | Occlusions | Audio streams | Review |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| deep | `full_0001` | `c3243c46bad68d3b2772e82648e45b68e75a1893b0ce27edecd450226464c1e9` | fixed:`full_0001` | `10.933333333333334` | 7 | 1 annotation record / 3 semantic events | 1 | `machine_only` |
| occluded_many | `full_0002` | `6741b6184b847b6096e4282b1e0f76142714870b258ca16a19606e0441f0973f` | occlusion≥1, events>median | `14.7` | 8 | 2 annotation records / 6 semantic events | 1 | `machine_only` |
| short_few | `full_0024` | `a7a696bcdd835c083b27ca3705d13a2f22e069ebec9038581354fed39e6fbbe8` | occlusion=0, duration≤Q1, events≤median | `4.566666666666666` | 4 | 0 | 1 | `machine_only` |
| long_many | `full_0021` | `1cd6b0752bd3b7f1ca987d470d8e65d9247a4f0751c7f891c6824b63751eff05` | occlusion=0, duration≥median, events>median | `7.3` | 10 | 0 | 1 | `needs_review` |
| branch_diversity | `full_0004` | `34bc1833f713419c694c97af544f5c4148f0b03e02913415f4a3eb2be8660d09` | uncovered branch, quality flags preferred, event-distance-to-median | `8.8` | 6 | 1 annotation record / 2 semantic events | 1 | `machine_only` |

选择阈值为 duration Q1 `5.331888888888889` 秒、duration median `6.365` 秒、event-count median `5.0`。这 5 个样本只用于诊断，不代表完整数据集；其中 4 个 LAS reference 是 `machine_only`，另一个明确是 `needs_review`。因此 LAS 是对比参考，不是绝对人工 ground truth，也不能据此作数据集级排名或性能结论。

### 2.2 固定请求和运行约束

所有样本按冻结顺序、逐个提交，使用相同设置：operator `las_long_video_understand`、version `v1`、template `embodied_action_captioning`、model alias `qwen3-vl-8b-instruct`、`fps=2.0`、`media_resolution=medium`、`reasoning_effort=high`、`clip_context=high`，query 为仅描述可见操控动作和物体交互。没有调用付费 Ark/LAS，也没有第二模型裁判。终态失败不换样本、不把缺失结果记为零分。

### 2.3 对齐和计分

对每个 LAS 事件，以正时间重叠且 IoU 最大的本地片段作为 primary match；并列时依次选更早 start 和更小 segment index。

- `IoU = intersection_duration / union_duration`。
- any-overlap coverage：把所有与 LAS 事件相交的本地片段裁剪到该事件，合并重叠区间后，以 union duration 除以 LAS event duration。
- action-compatible coverage：先要求预冻结 action-family 映射相同，再按相同 union 方法计算。
- boundary error 是 LAS event 与 primary local segment 的 start/end 绝对误差。
- local extra segment 指与任何 LAS event 都没有正时间重叠的本地片段；语义不同但时间重叠的片段不是 extra，而是 semantic mismatch。

固定 action 映射把 LAS `move/transport` 和 local `move/lift/push/pull/rotate/place` 归为 motion，并分别映射 grasp、reach、release；LAS 的三类 occlusion 归为 occlusion，但 local schema 没有对应 family。固定 actor 映射只比较 single-manipulator 与 both-manipulators。target 只通过预冻结的 apple、basket、panel、container、track、block、ball、bottle、tool 中英别名比较。

未列出的 LAS type、对象 actor、local `unknown` 或没有共享别名的 target 都标记为 `not_comparable`，不进入该语义分母；它们既不是命中，也不是失败。本文所有语义结果都显示“matches/comparable”，避免用不同分母的百分比混淆结论。

## 3. 深入对比：`full_0001`

LAS reference 的显式对象清单包括：装有 4 个苹果的黄色篮、空透明容器、固定木块、可横移白色遮挡板和连接容器/木块的斜轨。其初态记录篮内 4 个苹果且容器为空；终态记录篮内剩 3 个、容器内有 1 个粉色苹果、遮挡板移到最右，斜轨和木块未位移；outcome 标为 success。Local 输出提供完整连续 fine-segment 时间轴和六字段 enrichment，但没有独立的对象 inventory、initial state、final state、occlusion state 或 outcome 字段，所以这些文档级事实不能做同字段一一对比。

| LAS event/time | Local matched segment/time | IoU | Common evidence | Difference | Interpretation |
| --- | --- | ---: | --- | --- | --- |
| `evt_001` move, `[0.2,3.3]`: 执行手将遮挡板从右下向左平移 | seg 1, `[0.841025641025641,1.682051282051282]`: “right hand grasps apple” | `0.27129859387923905` | 都识别到 single-manipulator activity；actor 可比且 match | LAS target 是 panel；local primary 描述 apple。target mismatch，结构化 skill `not_comparable` | 时间重叠存在，但 primary 细片段没有复现 LAS 的长时 panel-motion 语义；动作兼容 coverage 为 `0.2506203473945409` |
| `evt_002` grasp, `[2.0,3.3]`: 双手配合稳定遮挡板并抓起顶层粉色苹果 | seg 3, `[2.523076923076923,3.364102564102564]`: “right hand moves apple to ramp” | `0.569548872180451` | 两边自然语言都提到手与 apple | LAS 是双手/grasp；local 是单手/motion，actor、action、target 均 mismatch。target 评分比较 LAS `object_ids` 解析出的对象文本与 local 结构化 `target`，不按两段描述中共同出现的 “apple” 判定 | 时序接近，但角色数量和动作阶段粒度不同；动作兼容 coverage `0.40236686390532544` |
| `evt_003` occlusion_enter, `[3.3,3.4]`: 篮和苹果进入完全遮挡 | seg 3, `[2.523076923076923,3.364102564102564]`: “right hand moves apple to ramp” | `0.07309941520467861` | local 时间轴覆盖该瞬间 | local 没有 occlusion family；actor/target 不可比，skill mismatch | 明确的 schema gap，不应解释为普通动作分类错误；动作兼容 coverage `0.0` |
| `evt_004` occluded, `[3.3,8.6]`: 篮区域持续不可见 | seg 6, `[5.046153846153846,5.887179487179488]`: “right hand releases apple” | `0.15868408321238522` | local 连续时间轴覆盖整段 | LAS 表示持续不可见状态，local primary 表示 release；actor/target 不可比，skill mismatch | 长状态事件对单个 ≤1 秒动作片段，IoU 天然偏低；这是 schema 与粒度双重差异 |
| `evt_005` move, `[6.7,8.3]`: 苹果沿斜轨下滑并落入左侧容器后静止 | seg 8, `[6.728205128205128,7.569230769230769]`: “right hand approaches next apple” | `0.5256410256410252` | 边界起点只差 `0.028205128205128105` 秒，两边自然语言都涉及 apple | local 没有表达自主滑落/入容器；actor 不可比，action 与 target mismatch。target 评分仍使用 LAS `object_ids` 对 local 结构化 `target`，而不是描述词面重合 | 较高 IoU 不能代替语义一致；动作兼容 coverage 仍为 `0.0`，这是潜在视觉语义遗漏 |
| `evt_006` occlusion_exit, `[8.6,9.7]`: 遮挡板右移，篮和苹果重新露出 | seg 10, `[8.41025641025641,9.251282051282052]`: “right hand grasps next apple” | `0.5049701789264427` | local 覆盖重新显露时段 | local 无 occlusion-exit 表达；actor/target 不可比，skill mismatch | 较高时间重叠仍是 schema gap，不是显式遮挡命中 |
| `evt_007` move, `[8.6,9.8]`: 执行手将遮挡板右移到画面最右 | seg 10, `[8.41025641025641,9.251282051282052]`: “right hand grasps next apple” | `0.4686346863468642` | single-manipulator actor match，且时段相交 | panel-motion 与 apple-grasp 不同；action、target mismatch | actor 识别相近，但动作对象发生偏移；动作兼容 coverage `0.4572649572649567` |

深样本的 7 个 LAS 事件全部有时间重叠，local 有 13 个连续、相邻、完整覆盖 `[0.0,10.933333333333334]` 的片段，最长不超过 1 秒，六个 enrichment 字段齐全；其中 1 个 local segment 与任何 LAS event 无正重叠。最大 IoU 均值/中位数为 `0.36741097934158373` / `0.4686346863468642`，`4/7` 达到 0.3，`3/7` 达到 0.5；start/end 绝对误差中位数为 `0.523076923076923` / `0.5487179487179485` 秒。

语义上 actor 为 `2/3`，action 为 `0/6`，target 为 `0/4`。三个 LAS occlusion semantic events 均无 local explicit occlusion match。Local 的优势是完整、细粒度、可直接训练导出的时间拓扑；LAS 的优势是能表达重叠的长事件、遮挡状态、对象清单、初终态和结果。Local 的连续细分也带来冗余风险，同一个 LAS 长事件可能横跨多个片段；反过来，primary-match 只取一个片段，可能低估组合后的语义。所有描述都必须结合结构化字段和可见证据复核，不能因为语言流畅就排除对象混淆或 hallucination 风险。本次 deep 结果严格首轮通过，未使用 `unknown` normalization。

## 4. 五个样本的结果

### 4.1 每个样本：时间指标

`full_0021` 的 10 个 LAS 事件已入选，但没有本地结果，因此是 selected 而非 scored。

| Sample | Terminal | LAS selected/scored | Local segments | Any-overlap mean | Action-compatible mean | Max-IoU mean / median | IoU≥0.3 | IoU≥0.5 | Median start/end error (s) | Uncovered / local extra |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |
| `full_0001` | COMPLETED | `7/7` | 13 | `1.0` | `0.15860745265211756` | `0.36741097934158373 / 0.4686346863468642` | `4/7` | `3/7` | `0.523076923076923 / 0.5487179487179485` | `0/1` |
| `full_0002` | COMPLETED | `8/8` | 17 | `1.0` | `0.25` | `0.37232128042927864 / 0.34502923976608185` | `4/8` | `2/8` | `0.17857142857142816 / 0.4357142857142855` | `0/2` |
| `full_0024` | COMPLETED | `4/4` | 6 | `1.0` | `0.0` | `0.44689672825277577 / 0.41171215074723844` | `3/4` | `1/4` | `0.3222222222222221 / 0.438888888888889` | `0/0` |
| `full_0021` | FAILED | `10/0` | — | — | — | — | — | — | — | — |
| `full_0004` | COMPLETED | `6/6` | 10 | `1.0` | `0.0` | `0.5397522099516687 / 0.5422413793103448` | `5/6` | `4/6` | `0.2100000000000004 / 0.3699999999999999` | `0/3` |

### 4.2 每个样本：语义与 schema 指标

| Sample | Actor match/comparable | Action match/comparable | Target match/comparable | LAS occlusion / local explicit | Normalized fields | Runtime note |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `full_0001` | `2/3` | `0/6` | `0/4` | `3/0` | 0 | 首轮 strict enrichment |
| `full_0002` | `1/1` | `2/8` | `3/4` | `6/0` | 0 | 首轮 strict enrichment |
| `full_0024` | `4/4` | `0/0` | `0/0` | `0/0` | 0 | action 分母为 0，因为 LAS `approach/contact/push/stop` 不在冻结 action 映射中；target 分母独立为 0，因为 LAS `object_ids` 文本与 local 结构化 `target` 没有共享的冻结别名 |
| `full_0021` | — | — | — | — | 0 | Pass B initial + repair 均被拒绝；无结果、不评分 |
| `full_0004` | `4/4` | `0/1` | `0/0` | `2/0` | 2 × `skill` | Pass B 经一次 repair；enrichment repair 对 2 个 skill 作保守 `unknown` normalization |

### 4.3 四个补充样本与全部五个样本的聚合

“四个补充样本”选择了 4 个，但只评分 3 个；“全部五个样本”选择了 5 个，但只评分 4 个。所有 coverage、IoU、boundary 和 semantic 数字只聚合真实 COMPLETED 结果。失败样本的 10 个 LAS events 只进入 selected denominator，不进入 scored denominator。

| Metric | Four selected / three scored | Five selected / four scored |
| --- | ---: | ---: |
| Selected / scored LAS events | `28 / 18` | `35 / 25` |
| Completed / failed samples | `3 / 1` | `4 / 1` |
| Local segments | `33` | `46` |
| Mean any-overlap coverage | `1.0` | `1.0` |
| Mean action-compatible coverage | `0.1111111111111111` | `0.12441008674259292` |
| Mean / median maximum IoU | `0.44470391200863024 / 0.450163398692811` | `0.42306189086185725 / 0.4558823529411774` |
| IoU at least 0.3 | `12 / 18` | `16 / 25` |
| IoU at least 0.5 | `7 / 18` | `10 / 25` |
| Median start / end error (s) | `0.2000000000000001 / 0.4357142857142855` | `0.20000000000000018 / 0.4571428571428573` |
| Actor matches / comparable | `9 / 9` | `11 / 12` |
| Action matches / comparable | `2 / 9` | `2 / 15` |
| Target matches / comparable | `3 / 4` | `3 / 8` |
| Uncovered LAS events / local extras | `0 / 5` | `0 / 6` |
| LAS occlusion events / explicit local matches | `8 / 0` | `11 / 0` |

这里 `any-overlap=1.0` 主要说明 local 连续时间轴覆盖了 LAS 事件所在时段，并不代表语义正确。LAS 常用一个长区间表达完整动作或持续状态，local 则硬性拆成不超过 1 秒的连续片段，所以单片段最大 IoU 会受到粒度上限压低；应同时看 union coverage、boundary error 和语义指标。action-compatible coverage 明显低于 any-overlap coverage，说明差异不能只用切段粒度解释。

### 4.4 失败、schema gap 与 normalization 率

- 终态结果可用率为 `4/5`（80%），真实 pipeline failure rate 为 `1/5`（20%）。在四个补充样本块中，可用率为 `3/4`（75%），失败率为 `1/4`（25%）。`full_0021` 的闭合错误码为 `SEGMENT_TOO_LONG` 和 `SEGMENT_BOUNDARY_NOT_ADJACENT`；没有替换、重提或伪造结果。
- 4 个完成样本中 1 个使用过保守 enum fallback，即 `1/4`（25%）；按全部入选样本是 `1/5`（20%）。最终 46 个 local segments 中有 2 个 `skill` 字段被归一为 `unknown`，即 `2/46`（约 4.35%）；这些字段按规则不进入 action semantic denominator。
- 已评分的 25 个 LAS events 中有 11 个 occlusion events，即 `11/25`（44%），local explicit occlusion matches 为 `0/11`。这首先暴露 schema gap；只有在 local schema 增加遮挡表示后，才适合评估模型是否真正漏看遮挡。
- actor/action/target 的 comparable denominators 分别只有 12、15、8，远小于 25；直接用总事件数计算“准确率”会把 `not_comparable` 错当 mismatch。

## 5. 相同点与差异

### 相同点

- 两者都把视觉事件绑定到时间区间；4 个完成样本中，local 时间轴对所有 25 个 LAS events 都有正时间覆盖。
- 两者都能描述操控者、动作与对象证据。固定 ontology 映射后，actor 在可比事件中达到 `11/12`，说明 single/both manipulator 层面大体一致。
- 两者都保留不确定性：LAS 标出 `machine_only`/`needs_review`，local schema 允许 `unknown`，且严格失败不会伪装成成功。

### 差异

- LAS reference 允许重叠的长事件、瞬时 enter/exit 与持续 occluded 状态，并有对象 inventory、initial/final state 和 outcome；local 输出以无缝、≤1 秒的动作片段为核心，没有这些独立字段。
- local 的粒度更适合 frame-contiguous 训练导出，但会把一个 LAS 语义事件拆成多个片段；当前 primary-single-segment IoU 因而偏保守。
- 遮挡是最大结构差异：11 个 LAS 遮挡事件对应 0 个 local 显式遮挡匹配。动作与 target 的低可比命中（`2/15`、`3/8`）还显示出 schema 之外的对象/动作语义偏移，deep case 中尤其明显。
- LAS reference 本身多为 machine-only，且不同样本含有冻结映射未覆盖的 `approach/contact/push/stop` 等类型；这些不应被偷偷映射或当作 local 错误。

## 6. 失败历史与修复

以下失败均被保留，没有通过修改标注、裁剪模型结果或放宽最终 schema 来制造成功：

1. 早期 GPU 验收先暴露完整时长遗漏、Pass B cardinality/hard-cap 和 enrichment cardinality 问题。修复集中在提示词中的可信数值约束、binary64-safe 可行窗口、确定性拓扑和完整 enrichment skeleton；严格 validator 保持不变。
2. `full_0001` 最初进行了两个 task attempts；每次都有 enrichment initial output 与唯一一次内置 repair output，合计四个 enrichment outputs 全部返回通用 enum rejection。随后先增加只暴露字段 family、不暴露非法 token 的诊断，定位到 skill；公共 skill vocabulary 增加了语义合理的 `touch`。
3. 新运行又出现 actor-state/skill enum rejection。实现了严格、repair-only、仅 enum、allowlist 字段的 `unknown` fallback，并要求封闭 envelope、再次严格验证和公开 warning 一致；首次输出仍必须 strict。
4. fallback 后的首个 `full_0001` 虽然 pipeline COMPLETED，但使用 container duration，末端 `10.934` 比可见视频流 `10.933333333333334` 多 `0.0006666666666657051` 秒，analyzer 正确拒绝。媒体探测先改为 video stream authority，随后发现 FFprobe 的 direct duration 只有六位小数；最终使用 `duration_ts × time_base` 精确 rational，得到一致终点并完成 deep run。
5. 最终 deep run 13 段首轮 strict enrichment 成功，无 normalization。补充样本中 `full_0004` 通过一次 Pass B repair 和 2 个 skill normalization 完成；`full_0021` 的 Pass B 初次/repair 均违反 segment 上限和邻接约束，保持 FAILED 且不评分。
6. 运行环境重启曾导致易失内容丢失。重建后对 wheel、安装树、模型 manifest、数据库和结果做一致性验证；持久化敏感状态采用 authenticated encryption，权限不足的对象存储上不保留明文。

## 7. 是否需要微调

### 7.1 先做 schema 与提示词改进

优先级最高，且不需要改模型权重：

- 增加 `occlusion_enter/occluded/occlusion_exit`、initial/final state、object inventory 和 outcome 的显式输出；否则 11 个遮挡事件永远无法成为同 schema 的命中。
- 增加 `left_hand/right_hand/both_hands` 等人手 actor，不把人手语义损失地别名为 robot gripper；同时保留 single/both 的评估投影。
- 在 fine segments 之上生成可重叠、可跨片段的 grouped semantic events，覆盖“抓取—运输—释放”和持续状态；提示词强化 manipulated object 与 destination 的区分。
- 继续使用封闭错误 family 和 conservative `unknown`，不要把未经证明的别名自动改成一个“看似合法”的枚举。

### 7.2 改进确定性评估

- 在不修改原始 local segments 的前提下，按 action/target/actor 兼容性把相邻片段分组，再同时报告 grouped-event IoU 与现有 single-segment IoU。
- 建立更大的、预先冻结、按遮挡/时长/动作类型分层的盲测集，并引入人工复核 reference。分别报告 `machine_only` 与 human-reviewed 结果。
- 保留 selected、completed、scored 三种分母；pipeline failure 单独计成功率，不能混入语义分数。

### 7.3 可能的针对性微调候选

只有在上述 schema/prompt/grouping 改动后仍在更大人工复核集稳定出现，才考虑：

- 对可见遮挡边界和持续不可见状态做专门 LoRA/SFT；当前 `0/11` 只能证明输出 schema 没有显式 family，尚不能区分“模型没看见”和“无字段可写”。
- 对 manipulated object、destination 和自主物体运动的区分做训练；deep case 的 panel/apple 与 apple/ramp/container 混淆，以及总体 action `2/15`、target `3/8`，是值得继续验证的候选。
- 对系统性 boundary bias 做训练前，应先收集有符号误差。当前只有绝对误差中位数（start `0.20000000000000018` 秒、end `0.4571428571428573` 秒），不能判断偏早还是偏晚，也不能排除 ≤1 秒切段策略本身的影响。

### 7.4 当前证据不支持的结论

5 个入选样本、4 个已评分结果不足以支持广泛模型微调、模型排名、相对 LAS 优劣结论或任何数据集级性能声明。尤其 LAS reference 不是人工 ground truth。当前建议是“暂不做广泛微调，先改 schema/prompt/eval；再根据更大盲测中的重复视觉语义缺陷决定是否微调”。

## 8. 可复现性边界

机器可复现的部分包括：冻结样本 digest、固定请求、视频/hash/时长验证、严格结果拓扑、预冻结映射、IoU/coverage/boundary 算法、canonical metrics 再生成和独立算术复核。最终 metrics 文件两次生成 byte-identical，SHA-256 为 `833abb368769105d114d3725fbf4c6d198aa7c39b08d22418c0697efff99050c`。

仍需人工复核的部分包括：LAS `machine_only` 标注的事实性、同义词 alias 是否充分、长事件与细片段应如何成组、描述中的对象混淆与 hallucination、以及 `needs_review` 样本的 reference 质量。原始视频、LAS JSON、本地原始结果、运行任务标识、服务坐标和凭据均不进入 Git；本报告只提交去敏后的方法与汇总证据。
