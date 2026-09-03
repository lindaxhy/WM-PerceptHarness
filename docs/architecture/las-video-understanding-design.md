# LAS 视频精细理解复现设计

日期：2026-09-02

## 1. 目标

实现一个完全自托管的视频精细理解服务。服务复现《LAS 视频精细理解算子 × 方舟模型接入指南》中的异步调用方式，但模型推理不调用、不转发方舟或其他外部模型 API。

第一版交付必须满足：

- 提供兼容的 `POST /api/v1/submit` 与 `POST /api/v1/poll` 接口。
- 支持 `operator_id=las_long_video_understand` 和
  `operator_id=las_video_understanding`，版本严格为 `operator_version=v1`。
- 支持显式 `task_template` 模式以及快速入门使用的非空 `query` 模式；仅提供
  `query` 时，本地有效模板为 `general_video_captioning`。
- 支持 `task_template=general_video_captioning`。
- 支持 `task_template=embodied_active_object_detection`，输出稳定的主要交互物体词表。
- 支持 `task_template=embodied_action_captioning`，输出任务摘要、连续动作时间片段和具身动作结构化字段。
- 使用本地部署的视频多模态模型理解视频。
- 对长视频进行分段理解，并输出带时间范围的精细描述与全局摘要。
- 异步任务状态和结果在服务重启后仍可恢复。
- 可以在无 GPU 的开发环境中用伪模型运行接口和流程测试。
- 可以在目标 GPU 服务器上完成真实模型端到端测试。

## 2. 非目标

第一版不包含：

- 从零预训练视频多模态基础模型。
- 在尚无评测基线和标注集时进行 LoRA/SFT 微调。
- 完整复刻 LAS 未公开的内部算法、响应字段或计费系统。
- 多机任务调度、高可用数据库和租户计费。
- 转发或保存客户的方舟 API Key。
- 处理音轨、语音识别或把音频内容作为模型证据。

## 3. 已确认的运行环境

目标服务器当前配置：

- 4 张 NVIDIA GeForce RTX 5090，每张约 32 GB 显存。
- 15 个可用 CPU 核心。
- 117 GiB 内存。
- PyTorch 2.10.0、CUDA 12.8，四张 GPU 均可被 PyTorch 识别。
- FFmpeg/FFprobe 4.4.2。
- 根盘约 93 GB 可用；共享存储约 273 GB 可用。
- 服务器不能直接访问 Hugging Face，模型权重和依赖需要通过可联网机器下载后传输，或使用可达的内部/国内镜像。

模型权重、数据集、视频缓存和结果归档应放在大容量持久化存储；根盘只放代码、虚拟环境和小型运行文件。

## 4. 方案选择

### 4.1 方案一：整段视频直接输入模型

优点是实现简单。缺点是长视频会产生大量视觉 token，容易超出显存或上下文限制，也不利于定位遗漏事件和并行执行。

### 4.2 方案二：分层长视频理解管线（采用）

先解码和切段，再让模型逐段生成结构化视觉事件，最后按时间线聚合。该方案可控制显存、并行使用四张 GPU、保留时间定位，并可独立评测每一阶段。

### 4.3 方案三：先微调专用模型

微调可能提高特定领域效果，但必须以基线评测和标注数据为前提。第一版先建立可测量的零微调基线，随后只针对稳定缺陷开展 LoRA/SFT。

## 5. 总体架构

系统分为控制面、任务存储、预处理、模型推理和结果聚合五个边界清晰的组件。

### 5.1 API 控制面

FastAPI 接收 Submit/Poll 请求，完成鉴权、字段校验、任务创建和状态查询。API 进程不加载模型，也不执行耗时视频处理。

任务状态机为：

`PENDING -> RUNNING -> COMPLETED`

任务在校验后或运行中失败时进入 `FAILED`。终态为 `COMPLETED` 或 `FAILED`，不能再回到非终态。

### 5.2 任务存储与本机队列

SQLite 采用 WAL 模式，保存顶层任务、段级作业、状态、阶段进度、结果、错误摘要和时间戳。协调 worker 通过短事务原子领取顶层任务，完成下载、预处理、创建段级作业和最终聚合；GPU worker 只领取已就绪的段级作业。两类领取都使用租约，避免重复执行并支持崩溃恢复。

第一版面向单台四卡服务器，不引入 Redis。任务存储封装为接口，以便未来替换为 PostgreSQL/Redis 队列而不改变 API 和推理管线。

### 5.3 视频预处理

预处理阶段执行：

1. 将允许的视频来源解析为本地只读文件。
2. 使用 FFprobe 获取时长、帧率和分辨率等视频元数据。
3. 按最大段长和镜头边界生成重叠视频段。
4. 为每段进行限额抽帧，保留相对与绝对时间戳。

第一版视频来源支持本地路径和 HTTP(S)。`tos://` 通过独立下载适配器处理；未配置 TOS 凭据时返回明确错误，不静默改写地址。

CPU 预处理并发默认不超过 10，保留核心给 API、SQLite、系统和 GPU 数据装载。所有并发数都可通过配置修改。

### 5.4 本地模型推理

默认模型为 `Qwen/Qwen3-VL-8B-Instruct`，通过统一 `VideoModel` 接口加载。模型适配器接收一组带时间戳的帧和任务提示词，返回严格校验的结构化段结果；不提取或处理音轨。

四个 GPU worker 各自固定到一张 GPU。每个 worker 串行运行一个段级模型推理作业；同一顶层任务的不同视频段可以跨 GPU 并行，但同一段只由一个 worker 处理。协调 worker 等待该任务的全部段级作业进入终态后执行聚合。若后续模型无法单卡运行，可增加张量并行适配器，而不改变上层任务协议。

开发和测试使用 `FakeVideoModel`，它不下载权重、不需要 GPU，并产生确定性结果。

### 5.5 分层聚合

段级结果至少包含：

- `start_time`、`end_time`
- 场景与环境
- 人物或主体
- 动作与事件
- 屏幕文字或字幕
- 不确定性说明

聚合器按绝对时间排序，合并重叠段的重复事件，保持不确定性而不捏造缺失信息，并生成：

- 视频总体摘要
- 按时间排列的事件时间线
- 视频与处理元数据
- 每阶段告警

第一版使用确定性规则完成去重和时间合并，再使用同一本地模型生成全局自然语言摘要。结构化时间线必须在模型汇总失败时仍可返回。

### 5.6 具身动作时序管线

《LAS 视频精细理解 API：具身动作时序标注》和《逆矩阵 Pipeline · 调用方法》作为具身任务的补充协议。若同一文档中的版本互相冲突，采用日期最新的 `0805 update：两阶段 Actions + Boundary Fine Segments`；较早版本仅作为回归样例和成本对照。

具身管线由以下阶段组成：

1. **Active Objects（可选）**：`embodied_active_object_detection` 对腕部近景或左右腕水平拼接视频生成 `objects[].category` 与稳定的 `instances[].name/description`。结果可由调用方整理后放入动作任务的 `task_context.prompt_context`。
2. **Pass A — Coarse Actions**：对完整主视角视频生成英文 `task_description` 与覆盖完整时间轴的粗动作 `actions`。每个 action 具有 `action_index/start/end/description/event_type`。
3. **Pass B — Boundary Fine Segments**：输入同一视频和 Pass A JSON，为每个 action 产生基于可见物理状态变化的 `boundary_points`，再从相邻边界生成不超过 1 秒的 `fine_segments`。
4. **Stage 4 — Dataspec Enrichment**：在本地代码固定时间边界和英文 caption 后，模型只补全 `actor/actor_state/skill/target/visual_motion_state/confidence` 六个字段。
5. **Postprocess/Export**：确定性代码校验并展平结果，生成 API 的 `task_description/segments`，同时可导出训练数据格式 `action_captions.jsonl`。

本地验证器而非模型拥有最终时间边界。它必须保证 action 和 fine segment 按时间升序、无空隙、无重叠、完整覆盖父区间；边界 ID 引用存在且时间相等；段长不超过配置上限（默认 1.0 秒）。模型结果违反约束时先做一次带错误详情的修复请求，仍失败则任务进入 `FAILED`，不能用均匀网格静默替换模型边界。

方舟文档中的 `use_responses_api`、`previous_response_ids` 和 `expire_in` 是云端输入缓存协议。本地服务为兼容已有请求可以接收这些字段，但必须像方舟密钥字段一样在校验后丢弃并返回忽略告警：不写数据库、不参与推理。本地实现使用 `VideoSession` 在同一任务的多个阶段复用视频元数据、抽样帧与模型预处理结果；模型适配器可进一步实现视觉特征缓存，但不能把方舟缓存 ID 暴露为本地依赖。

## 6. API 契约

### 6.1 Submit

请求形状与指南保持兼容：

```json
{
  "operator_id": "las_video_understanding",
  "operator_version": "v1",
  "data": {
    "video_url": "https://example/video.mp4",
    "query": "按时间顺序描述画面中可见的动作",
    "model_name": "qwen3-vl-8b-instruct"
  }
}
```

`operator_id` 只接受 `las_long_video_understand` 或
`las_video_understanding`，`operator_version` 只接受 `v1`。Submit `data`
必须提供三个受支持 `task_template` 之一，或提供非空白 `query`。省略模板且
`query` 非空时，持久化和本地路由使用有效模板
`general_video_captioning`；缺少两者或仅提供空白 `query` 时拒绝请求。

本地推理显式接受并持久化 `query`、有限且大于零的 `fps`，可选的
`media_resolution/reasoning_effort/clip_context`（均为
`low|medium|high`），以及成对出现的有限 `start/end`（满足
`0 <= start < end`）。显式模板仍只接受 `general_video_captioning`、
`embodied_active_object_detection` 和 `embodied_action_captioning`。

成功创建后返回：

```json
{
  "metadata": {
    "task_id": "<uuid>",
    "task_status": "PENDING",
    "business_code": "0",
    "error_msg": ""
  }
}
```

`model_name` 映射到本地允许列表，不能作为任意文件路径。为兼容已有客户端，请求中可以出现 `ark_api_key`、`ark_endpoint_id`、`use_responses_api`、`previous_response_ids` 与 `expire_in`，但服务在请求校验后立即丢弃这些字段：不写日志、不写数据库、不参与推理、不转发网络请求。如果出现这些字段，Submit 响应在 `metadata.warnings` 可选字符串数组中说明它们已被忽略；未出现时省略该数组。

`task_context.prompt_context` 是可选的短文本命名提示。服务把它作为上下文而不是动作 SOP：上下文中出现但画面没有交互的物体不能被强行写入结果。

### 6.2 Poll

Poll 以 `operator_id`、`operator_version` 和 `task_id` 查询任务。服务把请求中的算子 ID 与该任务提交时持久化的算子 ID 比较，而不是统一改写为某个别名。非终态返回当前状态和阶段进度；完成时在 `data` 返回结果；失败时返回稳定的业务错误码和不含敏感信息的错误摘要。

不存在的任务、算子不匹配、版本不匹配分别返回可区分的错误。HTTP 状态码表达协议层错误，`business_code` 表达业务执行结果。

## 7. 安全与错误处理

- API Key 只通过哈希比对，不以明文保存。
- 日志过滤所有名称包含 `key`、`token`、`authorization` 的字段。
- HTTP 下载限制协议、最大文件大小、连接/读取超时和重定向次数，并阻止访问回环、链路本地和私有网段，降低 SSRF 风险。
- 本地视频路径必须位于配置的允许目录中，解析符号链接后再次校验。
- 每个任务使用独立工作目录；成功、失败或取消后按保留策略清理临时帧和视频片段。
- 模型输出必须经过 JSON Schema/Pydantic 校验。修复重试耗尽后保留阶段错误，不能把未校验文本伪装成成功结果。
- Worker 崩溃后，超过租约时间的 `RUNNING` 任务可重新领取；每个阶段以幂等产物标识避免重复执行已完成工作。

## 8. 配置

配置由环境变量和可选配置文件提供，至少包括：

- 服务监听地址和 API Key 哈希。
- SQLite 路径、工作目录、模型目录、缓存目录。
- 本地视频允许目录和 HTTP 下载限制。
- 模型别名、模型路径、推理精度和每段最大帧数。
- 视频段长、重叠时长、CPU 并发和 GPU worker 映射。
- 任务租约、超时、重试和临时文件保留策略。

仓库只提供 `.env.example`，不提交真实密钥、服务器地址、私钥、模型权重、视频或任务数据库。

## 9. 测试与验收

### 9.1 单元测试

- 两个算子 ID、严格 `v1`、显式模板/查询模式、调优字段边界、状态机和错误码。
- 敏感字段过滤。
- SQLite 顶层任务/段级作业的原子领取、租约恢复和持久化。
- 时间段生成、重叠去重和时间线排序。
- 具身 action/fine segment 的连续覆盖、边界引用、最大时长和枚举校验。
- 22 维历史 dataspec 输入兼容到六字段最新输出的映射与拒绝路径。
- 模型输出校验与失败路径。
- URL/本地路径安全校验。

### 9.2 集成测试

- 使用 FakeVideoModel 分别以两个算子 ID、查询模式和显式模板完成 Submit -> Worker -> Poll 全流程，并验证 Poll 匹配任务保存的算子 ID。
- 使用 FakeVideoModel 完成 active objects -> Pass A -> Pass B -> Stage 4 -> JSONL 导出的具身全流程。
- 服务重启后仍能查询已创建和已完成任务。
- 使用无音频的短视频夹具验证 FFmpeg 视频元数据和抽帧。
- 并发提交时任务 ID 唯一、任务不重复领取。

### 9.3 GPU 验收

在目标服务器上：

1. 加载本地 Qwen3-VL-8B 权重并验证四张 GPU 可分别推理。
2. 对含明显画面变化的无音频短视频完成端到端任务。
3. 对超过单段长度的视频验证切段、跨段排序、重叠去重和全局摘要。
4. 检查显存峰值、处理耗时、临时空间和失败后的显存释放。
5. 断开外网或监控出站连接，证明模型推理不调用方舟及其他外部模型服务。

## 10. 微调决策门槛

第一版建立固定评测集，至少测量事件覆盖率、时间定位误差、人物/主体一致性、描述事实性和结构化输出成功率。

只有当同类错误在评测集中稳定复现，且通过提示词、抽帧、切段或聚合策略仍无法解决时，才进入 LoRA/SFT。训练数据必须保留视频、时间段、问题/任务、目标结构化结果和数据许可信息。

## 11. Git 与交付纪律

- 初始化 Git 后先提交原始指南和本设计。
- 实现按可独立验证的功能切分提交，不把模型权重或运行数据提交进 Git。
- 每个实现提交前运行相关测试；最终提交前运行完整测试和 GPU 端到端验收。
- README 记录本地 Fake 模式、GPU 模式、API 示例、配置、安全限制和部署步骤。
