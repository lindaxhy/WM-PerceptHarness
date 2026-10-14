# 顶层模块逐函数审查

审查基线：`5ac95b1`（PR #33 第一轮清理之后）。覆盖用户列出的 11 个模块，包括类方法、协议声明、Pydantic/日志回调及嵌套函数；不把“仅一次调用”或“通过回调调用”当成无用代码。
共逐项检查 **97 个显式函数/方法定义**。dataclass 自动生成的方法不计入。

## 结论

- 直接删除：无调用方的 `get_inference_job`（协议和实现）及 `allowed_model_aliases`。
- 内联：`Settings.from_env`、`_decimal_text`；移除 `_schema_validation_context` 的未用参数。
- 合并模块：执行、数据记录、运行指标归入 `runner.py`；权重检查归入轻量的 `video_metrics` 包。
- 去掉一次重复归档校验；写标注时仅计算一次 sidecar 路径。
- 用严格格式匹配加标准库 Fraction 替代 time_base 的手工分割/整数解析，保持接受范围和时长计算。
- 保留：批处理/限流/资源释放、请求缓存、结果恢复、时间戳及模式校验。
- 后续候选：未激活的视频会话链和未被 pipeline 使用的 task_dir；以及同步执行仍保留的 job/wait 接口。本轮不以删除这些接口改变自定义执行路径。尤其 `wait_for_jobs(timeout)` 并不实现超时，真正的请求超时属于模型传输层。

## 逐函数清单

### `__init__.py`

没有显式函数。仅包说明及版本号，保留。

### `batch.py`

| 函数/方法（原行号） | 用途/调用证据 | 处理 |
|---|---|---|
| `RequestGate.__init__` (33) | 初始化限流器、日志处理器或 GPU worker（见所属类） | 保留 |
| `RequestGate.__enter__` (40) | 请求进入限流 gate，超过并发限制时等待 | 保留 |
| `RequestGate.__exit__` (46) | 释放限流容量并逐步增加健康并发 | 保留 |
| `RequestGate.throttled` (56) | HTTP 429 后降低并发 | 保留 |
| `RequestGate.snapshot` (62) | 加锁读取进度计数 | 保留 |
| `_ThrottleHandler.__init__` (68) | 初始化限流器、日志处理器或 GPU worker（见所属类） | 保留 |
| `_ThrottleHandler.emit` (72) | logging 回调；识别模型 429 通知 | 保留 |
| `read_manifest` (77) | 校验任务 ID、重复项及视频路径 | 保留 |
| `assignment` (98) | 分片选择和 GPU 分配共用的稳定哈希 | 保留 |
| `_cv_worker` (102) | spawn 子进程入口，拥有 SAM 和 IPC | 保留 |
| `GpuSlot.__init__` (136) | 初始化限流器、日志处理器或 GPU worker（见所属类） | 保留 |
| `GpuSlot.receive` (147) | 接收 IPC 并检测 worker 死亡 | 保留 |
| `GpuSlot.wait_ready` (156) | 验证 worker 启动状态 | 保留 |
| `GpuSlot.__call__` (163) | 串行化同 GPU 访问并发送任务 | 保留 |
| `GpuSlot.close` (171) | 停止 GPU 进程并关闭连接 | 保留 |
| `archive_cv_evidence` (181) | 在有限缓存外持久化 SAM 证据 | 保留 |
| `_verify_cv_archive` (208) | 验证临时/已有归档；删除碰撞分支中的一次重复调用 | 保留 |
| `required_event_phases` (225) | 校验 reviewed/confidence-gated 所需阶段 | 保留 |
| `ReviewRequiredRunner.evaluate` (237) | ReviewRequiredRunner 对新结果验证 schema/阶段完整性 | 保留 |
| `run_manifest` (255) | manifest 入口，协调 GPU、线程、限流及进度 | 保留 |
| `run_manifest.status` (289) | 输出启动/运行/结束状态 | 保留 |
| `run_manifest.reporter` (305) | 监控线程周期性写进度 | 保留 |
| `run_manifest.worker` (309) | 线程池工作函数，独立模型客户端和任务队列 | 保留 |
| `run_manifest.worker.generate` (313) | 包裹模型 generate，让请求通过共享 gate | 保留 |
| `run_manifest.worker.protected_cv` (318) | 调用 CV 前登记活跃 cache key，防止过早回收 | 保留 |

### `cli.py`

| 函数/方法（原行号） | 用途/调用证据 | 处理 |
|---|---|---|
| `main` (42) | percept 入口；分发评测及标注 | 保留 |
| `_parser` (63) | 构建参数、默认值及 help | 保留 |
| `_eval` (131) | 标注入口，区分 manifest/单进程并映射退出码 | 保留 |
| `_load_backend` (204) | CLI 和批处理线程创建独立模型客户端 | 保留 |
| `_single_alias` (254) | fake/OpenAI 共用默认模型选择，拒绝歧义 | 保留 |
| `annotation_cv_preflight` (262) | CLI/worker 共用 provider 策略及资源检查 | 保留 |
| `_cv_executor` (277) | 按需加载 CV 的外层资源生命周期 | 保留 |
| `_cv_executor.execute` (282) | 首次请求才实例化 CV executor 的闭包回调 | 保留 |
| `_loaded_cv_executor` (294) | 已加载 CV provider/cache 的获取和释放 | 保留 |
| `_configure_execution_chunk_frames` (345) | 设置并检查 chunk 大小；单一调用，可内联但保留可读性 | 保留 |

### `config.py`

| 函数/方法（原行号） | 用途/调用证据 | 处理 |
|---|---|---|
| `_integer` (14) | Pydantic 回调：拒绝布尔/小数并解析整数字符串 | 保留 |
| `_finite_number` (24) | Pydantic 回调：拒绝布尔、NaN 和无穷值 | 保留 |
| `_strict_boolean` (36) | 保持环境变量 true/false 的严格语义 | 保留 |
| `Settings.openai_model_aliases` (113) | 模型 ID/别名映射；CLI 和配置校验调用 | 保留 |
| `Settings.allowed_model_aliases` (125) | 无调用方 | 删除 |
| `Settings.validate_model_configuration` (129) | Pydantic 自动调用；检查互斥配置、别名冲突及 URL | 保留 |
| `Settings.validate_cv_configuration` (144) | Pydantic 自动调用；检查采样 FPS 上下界 | 保留 |
| `Settings.from_env` (150) | 仅返回 cls()；CLI/batch 可直接 Settings() | 内联 |

### `execution.py`

| 函数/方法（原行号） | 用途/调用证据 | 处理 |
|---|---|---|
| `InferenceJobFailed.__init__` (37) | 初始化异常、缓存或执行器状态（见所属类） | 保留，合入 runner |
| `JobStore.create_inference_jobs` (49) | pipeline 的同步阶段入口；执行模型/CV 并保留终态 | 保留，合入 runner |
| `JobStore.get_inference_job` (53) | 仅存在声明/实现，没有调用方 | 删除 |
| `JobStore.list_inference_jobs` (55) | wait_for_jobs 按 task_id 取回执行记录 | 保留，合入 runner |
| `CvJobExecutor.__call__` (61) | CV executor 的 callable 类型协议 | 保留，合入 runner |
| `SemanticResultCache.__init__` (88) | 初始化异常、缓存或执行器状态（见所属类） | 保留，合入 runner |
| `SemanticResultCache.key` (91) | 根据请求、输入元数据及模型身份生成缓存键 | 保留，合入 runner |
| `SemanticResultCache._entry_path` (114) | load/publish 共用分片缓存路径 | 保留，合入 runner |
| `SemanticResultCache.load` (117) | 读取请求缓存；损坏或旧格式视为未命中 | 保留，合入 runner |
| `SemanticResultCache.publish` (131) | 原子写入请求缓存；写失败时允许继续推理 | 保留，合入 runner |
| `SyncJobStore.__init__` (165) | 初始化异常、缓存或执行器状态（见所属类） | 保留，合入 runner |
| `SyncJobStore.create_inference_jobs` (194) | pipeline 的同步阶段入口；执行模型/CV 并保留终态 | 保留，合入 runner |
| `SyncJobStore.get_inference_job` (237) | 仅存在声明/实现，没有调用方 | 删除 |
| `SyncJobStore.list_inference_jobs` (240) | wait_for_jobs 按 task_id 取回执行记录 | 保留，合入 runner |
| `SyncJobStore.close` (243) | 释放已附加的视频会话；当前内建 pipeline 不提供 video_metadata | 保留，合入 runner |
| `SyncJobStore._execute_cv` (258) | 调用 CV executor 并检查是否已配置；可内联但不是死代码 | 保留，合入 runner |
| `SyncJobStore._execute_model` (263) | 构造模型请求、验证输出、重放/写入缓存及请求清理 | 保留，合入 runner |
| `SyncJobStore._attach_session` (331) | 仅当 payload 带 video_metadata 时附加会话；当前内建 pipeline 未激活 | 当前未使用/未激活；后续候选 |
| `SyncJobStore._remember_session` (342) | 记录附加会话；当前内建 pipeline 未激活 | 当前未使用/未激活；后续候选 |
| `wait_for_jobs` (347) | 同步结果收集及失败传播；不等待，也不实施 timeout | 保留，合入 runner |
| `_model_request` (379) | 将 stage payload 转为已验证的 ModelRequest | 保留，合入 runner |
| `_model_request_metrics` (420) | 读取 provider 指标并添加推理耗时 | 保留，合入 runner |
| `_schema_validation_context` (433) | 校验 schema_context 是 mapping；request 参数未使用 | 保留，删未用参数 |
| `_video_session_metadata` (445) | 会话元数据解析；当前内建 pipeline 未激活 | 当前未使用/未激活；后续候选 |
| `_finite_number` (469) | 模型时间段/FPS 的有限实数校验 | 保留，合入 runner |

### `media.py`

| 函数/方法（原行号） | 用途/调用证据 | 处理 |
|---|---|---|
| `TimeSpan.__post_init__` (47) | dataclass 自动调用，保证时间段有限且有效 | 保留 |
| `probe_video` (60) | 标注读取时长、分辨率、帧率 | 保留 |
| `playback_origin` (113) | 视频读取及 CV timeline 共同使用的播放起点 | 保留 |
| `_video_duration` (126) | 处理视频流相对容器起点的延迟 | 保留 |
| `_timestamp_duration` (144) | 将 duration_ts/time_base 转成时长，并处理缺失/坏值 | 保留 |
| `_positive_finite_float` (184) | 时长解析的共享容错规则 | 保留 |
| `extract_frames` (196) | OpenAI 模型适配器抽取带时间戳的视频帧 | 保留 |
| `_video_frame_rate` (270) | 平均帧率无效时回退到 r_frame_rate | 保留 |
| `_decimal_text` (284) | 只有一个调用方；仅执行 format(value, "f") | 内联 |

### `metrics.py`

| 函数/方法（原行号） | 用途/调用证据 | 处理 |
|---|---|---|
| `validate_inference_job_metrics` (41) | 阶段完成时校验指标键、值类型、范围及序列化大小 | 保留，合入 runner |

### `results.py`

| 函数/方法（原行号） | 用途/调用证据 | 处理 |
|---|---|---|
| `_unique_pairs` (11) | JSON decoder 回调：拒绝重复键 | 保留 |
| `read_record` (20) | 读 JSON object；拒绝无穷值、坏文件和非对象 | 保留 |
| `audit_path` (29) | 读取、写入和路径分配共用 sidecar 命名规则 | 保留 |
| `is_annotation_record` (33) | 区分 audit sidecar 和恰好以 .audit.json 结尾的标注 | 保留 |
| `annotation_record` (38) | 评分和 resume 读取公共 JSON + audit 或旧 envelope | 保留 |
| `write_annotation` (56) | 以 audit 状态为提交标记，处理两文件写入中断 | 保留 |
| `result_path` (78) | 识别归属并避免同名视频覆盖 | 保留 |
| `completed_result` (101) | 校验 source/status/schema，决定是否复用 | 保留 |
| `atomic_text` (115) | JSON 和 JSONL 共用的原子文件替换 | 保留 |
| `write_json` (130) | 结果、批处理进度和归档元信息共用 JSON 写入 | 保留 |

### `runner.py`

| 函数/方法（原行号） | 用途/调用证据 | 处理 |
|---|---|---|
| `SyncRunner.__init__` (42) | 初始化模型、配置和可选请求缓存 | 保留 |
| `SyncRunner.evaluate` (66) | 单视频标注、上下文创建及失败结果封装 | 保留 |
| `collect_videos` (133) | CLI 展开视频目录、排序和去重 | 保留 |
| `run_batch` (156) | 顺序执行、复用已完成结果及写汇总 | 保留 |
| `_task_record` (189) | 构造 pipeline 输入与任务 ID；可内联但独立职责清楚 | 保留 |
| `_task_dir` (208) | 创建并清理任务临时目录；当前 pipeline 未读取 task_dir | 当前未使用/未激活；后续候选 |
| `_write_result` (219) | 单视频及 manifest 失败修正共用的结果序列化 | 保留 |
| `_write_aggregate` (232) | JSONL 汇总，调用原子写入 | 保留 |

### `types.py`

没有显式函数。包含阶段状态/请求/结果及任务输入 dataclass；仍有调用方，定义合入 runner。

### `weights.py`

| 函数/方法（原行号） | 用途/调用证据 | 处理 |
|---|---|---|
| `check_weights` (14) | CLIP-IQA、motion 和安装检查共用权重路径/非空检查 | 保留，归入 video_metrics |

## 容易误删的行为

- `ReviewRequiredRunner.evaluate` 与 worker 都检查阶段完整性：前者覆盖新推理，后者还覆盖跳过推理的 resume 结果，不能直接删第二次检查。
- `_cv_executor` 与 `_loaded_cv_executor` 分别承担延迟加载与已加载资源释放，不是重复实现。
- `read_record` 的有限 JSON 校验还会捕获嵌套无穷值和数值溢出，不能只保留 `json.loads`。
- 配置的 Pydantic validators、日志 `emit`、上下文管理器和 worker target 都是隐式调用，不应按普通调用次数判断。

## 与 WBench 抽帧工具的区别


WBench `src/utils/video_utils.py`（对照 checkout `753cc98`）顺序解码视频，按帧索引选择并返回 PIL 图片；这个函数不返回帧时间戳。Percept 的 media 模块还承担元数据、播放起点和带时间戳的抽帧结果，供 VLM 提示与原始视频/SAM 时间轴配合使用。

这些职责解释了部分代码量，但不能证明现有实现都必要。当前每帧单独运行 FFmpeg，进程/重复 seek 成本可以优化；返回的 timestamp 是请求 seek 的时间，不是实际解码帧的 PTS。单次解码替换应验证延迟视频、容器偏移、变帧率、片段边界及选中图像，不能直接用 index/fps 替换后声称等价。本 PR 保留抽帧选择行为，仅简化时间基解析。
