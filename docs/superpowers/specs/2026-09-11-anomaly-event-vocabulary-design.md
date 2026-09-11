# 异常事件词表与校验违背分流设计（Anomaly Event Vocabulary）

日期：2026-09-11
状态：草案（待评审；§5 与 §6 各含一个需拍板的开放决策）
适用分支：`feat/sam31-evidence-integration`
前置阅读：`2026-09-06-semantic-contract-remediation-design.md`（三态处置语义）、cohort-v5 diag 归因记录（`OCCLUSION_OCCLUDER_NOT_PROPOSED` 出题错案例）

---

## 1. 问题陈述

las_repro 标注管线的全部本体假设是：**视频记录了一个连贯的物理世界**。物体是持久实体（entity table 一经 CV 建立即闭合）、身份不分裂不合并（identity 模块只做保留几何假设）、时间连续（boundary 拓扑校验）、遮挡是唯一合法的"消失"解释（occlusion 候选闭集）。在真实视频上这些假设成立，校验器因此可以把一切违背归咎于**模型输出错误**，走 repair 重试或本地规范化。

在生成视频（世界模型评测的输入）上，这个前提倒置：违背本体假设的可能不是标注模型，而是**视频本身**。当前管线在这种输入下有三个失效形态：

- **形态 A（工具退化）**：伪影把 SAM3.1 / 光流带崩，产生碎裂 mask、身份跳变。这是测量噪声，已有部分信号承接（`low_confidence`、`visibility_runs`、`missing_intervals`、各 `*_complete` 标志），但这些信号按真实视频的失效模式设计，对生成伪影的敏感性未标定。
- **形态 B（本体违背，本设计的核心）**：视频内容本身违反连贯世界假设——物体凭空消失、身份分裂、瞬移、形变。此时"正确标注"在现有 schema 中**无法表达**：模型如实描述异常会触发校验码，然后被 repair 掉或规范化掉。cohort-v5 的灰板案例是精确预演：full_0002 中第二块遮挡板真实存在但不在 CV 实体表，模型如实写 `gray_flat_board_with_dark_trim`，被 `OCCLUSION_OCCLUDER_NOT_PROPOSED` 判拒。在真实视频上这是出题错（CV 漏检）；在生成视频上，"实体表外的物体出现了"可能正是要测的生成错误——而 v6 的 `_normalize_unproposed_occluders` 会把它静默降为 `unknown`。
- **形态 C（VLM 先验归一化）**：VLM 倾向把轻微不连贯的视频补全成合理叙事，标注照常产出、校验照常通过，系统性偏向"视频没问题"。文献中已被命名为 Blind Trust Problem（Robust-TO, arXiv 2606.26904：退化输入下掉 15–30 个百分点且模型不自知）。schema 层无法根治，只能靠 §8 的注入实验 + 人工盲评度量。

本设计回答形态 B：**让"视频不连贯"成为可陈述的标注内容，而不是被修复的输出错误**。形态 A 在 §7 给出信号对照表，形态 C 在 §9 作为已知风险记录。

### 1.1 为什么不能扩现有 `Skill` 枚举

直觉方案是把异常词加进 `Skill`（`pipelines/validators.py:53-74`）。否决，三个理由：

1. **Parity 契约**。该枚举 docstring 明写 "The official LAS semantic event vocabulary"，是 las-parity 对标的锚点（commit d212862 刚把词表统一回官方 19 词，撤销了 roll/static 的私自扩充）。扩词表直接破坏对标目标。
2. **别名耦合**。`SceneEventType = Skill`（`pipelines/scene_semantics.py:26-27`），改一处同时影响 embodied 与 scene 两个分支的全部校验、prompt 契约和已冻结的对齐映射（`evaluation/config/las_alignment_mapping_v1.json` 明确只做精确归一化标签匹配）。
3. **静默吞词风险**。词表在 prompt 中是硬编码散文（`prompts/scene_semantics.txt:55` 与 output schema 段、`embodied_enrichment.txt` 各一份），漏同步时新词不会报错：enrichment 分支在最后一次尝试被 `_normalize_enrichment_enums`（`output_validation.py:1437-1496`）降为 `unknown`，只留一条 `ENRICHMENT_ENUM_NORMALIZED_TO_UNKNOWN` 告警；scene relation 谓词越界则被 `normalize_scene_choice_mechanics`（`scene_choices.py:232-245`）**整行丢弃**。异常事件恰恰是低频关键信号，经不起这种静默损耗。

**结论：新增平行的 `anomaly_events` 输出层**，与 `semantic_events` 并列，官方词表零改动。正常标注分支的行为、缓存 key、parity 指标全部不受影响；异常层可以独立开关（默认关，评测模式开）。

---

## 2. 词表 v0：三源先验合并

词表不闭门发明，从三套已发表分类法取**时序-结构层交集**，再按"能否锚定到本管线的事件/证据结构"筛选：

| 来源 | 取用层 | 弃用层及理由 |
|---|---|---|
| Artifact-Bench（arXiv 2605.18984，三层分类法，30 细类） | Structural Defects 的 identity/morphology 族、Temporal-Semantic Violations 的 motion/causality/scene continuity 族 | Surface Artifacts（色彩/镜头/纹理）不是事件，属观测质量通道 |
| DVAR/GenVID（arXiv 2601.20297，10 类，8 万视频二值标注） | Motion 轴（物体突然显隐、非物理运动）、Appearance 轴的 object deformation | flicker/texture corruption → 质量通道；Camera 轴（不稳定轨迹）v0 不做——需相机运动估计，管线暂无此工具 |
| Spotlight（arXiv 2511.18102，六类带时间区间） | appearance/disappearance、motion、physical violation | adherence（提示遵循）超出感知层职责；anatomy 并入 morphological_change |
| Physion-Eval Glitch_Category | 交叉验证用，不单独引入新类 | 类别粒度与上三者重叠 |

### 2.1 v0 异常事件类型（`AnomalyEventType`，9 值闭集 + 自由文本兜底）

**第一组：实体连续性（entity continuity）** —— 判定锚点是 entity table 与 track 结构，CV 可仲裁性最强。

| 值 | 定义 | 所需 CV 证据 | 闭集可判定 |
|---|---|---|---|
| `entity_appearance` | 实体表外的物体在非入画、非去遮挡情形下出现 | 该区域无既有 track；`visibility_runs` 无先行 missing→visible 转换；非 `edge_departure` 反向情形 | 是（三个排除条件都是既有结构字段） |
| `entity_disappearance` | 已建 track 的实体在非出画、非遮挡情形下消失 | `VisibilityGap` 存在且无 occlusion 候选覆盖该区间、无 `edge_departure` | 是 |
| `identity_split` | 一个实体的 track 在同一帧后对应两个空间分离的观测 | `CrossLabelCue` / 同 entity 双 track 并存（identity 模块已产出此信号，现被仅用于遮挡排序） | 是 |
| `identity_merge` | 两个已区分实体合并为单一观测 | 两条 track 同帧终止 + 单一后继观测同时匹配两者 | 部分（需新增 track 汇合检测） |
| `teleportation` | 实体位置帧间跳变超出运动连续性界限 | 相邻观测 `center_xy` 位移 / 帧间隔超阈值，且非 track 断裂重锚 | 是（纯几何，阈值需 §8 标定） |

**第二组：形态与物理（morphology & physics）** —— 判定需要语义判断，CV 只能提供一致性佐证。

| 值 | 定义 | 所需 CV 证据 | 闭集可判定 |
|---|---|---|---|
| `morphological_change` | 实体形态非物理地改变（对应 Artifact-Bench morphology、Spotlight anatomy） | `area_fraction` / bbox 纵横比突变佐证；判定主体是 VLM | 否（VLM 主判 + CV 佐证） |
| `physical_violation` | 可见的物理因果违背：穿透、悬空、不可逆过程逆转（Artifact-Bench causality / Irreversibility Violation） | `SpatialRelation` 的 inside/overlap 结构佐证 | 否 |

**第三组：时间连续性（temporal continuity）**

| 值 | 定义 | 所需 CV 证据 | 闭集可判定 |
|---|---|---|---|
| `temporal_discontinuity` | 场景级帧间跳变（非剪辑意图的内容突变） | 全帧级信号，v0 由 VLM 主判；后续可加帧间全局特征距离 | 否 |
| `unknown_anomaly` | 无法归入上述类别的可陈述异常 | — | 兜底 |

每条异常事件的记录结构（与 `semantic_events` 行对齐，便于复用现有 provenance 校验）：

```
anomaly_type      : AnomalyEventType
target_object_id  : ObjectId | "unknown"     # 复用现有 ObjectId 正则
start, end        : Timestamp                # 允许 start == end（点异常，见下）
description       : StrictStr (max 1024)     # 自由文本兜底，开放集全靠它
evidence_basis    : Literal["cv_supported", "vlm_only", "cv_contradicted"]
confidence        : float [0,1]
+ 现有 7 字段 provenance（branch="anomaly", model_stage, source_track_ids, ...）
```

设计要点：

- **点异常合法**（`start == end`）。瞬移、突变是点事件，现有 `*_NONPOSITIVE_DURATION` 校验对 anomaly 行放宽为 `end >= start`。这吸取了飞书调研文档 §4 的教训：单点不能静默扩成区间。
- **`evidence_basis` 三值是词表之外最重要的字段**。它把"模型声称"与"证据支持"显式分离：`cv_supported` = 仲裁通过；`vlm_only` = CV 无对应信号但也不矛盾（形态 C 风险区，评分时可单独降权）；`cv_contradicted` = CV 信号与陈述矛盾但保留记录（供人工审计，不进自动指标）。
- **开放集靠 `description`，不靠追加枚举**。9 值闭集覆盖三套先验分类法的可映射交集；未见过的异常落 `unknown_anomaly` + 自由文本，经 §8 的 invalid-outputs 归纳轮次后再决定是否晋升为闭集值。这就是"不需要预判"的机制保证。

### 2.2 与官方 19 词的关系

异常层与正常层可以共存于同一时间区间且互不占位：一段 `transport` 期间发生 `teleportation`，两个分支各记各的。唯一的语义交界是 `occlusion_*` 三词——"消失"必须先排除遮挡解释才能落 `entity_disappearance`，这个排除逻辑由仲裁器执行（§4），不交给 VLM 自由裁量。

---

## 3. 现有校验码的异常语义重审

逐条审查现有校验码在生成视频输入下的语义。绝大多数码保持原判（机械错就是机械错，与输入无关）；下表列出**语义发生倒置或模糊的码**：

| 校验码 | 现处置 | 真实视频语义 | 生成视频下的备选语义 | 分流判定 |
|---|---|---|---|---|
| `OCCLUSION_OCCLUDER_NOT_PROPOSED` | normalize（降 `unknown`，每次尝试） | CV 漏检或模型幻觉 | 实体表外物体真实出现（→ `entity_appearance`） | **进入仲裁**（§4）|
| `SCENE_EVENT_UNKNOWN_OBJECT` | normalize（降 `unknown`，每次尝试） | 模型悬空引用 | 模型看到了 CV 没建 track 的新物体 | **进入仲裁** |
| `SCENE_REQUIRED_OBJECT_MISSING` | retry | 模型漏抄 trusted skeleton | 该物体在视频中途消失，模型如实不再提及 | **进入仲裁** |
| `COARSE_PLAN_ENTITY_UNKNOWN_NAME` | retry | 模型偷懒 | 物体形态异常到无法命名 | v0 不分流（信号太弱），记录频次 |
| `OCCLUSION_INTERVAL_NOT_ALLOWED` | retry | 模型重组端点 | 消失/重现的真实时刻不在 CV 提出的 offer 内（CV 的 visibility 结构本身被伪影破坏） | v0 不分流；§8 注入实验专门观测此码频次变化 |
| `SCENE_SPATIAL_TRACKS_INVALID` | retry | 引用不存在的 track | 同上，track 结构被伪影破坏 | v0 不分流，同上 |

**仲裁范围收敛为 3 个码**。理由：这 3 个码的共同形状是"模型陈述了 CV 闭集之外的实体存在性变化"，恰好是 §2.1 第一组（可仲裁性最强）的镜像；后 3 个码的异常解释依赖"CV 结构本身已被破坏"这个更深的失效，v0 先用注入实验采集频次证据，不急于分流。

**重要不变式**：分流只在**评测模式**（新增开关 `anomaly_detection_enabled`，默认 False）生效。默认关闭时管线行为与当前逐字节一致——parity 实验、缓存 key、已冻结的 cohort 结果零影响。

---

## 4. 校验违背分流：第四处置态

### 4.1 现状（三态）

```
模型输出 → sanitize()
  ├─ valid       → 通过
  ├─ normalized  → 本地修复 + 审计告警（NormalizedSchemaOutput）
  └─ invalid     → repair 重试（max_attempts 2~3）→ 耗尽 → 降级（*_UNAVAILABLE）
```

### 4.2 新增分流（第四态：anomaly-suspect）

```
模型输出 → sanitize()
  ├─ valid / normalized / invalid  （原样）
  └─ invalid 且 anomaly_detection_enabled 且 issue_codes ⊆ 仲裁码集：
       → CV 仲裁器 arbitrate(issue, cv_summary)
            ├─ SUPPORT     → 该 issue 对应内容转录为 anomaly_event（evidence_basis=cv_supported）
            │                其余字段照常走 normalize/repair；不再因该 issue 重试
            ├─ NEUTRAL     → 同上转录，evidence_basis=vlm_only；同时保留原 issue 走原路径
            │                （即：既修复输出，又留下异常嫌疑记录——两边都不丢）
            └─ CONTRADICT  → 原路径照常（repair/normalize），另记 evidence_basis=cv_contradicted
                             的影子记录进 anomaly 分支（不进指标，供审计）
```

三个仲裁判据（对应 3 个分流码，全部只读现有 CV 结构，不新增推理调用）：

1. **`OCCLUSION_OCCLUDER_NOT_PROPOSED`**：取模型写的 occluder 名，若该名无法绑定到任何 entity（复用 `scene_provenance.scene_object_bindings` 的规范化 token 匹配），检查目标 track 在该区间是否确有 `VisibilityGap` 且 gap 无既有候选覆盖 → SUPPORT（"某物遮挡了它，但那个某物不在实体表"至少与"有未建模实体存在"一致）；gap 不存在 → CONTRADICT；gap 存在但已有候选可解释 → NEUTRAL。
2. **`SCENE_EVENT_UNKNOWN_OBJECT`**：模型引用的 object_id 悬空。查该事件时间区间内是否存在**未分配给任何 scene object 的 track 活动**（tracks 中 entity 绑定失败的观测）→ SUPPORT；区间内全部观测已绑定 → CONTRADICT。
3. **`SCENE_REQUIRED_OBJECT_MISSING`**：trusted skeleton 的物体被模型丢弃。查该物体对应 track 的 `visibility_runs`：若后段有长 missing run 且无遮挡候选、无 `edge_departure` → SUPPORT（物体确实从证据中消失了）；track 全程 visible → CONTRADICT（模型就是漏抄了）。

### 4.3 与现有机制的接缝

- **仲裁不消耗模型调用**：三个判据全是对 `CvEvidenceSummary` 既有字段的规则查询。成本为零，且确定性（同输入同判）——这保持了语义缓存的可用性。
- **NEUTRAL 分支两边都走**是有意设计：repair 后的干净输出保证正常分支指标可算，影子异常记录保证信号不丢。代价是这类记录里混有模型幻觉，靠 `evidence_basis=vlm_only` 标记 + 评分降权 + 人工抽检兜住。
- **repair prompt 不感知分流**。分流发生在 sanitize 之后、重试决策之前，`VALIDATION_REPAIR_JSON` 契约（只含 issue_codes）不变。
- **新增审计告警码**（进 `hybrid_result.py:60-70` `AUDIT_WARNING_CODES`，字段形状仿 `SCENE_MECHANICS_NORMALIZED`）：
  - `ANOMALY_ARBITRATED` → `{code, issue_codes, support_count, neutral_count, contradict_count}`
  - `ANOMALY_EVENTS_RECORDED` → `{code, count}`（与 `annotation_branches.anomaly.events` 长度强制相等，仿 degradation_count 口径校验）

---

## 5. 异常事件的产出通道【需拍板：方案 A vs B】

分流（§4）只覆盖"模型撞上校验码"这一被动触发面。主动检测面——模型看到异常但没有撞码（比如瞬移不触发任何现有校验）——需要一个产出通道。两个方案：

### 方案 A：开放陈述 + CV 仲裁（推荐）

在 scene_semantics 阶段的输出 schema 中并列增加 `anomaly_events` 列表（默认允许为空），prompt 增加一段异常词表说明。模型自由陈述，每条产出后经 §4 仲裁器标 `evidence_basis`。

- 优点：覆盖全部 9 类（含 VLM 主判的 morphology/physics 组）；不需要 CV 侧预生成候选（CV 根本不知道"哪里该有异常"）；改动集中在 scene 阶段一个 schema。
- 缺点：形态 C 正面暴露——模型可能漏报（把异常合理化）也可能多报（把伪影当异常）。漏报无解（任何方案都无解），多报靠 `evidence_basis` 分层 + 正常视频误报率指标（飞书调研文档 §7 已列此指标）控制。
- 成本：scene prompt 变长约 400 token；无新增模型调用。

### 方案 B：CV 预生成候选 + 模型闭集选择（仿 occlusion 契约）

在 `cv/summary.py` 新增异常候选生成（如 `VisibilityGap` 无遮挡解释 → `entity_disappearance` 候选；`CrossLabelCue` → `identity_split` 候选），模型只对候选做 confirm/reject，仿 `OcclusionCandidate` + `AllowedEventInterval` 的闭合模式。

- 优点：延续管线"模型只选不写"的核心风格，输出可控性最高，v4 的教训（约束越复杂模型越不稳）风险最小。
- 缺点：**只能覆盖第一组 5 类**（CV 可从结构信号预判的），morphology/physics/temporal 三组 CV 无法出题；且召回上限被 CV 信号质量锁死——伪影恰恰会破坏 CV 信号本身，最需要检测的样本上候选生成最不可靠。
- 成本：`cv/summary.py` 新增候选管线 + 新 prompt 阶段（多一次模型调用/样本）。

### 建议

**v0 走方案 A**，理由：覆盖完整词表、零新增调用、把"CV 可判定"的价值放在仲裁端而非出题端（仲裁对 CV 信号质量的依赖是软的——信号坏了顶多标 `vlm_only`，不会锁死召回）。方案 B 的闭集候选可在 §8 实验证明方案 A 多报严重时，作为第一组 5 类的收紧手段引入（两方案对第一组不互斥）。

**此处需要用户确认后再进入 plan。**

---

## 6. 质量通道与真实性通道的显式分离

形态 A（工具退化）不进异常词表，但必须可见，否则评分时无法区分"视频有异常"与"仪器看不清"。现有信号盘点（代码事实，非设想——注意管线**没有** `low_texture` / `track_interruption` 命名字段）：

| 现有信号 | 位置 | 承接的退化 | 误判风险（若忽略） |
|---|---|---|---|
| `low_confidence` | `OcclusionCandidate`（`cv/summary.py:805`） | 检测置信度低 | 低置信 gap 被当作 `entity_disappearance` |
| `edge_departure` | 同上 :806 | 出画歧义 | 出画被当作消失异常 |
| `visibility_runs` / `missing_intervals` | `SummaryTrack`（:227-244） | 追踪中断的显式区间表达 | 追踪断裂被当作显隐异常（**形态 A 最大风险点**） |
| `*_complete` 标志族 | summary/identity 多处 | 有界扫描触顶、证据截断 | 不完整证据被当作"无证据支持"→ 误标 CONTRADICT |
| `SummaryWarning` 14 码 | `cv/summary.py:345-360` | 截断审计 | 同上 |
| `EvidenceStatus` 三态 | `cv/contracts.py:125-128` | CV 整体可用性 | — |

设计规则（写死进仲裁器实现）：

1. **仲裁器在任何 `*_complete == False` 或相关 `SummaryWarning` 覆盖目标区间时，禁止产出 CONTRADICT**——证据不完整时最多 NEUTRAL。CONTRADICT 必须建立在完整证据上。
2. **`low_confidence` 或 `minimum_confidence` 低于阈值（§8 标定）的 `VisibilityGap` 不得作为 SUPPORT 的唯一依据**。
3. 顶层结果新增 `observation_quality` 摘要块（进 `annotation_branches.anomaly` 旁），聚合上表信号为样本级三档 `reliable / degraded / unreliable`，供评分层按档分层报告——这直接对应飞书调研文档 H2（区分观测质量与事件真实性）。

**待标定项**：上述信号对生成伪影的实际敏感性未知。文献只有图像级腐蚀基准（SAM 在 15 类腐蚀下 mIoU ~0.7，zoom blur 除外；arXiv 2306.07713）和 SAM2 真实视频时序失效刻画（贪心记忆更新级联），**没有任何 SAM 系×生成视频的公开测量**。§8 的注入实验同时为本表做首次标定——这也是该实验可独立成为论文贡献点的原因。

---

## 7. 落地改动点清单

按依赖序。全部改动在 `anomaly_detection_enabled=False` 时零行为差异。

| # | 改动 | 位置 | 量级 |
|---|---|---|---|
| 1 | `AnomalyEventType` 枚举（9 值）+ `AnomalyEvent` 模型（含 `evidence_basis`、点区间放宽） | 新文件 `pipelines/anomaly.py` | 小 |
| 2 | 仲裁器 `arbitrate()`：3 判据 + §6 两条禁止规则；纯函数，输入 `(issue, cv_summary, bindings)` | 同上 | 中 |
| 3 | 分流接线：`embodied.py` 的 `_run_validated_stage` 在 invalid 分支前插入仲裁调用（仅 3 码、仅开关开启时） | `pipelines/embodied.py:504` 附近 | 中 |
| 4 | 方案 A schema：`SceneSemanticsChoices` 增加 `anomaly_events` 字段 + 对应校验码族 `_schema_codes("ANOMALY_EVENT")` + 时序码（放宽 nonpositive duration） | `pipelines/scene_choices.py`、`output_validation.py` | 中 |
| 5 | Prompt 同步：`scene_semantics.txt` 增异常词表段（9 值 + 定义 + "宁可漏报勿编造"指令）；版本号轮换（有意 bust 语义缓存） | `prompts/scene_semantics.txt` | 小 |
| 6 | 结果结构：`annotation_branches.anomaly = {status, events, observation_quality}`；`HYBRID_KEYS` 与精确字段集校验同步；新增 2 个审计告警码及字段形状校验 | `pipelines/hybrid_result.py:35-58, 60-70, 555-566` | 中 |
| 7 | invalid-outputs 通道扩展：`_record_invalid_output` 增记 `status=="normalized"` 且 issue ∈ 仲裁码集的样本（当前只记 invalid，会漏掉被 normalize 吞掉的异常嫌疑） | `workers.py:339-341` | 小 |
| 8 | 评测指标：`las_alignment.py` 不动（parity 口径冻结）；异常层指标另立 `evaluation/anomaly_metrics.py`（事件级 P/R/F1 + 时间 IoU + 正常视频误报率 + 按 `evidence_basis` 分层） | 新文件 | 中 |

**显式不做**：不改 `Skill`/官方 19 词；不改三态处置的既有语义（第四态是纯增量）；不做伪影检测器（异常层陈述"发生了什么"，不陈述"画质如何"——画质归 §6 质量通道）；不动 occlusion 候选生成（方案 B 若启用另行设计）。

---

## 8. 验证方案：受控破坏注入

在上生成视频之前，先在真值已知的环境里验证词表表达力和仲裁准确性。

**材料**：5 个冻结样本（full_0001/0002/0004/0021/0024），标注真值与 CV 行为已有三轮 cohort 基线。

**注入操作**（每种对应词表中至少一个目标类）：

| 注入 | 实现 | 应触发 | 应不触发 |
|---|---|---|---|
| 中段抠除某实体可见帧（非遮挡区间） | ffmpeg 局部替换为背景补全帧 | `entity_disappearance` (cv_supported) | occlusion 分支误报 |
| 拼接同场景另一段，中途换入外观相近新物体 | 片段拼接 | `entity_appearance` | — |
| 把某实体后半段 track 区域平移 N 像素再合成 | 逐帧合成 | `teleportation` | — |
| 时间反转某个不可逆动作段（如放置→拿起） | 帧序反转 | `physical_violation` (vlm_only 预期) | — |
| 局部高斯噪声/压缩伪影（不改内容） | ffmpeg 滤镜 | **什么都不触发**（负对照，考核形态 A 隔离） | 任何 anomaly 事件 |
| 原始视频不动 | — | 全零（误报率基线） | 任何 anomaly 事件 |

**通过判据**（对齐 cohort pilot 的验收风格）：

1. 每类注入在 ≥4/5 样本上产出正确 anomaly_type 且时间区间 IoU ≥ 0.5（点异常改用边界误差 ≤ 1s）；
2. 负对照与原始视频的异常误报 ≤ 1 事件/样本，且误报全部落在 `vlm_only` 档（cv_supported 档零误报——仲裁器的硬指标）；
3. 仲裁三判据在注入样本上的 SUPPORT 判定与注入事实一致率 ≥ 80%；
4. 噪声注入组的 §6 质量信号（`low_confidence`、`missing_intervals` 密度、`SummaryWarning` 频次）相对原始组有可测上移——这是形态 A 敏感性的首次标定数据，单独成表。

**归纳轮次**：全程开 `LAS_DEBUG_INVALID_OUTPUTS=1`，实验后人工审 invalid-outputs.jsonl 与全部 `vlm_only`/`cv_contradicted` 记录，归纳未被词表接住的违背模式 → 回填 §2.1 或确认留在 `unknown_anomaly`。此后再上真实世界模型生成视频做第二轮（无真值，人工盲评抽检 30–50 条）。

---

## 9. 开放问题与已知风险

1. **形态 C 无 schema 解**。模型把异常合理化导致的漏报，词表和仲裁都接不住，只能靠 §8 注入实验量化漏报率、人工盲评长期监控。若漏报率高，缓解方向是异常专用的第二次审视 pass（成本换召回），不在 v0 范围。
2. **仲裁器可能被伪影骗**。SUPPORT 判据依赖 CV 结构，而伪影同时破坏 CV 结构——§6 规则 1/2 把这种情形压到 NEUTRAL，代价是伪影重的样本上 `cv_supported` 召回下降。这是有意的保守取舍：宁可降档不可误判。
3. **异常层无官方真值**。las_alignment 的 parity 口径不适用；注入实验提供合成真值，生成视频上只能对标 Spotlight/Physion-Eval 的人工标注做外部效度检验（协议差异见飞书调研文档 §5，两套评分协议并存的做法直接沿用）。
4. **`morphological_change` 与 `physical_violation` 的边界模糊**（形变到穿透是连续谱）。v0 允许双标（多标签，与 Artifact-Bench 的 diagnostic 定位一致），评分按主类计。
5. **点异常的匹配协议**。现有 `match_events` 基于区间 IoU，点事件需要边界误差匹配分支——`evaluation/anomaly_metrics.py` 设计时定，先记录。

## 10. 决策请求汇总

- **§5：产出通道方案 A（开放陈述+仲裁，推荐）还是方案 B（CV 出题闭集选择）**——影响改动点 4/5 的形状。
- **§3：分流码集 v0 收敛为 3 个码**，后 3 个观察码是否同意先不分流、只采集频次。
- §8 注入实验是否需要在扩大生成视频规模之前作为硬性门槛（建议是，对齐"阶段 A 资源与协议审计"的推进风格）。
