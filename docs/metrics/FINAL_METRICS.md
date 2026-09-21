# 最终指标清单（冻结版）

更新时间：2026-09-20

本文件与同目录 `FINAL_METRICS.json` 完整记录 2026-09-20 冻结的指标选择。下方清单是公开的选择依据，不需要访问内部资料。选择范围不等于已实现范围；当前入口及验证证据见 [指标目录](README.md)。

## 最终交付范围

### 21 项主榜

1. MUSIQ / Imaging Quality
2. DOVER
3. Motion Smoothness / AMT
4. Temporal Flickering
5. Dynamic Degree
6. CLIPScore
7. ViCLIP / Overall Consistency
8. Instruction Following
9. Action Binding
10. Motion Binding
11. Motion Order Understanding
12. Motion Rationality
13. Object Interactions
14. PhyGenEval PCA
15. VideoPhy-2 PC
16. VideoPhy-2 SA
17. Mechanics
18. Thermotics
19. Material
20. Subject Consistency
21. Background Consistency

Mechanics、Thermotics、Material 仍属于最终保留的物理专项，但按第 10 节约定只做专项报告，不并入统一主榜总分。

### 5 项固定附表

- PSNR
- SSIM
- LPIPS
- FVD
- FID

这五项分别属于有参考质量和分布质量报告，不与 21 项主榜平均。没有配对真实后续帧时，PSNR/SSIM/LPIPS 报 `N/A`；没有匹配真实集合或统计文件时，FVD/FID 报 `N/A`。

## 单独标记、但不计入上述 26 项

- **Aesthetic Quality**：VBench 诊断项；与 DOVER 的审美分支重叠，不进入主榜。
- **Long Subject Consistency**：长视频附加/诊断报告；不进入 21 项主榜，也不计入固定 5 项附表。
- **Long Background Consistency**：长视频附加/诊断报告；不进入 21 项主榜，也不计入固定 5 项附表。
- **CLIP-IQA+**：图像帧级备用指标；已有提交代码，但按去重决定不进入最终主榜。
- **Physical Adherence**：删除主榜，保留其官方数据/代码作为复核备用；不进入最终名单。
- **VideoPhy-2 Joint**：只按同一 `video_id` 对齐 PC、SA 后计算 `SA >= 4 且 PC >= 4`，是分析字段，不是独立模型或第 22 项指标。

## 实现状态

名单冻结不等于每项代码已经完成。现有 PR 中已提交并完成单视频验证的只有 Motion Smoothness / AMT；其余 20 项主榜和 5 项附表属于最终范围，仍需按各自官方仓库、权重、环境和输入格式补齐。`FINAL_METRICS.json` 的 `implementation_status` 记录这一点，不能把 `to_complete` 误读成“已实现”。

## 官方来源固定版本

- VBench（基础指标、VBench-Long、VBench-2.0）：`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`
- DOVER：`f1ddc96215bc7fbcf8f315c65d47905f339c3419`
- Google Research（MUSIQ、FVD）：`4700efb9afa54286b0e04473ba80a13e8461e25f`
- IQA-PyTorch 依赖：`18dd7a19694e94aac21019170e3f5e63d6b4e19e`
- WorldModelBench、T2V-CompBench、PhyGenBench、VideoPhy-2：按官方代码包内 provenance 记录的仓库和版本执行；不得用其他裁判模型替代。

模型权重、环境和输入模板必须继续遵循各自官方发布物；本清单只冻结“选哪些指标”，不授权改写评分公式或替换模型。
