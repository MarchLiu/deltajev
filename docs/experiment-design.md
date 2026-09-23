# deltajev 实验设计（冻结前草案 v0.1）

> 提示词、样本 id、指标在首个 frozen run 之前定稿；之后不得改动，只允许追加
> 新的协议版本。每条预测记录 prompt 的 sha256，报告要求可字节级复现。

## 1. 研究问题

| # | 假设 | 度量 | 对照 |
|---|---|---|---|
| H1 | GDN 混合架构的 fresh 模式延迟随 state 长度增长更慢 | 前向耗时 vs input tokens（254 / 1.4k / 5k / 7.7k 四档） | 同协议下 Qwen3.5-4B / Qwen3.8-27B 逐层消融（关 GDN 不可能，退而对照 dense 模型阶梯） |
| H2 | shared-state 吞吐优势在 GDN 上被放大 | decisions/s（fresh vs shared），KV/分支增量显存 | SemIf 公布的 4B 数字（20.03 decisions/s） |
| H3 | 27B 决策质量逼近 frontier-agreement | balanced acc.、TVD、与 TypeSafe 公开子集一致性 | SemIf 4B：0.813 / 0.177 / 0.845；公开 Jev 值 0.883 / 0.127 |

## 2. 评测矩阵（冻结）

- **自建集**：8 领域 × 13 决策（沿用 SemIf authored144 的形状：客服、内容审核、
  代码评审、事故分级、邮件意图、合规、信贷、工单优先级），先跑 104 行快评。
- **扰动集**：选项顺序反转、缺证据（应答 insufficient）、同义改写。
- **TypeSafe 公开 workflow evals 可选公开行**：只对齐可选行，不聚合宣称 711 行。
- **公共基准**：LocalLLaMA/typed-decisions（2,000 决策 / 4 workflows，标注者
  概率分布）用于 TVD 与 soft-label 对比。
- **延迟探针**：合成 state，4 个 token 长度档，各 20 次取中位数。

## 3. 运行协议

1. `configs/eval.yaml` 先冻结（prompts/ids/metrics 哈希入 results/raw/manifest.json）。
2. 每模式（fresh / shared）全量跑一遍，逐行写 `results/raw/predictions_<mode>.jsonl`。
3. 报告数字只来自冻结矩阵；额外探索性结果单独标注 `exploratory`。
4. 已知风险（从复现生态继承，必须显式报告）：
   - softmax 概率 conditional on declared options，**未经校准**，不得当操作置信度；
   - 选项顺序敏感性：报告反转导致的 argmax 翻转次数；
   - 27B BF16 内存 ~55GB；Mac 统一内存 <48GB 用 4-bit 量化并单独报告。

## 4. 消融与后续

- A1：shared 前缀长度对分支成本（验证 GDN 层 KV 固定的论断）。
- A2：`logits_to_keep=1` vs 全 logits 的内存/延迟。
- A3（可选）：soft-label 蒸馏 + 温度校准（对齐 Open Jev 的 ECE 路线）。
- A4（可选）：MLX / GGUF 量化档位对决策一致性的影响。

## 5. 报告模板

results/report.md：协议哈希 → 主表（H1/H2/H3）→ 扰动与失败案例 → limitations。
所有与公开数字的对比标注来源与协议差异（如 102 行可选子集 ≠ 711 行聚合）。
