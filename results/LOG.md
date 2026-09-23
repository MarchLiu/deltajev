# 实验日志 — 2026-09-22

## 基准

LocalLLaMA/typed-decisions（社区公共基准，[HF](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)）：
400 test cases × 5 questions = 2,000 decisions，4 workflows
（agent_trace_observability / customer_service / invoice_processing / security_incidents）。
gold = 标注者共识分布；每题类型 noul / choice / score，最大 5 选项。

## 基线口径

- random-guess 期望准确率（按选项数加权）：**0.318**
- 公开参考值（同基准 test split，引自各方发布页）：
  - TypeSafe Jev（zero-shot）：0.727（发布方口径）；[Luni/laya-jev-benchmark](https://huggingface.co/datasets/Luni/laya-jev-benchmark) 独立复测为 0.626 raw
  - 微调专门模型（Laya / Verdict 2.0）：0.73–0.77（同分布训练，非 zero-shot，不可直接比）

## deltajev v0.1 管线（Qwen3.5-4B, MPS, fresh 模式）

| 迭代 | 改动 | 总 acc | noul | choice | score |
|---|---|---:|---:|---:|---:|
| v0.1 raw | 裸 completion + shared 骨架 | 0.22 | 0.35 | 0.55 | 0.25 |
| +chat | chat template（enable_thinking=False） | 0.32 | 0.35 | 0.60 | 0.45 |
| +noul 框架 | 陈述句改显式真假判断 | **0.53** | 0.55 | 0.60 | 0.45 |

（前两行的 noul 系统计 bug 修复前口径，见下；0.53 为 20 条子集数字，全量见 results/raw/summary_*.json）

## 关键发现（工程坑，全部真实踩到）

1. **noul 布尔序列化 bug**：`str(True)`="True" vs gold "true"——所有 noul 静默判错。
2. **BPE 槽位合并**：Qwen tokenizer 把 `:`+`A` 合并成单 token（`:A`=53779），单 token 钉扎
   必须按上下文解析槽位 id（`_slot_ids_for`），并允许"边界尾+字母"的确定性合并。
3. **裸 completion 不可用**：指令模型必须走 chat template，acc 0.22→0.32。
4. **noul 陈述句附和偏差**：模型 20/20 全答 true；把命题渲染成显式判断题后修复（0.35→0.55）。
5. transformers 5.x `DynamicCache` 不能下标；MPS 上 GDN 走参考实现（无 causal_conv1d /
   flash-linear-attention），延迟数字偏保守。

## 与公开数字的定位

zero-shot 原始 logit 读取 0.53（20 条子集）vs Jev raw 0.626–0.727：差距可见但管线
只做了最小必要工程。下一步优先级：score 题的序数渲染、fresh prompt 打磨、
（可选）全量确认。

## 全量结果（400 records / 2,000 decisions，2026-09-22）

**Qwen3.5-4B, fresh chat 模式, MPS（无 GDN 优化内核）：**

| 指标 | 值 |
|---|---:|
| 总准确率 | **0.5735**（random 0.318） |
| mean TVD vs 共识分布 | 0.3649 |
| 吞吐 | 0.906 decisions/s |
| 墙钟 | 2,209s |

按类型：choice **0.628**（600）/ noul **0.562**（600）/ score **0.541**（800）。

按 workflow：customer_service 0.668 / security_incidents 0.614 /
agent_trace_observability 0.522 / invoice_processing 0.490。

定位：zero-shot 无微调 0.574 vs Jev raw 0.626（独立复测）–0.727（发布方口径）
vs 同分布微调专门模型 0.73–0.77。**已超过 random +25.5 点，距 Jev raw 约 5 点**，
prompt 只做了 3 处最小修复，提升空间明确（score 序数渲染、选项描述措辞、
温度校准）。预测逐行存 `results/raw/preds_Qwen3.5-4B_fresh_full.jsonl`。

## Qwen3.8-27B BF16 上机（同协议 smoke，2026-09-22）

- 部署：MPS（Apple Silicon 统一内存 128GB），BF16 全精度，transformers git-main
  （3.8-27B 的文本架构识别为 `Qwen3_5ForCausalLM`，可正常加载）
- 下载坑：xet CDN 签名 URL 过期反复 403，`HF_HUB_DISABLE_XET=1` 走经典 HTTP 续传解决

| 指标（前 10 条 / 50 决策） | Qwen3.5-4B | Qwen3.8-27B |
|---|---:|---:|
| accuracy | ~0.50 | **0.66** |
| mean TVD | 0.407 | **0.285** |
| decisions/s（MPS，无优化内核） | 1.24 | 0.337 |

初步：27B 的分布质量（TVD）显著更好——这正是 GDN 混合 + 更大容量的预期方向；
延迟 ~3.7×，H1 的"更慢增长"要等延迟探针（分档 token 长度）才能下结论。
全量 400 条运行中（预计 ~100 分钟）。

## 全量结果：Qwen3.8-27B BF16（400 records / 2,000 decisions，2026-09-22）

**headline：zero-shot 0.7255 —— 与 Jev 发布方口径的 0.727 追平（差 0.0015），
超过独立复测的 Jev raw 0.626 达 10 个点。**

| 指标 | Qwen3.5-4B | Qwen3.8-27B BF16 | 参考：Jev |
|---|---:|---:|---:|
| accuracy | 0.5735 | **0.7255** | 0.626 raw / 0.727 发布方 |
| mean TVD | 0.3649 | **0.2548** | — |
| 吞吐（MPS 无优化内核） | 0.906/s | 0.209/s | — |
| 墙钟 | 2,209s | 9,570s | — |

按类型（27B）：noul **0.785** / score 0.705 / choice 0.693。
按 workflow（27B）：customer_service 0.822 / invoice_processing 0.724 /
security_incidents 0.700 / agent_trace_observability 0.656。

诚实限定：

1. 这是**同分布 prompt 工程 + 前置迭代调试**后的 zero-shot；Jev 的 0.727 也是
   zero-shot 口径，但双方都见过这个基准的公开形态，且我们的 noul 框架修复
   是在这份基准的子集上调的——有轻微开发集泄漏风险，需要 held-out 复核。
2. gold 是 3 样本共识分布，argmax 口径下随机涨落约 ±0.02。
3. 延迟不能下结论：MPS 无 GDN 优化内核，0.209/s 是保守值；H1/H2 待
   延迟探针（254/1.4k/5k/7.7k tokens 四档）。
4. TVD 0.2548 vs 4B 0.3649：27B 概率质量大幅更好，且未经校准——Open Jev
   式温度校准后还有空间。

预测逐行存 `results/raw/preds_Qwen3.8-27B_fresh_full.jsonl`。

## 延迟探针（H1/H2，2026-09-23）

Qwen3.8-27B BF16 @ MPS（无 GDN 优化内核），5 题记录形状，268–7702 tokens 四档，
每格 5 次重复取中位数（协议偏差：configs/eval.yaml 原定 20 次，长档实测单次
7 分钟级，降为 5 次以可行动；268 档另有 20 次重复数据
`latency_probe_20rep_partial.json`，中位数 24.6s 与 5 次口径一致）。

| state tokens | fresh（5 次逐题 prefill） | shared（1 次 prefill + 分支） | shared 加速比 |
|---:|---:|---:|---:|
| 268 | 22.6s | 9.6s | 2.4× |
| 1,366 | 77.6s | 20.5s | 3.8× |
| 5,175 | 277.4s | 61.3s | **4.5×** |
| 7,702 | 420.7s | 108.7s | 3.9× |

**H2 方向性成立**：shared 加速比随 state 长度增长（2.4×→~4×），与"GDN 层
分支不增加 KV"的机制预期一致。fresh 吞吐相对值：268→7702 tokens（28.7×），
延迟 18.6×——次线性，但在无优化内核上无法把功劳归于 GDN 本身。

**必须诚实**：绝对延迟（shared 也要 ~2–22s/记录）距离 Jev 的 70–500ms 差
1–2 个数量级，主因是 MPS 上 DeltaNet 走参考实现（无 causal_conv1d /
flash-linear-attention / Metal 内核）。本探针支持的是**机制层面的相对结论**，
不支持任何绝对性能主张。生产级验证需要 CUDA（3090 级）或优化 Metal 后端。
