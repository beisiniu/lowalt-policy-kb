# K=1~20 三裁判复核结果（2026-10-30）

本目录是 K 敏感性扫描的**裁判复核结果**，不是重新生成的回答。

## 数据来源

- 原始答案：`results/ksweep/k1.json` ~ `k20.json`，20 个 K ×31 题 × Naive/Temporal 两支 = 1240 条 DeepSeek 答案。
- Qwen 裁判：Qwen-3.8-Flash，全量 1240 条。
- GLM 裁判：GLM-5.3-Flash，只判 DS/Qwen 分歧的 151 条。
- 合并：1240 条，未决 0。
- DS/Qwen 一致率：87.82%，Cohen's κ=0.8204。
- GLM 在分歧题上的成对一致率：46.36%，κ=0.2197（样本为专门挑出的分歧题，存在选择偏差）。
- 11 条三票各不相同，已由作者逐条人工复核：7 条改判、4 条维持。

## 关键结果（三裁判多数票）

| K | 现行召回 | 旧穿透 | Naive 毒化 | Temporal 毒化 | Temporal 正确 | Temporal 拒答 | 严格拯救 | 加权增益 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 67.7% | 93.5% | 21/31 | 0/31 | 18/31 | 12/31 | 52.4% | +1.123 |
| **9** | **90.3%** | **96.8%** | **20/31** | **0/31** | **20/31** | **11/31** | **65.0%** | **+1.142** |
| 10 | 90.3% | 96.8% | 19/31 | 0/31 | 21/31 | 10/31 | 57.9% | +1.129 |
| 17 | 96.8% | 96.8% | 21/31 | 0/31 | 26/31 | 5/31 | 81.0% | +1.336 |
| 20 | 96.8% | 96.8% | 21/31 | 0/31 | 24/31 | 7/31 | 76.2% | +1.252 |

K=9 仍然是“现行召回≥90%、旧穿透≥90%”的**最小 K**，选择只看检索层指标；K 扫描的行为列只用来观察趋势，不参与 K 选择，也不能与 A 层三裁判主表混算。

## 文件

| 路径 | 内容 |
|---|---|
| `merged_summary.json` / `.md` | 1240 条合并结果、逐 K 指标、裁判一致性 |
| `human_review.json` | 11 条无多数票的人工复核记录（7 改判、4 维持） |
| `data/ksweep_3judge.json` | 论文用的 20 个 K 点结构化数据 |
| `evidence/ksweep_3judge_table.md` | 论文用完整 Markdown 表 |
| `records/qwen/` | Qwen 对 1240 条答案的裁判记录 |
| `records/glm/` | GLM 对 151 条分歧的裁判记录 |
| `records/merged/` | 三裁判合并后的逐条记录（含 `auto_status`） |
| `scripts/` | 复核、合并、表格生成脚本 |

## 复现

```bash
# 需要 API；当前结果已落盘，通常不需要重跑
py -3.10 results/ksweep_3judge_20261030/scripts/rejudge_ksweep.py --judge qwen ...
py -3.10 results/ksweep_3judge_20261030/scripts/rejudge_ksweep.py --judge glm --only-disagree ...
py -3.10 results/ksweep_3judge_20261030/scripts/merge_ksweep_judges.py
py -3.10 results/ksweep_3judge_20261030/scripts/make_ksweep_3judge_table.py
```

脚本默认路径仍指向论文工作区；迁移到项目目录后需按本目录调整 `PROJECT`/`OUT` 路径。

## 边界

- 回答没有重跑，只对同一批 DeepSeek 答案补 Qwen/GLM 裁判。
- 11 条无多数已人工复核；其余 1229 条按 DS/Qwen 两票或三裁判多数合并。
- K 扫描是独立 run，不能与 A 层主表的 22/31、23/31、8/31 混算。
