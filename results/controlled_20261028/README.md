# 可控实验目录（2026-10-28，已完成开发轮）

## 快照

- `snapshot.json`：31 题、基准日 2026-10-27、Top-9、3166 切片的冻结检索快照。
- 校验：旧目标召回 30/31，新目标召回 28/31；源 Chroma 冻结前后 SHA-256 一致。
- `freeze_log.json`：冻结时间、源 sqlite 哈希、快照指纹。
- `audit/`：离线条件/上下文审计（不调用模型）。
  - `audit_baseline.json`
  - `audit_noise.json`（temporal_filter 全噪声条件）
  - `audit_noise_corpus_pool.json`（全库注入池敏感性）
  - `audit_noise_defense.json`（多防御、50/100% 联合损坏）
  - `audit_noise_jm50_s43.json`（50% seed43 联合缺失）

## 已完成运行

| 目录 | 规模 | 失败 | 内容 |
|---|---|---|---|
| `run_smoke/` | 12 | 0 | 4 题 ×3 方法 API 冒烟 |
| `run_baseline/` | 186 | 0 | 31 题 ×6 方法，干净元数据 |
| `run_noise_100/` | 372 | 0 | 31 题 ×6 方法 ×2 种 100% 联合损坏 |
| `run_noise_50/` | 186 | 0 | 31 题 ×6 方法 ×50% joint_missing（seed43） |
| `run_reliability_smoke/` | 12 | 0 | 3 道硬题 ×4 方法：族谱扩展/不确定门禁冒烟 |
| `reliability_noise_smoke.json` | 3 | 0 | 100% chunk metadata 缺失 + family_uncertainty |

最终合并表：`combined_summary.md` / `combined_summary.json`。
正式报告：`../../output/控制实验报告_20261028.md`。
扩展说明：`../../README_时序可靠性扩展.md`；全量离线审计：`audit/audit_reliability.json`。

## 关键结果（开发轮，单裁判 DeepSeek）

- 干净元数据：Naive 毒化 58.1%；prompt-only 0%；date-rerank 41.9%；downweight 41.9%；llm_gate 0%；temporal_filter 0%。
- prompt-only 正确 64.5%/拒答 35.5%；temporal/llm_gate 正确 71.0%/拒答 29.0%。
- 50% 联合缺失：prompt 9.7%；temporal/llm_gate 19.4%；date 54.8%；downweight/naive 58.1%。
- 100% 联合缺失：temporal/downweight/llm_gate/naive 均 58.1%；date 48.4%；prompt 35.5%。
- 100% 伪造未来：temporal/downweight/llm_gate/naive 均 58.1%；date 48.4%；prompt 45.2%。
- 单字段 noise（只删日期或只删 status）被另一字段冗余拦住，机制上不改变过滤结果。

## 身份与缓存

- `case_key = hash(题内容, 条件)`；结果同时保存题内容、条件、快照、检索与上下文指纹。
- 请求级缓存按 provider/model/endpoint/messages/参数哈希；命中标记来源 case_key。
- 噪声只修改内存 Clause.metadata；正文、ID、检索顺序、干净真值均不变。
- 未写 Chroma；冻结时复制到临时目录后关闭。

## 注意

- 31 题高度集中于黄埔（25 题），比例不可外推为全国总体。
- 单裁判且与答题同供应商；正式主表应三裁判 + 人工抽检。
- 50% 名义比例实际曝光 3 政策/61 切片/12 题；100% 为 6 政策/218 切片。
- 结果不可与旧 `results/tq3_temporal_eval_k9_3judge.json` 直接混算：裁判提示和 baseline 集合不同。
- 代码与结果已提交并推送到 GitHub：`https://github.com/beisiniu/lowalt-policy-kb`（`main`，commit `92587c0`）。请求缓存被 `.gitignore` 排除，完整备份/结果包保留缓存。
