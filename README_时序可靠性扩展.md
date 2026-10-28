# 时序可靠性扩展（开发中）

最后更新：2026-10-28　｜　状态：核心模块、离线全量审计、3 题 API 冒烟、31 题 ×6 方法全量 API 已完成；全量 API 结果见 §2.2

这是主项目的新分支方向：从“Naive vs Temporal 的过滤器”升级为“时序知识可靠性”框架。旧题库、旧主表、旧结果和现有 runner 的默认行为保持不变。

## 1. 已实现什么

### 模块一：不确定性时序门禁

`rag_eval/validity.py`

每条条款返回三态：

```text
valid       明确可作为 as_of 时点依据
invalid     明确不可用（废止/失效/未生效/暂停）
uncertain   时效无法核实，不能当作确定依据
```

判定同时记录 `reason / confidence / basis`。字段缺失时查政策族谱的政策级状态；族谱也没有证据时返回 `uncertain`，而不是默认有效。

`rag_eval/temporal_graph.py` 新增 `mode="uncertainty"`：

- valid → 生成上下文；
- invalid → 剔除；
- uncertain → 单独记录，不进入生成上下文；
- 没有 valid、只有 uncertain → 拒答，话术说明“时效无法核实”。

旧 `filter` / `downweight` 行为保持不变。

### 模块二：政策族谱与同源扩展

`rag_eval/policy_family.py`

- 从条款分组构建政策节点（policy_id/name/region/status/valid_from/valid_to）；
- 边来源：4 条人工 seed + 覆盖表 `superseded_by` 自动匹配，共 8 条；
- `expand()` 在命中旧政策时，按“旧条款-新条款文本对应 + 问题相关性”选继任条款，默认补 2 条；
- 不是盲目增大 k：k=20 对 TQ3-030/032/041 仍召不回，族谱按政策身份定向补。

`rag_eval/experiments.py` / `tools/run_controlled_experiments.py`

新增 `reliability` 条件组：

```text
naive
prompt_only
temporal_filter
uncertainty_filter
family_filter
family_uncertainty
```

## 2. 已得到的验证结果

### 族谱补召回（全量离线，31 题）

| 方法 | 目标现行条款在上下文 | 族谱补入条款数 |
|---|---|---|
| temporal_filter | 28/31 | 0 |
| family_filter | **31/31** | 57 |
| family_uncertainty | **31/31** | 57 |

缺失的 3 道正是 TQ3-030/032/041；族谱扩展把目标新条款全部补回。

### 干净元数据小样本 API（3 道硬题 ×4 方法）

| 方法 | 结果 |
|---|---|
| naive | 3/3 毒化 |
| temporal_filter | 3/3 拒答 |
| family_filter | **3/3 正确** |
| family_uncertainty | **3/3 正确** |

具体答案：TQ3-041 由 family_filter 回答“**300 万元**”，不再答旧值 200 万；TQ3-030 答“实际运营费用 50%，每年最高 300 万”；TQ3-032 答“最高 1000 万”。

### 全量 API（31 题 ×6 方法，干净元数据）

运行目录 `results/controlled_20261028/run_reliability/`；186/186 条，0 失败，`check` 通过。答案与裁判均为 DeepSeek，温度 0，单裁判开发轮。

| 方法 | n | 毒化 | 正确 | 拒答 | 其他 | 加权分 | 旧在上下文 | 无效入上下文 | 族谱补入 | 严格拯救 |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 31 | 64.5% | 22.6% | 9.7% | 1 | -0.400 | 30 | 218 | 0 | — |
| prompt_only | 31 | 0.0% | 64.5% | 35.5% | 0 | +0.716 | 30 | 218 | 0 | 55.0% |
| temporal_filter | 31 | 0.0% | 71.0% | 29.0% | 0 | +0.768 | 0 | 0 | 0 | 65.0% |
| uncertainty_filter | 31 | 0.0% | 71.0% | 29.0% | 0 | +0.768 | 0 | 0 | 0 | 65.0% |
| family_filter | 31 | 0.0% | **77.4%** | 22.6% | 0 | **+0.819** | 0 | 0 | 57 | **80.0%** |
| family_uncertainty | 31 | 0.0% | **77.4%** | 22.6% | 0 | **+0.819** | 0 | 0 | 57 | **80.0%** |

关键现象：

1. **family 相对 temporal 净增 2 题**：多对 TQ3-011/030/032/041，少对 TQ3-006/010。四道增益全是“过滤后目标新条款不在 Top-9”的题；两道损失都是**拒答而非毒化**，答案已引用正确新政策但表述偏保守（TQ3-006 称“无法确定是否还能按该比例获得奖励”，TQ3-010 称“无法确定‘交付满50架’这一条件”），被严格裁判记为 REFUSED。不能把这 2 题说成“答错”，但说明族谱扩展会增加上下文（平均 521→1075 字，约 2.1 倍），并可能诱发保守表述。
2. **uncertainty_filter 与 temporal_filter 在干净元数据下 31/31 答案完全相同**：没有缺失/未知元数据时三态门禁退化为原硬过滤，这是预期行为；它的价值要在 100% 缺失条件下才显现。
3. **naive 存在运行间波动**：`run_baseline` 中 naive 毒化 18/31（58.1%），本次 20/31（64.5%）；prompt_only 与 temporal_filter 两次运行的裁判状态完全一致，但 naive 有 3 题状态变化（TQ3-001 REFUSED→CORRECT、TQ3-021 REFUSED→POISONED、TQ3-031 CORRECT→POISONED）。因此论文中应以**同一 run 内的 naive**作严格对照，并把 58.1%–64.5% 作为开发轮波动范围，不把两次差异当成方法效应。
4. **族谱并非每题都补**：TQ3-001/002 补入 0 条，其余 28 题补 2 条、TQ3-006 补 1 条；57 条补入与 Top-9 无重复。

边界：单裁判且与答题同供应商；31 题中 25 题来自黄埔；TQ3-006/010 的拒答判定需要第二裁判或人工复核。该表是开发轮结果，不作为最终论文主表。

### 100% chunk metadata 联合缺失（离线全量）

前提：只破坏 chunk 级 status/valid_to，政策族谱使用干净注册表；这是“chunk 元数据损坏 + 独立政策注册表”的组合，不是“注册表也损坏”。

| 方法 | 目标新条款召回 | 旧条款在上下文 | 真实无效切片入上下文 | 可生成 |
|---|---|---|---|---|
| temporal_filter | 28/31 | 30 | 218 | 31 |
| uncertainty_filter | 28/31 | 0 | 0 | 29 |
| family_filter | 31/31 | 30 | 218 | 31 |
| family_uncertainty | **31/31** | **0** | **0** | **31** |

小样本 API（TQ3-030/032/041，100% 缺失 + family_uncertainty）：**3/3 正确，0 个无效切片进入上下文**。

结论：

- 族谱扩展解决“过滤后没牌”；
- 不确定性门禁解决“元数据缺失时不要把旧条款当有效”；
- 两者合起来，在 chunk 元数据全损坏但政策注册表仍可用的条件下，能同时保住现行条款并挡住旧条款；
- 如果政策注册表也一起损坏，需要另做 `policy_registry_corrupt` 实验，不能把当前结果当成完全无先验的鲁棒性。

## 3. 文件映射

| 文件 | 作用 | 状态 |
|---|---|---|
| `rag_eval/validity.py` | 三态时效判定、证据与置信度 | 已完成 |
| `rag_eval/policy_family.py` | 政策节点、替代边、同源扩展 | 已完成 |
| `rag_eval/temporal_graph.py` | uncertainty 模式；旧 filter/downweight 不变 | 已完成 |
| `rag_eval/experiments.py` | reliability 方法、family 参数、扩展条件 | 已完成 |
| `tools/run_controlled_experiments.py` | reliability 审计/运行/检查/汇总；族谱补入校验 | 已完成 |
| `tests/test_temporal_reliability.py` | 13 项三态/族谱/门禁/组合测试 | 已完成 |
| `results/controlled_20261028/audit/audit_reliability.json` | 31 题全量离线审计 | 已完成 |
| `results/controlled_20261028/run_reliability_smoke/` | 3 题 ×4 方法 API 冒烟 | 已完成 |
| `results/controlled_20261028/run_reliability/` | 31 题 ×6 方法全量 API（186 条、0 失败） | 已完成 |
| `results/controlled_20261028/reliability_noise_offline.json` | 100% chunk metadata 缺失 + 四方法离线全量统计 | 已完成 |
| `results/controlled_20261028/reliability_noise_smoke.json` | 100% 缺失 + family_uncertainty 小样本 | 已完成 |

## 4. 命令

```bash
VENV=C:/Users/86139/.workbuddy/binaries/python/envs/default/Scripts/python.exe
cd D:/lowalt-policy-kb_backup_20261028

# 新模块测试
$VENV -B tests/test_temporal_reliability.py

# 全量离线审计（不调用 API）
$VENV -B tools/run_controlled_experiments.py audit --suite reliability --family-extra 2 \
  --out results/controlled_20261028/audit/audit_reliability.json

# 3 题小样本（可复跑；已有 run-reliability-smoke）
$VENV -B tools/run_controlled_experiments.py run --suite reliability \
  --methods naive,temporal_filter,family_filter,family_uncertainty \
  --item-ids TQ3-030,TQ3-032,TQ3-041 --family-extra 2 \
  --run-dir results/controlled_20261028/run_reliability_smoke

# 31 题 ×6 方法全量（可复跑；已有 run_reliability）
$VENV -B tools/run_controlled_experiments.py run --suite reliability \
  --methods naive,prompt_only,temporal_filter,uncertainty_filter,family_filter,family_uncertainty \
  --family-extra 2 --run-dir results/controlled_20261028/run_reliability

$VENV -B tools/run_controlled_experiments.py check --suite reliability \
  --run-dir results/controlled_20261028/run_reliability
$VENV -B tools/run_controlled_experiments.py summarize --suite reliability \
  --run-dir results/controlled_20261028/run_reliability
```

## 5. 边界与防错

- `check` 现在把 `audit.retrieved_ids` 与“原始 Top-9 + family_added”比对；不要把合法的族谱补入误判为检索快照错位。
- 旧 filter/downweight 测试和新控制实验 25 项测试均保持通过。
- 族谱从干净快照构建；噪声只改 chunk metadata。若要做“注册表也损坏”的实验，应另设实验组，不混入当前数字。
- 不覆盖 `run_baseline` / `run_noise_50` / `run_noise_100`；新实验使用 `run_reliability_*`。
- 不写 Chroma；扩展只在快照/内存上做。
- 当前扩展代码与结果已在本地提交，但未推送远程；交接时先看本文件与 `git status`。

## 6. 下一步

- [x] 跑 `reliability` 全量 31 题 API（6 方法 ×31 =186 条），得到正式开发轮表。
- [ ] 增加 `reliability_noise` 组：family/uncertainty 在 0/50/100% 缺失下的行为。
- [ ] `policy_registry_corrupt`：同时破坏 chunk 和族谱，量化“干净注册表”这一前提。
- [ ] 把 `family_added` 的上下文长度、延迟和 API 用量纳入 runner 汇总（本次只手工统计了上下文字符数）。
- [ ] 对 TQ3-006/010 的拒答判定做第二裁判或人工复核；不要为了让 family 好看而改单裁判口径。
- [ ] 更新主 README 和 `output/实验交接_20261028.md`，再决定是否推送扩展提交。
