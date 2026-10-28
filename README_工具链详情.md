# 低空政策知识库 · 增量更新工具链

面向"低空政策知识库动态更新"课题的四阶段流水线：

**增量抓取 → 大模型打标 → 本地知识库（LangGraph）→ 时效性评估**

每一阶段都带离线自检脚本（`tests/`），不需要真实站点、不需要 API Key 就能验证代码是否正常。

## 技术选型：为什么是 LangGraph + 自建知识库

| 候选 | 结论 | 理由 |
| --- | --- | --- |
| **LangGraph** | ✅ 用它 | 问答链路是 **有分支、有回路的状态机**（检索 → 时效门禁 → 答/拒答/放宽重检）。`gate` 节点既是"拦截过期政策"的逻辑，也是论文里可量化的指标（stale_blocked）。 |
| LangChain 组件 | ✅ 用它 | `Document`、`TextSplitter`、prompt、模型封装这些**积木**直接复用，但不用 `Chain` 去拼流程——链路一复杂，Chain 的嵌套比状态机难读。 |
| Dify | ❌ 已移除 | 需要额外部署实例 + 三把密钥，元数据过滤要手工配，平台版本还会漂移。旧实现在 `legacy/dify/`。 |
| 向量库（Chroma/FAISS） | ⏸ 预留 | 政策问题术语高度精确（"真高120米""提前3个工作日"），BM25 更稳且零依赖。要语义泛化时在 `PolicyRetriever` 加一路向量召回即可。 |

一句话：**LangGraph 管流程，LangChain 管组件；生产检索用 Chroma，旧实验保留 BM25；DeepSeek 只管"读懂并组织答案"。**

## 目录结构

```
lowalt-policy-kb/
├── incremental_crawler.py      阶段一 CLI：增量爬虫
├── policycrawler/              爬虫核心包
│   ├── config.py               配置加载
│   ├── fetcher.py              UA 伪装 / 超时重试 / 编码自适应 / 限速
│   ├── parser.py               列表页 & 详情页解析（多候选 selector）
│   ├── state.py                crawled_history.json 状态管理（两层去重）
│   ├── storage.py              落盘（按政策名命名的 txt + json）
│   ├── pipeline.py             增量主流程
│   ├── llm.py                  DeepSeek 客户端（JSON 强约束 + 模型名降级）
│   └── extract_rules.py        规则版抽取（离线兜底）
├── kb/                         阶段三：本地知识库
│   ├── splitter.py             按「第X条」切分（LangChain TextSplitter 子类）
│   ├── schema.py               时效判定 / 地域匹配 / 中文分词
│   ├── store.py                索引落盘（chunks.jsonl + manifest）
│   ├── retriever.py            BM25 + 元数据硬过滤 + 新法优先
│   ├── answer.py               抽取式 / 生成式 / 拒答三种回答
│   └── graph.py                LangGraph 问答图
├── kb_ingest.py                阶段三 CLI：建库
├── preflight.py                开跑前环境自检（依赖/密钥/配置/索引）
├── run_pipeline.py             一键跑完 抓取→打标→建库→评估
├── tools/inspect_site.py       给个列表页 URL，自动猜选择器
├── tools/import_manual.py      人工下载的政策 → 导入成标准格式
├── data/manual_inbox/          人工数据投放目录（你自己往里丢文件）
├── config.beijing.json         真实站点示例（北京市政策文件，实测可用）
├── .env.example                密钥模板
├── extract_policy.py           阶段二：结构化抽取
├── build_benchmark.py          阶段四：从真实库自动构建新旧冲突 benchmark
├── build_qa_probes.py          从真实库按地域生成覆盖性 QA 探针
├── evaluate.py                 阶段四：Naive RAG vs Temporal LangGraph 双分支评估
├── evaluate_qa_probes.py       覆盖性 QA：Recall@K + 回答正确率
├── build_chroma_db.py          全国 Chroma 构建脚本（56 篇 / 1294 chunk）
├── sources/lowalt_policy_urls.json  87 个政策 URL 种子
├── tools/harvest_caac.py       从民航局站内检索补充全国性政策种子
├── tools/build_url_seed.py     从公开政策索引生成 URL 清单
├── tools/crawl_policy_urls.py  跨站抓取 + HTTP/URL 台账
├── data/national/              crawl_manifest + corpus_manifest + 时效覆盖表
├── rag_eval/                   检索适配 / 时间元数据 / Temporal_Check 图
│   ├── backends.py             Chroma + 本地 BM25 可替换后端
│   ├── temporal.py             valid_from/valid_to 规范化与客观过期判定
│   ├── temporal_graph.py       入口即 Temporal_Check 的 LangGraph
│   └── models.py               统一 Clause 对象
├── demo/temporal_fixture.json  2 旧 + 1 新，最小可跑样例
├── test_set.json               旧版 20 题（历史实验，保留）
├── config.json                 真实站点配置（默认民航局政策法规栏目）
├── demo/                       本地演示站（4 篇仿真政策，纯静态 HTML）
├── legacy/                     已归档的 Dify 版与旧时效评估（不参与新链路）
└── tests/                      四阶段自检 + Mock 服务 + 时序评估验收
```

## 0. 环境准备

```bash
pip install -r requirements.txt
```

Python 3.10+（实测 3.13）。`lxml` 装不上时爬虫会自动退回 `html.parser`，功能不受影响，只是慢一点。

## 0.5 三种跑法

```bash
# ① 先看环境齐不齐（30 秒，强烈建议）
python preflight.py

# ② 一键跑完四个阶段（没有 Key 也能跑，只是数据不能进论文）
python run_pipeline.py --all --offline --no-llm

# ③ 有 Key 时的正式跑法
cp .env.example .env        # 填 DEEPSEEK_API_KEY
python run_pipeline.py --all
```

`preflight.py` 会逐项告诉你：依赖缺不缺、Key 有没有、`list_url` 是不是还停在示例值、
选择器是否为空、索引里有多少已失效的 chunk、测试集题型分布。有问题它会直接给出下一步命令。

`run_pipeline.py` 只对**新增或改过的正文**做打标（对比 txt 与产出 json 的 mtime），
不会每次把整个库重烧一遍 token；任一步失败立即停止。

改目标站点时，先用探测工具拿选择器，别手写：

```bash
python tools/inspect_site.py https://www.beijing.gov.cn/zhengce/zhengcefagui/
```

它会按「命中条目数」排序给出候选 `item` 选择器，并附上样例标题和链接。

---

## 阶段一 · 增量监控爬虫

### 1.1 改成你的目标站点

打开 `config.json`，三处必改：

| 字段 | 说明 |
| --- | --- |
| `source.list_url` | 政策列表页地址 |
| `source.base_url` | 用于把相对链接补全成绝对链接 |
| `list_selectors` / `detail_selectors` | CSS 选择器，**支持多个候选，按顺序尝试** |

拿选择器：F12 → 选中一条记录 → Copy selector，拆成"容器 / 标题 / 链接 / 日期"四层填进去。日期取不到时会用正则从条目文本里捞 `2026-10-20` 这类字符串。

> 动态渲染页面：先看 Network 有没有返回 JSON 的列表接口，有就直接请求接口；实在没有再用 Playwright（`requirements.txt` 里已注释）。

### 1.1.1 推荐监控源（2026-10 实测）

| 优先级 | 栏目 | 状态 | 说明 |
| --- | --- | --- | --- |
| ⭐ 首选 | 广东省政府「广东部署低空经济高质量发展」→ 政策文件 | ✅ 已跑通 | `config.guangdong.json`。15 条省级+地市低空政策，抓到 14 篇（1 篇表格类公告跳过） |
| ⭐ 备选 | 北京市政府「政策文件」 | ✅ 已跑通 | `config.beijing.json`。综合政策栏目，999 条，需要再筛低空相关 |
| 补充 | 广州市海珠区「海珠低空经济」专题 | ✅ 可抓 | `#Ulist li` 18 条，含《广州市低空经济发展条例》等 |
| 人工 | 深圳（市政府 / 交通局 / 司法局） | ⚠️ 前端渲染 | 列表由 AJAX 加载，静态页 0 条；且 https 报 `BAD_ECPOINT`（已自动降级 http）。走 `tools/import_manual.py` |
| 补充 | 中国民用航空局·信息公开 | ✅ 站内检索可抓 | 栏目页是 JS 渲染，但 `/was5/web/search` 返回静态 HTML，可分页；用 `tools/harvest_caac.py` 采集 |
| 备选 | 中国政府网·政策文件库 | ⚠️ 403 | 有反爬，需 Cookie |

广东省那份实测结果：14 篇政策 → 14 篇打标成功，地域判定全部正确
（广东省 / 广州市 / 中山市 / 惠州市…），建库 150 个 chunk，平均 194 词。
覆盖《广东省支持低空经济高质量发展若干措施》《广州市低空经济发展实施方案》
《中山市低空经济高质量发展行动方案》等真实文件。

> 北京政务网的详情页会把「实施日期」排在「成文日期」前面，很容易把发布日抓成实施日。
> 现在的规则是：**发布日期一律以列表页那一列为准**，详情页解析到的日期只做对照（不一致会打 `[INFO]`）。

### 1.1.2 只收「低空 / 无人机」相关

综合栏目（比如北京市政策文件 999 条）里 99% 跟低空无关。在 `config.json` 里配：

```json
"filter": {
  "include_any": ["低空", "无人机", "无人驾驶航空器", "eVTOL", "飞行汽车", "通用航空", "空域管理", "民用航空器"]
}
```

列表页解析完就按**标题**筛，命中任一关键词才去抓详情——省带宽，也不会把无关政策灌进知识库。
日志会打 `→ 关键词过滤：剔除 N 条…`，被剔除的数量可见。
留空数组 = 不过滤（专题栏目本身就是精准的）。

> 只看标题是刻意的：按正文筛就得先把详情页全抓一遍，等于白花 N 次请求。
> 标题里没关键词但正文有的，走下面的「人工导入」补。

### 1.1.3 人工导入（反爬 / 前端渲染站点的入口）

中国政府网、民航局、深圳政府在线这几个要么 403、要么 XHR 渲染，自动抓不到。
你自己在浏览器里打开、另存为网页（或直接复制正文存成 txt），丢进 `data/manual_inbox/`：

```bash
python tools/import_manual.py data/manual_inbox --dry-run   # 先看会导入什么
python tools/import_manual.py data/manual_inbox             # 正式导入

# 仓库里 demo/manual_inbox/ 放了两个示例文件，可以先拿它试：
python tools/import_manual.py demo/manual_inbox --dry-run
```

- `.html` 整页另存 → 自动抽标题（剥掉"…_深圳政府在线"这类后缀）、日期、正文
- `.txt` / `.md` → 首行当标题，正文里**最后一个**日期当发布日期（公文落款）
- 走 `sha256(正文)` 判重，重复导入只会打 `[DUP]`，可以放心反复跑
- 产出格式与爬虫**完全一致**（同样的头、同样的文件名规则），后续 `extract_policy.py` / `kb_ingest.py` 不用改

实测：2 个人工文件（一个深圳行动计划 HTML、一个物流试点 txt）导入后，与广东省 14 篇一起建库，
`在深圳飞 249 克无人机需要提前申请吗` 能正确命中深圳那两篇（地域判定 = 深圳市）。

### 1.1.4 全国政策 URL 种子库与 Chroma（2026-10-26 首版）

现有广东 14 篇之外，新增了一条按 URL 抓取跨省政府网站的通道：

```bash
# 从人民网低空经济政策库的省级/地方级索引生成 URL 清单（含分页）
python tools/build_url_seed.py

# 从民航局站内检索（TRS）补充全国性规章与规范性文件
python tools/harvest_caac.py --max-pages 8

# 逐条抓取，记录 HTTP 状态/失败原因/正文哈希/URL
python tools/crawl_policy_urls.py --resume --delay 0.5
```

台账：

- `sources/lowalt_policy_urls.json`：87 个种子（人民网 2 个索引页含分页 + 民航局检索 + 手工新旧对照）
- `data/national/crawl_manifest.jsonl`：每次抓取尝试（含失败）
- `data/national/crawl_manifest.csv`：按 URL 的最新状态，Excel 可直接开
- `data/national/corpus_manifest.json`：最终入库政策、来源 URL、chunk 数
- 实测：87 URL → 60 成功、18 网络/反爬失败（含深圳 https 握手失败、部分站点 404/412）、9 只抓到摘要；失败原因全部记录，不静默丢弃
- 最终 Chroma：**56 篇政策 / 1294 个 chunk / 35 个地域**（国家 306 / 省级 309 / 市级 679 个 chunk）
- 3 篇明确过期政策（福州 2023 意见、江苏 2023—2025 计划、广州黄埔 2023 措施），用于时序对照

抓取 HTML、`.docx` 和文字型 PDF；解析正文时会跳过政策解读/征求意见/门户导航，
并保留 `source_url`，任何 chunk 都能回到原文。

### 1.1.5 中文 BGE 嵌入与 Chroma 构建

Chroma 内置 all-MiniLM 是英文模型，中文政策排序不可用（实测“中山 2028 目标”会排错）。
项目改用 ModelScope 的 `Bge-small-zh-v1.5` ONNX 量化版（512 维，约 24MB，首次自动下载）：

```bash
python build_chroma_db.py \
  --extracted-dir data/extracted_gd_llm \
  --extracted-dir data/extracted_national \
  --chroma-dir ./chroma_lowalt --collection lowalt_policy \
  --embedding-function bge-zh --rebuild
```

- 每个 chunk 带 `region / topic / valid_from / valid_to / status / source_url`
- `valid_to=""` 表示长期有效（Chroma metadata 不接受 null）
- 旧政策被新政策明文废止但自身未写日期时，用 `data/national/validity_overrides.json`
  补 `expiry_date`，并为每条覆盖记录 `evidence_url`
- 构建前会去掉重复正文、政策解读、征求意见稿、过短摘要

### 1.2 运行

```bash
python incremental_crawler.py                 # 抓一轮
python incremental_crawler.py --dry-run       # 只判定新增/更新，不写文件
python incremental_crawler.py --limit 10
python incremental_crawler.py --no-check-updates   # 只抓新链接，不校验老政策修订
python incremental_crawler.py --stats
python incremental_crawler.py --reset         # 清空状态（会备份 .json.bak）
```

```
[1/12] NEW    2026-10-20  深圳市低空经济产业促进条例  5321字
[2/12] SKIP   2026-06-01  民用无人驾驶航空器运行安全管理规定  4802字  (正文未变化)
[3/12] UPDATE 2025-05-10  深圳市低空飞行活动管理办法  3512字 rev2  (正文变化，已存 rev2)
----------------------------------------
列表 12 条 → 新增 2 | 更新 1 | 跳过 9 | 失败 0 | 耗时 8.4s
```

### 1.3 增量逻辑：两层去重

`data/crawled_history.json` 是唯一的"记忆"：

```json
{
  "records": {
    "aa5e46348f8fffda": {          // = sha256(详情页URL)[:16]，第一层去重主键
      "url": "https://.../p1.html",
      "title": "深圳经济特区低空经济产业促进条例",
      "publish_date": "2026-03-15",
      "content_hash": "98af4d64...", // 第二层：正文哈希
      "content_hashes": ["98af4d64...", "3c1b..."],
      "revision": 2
    }
  }
}
```

- **URL 去重**：`sha256(url)[:16]` 做字典键，O(1) 判断"抓过没"。
- **内容哈希**：`sha256(normalize(正文))`。一致 → `SKIP`；变了 → `UPDATE`，存 `_rev2.txt`，**不覆盖历史版本**（政策沿革要留痕）。
- **哈希前必须归一化**：NFKC + 压空白。否则站点加个 `&nbsp;`，全部政策都会被判"已更新"。
- **跨 URL 去重**：同一正文被转载到多个栏目 → `DUP`，只登记不落盘。
- **原子写状态**：tmp → `fsync` → `os.replace`，Ctrl+C 也不会写坏。
- 每轮写 `data/logs/run_YYYYmmdd_HHMMSS.json`。

### 1.4 验证增量效果（不联网）

```bash
python tests/verify_phase1.py
```

脚本起本地 HTTP 服务托管 `demo/`，依次制造 5 个场景：

| 场景 | 期望 |
| --- | --- |
| S1 首次运行 | 新增 3 条 |
| S2 原样再跑 | 新增 0，跳过 3 |
| S3 列表页新增 1 条 | 新增 1，跳过 3 |
| S4 老政策正文修订 | 更新 1（rev2），跳过 3 |
| S5 再跑一次 | 全部跳过（幂等） |

边界用例（`python tests/verify_phase1_edges.py`）：

| 场景 | 期望行为 |
| --- | --- |
| 状态文件被写坏 | 自动备份成 `.json.broken`，按空历史继续跑，不崩 |
| 单条详情页 404 | 只记 failed，其余照常入库，退出码 2 |
| 列表页有 5 条但日期全缺 | 打印 `[WARN] …全部因日期无法解析被丢弃`，不静默返回 0 条 |
| item 选择器全部失配 | 打印 `[WARN] …选择器可能失配（站点改版）` |
| 同正文挂在两个链接下 | 识别为 `DUP`，只落一份文件 |

---

## 阶段二 · 大模型自动打标

### 2.1 配置 Key

```powershell
# Windows PowerShell（当前窗口有效）
$env:DEEPSEEK_API_KEY = "sk-xxxxxxxxxxxxxxxx"
# 永久生效
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sk-xxxx", "User")
```

```bash
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx   # Linux / macOS
```

Key 在 <https://platform.deepseek.com> → API keys。

**模型名注意**：官方模型名会滚动调整，**不要写版本号**。

| 时间点 | 可用 model | 说明 |
| --- | --- | --- |
| 现在 | `deepseek-flash` | 实际由 DeepSeek-V4.1-Flash 提供 |
| 现在 | `deepseek-v4-pro` | 更强的推理档 |
| 2026-07-24 起废弃 | `deepseek-chat` / `deepseek-reasoner` | 旧名，可能直接报 Model Not Exist |

代码默认 `deepseek-flash`，并内置降级链（`policycrawler/llm.py` 的 `MODEL_FALLBACKS`）：遇到 "Model Not Exist" 自动换下一个名字重试并打印 WARN。换模型：`DEEPSEEK_MODEL=deepseek-v4-pro` 或 `--model deepseek-v4-pro`。

### 2.2 运行

```bash
python extract_policy.py "data/policies/2026-03-15_xxx.txt"
python extract_policy.py data/policies/           # 批量
python extract_policy.py data/policies/ --quiet   # stdout 只留 JSON
python extract_policy.py data/policies/ --offline # 不调模型，规则抽取
```

输出（同时落盘到 `data/extracted/<同名>.json`）：

```json
{
  "policy_name": "深圳经济特区低空经济产业促进条例",
  "issue_date": "2026-03-15",
  "effective_date": "2026-05-01",
  "expiry_date": "无",
  "apply_region": "深圳市",
  "status": "现行有效",
  "summary": "25 千克以上无人机须提前 3 个工作日申请飞行计划并获批。",
  "_engine": "llm:deepseek-flash"
}
```

### 2.3 工程细节

- `response_format={"type":"json_object"}` + `temperature=0`，从协议层保证只出 JSON。
- 模型仍可能裹 ` ```json ` 围栏 → `extract_json()` 先剥围栏再取首尾大括号，失败再发一次"JSON 修复"请求。
- `normalize_result()` 做**强 schema 校验**：日期统一 `YYYY-MM-DD`、`status` 只允许两值、缺失填 `"无"`。下游建库和评估都依赖这个契约。
- 长文截成"前 6000 字 + 后 1500 字"：发文机关、文号、施行日期在头，有效期在尾，省 60%+ token。
- 没 Key / 限流兜底：`--offline` 走 `policycrawler/extract_rules.py`。

验证：`python tests/verify_phase2.py`

---

## 阶段三 · 本地知识库（LangGraph）

### 3.1 建库

```bash
python kb_ingest.py data/extracted/ --rebuild      # 全量重建
python kb_ingest.py data/extracted/xxx.json --append   # 追加一篇
python kb_ingest.py --stats                        # 看索引现状
```

```
入库 4 篇政策：
  深圳经济特区低空经济产业促进条例
      8 个 chunk ｜ 深圳市 ｜ 生效 2026-05-01 ｜ 现行有效 ｜ 可用
  深圳市低空飞行活动管理办法（2026年修订）
      5 个 chunk ｜ 深圳市 ｜ 生效 2026-09-01 ｜ 现行有效 ｜ 可用
```

产物是纯文本，可 diff、可进 git：

- `kb/index/chunks.jsonl` —— 每个 chunk 一行：正文 + 8 个元数据 + 分词结果
- `kb/index/manifest.json` —— 政策清单与时效信息

**为什么按「第X条」切**：政策的价值单位是"条"。用固定字数切，会把"第九条 …无需提出飞行计划申请"从中间劈开，检索出来的 chunk 只剩半句话，模型要么答错要么编。`PolicyChunkSplitter` 继承 LangChain 的 `TextSplitter`，先按条切，单条超长再按句号切并保留一句 overlap。

### 3.2 检索：过滤比打分重要

`PolicyRetriever`（BM25 + 元数据硬过滤）做三件事：

1. **硬过滤**（不是降权，是直接剔除）：
   - `status` 必须「现行有效」
   - `effective_date <= 今天`（还没生效的不能用来回答"现在要不要审批"）
   - `expiry_date == 无` 或 `>= 今天`
   - 地域：问深圳不会召回上海文件；「全国」文件对任何地方都适用；**省级文件适用于省内城市**（深圳市 ← 广东省，靠 `schema.PROVINCE_OF` 映射，不靠字符串前缀猜）
2. **新法优先**：`score × 0.65 + 新鲜度 × 0.35`。新鲜度用**绝对 3 年窗口**算，不用候选池内 min-max——否则"相差两个月"会被放大成 0→1 的差距，反而盖过相关性。
3. **意图重排**：
   - 问"要不要审批"时，含 `申请/审批/批准/工作日/无需` 的条款加权
   - **罚则条款降权**：问"4 千克要不要执照"时，最像的往往是"未取得执照罚 5000 元"，答非所问；除非问题本身问处罚，否则含 `处罚/罚款/责令/违反` 的条款 ×0.55
   - **数字对齐**：问题里有具体数字（1500米 / 25千克）时，含数字的条款加权，避免"在境内适用本规定"这种空条款抢占第一条

> 为什么过滤优先：**已废止的文件字面上往往和问题最像**。2025 旧办法"一律提前 5 个工作日"和"深圳飞无人机要申请吗"的相似度，比 2026 新条例"25kg 以下无需申请"更高。纯靠打分，top-1 一定是旧法——这正是"时效性衰减"的技术根源。

### 3.3 问答图

```
        parse ──> retrieve ──> gate ──┬─(命中且有效)──> answer ──> END
                                       ├─(只命中失效)──> refuse  ──> END
                                       └─(没命中)──> expand ──> retrieve(二次)
```

- `parse`：抽地域、规范化 query
- `retrieve`：带过滤的 BM25 召回，按政策去重
- `gate`：**时效门禁**。有可用条文 → 答；只有失效条文 → 明确拒答（"最相关的是《X》但已失效，不能作为当前依据"）；完全没命中 → 放宽地域再检一次
- `expand → retrieve` 是真实回路，也是选 LangGraph 而不是 Chain 的理由

拒答 ≠ 失败。前者是"时效风险被正确拦截"，后者是"检索没找到"——评估里这两类要分开统计。

```python
from kb.store import load_index
from kb.retriever import PolicyRetriever
from kb.graph import PolicyQA

docs, tokens, _ = load_index()
qa = PolicyQA(PolicyRetriever(docs, tokens), mode="extractive")
out = qa.ask("在深圳市用 249 克无人机飞行需要申请吗")
# {'answer': '根据《深圳经济特区低空经济产业促进条例》第九条…',
#  'route': 'answer', 'steps': ['parse','retrieve:3','gate>answer','answer'],
#  'cited': [...], 'blocked_stale': []}
```

`mode="llm"` 时把检索到的条文交给 DeepSeek 生成（需要 Key）；`mode="extractive"` 直接摘条文，作为**检索质量对照组**——连抽取式都答不对，说明是检索/数据的问题，换更大的模型也没用。

验证：`python tests/verify_phase3.py`（建库 → 地域过滤 → 新法优先 → 时效门禁 → 图分支/回路）

边界用例（`python tests/verify_phase3_edges.py`）：

| 场景 | 期望行为 |
| --- | --- |
| 单条超过 chunk 上限 | 按句号二次切分，每块 ≤ 上限，条号保留 |
| 省级政策 vs 市级提问 | 广东省文件能命中深圳市提问，但不会串到上海 |
| 政策还没生效（施行日在未来） | 过滤掉，不能回答"现在要不要审批" |
| 政策已废止但没写失效日期 | 按 `status` 过滤掉 |
| 同一篇政策重复入库 | 按 `policy_id` 整体替换，chunk 数不变 |
| 完全无关的问题 | `expand` 只放宽 1 次，然后 `refuse`，不死循环 |
| 通篇没有「第X条」的通知 | 按段落装包切分，不丢句子、不超长 |
| 正文解析后 0 个 chunk | 明确 `[WARN]`，不静默入库一份空政策 |

---

## 阶段四 · 时序 RAG 对抗评估（当前版本）

目标不是再跑一遍固定题，而是从真实库**自动挖新旧冲突 → 自动出题 → 同一 Top-5 双分支对比**。

### 4.1 字段与后端

| 你的 Chroma 字段 | 本项目字段 | 用途 |
| --- | --- | --- |
| `valid_from` | `effective_date` | 尚未生效判断 |
| `valid_to` | `expiry_date` | `None` = 长期有效；早于评估时点 = 过期 |
| `region` | `apply_region` | 地域分组/统计 |
| `topic` | 无（按主题词典兜底） | 新旧条款配对 |

`valid_to` 非空**不等于**一定过期：未来到期仍可用；`null` 才表示长期有效。判断逻辑在
`rag_eval/temporal.py`，不依赖 benchmark 标签。

Chroma 是默认后端；当前仓库没有 Chroma 时可用 `--backend local` 跑同一套逻辑：

```bash
# 你的真实库
python build_benchmark.py --chroma-dir ./chroma_db --collection lowalt_policy

# 当前仓库的 BM25 索引（没有新旧 valid_to 时只能验证流程）
python build_benchmark.py --backend local --index-dir kb/index
```

Chroma embedding 必须与建库时一致。通常 `--embedding-function auto` 即可；如果库不是默认模型，
用 `--embedding-function module:factory` 传入同一个 EmbeddingFunction。

### 4.2 自动构建 benchmark

```bash
python build_benchmark.py \
  --chroma-dir ./chroma_db --collection lowalt_policy \
  --as-of 2026-10-26 \
  --max-pairs 80 --max-items 30 \
  --output benchmark.json
```

流程：按 `(region, topic)` 分组 → 旧条款 `valid_to < as_of` × 新条款 `valid_to is null`
→ 词面相似度只用于挑候选（省 LLM 成本）→ DeepSeek 判断是否“同一现实问题会得到不同答案”
→ 生成不含政策名/日期/旧新提示的迷惑性问题与只依据新条款的标准答案。

输出 `benchmark.json`（列表），字段包括：

```json
{
  "topic": "飞行计划审批",
  "region": "深圳市",
  "question": "现在在深圳适飞空域飞249克无人机，还要提前5个工作日申请吗？",
  "ground_truth": "无需申请，完成实名登记与飞行前信息报备即可。",
  "old_clause_text": "……",
  "new_clause_text": "……",
  "old_clause_id": "…", "new_clause_id": "…",
  "conflict_type": "approval_removed"
}
```

旁路 `benchmark.manifest.json` 记录候选对、跳过数、成本参数，便于论文复现。
生成阶段每接受一题就原子落盘，限流中断不会丢掉已花的调用。

### 4.3 双分支评估

```bash
python evaluate.py \
  --chroma-dir ./chroma_db --collection lowalt_policy \
  --benchmark benchmark.json --top-k 5 \
  --as-of 2026-10-26 \
  --out results/temporal_poisoning_eval.json \
  --figure output/figures/temporal_poisoning_compare.png \
  --judge-repeat 1
```

- **只检索一次**：`similarity_search(question, k=5)` 的结果同时给两组，禁止各自重检。
- **Naive RAG**：Top-5 原样拼接。
- **Temporal LangGraph**：同一 Top-5 作为初始 State，入口节点直接是 `temporal_check`；
  该节点拿不到 `old_clause_id`，只按 `valid_from/valid_to/status` 过滤。
- 两组用同一个回答 Prompt、同一个模型、同一 temperature。
- 调用失败会中止并保留已完成结果，**不会把 API 错误伪装成答错**。
- `--resume` 在同一 benchmark、模型、top-k、as-of 下跳过已完成题。

把外部 Top-5 塞进图的写法：

```python
out = graph.invoke({
    "question": question,
    "as_of": "2026-10-26",
    "retrieved_docs": top5,   # 外部检索结果
    "steps": [],
})
```

对应 State 必须有 `retrieved_docs / filtered_docs / dropped / route / answer`，
入口设为 `temporal_check`，条件边指向 `generate` 或 `refuse`。完整实现见
`rag_eval/temporal_graph.py`。

### 4.4 指标

```text
过期政策穿透率 = Top-5 含旧条款 / N
Naive 毒化率   = Naive POISONED / N
Temporal 毒化率 = Temporal POISONED / N
严格拯救率     = Naive POISONED 且 Temporal CORRECT / Naive POISONED
毒化消除率     = Naive POISONED 且 Temporal 不再 POISONED / Naive POISONED
```

裁判为四类：`POISONED / CORRECT / REFUSED / OTHER_ERROR`。没有使用二分类，因为
Temporal 过滤后若没有现行条款会**安全拒答**；把拒答硬判成 CORRECT 会虚高，
硬判成 POISONED 又会误报。毒化率只统计 POISONED，拒答率另报。

### 4.5 全国种子库首轮实测

`demo/temporal_fixture.json` 只用于验证链路；下面是当前真实全国种子库上的结果。

**数据覆盖验证**（`qa_probes.json`，30 题，23 个地域）：

| 指标 | 结果 |
| --- | --- |
| 条款 Recall@1 | 70.0%（21/30） |
| 条款 Recall@5 | **96.7%（29/30）** |
| 政策 Recall@5 | 100% |
| 回答正确率（DeepSeek 裁判） | **96.7%（29/30）** |

唯一一题拒答是**检索没召回**，不是模型不会：问“朝阳市 2030 年通用航空直接产值”，
目标句在源文件里确实存在（200 亿元 / 拉动 1000 亿元），但该 chunk 没进 Top-5。
这说明 Recall@1 只有 70% 时，Hotpot 式的跨句问题仍会掉分，需要靠 rerank 或提高 top_k。

**时序冲突验证**（真实新旧对照，7 题，广州黄埔 2023→2025 + 福州 2023→2024）：

| 指标 | 结果 |
| --- | --- |
| 过期政策穿透率 | 100%（7/7） |
| 现行条款在 Top-5 | 71.4%（5/7） |
| Temporal 旧条款拦截率 | 100%（7/7） |
| Naive RAG 毒化率 | **57.1%（4/7）** |
| Temporal 毒化率 | **0%（0/7）** |
| 严格拯救率 | 50%（2/4） |
| 毒化消除率 | 100%（4/4） |
| Naive / Temporal 正确率 | 14.3% / 57.1% |
| Naive / Temporal 拒答率 | 28.6% / 42.9% |

结论方向成立，但必须同时看到代价：

- Naive 在 7 题中 4 题直接沿用旧补贴标准（旧条款均排第 1）；
- Temporal 过滤后 3 题用现行条款答对（严格拯救 75%）；
- 另 1 题因为 Top-5 里没有对应现行条款而**安全拒答**；
- Naive 有 2 题拒答、1 题答对（TQ-0003 规则延续，旧意见的 10 家/20 万与现行政策一致）；
  其余 4 题被旧条款带偏。裁判用 3 次投票后，TQ-0003 从“毒化”纠正为“正确”——
  **引用过期文件但结论正确，不应算毒化**。

N=7 只能证明链路和方向，**不能作为论文最终效应量**。真正的瓶颈不是代码，而是**已废止政策的存量**：
当前 56 篇里只有 3 篇有明确失效依据（福州 2023、江苏 2023—2025、黄埔 2023），
37 条旧条款配对 1257 条现行条款，经 LLM 二审只通过 7 组。要继续扩 N，必须先扩“旧政策”。

重画图不需要再调 API：

```bash
python evaluate.py --plot-only --out results/temporal_national_eval.json \
  --figure output/figures/temporal_national_compare.png
```

---

## 附录 · 旧版时效性衰减评估（已归档）

> 以下脚本已移到 `legacy/evaluate_time_decay.py`，不再作为主评估入口。
> 旧数字（0% → 65% 等）来自仿真政策 + 抽取式回答 + 规则裁判，**不能作为论文指标**。
> 保留本节只为说明历史实验口径。

### 旧版测试集与指标

#### 4.1 测试集

`test_set.json` 20 题，两类：

```json
{
  "id": "Q01",
  "question": "现在在深圳市用大疆 Mini 3（249 克）在适飞空域内飞行，需要提前申请飞行计划吗？",
  "answer_type": "审批判定",
  "expected_status": "无需审批",
  "ground_truth": "不需要。按条例第九条…",
  "trap": "stale"
}
```

- `answer_type: 审批判定` —— 比对"需要审批 / 无需审批"
- `answer_type: 信息查询` —— 比对 `key_facts`（如 `["120米"]`、`["5年"]`），不看审批结论
- `trap: "stale"` —— 标出"旧政策会把人带偏"的题，单独统计过期误导率

### 4.2 跑三次对比

```bash
# ① 更新前：库里只有当时有效的旧办法（data/extracted_before 是快照）
python kb_ingest.py data/extracted_before --index-dir kb/index_before --rebuild
python evaluate.py --tag before --backend local --no-llm --index-dir kb/index_before

# ② 更新后：新政策已入库
python evaluate.py --tag after  --backend local --no-llm --index-dir kb/index

# ③ 出对比图
python evaluate.py --compare results/eval_before.json results/eval_after.json
```

裁判判定"某条政策是否过期"时看的是 `--stale-ref-index`（默认当前全量库），**不看被测知识库自己的状态**——否则系统把旧法标成"现行有效"就能自我免责，测不出真实错误。

后端三选一：

| `--backend` | 说明 | 需要 Key |
| --- | --- | --- |
| `local` | 本地知识库 + LangGraph（默认） | `--no-llm` 时不需要 |
| `llm` | 纯大模型直答，不检索（对照组） | 需要 |
| `mock` | 生成模拟数据 | 不需要 |

裁判两种：`--judge llm`（DeepSeek，论文用）/ `--judge rules`（关键词规则，只够跑通链路）。`auto` 会按有没有 Key 自动选。

LLM 裁判本身有随机性，同一份回答判两次可能给不同标签。论文数据建议加 `--judge-repeat 3`：同一回答判 3 次取多数，并输出 `judge_agreement`（一致性）。代价是裁判调用量 ×3。

### 4.3 判定标准

| 标签 | 含义 |
| --- | --- |
| `Correct` | 核心结论与标准答案一致 |
| `Stale_Fact_Error` | 审批建议错误，且原因是用了**已失效/已修订/被取代**的旧政策 |
| `Refusal` | 拒答或明确表示无法确定 |
| `Incorrect_Other` | 结论错误，但非过期政策导致 |

四类相加 = 100%。另有 `detect_status()` 的**客观**指标"审批结论一致率"，不依赖大模型判断，可作交叉验证。

### 4.4 论文图

`--compare` 生成 300 dpi、`bbox_inches="tight"`、去上右边框的成图，同名 `.csv`（UTF-8 BOM，Excel 直接开）可做正文表格。

`output/figures/timeliness_decay.png` 目前是**真实数据**（4 篇仿真政策，`--no-llm` 抽取式 + 规则裁判）：

| 指标 | 更新前 | 更新后 |
| --- | --- | --- |
| 正确率（严格） | 0% | 65% |
| 有效应答正确率（拒答不计分母） | 0% | 72.2% |
| 过期政策误导率 | 70% | 0% |
| 拒答率 | 20% | 10% |
| 其中·审批判定 | 0/11 | 6/11（54.5%） |
| 其中·信息查询 | 0/9 | 7/9（77.8%） |

**为什么要两个正确率**：只报严格口径，会惩罚一个"宁可拒答也不乱说"的系统；只报有效应答口径，又容易被质疑在挪分母。两个都报，把取舍摊开——这篇的重点不是把正确率做大，而是不让过期政策产生错误建议。

**为什么图里去掉了「其他错误率」**：四类判定互斥，过期误导率降到 0 后，残余错误会自动归到"其他错误"，看起来像"更新后变差了"。真正可比的是上面这几项；其他错误率仍保留在终端输出和 CSV 里。

更新前之所以「正确率 0%」：深圳相关问题全部命中 2025 旧办法，而裁判按参考索引判定它已废止，所以这些答案即使字面对得上标准也计为过期误导——这正是"时效性衰减"要量化的东西。

这组数字只说明"链路能跑通、方向正确"，**不能当论文指标**——写论文时要换成 LLM 回答 + LLM 裁判。

---

## 一键自检

```bash
python tests/run_all.py
```

六个套件依次跑完并汇总：

- 增量逻辑（URL 去重 + 内容哈希 + 修订留痕 + 幂等）成立
- 打标 schema 稳定、JSON 围栏能剥离、失效日期有原文审计
- 建库切分、地域过滤、新法优先、时效门禁、图的分支与回路都正常
- Chroma 全链路：自动出题 → 同一 Top-5 → 双分支 → 裁判 → 300dpi 图
- Temporal_Check 确实过滤旧条款，生成上下文里不出现旧条款全文或其 ID

---

## 常见问题

**Q：报 SSL 证书校验失败？**
政府站点证书过期/证书链不全很常见（也可能是公司代理在中间拦）。确认可信后，在配置里加
`"crawl": {"verify_ssl": false}`。默认是 `true`，不会悄悄帮你关掉校验。

**Q：发布日期抓成了实施日期？**
详情页上「成文日期 / 实施日期 / 印发日期」常常并排，选择器随手指一个就错。现在的规则是
**发布日期以列表页为准**，详情页日期只做对照。如果你的列表页没有日期列，才会退回用详情页的。

**Q：列表页 HTML 里没有条目，只有一堆 JS？**
说明是前端渲染。先 F12 → Network 找 XHR 接口，直接请求 JSON 比解析 DOM 稳；实在找不到再用
Playwright。`tools/inspect_site.py` 会明确告诉你"没找到像样的列表结构"，而不是给你一个错的选择器。

**Q：抓到的正文乱码？**
政府站点常见 `gb2312`/`gbk` 且不声明 charset。`fetcher.decode_html()` 按"配置强制 → header → meta → 自动探测"四级回退。仍乱码就在 `config.json` 里设 `"charset": "gbk"`。

**Q：列表页解析出 0 条？**
先看输出里有没有 `[WARN]`。`0 条` 有两种完全不同的原因：真的没有新政策（`stats.no_date == 0`），或者选择器失配/日期解析失败（会明确告警）。没有告警的 0 条才是真 0 条。改版后用 `--dry-run` + F12 重新取选择器。

**Q：跑起来被封 IP？**
`crawl.delay_seconds` 调到 2~3 秒，`max_items_per_run` 降到 20，别开多线程。

**Q：为什么检索不到刚入库的政策？**
先 `python kb_ingest.py --stats` 看 policy 数对不对。`--append` 时同 `policy_id` 会整体替换；`--rebuild` 才是全量重建。

**Q：`--no-llm` 的正确率只有 60% 多，正常吗？**
正常。抽取式只是把条文原样摘出来，不做推理，也不做多条文冲突裁决。它的价值是**定位问题出在检索还是生成**：抽取式对、LLM 也对应 → 检索没问题；抽取式对、LLM 错 → 模型在编；抽取式就错 → 检索/切分要改。

**Q：想加向量检索？**
在 `PolicyRetriever` 里加一路 `embedding` 召回，和 BM25 分数做 RRF 融合即可，其余代码不用动。政策问答里 BM25 通常不弱，建议先看 badcase 再决定。
