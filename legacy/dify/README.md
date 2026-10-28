# 归档：基于 Dify 的旧实现

这两个脚本是"爬虫 → Dify 知识库 → 评估"的旧链路，**已不在主流程中**，
仅保留作为方案对比材料（论文里可以写"为什么不用 Dify 而自建本地知识库"）。

- `dify_ingest.py`：把打标结果推入 Dify 知识库（create-by-text + documents/metadata 三步）
- `dify_client.py`：Dify API 客户端（知识库 / 元数据 / 对话）
- `evaluate_dify.py`：对着 Dify 应用做评估的旧版 evaluate.py

运行需要把 `dify_client.py` 放回 `policycrawler/dify.py`，并配置 `DIFY_API_KEY`。

现方案（`kb/` + `kb_ingest.py`）的区别：

| | Dify | 本地 kb |
| --- | --- | --- |
| 依赖 | 云/私有化实例 + 3 把密钥 | 无服务、无密钥 |
| 检索 | 平台内置（向量为主） | BM25 + 元数据硬过滤（可改） |
| 时效门禁 | 靠调用方手配过滤条件 | 代码里的 `gate` 节点，可测试 |
| 可复现 | 受平台版本影响 | `chunks.jsonl` 可 diff |
