#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段三 · 把打标结果动态推入 Dify 知识库

用法：
    # 0) 忘了 dataset_id？让脚本去问 Dify（需要「访问所有知识库」权限的 Key）
    python dify_ingest.py --list-datasets

    # 1) 单个政策
    python dify_ingest.py data/extracted/2026-03-15_xxx.json --dataset-id 3d1f...c8

    # 2) 批量推整个目录
    python dify_ingest.py data/extracted/ --dataset-id 3d1f...c8 --wait

    # 3) 演练：只用打印请求体，不真的调接口
    python dify_ingest.py data/extracted/ --dataset-id 3d1f...c8 --dry-run

环境变量：
    DIFY_API_KEY   知识库 API Key（知识库 → API 访问 → 创建密钥）
    DIFY_BASE_URL  默认 https://api.dify.ai/v1；私有化部署改成 https://your-dify/v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policycrawler.dify import DifyClient, DifyError, pretty

ROOT = Path(__file__).resolve().parent

# 推到 Dify 的元数据字段（阶段四的 Agent 就靠这些字段做过滤）
METADATA_MAP = {
    "policy_name": "policy_name",
    "issue_date": "issue_date",
    "effective_date": "effective_date",
    "expiry_date": "expiry_date",
    "apply_region": "apply_region",
    "status": "status",
}


def load_extracted(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):  # 兼容 extract_policy.py 批量输出
        if not data:
            raise ValueError(f"{path} 是空数组")
        data = data[0]
    for f in ("policy_name", "status", "apply_region"):
        if not data.get(f):
            raise ValueError(f"{path} 缺少必要字段 {f}，请先跑阶段二")
    return data


def load_body_text(meta: dict, override: str | None) -> str:
    """正文来源：--text-file > 打标结果里的 _meta.source_file > 同目录同名 .txt"""
    if override:
        p = Path(override)
    else:
        src = (meta.get("_meta") or {}).get("source_file", "")
        p = Path(src) if src else Path()
    if not p.exists():
        raise FileNotFoundError(
            f"找不到正文文件：{p}\n请用 --text-file 显式指定政策正文 txt。"
        )
    text = p.read_text(encoding="utf-8", errors="replace")
    # 去掉爬虫写的元信息头，只把干净正文喂给知识库
    if "--------------------" in text:
        text = text.split("--------------------", 1)[1].strip()
    return text


def compose_text(meta: dict, body: str, inline_meta: bool) -> str:
    """在正文前加一行紧凑元信息。

    这不是冗余：RAG 检索按 chunk 召回，一个 chunk 里如果没有任何时间信息，
    模型就无法判断"这条规定现在还作不作数"。加一行抬头能显著降低过期政策误用。
    """
    if not inline_meta:
        return body
    head = (
        f"【政策】{meta['policy_name']}｜发布：{meta.get('issue_date', '无')}｜"
        f"施行：{meta.get('effective_date', '无')}｜失效：{meta.get('expiry_date', '无')}｜"
        f"适用地域：{meta.get('apply_region', '全国')}｜状态：{meta.get('status', '现行有效')}"
    )
    return f"{head}\n\n{body}"


def build_metadata_values(meta: dict) -> dict[str, str]:
    values = {}
    for src, dst in METADATA_MAP.items():
        v = meta.get(src)
        if v not in (None, "", "无"):
            values[dst] = str(v)
    return values


def ingest_one(client: DifyClient | None, path: Path, args, fields_cache: dict) -> dict:
    meta = load_extracted(path)
    body = load_body_text(meta, args.text_file)
    text = compose_text(meta, body, not args.no_inline_meta)
    values = build_metadata_values(meta)

    create_payload = {
        "name": meta["policy_name"],
        "text": f"{text[:80]}…（{len(text)} 字）",  # dry-run 时别把全文打一屏
        "indexing_technique": args.indexing_technique,
        "doc_form": "text_model",
        "doc_language": "Chinese",
        "process_rule": parse_process_rule(args),
    }

    if args.dry_run:
        print(f"\n--- [dry-run] POST /v1/datasets/{args.dataset_id}/document/create-by-text")
        print(pretty(create_payload))
        print(f"--- [dry-run] POST /v1/datasets/{args.dataset_id}/documents/metadata")
        print(pretty({"operation_data": [{"document_id": "<返回的 document.id>",
                                          "metadata_list": [{"id": f"<{k} 字段id>", "name": k, "value": v}
                                                            for k, v in values.items()],
                                          "partial_update": True}]}))
        return {"ok": True, "dry_run": True, "name": meta["policy_name"], "metadata": values}

    assert client is not None
    created = client.create_document_by_text(
        dataset_id=args.dataset_id,
        name=meta["policy_name"],
        text=text,
        indexing_technique=args.indexing_technique,
        process_rule=parse_process_rule(args),
    )
    doc_id = (created.get("document") or {}).get("id")
    if not doc_id:
        raise DifyError(f"创建文档未返回 document.id：{pretty(created)[:400]}")
    print(f"  ✓ 文档已创建：{doc_id}（batch {created.get('batch', '-')}）")

    # 元数据：先确保字段存在，再写值。字段定义在知识库级别，缓存一次即可
    if args.dataset_id not in fields_cache:
        fields_cache[args.dataset_id] = client.ensure_metadata_fields(
            args.dataset_id, list(values.keys()), type_=args.metadata_type
        )
        print(f"  ✓ 元数据字段就绪：{', '.join(fields_cache[args.dataset_id].keys())}")
    fields = fields_cache[args.dataset_id]

    if values and fields:
        client.update_document_metadata(args.dataset_id, doc_id, values, fields, partial=True)
        print(f"  ✓ 元数据已写入：{values}")

    status = "skipped"
    if args.wait:
        status = client.wait_for_indexing(args.dataset_id, doc_id, timeout=args.wait_timeout)
        print(f"  ✓ 索引状态：{status}")

    return {"ok": True, "document_id": doc_id, "name": meta["policy_name"],
            "metadata": values, "indexing": status}


def parse_process_rule(args) -> dict:
    """分段规则：默认 automatic；给了分隔符就用 custom 按特定字符切。"""
    if args.separator:
        return {
            "mode": "custom",
            "rules": {
                "pre_processing_rules": [
                    {"id": "remove_extra_spaces", "enabled": True},
                    {"id": "remove_urls_emails", "enabled": False},
                ],
                "segmentation": {
                    "separator": args.separator,
                    "max_tokens": args.max_tokens,
                    "chunk_overlap": args.chunk_overlap,
                },
            },
        }
    return {"mode": "automatic"}


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段三 · Dify 知识库动态入库")
    ap.add_argument("paths", nargs="*", help="阶段二产出的 json 文件或目录")
    ap.add_argument("--dataset-id", default=os.getenv("DIFY_DATASET_ID", ""), help="知识库 ID")
    ap.add_argument("--list-datasets", action="store_true", help="列出所有知识库及其 ID 后退出")
    ap.add_argument("--text-file", default=None, help="显式指定政策正文 txt（默认取 _meta.source_file）")
    ap.add_argument("--indexing-technique", default="high_quality", choices=["high_quality", "economy"])
    ap.add_argument("--metadata-type", default="string", choices=["string", "number", "time"],
                    help="新建元数据字段的类型；日期建议 string（time 类型对格式要求更严）")
    ap.add_argument("--separator", default="", help="按特定字符分段，如 '\\n第' ；留空=自动分段")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--chunk-overlap", type=int, default=50)
    ap.add_argument("--no-inline-meta", action="store_true", help="不在正文前加元信息抬头")
    ap.add_argument("--wait", action="store_true", help="等待索引完成再返回")
    ap.add_argument("--wait-timeout", type=float, default=180)
    ap.add_argument("--dry-run", action="store_true", help="只打印请求体，不调接口")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--base-url", default=None)
    args = ap.parse_args()

    if args.list_datasets:
        try:
            client = DifyClient(args.api_key, args.base_url)
            rows = client.list_datasets()
        except DifyError as e:
            print(f"[ERROR] {e}")
            return 3
        if not rows:
            print("没有查到知识库。若 Key 是「仅限特定知识库」的 scoped key，请直接从浏览器地址栏复制 ID。")
            return 0
        print(f"{'dataset_id':38}  文档数  名称")
        for ds in rows:
            print(f"{ds.get('id', '-'):38}  {str(ds.get('document_count', '-')):>6}  {ds.get('name', '-')}")
        return 0

    if not args.paths:
        ap.error("请传入阶段二产出的 json 路径，或用 --list-datasets 查看知识库")
    if not args.dataset_id:
        print("[ERROR] 缺少 --dataset-id。\n"
              "  找法一：执行  python dify_ingest.py --list-datasets\n"
              "  找法二：Dify 后台打开该知识库，地址栏 /datasets/<这一段就是 dataset_id>/documents")
        return 3

    targets: list[Path] = []
    for p in args.paths:
        path = Path(p)
        if path.is_dir():
            targets.extend(sorted(path.glob("*.json")))
        elif path.exists():
            targets.append(path)
        else:
            print(f"[WARN] 路径不存在：{p}")

    if not targets:
        print("没有找到可入库的 json")
        return 1

    client = None
    if not args.dry_run:
        try:
            client = DifyClient(args.api_key, args.base_url)
        except DifyError as e:
            print(f"[ERROR] {e}")
            return 3

    fields_cache: dict[str, dict] = {}
    ok, failed = 0, 0
    report = []
    for i, t in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {t.name}")
        try:
            r = ingest_one(client, t, args, fields_cache)
            report.append({"file": t.name, **r})
            ok += 1
        except (DifyError, ValueError, FileNotFoundError, json.JSONDecodeError) as e:
            failed += 1
            print(f"  ✗ 失败：{e}")
            report.append({"file": t.name, "ok": False, "error": str(e)})

    out = ROOT / "data" / "dify_ingest_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"at": datetime.now().isoformat(timespec="seconds"), "results": report},
                              ensure_ascii=False, indent=2), encoding="utf-8")

    print("-" * 60)
    print(f"成功 {ok} 个，失败 {failed} 个；报告：{out}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
