#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段三 · 建本地知识库（替代 Dify 入库，全程离线、不需要任何 API）

用法：
    python kb_ingest.py data/extracted/ --rebuild     # 全量重建索引
    python kb_ingest.py data/extracted/xxx.json       # 增量追加一篇
    python kb_ingest.py --stats                       # 看索引现状
    python kb_ingest.py data/extracted/ --rebuild --max-chars 400 --overlap 50

产物：
    kb/index/chunks.jsonl   条文级 chunk + 元数据 + 分词结果
    kb/index/manifest.json  索引清单（政策数、chunk 数、每篇的时效信息）
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from kb import store
from kb.schema import freshness
from kb.splitter import PolicyChunkSplitter, build_documents
from policycrawler.extract_rules import parse_header
from policycrawler.utils import resolve_source_path, sha256_text, url_key

ROOT = Path(__file__).resolve().parent
DEFAULT_EXTRACTED = ROOT / "data" / "extracted"


def read_body(path: Path) -> tuple[str, dict]:
    """读爬虫落盘的 txt：返回 (正文, 头部元信息)。

    头部与正文之间是一条 60 个短横线的分隔行；按**整行**匹配切掉，
    否则会残留一串 "-" 被当成正文（会污染第一个 chunk）。
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    header = parse_header(text)
    m = re.search(r"^-{10,}\s*$", text, re.MULTILINE)
    body = text[m.end():] if m else text
    return body.strip(), header


def ingest_file(json_path: Path, splitter: PolicyChunkSplitter, quiet: bool = False) -> list:
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    src = (meta.get("_meta") or {}).get("source_file", "")
    txt_path = resolve_source_path(str(src), (
        ROOT / "data" / "national_policies",
        ROOT / "data" / "gd_policies",
        ROOT / "data" / "beijing_policies",
    ))
    if txt_path is None:
        raise FileNotFoundError(f"{json_path.name} 里记录的正文文件不存在：{src}")
    body, header = read_body(txt_path)

    url = header.get("来源链接", "")
    doc_meta = {
        "policy_name": meta.get("policy_name", ""),
        "policy_id": url_key(url) if url else sha256_text(meta.get("policy_name", ""))[:16],
        "issue_date": meta.get("issue_date", "无"),
        "effective_date": meta.get("effective_date", "无"),
        "expiry_date": meta.get("expiry_date", "无"),
        "apply_region": meta.get("apply_region", "全国"),
        "status": meta.get("status", "现行有效"),
        "source_url": url,
    }
    docs = build_documents(body, doc_meta, splitter)

    if not quiet:
        ok, reason = freshness(doc_meta)
        flag = "可用" if ok else f"不可用：{reason}"
        print(f"  {meta.get('policy_name', json_path.stem)}")
        print(f"    {len(docs):>3} 个 chunk ｜ {doc_meta['apply_region']} ｜ "
              f"生效 {doc_meta['effective_date']} ｜ {doc_meta['status']} ｜ {flag}")
        if not docs:
            print(f"    [WARN] 正文解析后为 0 个 chunk（原文 {len(body)} 字），"
                  f"该政策不会进入检索，请检查正文文件：{src}")
    return docs


def collect(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            out.extend(sorted(path.glob("*.json")))
        elif path.exists():
            out.append(path)
        else:
            print(f"[WARN] 路径不存在：{p}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="阶段三 · 建本地政策知识库")
    ap.add_argument("paths", nargs="*", help="阶段二产出的 json 文件或目录")
    ap.add_argument("--rebuild", action="store_true", help="清空旧索引后重建")
    ap.add_argument("--append", action="store_true", help="保留已有索引，追加新政策（默认路径也给全量重建）")
    ap.add_argument("--index-dir", default=str(store.DEFAULT_DIR))
    ap.add_argument("--max-chars", type=int, default=600, help="单个 chunk 最大字数")
    ap.add_argument("--overlap", type=int, default=60, help="超长条款二次切分时的重叠字数")
    ap.add_argument("--stats", action="store_true", help="只看索引统计")
    args = ap.parse_args()

    index_dir = Path(args.index_dir)

    if args.stats:
        try:
            docs, _, manifest = store.load_index(index_dir)
        except FileNotFoundError as e:
            print(e)
            return 1
        print(f"索引目录：{index_dir}")
        print(f"构建时间：{manifest.get('built_at', '-')}")
        print(f"政策 {manifest.get('policies', 0)} 篇 / chunk {len(docs)} 个 / 平均 {manifest.get('avg_tokens')} 词")
        print("-" * 78)
        print(f"{'政策名称':38} {'地域':6} {'生效':12} {'失效':12} 状态")
        for p in (manifest.get("policy_list") or {}).values():
            name = (p.get("policy_name") or "")[:36]
            print(f"{name:38} {str(p.get('apply_region')):6} {str(p.get('effective_date')):12} "
                  f"{str(p.get('expiry_date')):12} {p.get('status')}")
        stale = sum(1 for d in docs if not freshness(d.metadata)[0])
        print("-" * 78)
        print(f"其中失效/未生效 chunk：{stale} / {len(docs)}（检索时会被硬过滤掉）")
        return 0

    targets = collect(args.paths or [str(DEFAULT_EXTRACTED)])
    if not targets:
        print("没有找到可入库的 json，请先跑：python extract_policy.py demo/data/policies")
        return 1

    splitter = PolicyChunkSplitter(max_chars=args.max_chars, overlap=args.overlap)

    existing = []
    if args.append and (index_dir / "chunks.jsonl").exists():
        existing, _, _ = store.load_index(index_dir)
        print(f"追加模式：已有 {len(existing)} 个 chunk")
    else:
        store.rebuild_from_scratch(index_dir)

    print(f"入库 {len(targets)} 篇政策：")
    new_docs = []
    for t in targets:
        try:
            new_docs.extend(ingest_file(t, splitter))
        except (OSError, json.JSONDecodeError, KeyError) as e:
            print(f"  [FAIL] {t.name}: {e}")

    # 按 policy_id 去重：同一篇政策重复入库时**整体替换**，不叠加
    replaced = {d.metadata.get("policy_id", "") for d in new_docs}
    by_policy: dict[str, list] = {}
    for d in existing:
        pid = d.metadata.get("policy_id", "")
        if pid not in replaced:
            by_policy.setdefault(pid, []).append(d)
    for d in new_docs:
        by_policy.setdefault(d.metadata.get("policy_id", ""), []).append(d)

    all_docs = [d for group in by_policy.values() for d in group]
    manifest = store.save_index(all_docs, index_dir)

    print("-" * 78)
    print(f"索引已写入 {index_dir}")
    print(f"  {manifest['policies']} 篇政策 / {manifest['chunks']} 个 chunk / 平均 {manifest['avg_tokens']} 词")
    print("\n下一步：python evaluate.py --backend local --test-set test_set.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
