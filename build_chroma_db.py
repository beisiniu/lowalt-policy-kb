#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把多来源已打标政策构建成可查询的 Chroma 低空政策库。

数据流：
    data/extracted_gd_llm/*.json  ┐
    data/extracted_national/*.json├─→ 正文按「第X条」切分
    （其他 extracted_* 目录）      ┘      → valid_from/valid_to/topic/region/source_url
                                         → Chroma collection

设计要点：
  1. 每个 chunk 都带 **source_url / source_name / valid_from / valid_to / topic**，
     后续既能追溯原文，也能做时序过滤和按地区统计。
  2. Chroma metadata 不支持 null，`valid_to` 用空字符串表示“长期有效”；
     rag_eval/temporal.py 已把空字符串识别为 None。
  3. embedding 默认使用 Chroma 内置 all-MiniLM-L6-v2（384 维，首次会自动下载）。
     如果你建库时用的是别的模型，用 --embedding-function module:factory 传同一个函数。

用法：
    # 先确保 data/extracted_national 已由 extract_policy.py 生成
    python build_chroma_db.py \
      --extracted-dir data/extracted_gd_llm \
      --extracted-dir data/extracted_national \
      --chroma-dir ./chroma_lowalt --collection lowalt_policy --rebuild

    python build_chroma_db.py --stats
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from kb.splitter import PolicyChunkSplitter, build_documents
from kb_ingest import read_body
from policycrawler.extract_rules import parse_header
from policycrawler.utils import resolve_source_path, sha256_text, url_key
from rag_eval.models import Clause
from rag_eval.temporal import effective_topic, iso_or_none

# 正文质量闸：拒绝把"网页源码 / PDF 乱码"当政策正文切块入库。
# 背景：曾有 1 份文件的抓取结果是整页 JS 源码，切成 2400+ 个无意义 chunk，占全库 39%。
try:
    from body_quality import deep_clean, judge as body_judge
except Exception:  # pragma: no cover - 缺失时降级为不检查
    body_judge = None
    deep_clean = None

ROOT = Path(__file__).resolve().parent
DEFAULT_CHROMA = ROOT / "chroma_lowalt"
DEFAULT_COLLECTION = "lowalt_policy"
LOWALT_KWS = ("低空", "无人机", "无人驾驶航空器", "通用航空", "eVTOL", "飞行汽车", "空天信息")
EXCLUDE_TITLE_KWS = ("政策解读", "征求意见", "解读丨", "一问一答")


def log(msg: str) -> None:
    print(msg, flush=True)


def norm_text(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", s or ""))


def infer_level(region: str) -> str:
    if region in ("全国", "国家", ""):
        return "national"
    if region.endswith(("省", "自治区", "特别行政区")):
        return "province"
    return "city"


def source_label(txt_path: Path, header: dict) -> str:
    if header.get("采集来源"):
        return header["采集来源"]
    parent = txt_path.parent.name
    return {
        "gd_policies": "广东省低空经济专题",
        "beijing_policies": "北京市政策文件",
        "national_policies": "人民网低空经济政策库",
    }.get(parent, parent)


def topic_of(text: str, metadata: dict) -> str:
    c = Clause("tmp", text, metadata)
    t = effective_topic(c)
    return t


def load_policy(json_path: Path, validity_overrides: dict | None = None) -> tuple[dict, str, str] | None:
    """读取一条打标结果 → (metadata, body, source_txt)。"""
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    src = (meta.get("_meta") or {}).get("source_file", "")
    txt_path = resolve_source_path(str(src), (
        ROOT / "data" / "national_policies",
        ROOT / "data" / "gd_policies",
        ROOT / "data" / "beijing_policies",
    ))
    if txt_path is None:
        return None
    body, header = read_body(txt_path)
    if deep_clean is not None:
        # 兜底：即使源文件残留导航栏/页脚/JSON 元数据，也保证入库正文是政策文本
        body = deep_clean(body)
    if len(body) < 500:
        # 前期抓取留下的短摘要/导航页不进入正式库
        return None

    name = str(meta.get("policy_name") or header.get("政策名称") or txt_path.stem).strip()
    if name in ("无", "未命名政策", ""):
        return None
    blob = name + "\n" + str(meta.get("summary") or "") + "\n" + body[:6000]
    if not any(k in blob for k in LOWALT_KWS):
        return None
    # 解读/征求意见不是生效规范，混进知识库会制造“貌似权威”的错误依据。
    # 模型有时会把解读页的 policy_name 洗成被解读文件的原名，所以文件名/页面标题也要查。
    title_markers = name + " " + txt_path.name + " " + str(header.get("政策名称") or "")
    if any(k in title_markers for k in EXCLUDE_TITLE_KWS):
        return None

    region = str(meta.get("apply_region") or "").strip() or "全国"
    source_url = str(header.get("来源链接") or "").strip()
    pid = url_key(source_url) if source_url else sha256_text(name)[:16]
    # 有些旧政策自身只写“有效期三年”，不知道后来被谁废止。
    # 用带 evidence_url 的人工覆盖表补上真实失效日，避免模型推断。
    ov = (validity_overrides or {}).get(source_url) or {}
    if ov:
        if ov.get("status"):
            meta["status"] = ov["status"]
        if ov.get("expiry_date"):
            meta["expiry_date"] = ov["expiry_date"]
    out = {
        "policy_id": pid,
        "policy_name": name,
        "issue_date": str(meta.get("issue_date") or "无"),
        "valid_from": iso_or_none(meta.get("effective_date")) or "",
        "valid_to": iso_or_none(meta.get("expiry_date")) or "",
        "region": region,
        "status": str(meta.get("status") or "现行有效"),
        "summary": str(meta.get("summary") or ""),
        "source_url": source_url,
        "source_name": source_label(txt_path, header),
        "level": infer_level(region),
        "doc_type": "政策文件",
        "source_file": str(txt_path.resolve()),
        "content_hash": sha256_text(norm_text(body)),
    }
    return out, body, str(txt_path)


def collect_extracted(dirs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for d in dirs:
        if not d.exists():
            log(f"[WARN] 打标目录不存在：{d}")
            continue
        files.extend(sorted(d.glob("*.json")))
    return files


def add_batch(collection, batch: list[dict[str, Any]]) -> None:
    if not batch:
        return
    collection.add(
        ids=[x["id"] for x in batch],
        documents=[x["text"] for x in batch],
        metadatas=[x["meta"] for x in batch],
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="构建 Chroma 全国低空政策库")
    ap.add_argument("--extracted-dir", action="append", default=[],
                    help="可重复；默认 data/extracted_gd_llm + data/extracted_national")
    ap.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA))
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--embedding-function", default="bge-zh",
                    help="bge-zh（中文 BGE-small，推荐）/ default（英文 MiniLM）/ hash（测试）")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--max-chars", type=int, default=450)
    ap.add_argument("--max-chunks-per-policy", type=int, default=250,
                    help="单片政策切块上限；超过视为正文污染并拒收（默认 250）")
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--manifest", default=str(ROOT / "data" / "national" / "corpus_manifest.json"))
    ap.add_argument("--validity-overrides", default=str(ROOT / "data" / "national" / "validity_overrides.json"),
                    help="人工核验的新旧替代/失效日期覆盖表（含 evidence_url）")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    try:
        import chromadb
    except ImportError:
        log("[ERROR] 未安装 chromadb：pip install -r requirements.txt")
        return 2

    from rag_eval.backends import _resolve_embedding_function

    chroma_dir = Path(args.chroma_dir)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if args.stats:
        try:
            col = client.get_collection(args.collection)
        except Exception as e:
            log(f"[ERROR] collection 不存在：{e}")
            return 2
        print(f"collection={args.collection} count={col.count()}")
        return 0

    dirs = [Path(x) for x in args.extracted_dir] if args.extracted_dir else [
        ROOT / "data" / "extracted_gd_llm",
        ROOT / "data" / "extracted_national",
    ]
    files = collect_extracted(dirs)
    if not files:
        log("[ERROR] 没有可入库的 extracted JSON")
        return 2

    if args.rebuild:
        try:
            client.delete_collection(args.collection)
            log(f"已删除旧 collection：{args.collection}")
        except Exception:
            pass

    ef = _resolve_embedding_function(args.embedding_function)
    kwargs = {"name": args.collection, "metadata": {"hnsw:space": "cosine"}}
    if ef is not None:
        kwargs["embedding_function"] = ef
    collection = client.get_or_create_collection(**kwargs)

    overrides = {}
    try:
        overrides = json.loads(Path(args.validity_overrides).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass

    splitter = PolicyChunkSplitter(max_chars=args.max_chars, overlap=args.overlap)
    seen_policy_hash: dict[str, str] = {}
    seen_policy_name: dict[tuple[str, str], str] = {}
    policy_rows: list[dict] = []
    skipped: Counter[str] = Counter()
    bad_bodies: list[dict] = []
    topic_counts: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    level_counts: Counter[str] = Counter()
    batch: list[dict] = []
    total_chunks = 0

    for i, jf in enumerate(files, 1):
        try:
            loaded = load_policy(jf, overrides)
        except Exception as e:
            skipped["parse_error"] += 1
            log(f"[WARN] {jf.name}: {e}")
            continue
        if loaded is None:
            skipped["not_lowalt_or_empty"] += 1
            continue
        meta, body, source_file = loaded

        # 质量闸 ①：正文是否像一份政策文本（非源码、非 PDF 乱码）
        if body_judge is not None:
            verdict, qm = body_judge(body)
            if verdict == "reject":
                skipped["bad_body_quality"] += 1
                bad_bodies.append({
                    "policy_name": meta["policy_name"],
                    "region": meta["region"],
                    "reason": qm.get("reason", ""),
                    "cjk_ratio": qm["cjk_ratio"],
                    "js_per_10k": qm.get("js_per_10k", 0),
                    "chars": qm["chars"],
                    "source_file": source_file,
                })
                log(f"[BAD] 正文质量不合格（{qm.get('reason', '')}｜汉字比 {qm['cjk_ratio']}）：{meta['policy_name'][:40]}")
                continue

        if meta["content_hash"] in seen_policy_hash:
            skipped["duplicate_content"] += 1
            log(f"[DUP] {meta['policy_name'][:42]} ← 与 {seen_policy_hash[meta['content_hash']][:50]} 相同")
            continue
        name_key = (norm_text(meta["policy_name"])[:40], meta["region"])
        if name_key in seen_policy_name:
            skipped["duplicate_name"] += 1
            log(f"[DUP] 同名同地域：{meta['policy_name'][:50]}")
            continue
        seen_policy_hash[meta["content_hash"]] = meta["policy_name"]
        seen_policy_name[name_key] = meta["policy_name"]

        doc_meta = {
            "policy_id": meta["policy_id"],
            "policy_name": meta["policy_name"],
            "issue_date": meta["issue_date"],
            "effective_date": meta["valid_from"] or "无",
            "expiry_date": meta["valid_to"] or "无",
            "apply_region": meta["region"],
            "status": meta["status"],
            "source_url": meta["source_url"],
        }
        docs = build_documents(body, doc_meta, splitter)
        if not docs:
            skipped["zero_chunks"] += 1
            continue

        # 质量闸 ②：单片政策切块数异常 → 多半是正文混入了与政策无关的整段文本
        if len(docs) > args.max_chunks_per_policy:
            skipped["chunk_explosion"] += 1
            bad_bodies.append({
                "policy_name": meta["policy_name"],
                "region": meta["region"],
                "reason": f"切片数 {len(docs)} 超过上限 {args.max_chunks_per_policy}",
                "cjk_ratio": 0,
                "js_per_10k": 0,
                "chars": len(body),
                "source_file": source_file,
            })
            log(f"[BAD] 切片数异常（{len(docs)} > {args.max_chunks_per_policy}）：{meta['policy_name'][:40]}")
            continue

        policy_chunks = 0
        for idx, d in enumerate(docs):
            topic = topic_of(d.page_content, d.metadata)
            md = {
                "policy_id": meta["policy_id"],
                "policy_name": meta["policy_name"],
                "article": d.metadata.get("article", ""),
                "chunk_index": idx,
                "issue_date": meta["issue_date"],
                "valid_from": meta["valid_from"],
                # Chroma 不接受 None；空串由 temporal.py 解释为长期有效
                "valid_to": meta["valid_to"],
                "region": meta["region"],
                "status": meta["status"],
                "doc_type": meta["doc_type"],
                "level": meta["level"],
                "topic": topic,
                "source_name": meta["source_name"],
                "source_url": meta["source_url"],
            }
            cid = hashlib.sha256(
                f"{meta['policy_id']}|{idx}|{d.metadata.get('article', '')}|{norm_text(d.page_content)[:120]}".encode("utf-8")
            ).hexdigest()[:24]
            batch.append({"id": f"clause_{cid}", "text": d.page_content, "meta": md})
            policy_chunks += 1
            total_chunks += 1
            topic_counts[topic] += 1
            region_counts[meta["region"]] += 1
            level_counts[meta["level"]] += 1
            if len(batch) >= args.batch_size:
                add_batch(collection, batch)
                batch.clear()

        policy_rows.append({
            "policy_id": meta["policy_id"],
            "policy_name": meta["policy_name"],
            "region": meta["region"],
            "level": meta["level"],
            "issue_date": meta["issue_date"],
            "valid_from": meta["valid_from"],
            "valid_to": meta["valid_to"],
            "status": meta["status"],
            "chunks": policy_chunks,
            "source_name": meta["source_name"],
            "source_url": meta["source_url"],
            "source_file": source_file,
        })
        log(f"[{i}/{len(files)}] {meta['region']:8} {meta['policy_name'][:44]:46} {policy_chunks:>3} chunks")

    add_batch(collection, batch)

    manifest = {
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "chroma_dir": str(chroma_dir.resolve()),
        "collection": args.collection,
        "embedding_function": args.embedding_function,
        "policies": len(policy_rows),
        "chunks": total_chunks,
        "by_region": dict(region_counts.most_common()),
        "by_level": dict(level_counts),
        "by_topic": dict(topic_counts.most_common()),
        "skipped": dict(skipped),
        "rejected_bodies": bad_bodies,
        "policies_detail": policy_rows,
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = manifest_path.with_suffix(".csv")
    fields = ["policy_id", "policy_name", "region", "level", "issue_date", "valid_from",
              "valid_to", "status", "chunks", "source_name", "source_url"]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(policy_rows, key=lambda x: (x["level"], x["region"], x["policy_name"])))

    print("\n" + "=" * 64)
    print(f"Chroma：{chroma_dir} / {args.collection}")
    print(f"政策 {len(policy_rows)} 篇 / chunk {total_chunks} 个")
    print(f"地域 {len(region_counts)} 个：{dict(region_counts.most_common(8))}")
    print(f"主题：{dict(topic_counts.most_common(8))}")
    print(f"跳过：{dict(skipped)}")
    if bad_bodies:
        print(f"⚠ 正文质量闸拦下 {len(bad_bodies)} 篇（明细见 manifest.rejected_bodies）：")
        for b in bad_bodies[:10]:
            print(f"    - {b['region']} / {b['policy_name'][:40]}：{b['reason']}")
    print(f"manifest → {manifest_path}")
    print(f"CSV      → {csv_path}")
    client.close()
    return 0 if total_chunks else 2


if __name__ == "__main__":
    raise SystemExit(main())
