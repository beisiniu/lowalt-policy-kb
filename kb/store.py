"""本地索引的落盘与加载（替代 Dify 的知识库存储）。

产物全部是纯文本/JSONL，可 diff、可进 git、可复现：
    kb/index/chunks.jsonl   每个 chunk 一行：文本 + 元数据 + 分词结果
    kb/index/manifest.json  构建信息：来源政策、chunk 数、构建时间
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from .schema import tokenize

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "kb" / "index"


def save_index(docs: Iterable[Document], out_dir: str | Path = DEFAULT_DIR) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, policies = [], {}

    for d in docs:
        md = dict(d.metadata)
        tokens = tokenize(d.page_content + " " + str(md.get("policy_name", "")))
        rows.append({"text": d.page_content, "meta": md, "tokens": tokens})
        key = md.get("policy_id") or md.get("policy_name")
        policies[key] = {
            "policy_name": md.get("policy_name"),
            "status": md.get("status"),
            "apply_region": md.get("apply_region"),
            "effective_date": md.get("effective_date"),
            "expiry_date": md.get("expiry_date"),
        }

    chunks_path = out / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    manifest = {
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "policies": len(policies),
        "chunks": len(rows),
        "avg_tokens": round(sum(len(r["tokens"]) for r in rows) / max(1, len(rows)), 1),
        "policy_list": policies,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def load_index(path_dir: str | Path = DEFAULT_DIR) -> tuple[list[Document], list[list[str]], dict]:
    """返回 (docs, tokens, manifest)。tokens 与 docs 一一对应，避免重复分词。"""
    d = Path(path_dir)
    path = d / "chunks.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"索引不存在：{path}\n请先运行：python kb_ingest.py data/extracted/ --rebuild"
        )
    docs, tokens = [], []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            docs.append(Document(page_content=row["text"], metadata=row["meta"]))
            tokens.append(row.get("tokens") or tokenize(row["text"]))

    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return docs, tokens, manifest


def rebuild_from_scratch(out_dir: str | Path = DEFAULT_DIR) -> None:
    """清空重建：删掉旧索引文件（不动源数据）。"""
    out = Path(out_dir)
    for name in ("chunks.jsonl", "manifest.json"):
        p = out / name
        if p.exists():
            p.unlink()
