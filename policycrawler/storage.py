"""落盘层：把抓到的政策存成 txt（给大模型读）+ json（结构化元数据）。"""
from __future__ import annotations

import json
from pathlib import Path

from .parser import Detail
from .utils import normalize_text, now_iso, sha256_text, slugify, url_key

HEADER_SEP = "-" * 60


def build_filename(detail: Detail, revision: int = 1) -> str:
    """政策名 → 文件名：`2026-10-20_深圳市低空经济促进条例.txt`。

    - 前缀发布日期，文件夹里按时间自然排序
    - revision > 1 时加 `_rev2` 后缀，**不覆盖**历史版本（政策修订要留痕）
    """
    date = detail.publish_date or "unknown-date"
    title = detail.title or "untitled"
    stem = f"{date}_{slugify(title)}"
    if revision > 1:
        stem += f"_rev{revision}"
    return stem


def save_policy(
    detail: Detail,
    policy_dir: str | Path,
    revision: int = 1,
    extra_meta: dict | None = None,
) -> tuple[Path, Path, str]:
    """写盘，返回 (txt 路径, json 路径, 正文 content_hash)。

    正文哈希只对**归一化后的正文**计算，header 不参与——否则每次抓取时间不同，
    哈希永远不同，第二层去重就失效了。
    """
    policy_dir = Path(policy_dir)
    policy_dir.mkdir(parents=True, exist_ok=True)

    # 落盘用原样正文（保留全角标点），哈希仍走 NFKC 归一化
    body = detail.content.strip()
    content_hash = sha256_text(normalize_text(detail.content))

    stem = build_filename(detail, revision)
    txt_path = policy_dir / f"{stem}.txt"
    json_path = policy_dir / f"{stem}.json"

    # 极端情况下同一天同名的两条不同政策会撞名：用 url 短哈希兜底
    if txt_path.exists() and json_path.exists():
        try:
            old = json.loads(json_path.read_text(encoding="utf-8"))
            if old.get("url") != detail.url:
                stem = f"{stem}_{url_key(detail.url, 6)}"
                txt_path = policy_dir / f"{stem}.txt"
                json_path = policy_dir / f"{stem}.json"
        except (OSError, json.JSONDecodeError):
            pass

    header = "\n".join(
        [
            f"政策名称：{detail.title}",
            f"发布日期：{detail.publish_date or '未知'}",
            f"来源链接：{detail.url}",
            f"抓取时间：{now_iso()}",
            f"正文哈希：sha256:{content_hash}",
            f"版本：rev{revision}",
            HEADER_SEP,
        ]
    )
    txt_path.write_text(f"{header}\n{body}\n", encoding="utf-8", newline="\n")

    meta = {
        "title": detail.title,
        "url": detail.url,
        "publish_date": detail.publish_date,
        "content_hash": content_hash,
        "content_length": len(detail.content),
        "revision": revision,
        "crawled_at": now_iso(),
        "txt_path": str(txt_path),
        "charset_note": "见 fetched_at",
        **(extra_meta or {}),
    }
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

    return txt_path, json_path, content_hash
