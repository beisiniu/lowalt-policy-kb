"""本地状态管理：crawled_history.json 的读写与增量判定。

================================ 核心概念 ================================
状态文件里做**两层去重**，这是增量爬虫的关键：

  第一层 · URL 去重（"这条政策我见过没有"）
      key = sha256(url)[:16]，直接作为 records 的字典键。
      命中 → 详情页抓过，不必再当新政策处理。

  第二层 · 内容哈希去重（"这条政策的内容改过没有"）
      content_hash = sha256(normalize(正文))
      政策会修订：同一 URL 正文变了，属于「更新」而不是「新增」。
      content_hash 一致 → SKIP，不重复落盘、不重复入库。

  ⚠ 为什么必须先 normalize 再算哈希？
     原始 HTML 转文本时，换行、全角空格、&nbsp; 都会变。若直接哈希，
     站点模板微调一次就会把所有政策判定为「已更新」，RAG 知识库被灌满重复文档。
==========================================================================
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .utils import atomic_write_json, now_iso, read_json, url_key

SCHEMA_VERSION = 1


class HistoryStore:
    """crawled_history.json 的读写封装。

    文件结构：
    {
      "schema_version": 1,
      "source": "中国民用航空局-政策法规",
      "updated_at": "2026-10-26T01:20:00+08:00",
      "records": {
        "3f9a...（url_key）": {
          "url": "https://...",
          "title": "深圳市低空经济促进条例",
          "publish_date": "2026-10-20",
          "content_hash": "sha256...",     <- 第二层去重依据
          "content_length": 5321,
          "content_hashes": ["sha256..."], <- 历史版本，用于识别"第几次修订"
          "revision": 1,
          "txt_path": "data/policies/xxx.txt",
          "json_path": "data/policies/xxx.json",
          "first_seen_at": "...",
          "last_crawled_at": "..."
        }
      }
    }
    """

    def __init__(self, path: str | Path, source: str = "", autosave: bool = True):
        self.path = Path(path)
        self.source = source
        self.autosave = autosave
        self.records: dict[str, dict] = {}
        self.load()

    # ---------------- 读写 ----------------
    def load(self) -> None:
        data = read_json(self.path, default=None)
        if not data:
            self.records = {}
            return
        self.records = data.get("records", {}) or {}
        # 兼容：万一有人手工把 records 写成 list
        if isinstance(self.records, list):
            self.records = {r["url_key"]: r for r in self.records if r.get("url_key")}

    def save(self) -> None:
        atomic_write_json(
            self.path,
            {
                "schema_version": SCHEMA_VERSION,
                "source": self.source,
                "updated_at": now_iso(),
                "count": len(self.records),
                "records": self.records,
            },
        )

    # ---------------- 查询 ----------------
    def key_of(self, url: str) -> str:
        return url_key(url)

    def get(self, url: str) -> Optional[dict]:
        """按 URL 取历史记录（第一层去重的入口）。"""
        return self.records.get(self.key_of(url))

    def has(self, url: str) -> bool:
        return self.key_of(url) in self.records

    def content_changed(self, url: str, new_hash: str) -> bool:
        """第二层去重：正文哈希是否变化。没记录过视为"变化"（需要保存）。"""
        rec = self.get(url)
        if not rec:
            return True
        return rec.get("content_hash") != new_hash

    # ---------------- 写入 ----------------
    def add_new(self, record: dict) -> None:
        """登记一条新政策。record 需含 url / content_hash 等字段。"""
        key = self.key_of(record["url"])
        rec = dict(record)
        rec.setdefault("revision", 1)
        rec.setdefault("content_hashes", [rec.get("content_hash")])
        rec.setdefault("first_seen_at", now_iso())
        rec["last_crawled_at"] = now_iso()
        self.records[key] = rec
        if self.autosave:
            self.save()

    def mark_updated(self, url: str, new_hash: str, **fields: Any) -> dict:
        """同一 URL 正文变了：递增 revision，并把新哈希追加进历史。"""
        key = self.key_of(url)
        rec = self.records.get(key)
        if rec is None:
            raise KeyError(f"{url} 不在历史中，不能标记为更新")
        rec["revision"] = int(rec.get("revision", 1)) + 1
        rec["content_hash"] = new_hash
        rec.setdefault("content_hashes", [])
        if new_hash not in rec["content_hashes"]:
            rec["content_hashes"].append(new_hash)
        rec["last_crawled_at"] = now_iso()
        rec.update(fields)
        if self.autosave:
            self.save()
        return rec

    def touch(self, url: str) -> None:
        """命中但内容没变：只更新 last_crawled_at（可选的探活记录）。"""
        rec = self.get(url)
        if rec:
            rec["last_crawled_at"] = now_iso()
            if self.autosave:
                self.save()

    # ---------------- 统计 ----------------
    @property
    def count(self) -> int:
        return len(self.records)

    def summary(self) -> str:
        by_month: dict[str, int] = {}
        for rec in self.records.values():
            month = (rec.get("publish_date") or "unknown")[:7]
            by_month[month] = by_month.get(month, 0) + 1
        lines = [f"历史记录共 {self.count} 条（{self.path}）"]
        for month in sorted(by_month, reverse=True)[:6]:
            lines.append(f"  {month}: {by_month[month]} 条")
        return "\n".join(lines)
