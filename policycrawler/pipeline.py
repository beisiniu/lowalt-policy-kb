"""增量抓取主流程：列表页 → 逐条判定（新增/更新/跳过）→ 落盘 → 更新状态。"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import Config
from .fetcher import Fetcher, FetchError
from .parser import diagnose_list, parse_detail, parse_list
from .state import HistoryStore
from .storage import save_policy
from .utils import CST, normalize_text, now_iso, parse_date_to_iso, sha256_text


@dataclass
class ItemResult:
    status: str  # new | updated | skipped | duplicate | failed
    title: str
    url: str
    publish_date: str = ""
    revision: int = 0
    content_length: int = 0
    saved_to: str = ""
    note: str = ""


@dataclass
class RunResult:
    started_at: str = field(default_factory=now_iso)
    finished_at: str = ""
    elapsed_sec: float = 0.0
    listed: int = 0
    items: list[ItemResult] = field(default_factory=list)

    @property
    def new_items(self) -> list[ItemResult]:
        return [i for i in self.items if i.status == "new"]

    @property
    def updated_items(self) -> list[ItemResult]:
        return [i for i in self.items if i.status == "updated"]

    @property
    def skipped(self) -> list[ItemResult]:
        return [i for i in self.items if i.status in ("skipped", "duplicate")]

    @property
    def failed(self) -> list[ItemResult]:
        return [i for i in self.items if i.status == "failed"]

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "listed": self.listed,
            "new": len(self.new_items),
            "updated": len(self.updated_items),
            "skipped": len(self.skipped),
            "failed": len(self.failed),
            "items": [asdict(i) for i in self.items],
        }


class IncrementalCrawler:
    def __init__(self, cfg: Config, check_updates: Optional[bool] = None, limit: Optional[int] = None):
        self.cfg = cfg
        crawl = cfg.crawl
        self.fetcher = Fetcher(
            timeout=crawl["timeout"],
            retries=crawl["retries"],
            backoff=crawl["retry_backoff"],
            delay=crawl["delay_seconds"],
            user_agents=cfg.politeness["user_agents"],
            charset=cfg.source.get("charset", ""),
            verify_ssl=crawl.get("verify_ssl", True),
        )
        self.store = HistoryStore(cfg.history_file, source=cfg.source.get("name", ""))
        self.check_updates = crawl["check_content_update"] if check_updates is None else check_updates
        self.limit = limit or crawl["max_items_per_run"]
        # 正文哈希 → URL 反查表：识别「同一份政策被转载到多个栏目」
        self._hash_index: dict[str, str] = {
            r["content_hash"]: r["url"] for r in self.store.records.values() if r.get("content_hash")
        }

    # ------------------------------------------------------------------
    def run(self, dry_run: bool = False) -> RunResult:
        t0 = time.time()
        result = RunResult()
        print(f"→ 列表页：{self.cfg.list_url}")
        html = self.fetcher.get_text(self.cfg.list_url)
        stats: dict = {}
        items = parse_list(html, self.cfg.list_selectors, self.cfg.base_url or self.cfg.list_url, stats)
        print(f"→ 解析到 {len(items)} 条列表记录（历史库存 {self.store.count} 条）")

        # 「0 条」必须区分是"今天没新政策"还是"选择器挂了"，否则会静默丢数据
        for msg in diagnose_list(stats):
            print(f"[WARN] {msg}")

        # 关键词过滤：综合栏目（如北京市政策文件）里 99% 与低空无关，
        # 在拉详情页之前就筛掉，既省带宽也避免知识库被无关政策污染
        items, filtered_out = self._apply_keyword_filter(items)
        if filtered_out:
            print(f"→ 关键词过滤：剔除 {filtered_out} 条与「{self.cfg.filter_any[0]}…」无关的政策")

        # 本次运行内按 URL 去重（列表页分页/置顶会让同一条出现多次）
        seen: set[str] = set()
        uniq = []
        for it in items:
            if it.url in seen:
                continue
            seen.add(it.url)
            uniq.append(it)
        result.listed = len(uniq)
        uniq = uniq[: self.limit]

        for idx, item in enumerate(uniq, 1):
            tag = f"[{idx}/{len(uniq)}]"
            try:
                res = self._handle_item(item, dry_run=dry_run)
            except FetchError as e:
                res = ItemResult(status="failed", title=item.title, url=item.url,
                                 publish_date=item.publish_date, note=str(e)[:160])
            except Exception as e:  # 单条异常不能中断整轮增量
                res = ItemResult(status="failed", title=item.title, url=item.url,
                                 publish_date=item.publish_date, note=f"{type(e).__name__}: {e}"[:160])
            result.items.append(res)
            self._print_item(tag, res)

        result.finished_at = now_iso()
        result.elapsed_sec = time.time() - t0
        if not dry_run:
            self._write_log(result)
        return result

    # ------------------------------------------------------------------
    def _apply_keyword_filter(self, items: list) -> tuple[list, int]:
        """按标题关键词保留「低空/无人机」相关条目。

        config.filter.include_any 为空时不过滤（专题栏目已经是精准的）。
        只看标题是刻意的：要按正文筛就得先抓详情页，等于白花 N 次请求；
        标题里没有"低空/无人机"但正文有的政策，走 tools/import_manual.py 人工补。
        """
        kws = self.cfg.filter_any
        if not kws:
            return items, 0
        kept = [it for it in items if any(k in it.title for k in kws)]
        return kept, len(items) - len(kept)

    def _handle_item(self, item, dry_run: bool) -> ItemResult:
        url = item.url
        existing = self.store.get(url)

        # ---------- 第一层：URL 去重 ----------
        if existing is not None and not self.check_updates:
            self.store.touch(url)
            return ItemResult("skipped", item.title, url, item.publish_date,
                              existing.get("revision", 1), existing.get("content_length", 0))

        detail = parse_detail(self.fetcher.get_text(url, referer=self.cfg.list_url), self.cfg.detail_selectors, url)

        # 标题：详情页的 <title> 常被站点后缀污染（"…的通知  广东省人民政府"），
        # 列表页那一列通常就是干净的政策名，所以列表标题优先；它被截断（带 …）时才用详情页的。
        if item.title and len(item.title) >= 6 and not item.title.endswith(("…", "...")):
            detail.title = item.title
        elif not detail.title:
            detail.title = item.title

        # 发布日期以**列表页那一列**为准。详情页上「成文日期」「实施日期」「印发日期」混在一起，
        # 随便命中一个就会把 2026-10-23 的发布日写成 2027-01-01 的实施日（北京政务网实测踩过）。
        detail_date = detail.publish_date
        if item.publish_date:
            if detail_date and detail_date != item.publish_date:
                print(f"  [INFO] 详情页日期 {detail_date} 与列表页 {item.publish_date} 不一致，采用列表页（发布日期）")
            detail.publish_date = item.publish_date
        body = normalize_text(detail.content)
        if len(body) < 50:
            raise FetchError(f"正文过短({len(body)}字)，疑似反爬/结构变化，跳过")
        new_hash = sha256_text(body)

        # ---------- 已抓过：第二层按内容哈希判断有没有改 ----------
        if existing is not None:
            if existing.get("content_hash") == new_hash:
                self.store.touch(url)
                return ItemResult("skipped", detail.title, url, detail.publish_date,
                                  existing.get("revision", 1), len(body), note="正文未变化")

            rev = int(existing.get("revision", 1)) + 1
            if dry_run:
                return ItemResult("updated", detail.title, url, detail.publish_date, rev, len(body), note="[dry-run]")
            txt_path, json_path, chash = save_policy(detail, self.cfg.policy_dir, revision=rev,
                                                     extra_meta={"previous_hash": existing.get("content_hash")})
            self.store.mark_updated(url, chash, title=detail.title, publish_date=detail.publish_date,
                                    content_length=len(body), txt_path=str(txt_path), json_path=str(json_path))
            self._hash_index[chash] = url
            return ItemResult("updated", detail.title, url, detail.publish_date, rev, len(body),
                              saved_to=str(txt_path), note=f"正文变化，已存 rev{rev}")

        # ---------- 同一份正文已在别的 URL 存过（政策转载） ----------
        dup_of = self._hash_index.get(new_hash)
        if dup_of and dup_of != url and len(body) > 200:
            if not dry_run:
                self.store.add_new({
                    "url": url, "title": detail.title, "publish_date": detail.publish_date,
                    "content_hash": new_hash, "content_length": len(body),
                    "revision": 1, "duplicate_of": dup_of, "txt_path": "", "json_path": "",
                })
            return ItemResult("duplicate", detail.title, url, detail.publish_date, 1, len(body),
                              note=f"与 {dup_of[:60]} 正文相同")

        # ---------- 全新政策 ----------
        if dry_run:
            return ItemResult("new", detail.title, url, detail.publish_date, 1, len(body), note="[dry-run]")
        txt_path, json_path, chash = save_policy(detail, self.cfg.policy_dir, revision=1)
        self.store.add_new({
            "url": url, "title": detail.title, "publish_date": detail.publish_date,
            "content_hash": chash, "content_length": len(body), "revision": 1,
            "txt_path": str(txt_path), "json_path": str(json_path),
        })
        self._hash_index[chash] = url
        return ItemResult("new", detail.title, url, detail.publish_date, 1, len(body), saved_to=str(txt_path))

    # ------------------------------------------------------------------
    _ICON = {"new": "NEW  ", "updated": "UPDATE", "skipped": "SKIP ", "duplicate": "DUP  ", "failed": "FAIL "}

    def _print_item(self, tag: str, r: ItemResult) -> None:
        title = r.title if len(r.title) <= 34 else r.title[:33] + "…"
        extra = f" rev{r.revision}" if r.status == "updated" else ""
        note = f"  ({r.note})" if r.note else ""
        print(f"  {tag} {self._ICON.get(r.status, r.status)} {r.publish_date or '----------'}  "
              f"{title}  {r.content_length}字{extra}{note}")

    def _write_log(self, result: RunResult) -> Path:
        self.cfg.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(CST).strftime("%Y%m%d_%H%M%S")
        path = self.cfg.log_dir / f"run_{stamp}.json"
        path.write_text(json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path
