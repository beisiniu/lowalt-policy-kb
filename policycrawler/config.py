"""配置加载：把 config.json 读成一个带默认值的对象，并解析相对路径。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "source": {"name": "未命名来源", "list_url": "", "base_url": "", "charset": ""},
    "list_selectors": {
        "item": ["li"],
        "title": ["a"],
        "link": ["a"],
        "date": ["span", "td", "em"],
    },
    "detail_selectors": {
        "title": ["h1", "title"],
        "publish_date": ["span", "div"],
        "content": ["article", "div.content", "div"],
    },
    "crawl": {
        "timeout": 15,
        "retries": 3,
        "retry_backoff": 2.0,
        "delay_seconds": 1.0,
        "max_items_per_run": 50,
        "check_content_update": True,
        "verify_ssl": True,
    },
    "storage": {
        "history_file": "data/crawled_history.json",
        "policy_dir": "data/policies",
        "log_dir": "data/logs",
    },
    "politeness": {"user_agents": ["Mozilla/5.0 (compatible; LowAltPolicyBot/1.0)"]},
    "filter": {"include_any": []},
}


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return list(value or [])


class Config:
    def __init__(self, raw: dict, base_dir: Path):
        self.raw = raw
        self.base_dir = base_dir
        self.source = raw["source"]
        self.crawl = raw["crawl"]
        self.storage = raw["storage"]
        self.politeness = raw["politeness"]

        # selector 统一成 list，解析时按顺序逐个尝试
        self.list_selectors = {k: _as_list(raw["list_selectors"].get(k)) for k in ("item", "title", "link", "date")}
        self.detail_selectors = {
            k: _as_list(raw["detail_selectors"].get(k)) for k in ("title", "publish_date", "content")
        }

        # 关键词过滤（空列表 = 不过滤）
        self.filter_any = list((raw.get("filter") or {}).get("include_any") or [])

        self.history_file = self._path(self.storage["history_file"])
        self.policy_dir = self._path(self.storage["policy_dir"])
        self.log_dir = self._path(self.storage["log_dir"])

    def _path(self, p: str) -> Path:
        """相对路径一律相对 **配置文件所在目录** 解析，保证从任何 CWD 调用都一致。"""
        path = Path(p)
        return path if path.is_absolute() else (self.base_dir / path)

    @property
    def list_url(self) -> str:
        return self.source["list_url"]

    @property
    def base_url(self) -> str:
        return self.source.get("base_url") or ""


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return Config(_merge(DEFAULTS, raw), path.parent)
