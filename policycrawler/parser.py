"""HTML 解析层：列表页抽取、详情页正文抽取、日期归一化。

设计要点：**每个字段支持多个候选 selector，按顺序尝试**。
政府网站改版很常见，写死一个选择器 = 半年后脚本必挂。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

from .utils import clean_space, parse_date_to_iso

_DROP_TAGS = ("script", "style", "noscript", "iframe", "form", "nav", "footer")


@dataclass
class ListItem:
    title: str
    url: str
    publish_date: str = ""


@dataclass
class Detail:
    title: str
    url: str
    publish_date: str = ""
    content: str = ""
    raw_length: int = 0
    extra: dict = field(default_factory=dict)


def _make_soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:  # 没装 lxml 时退回标准库解析器
        return BeautifulSoup(html, "html.parser")


def _first(node: Tag, selectors: list[str]) -> Optional[Tag]:
    """按顺序尝试 selector，返回第一个命中的节点。"""
    for sel in selectors:
        try:
            hit = node.select_one(sel)
        except Exception:  # 用户写了非法 CSS 时不要让整个脚本崩
            continue
        if hit is not None:
            return hit
    return None


def _text(node: Optional[Tag]) -> str:
    return clean_space(node.get_text(" ", strip=True)) if node else ""


def parse_list(html: str, selectors: dict, base_url: str, stats: Optional[dict] = None) -> list[ListItem]:
    """解析列表页 → [ListItem]。

    策略：
      1. 用 item 选择器命中「每条记录」的容器
      2. 容器内用 title/link 选择器取 <a>
      3. 日期优先用 date 选择器，取不到就用正则从条目文本里捞（列表页几乎都带日期）
      4. 相对链接用 urljoin 补成绝对链接，并去掉 #锚点

    stats：传入 dict 时回填丢弃统计。**没有这个统计，"0 条"到底是"今天没新政策"
    还是"站点改版了"就分不清**——静默丢数据比报错更危险。
    """
    stats = stats if stats is not None else {}
    stats.update({"candidates": 0, "no_link": 0, "short_title": 0, "no_date": 0, "kept": 0})

    soup = _make_soup(html)
    items: list[ListItem] = []

    candidates: list[Tag] = []
    for sel in selectors.get("item", []):
        try:
            hits = soup.select(sel)
        except Exception:
            continue
        if hits:
            candidates = hits
            break

    if not candidates:
        # 兜底：正文里所有 <li>，靠日期过滤，避免改版后直接 0 条
        candidates = soup.find_all("li")

    stats["candidates"] = len(candidates)

    for node in candidates:
        a = _first(node, selectors.get("title", ["a"])) or node.find("a", href=True)
        if isinstance(a, Tag) and a.name != "a":
            a = a.find("a", href=True) or a
        if not isinstance(a, Tag) or not a.get("href"):
            stats["no_link"] += 1
            continue

        href = (a.get("href") or "").strip()
        if not href or href.startswith(("javascript:", "#", "mailto:")):
            stats["no_link"] += 1
            continue

        title = _text(a) or clean_space(a.get("title", ""))
        if len(title) < 4:  # 「更多>>」「下一页」这类噪声
            stats["short_title"] += 1
            continue

        date_raw = _text(_first(node, selectors.get("date", [])))
        publish_date = parse_date_to_iso(date_raw) or parse_date_to_iso(node.get_text(" ", strip=True))
        if not publish_date:
            stats["no_date"] += 1
            continue

        items.append(ListItem(title=title, url=urljoin(base_url, href).split("#")[0], publish_date=publish_date))

    stats["kept"] = len(items)
    return items


def diagnose_list(stats: dict) -> list[str]:
    """把统计翻译成人话告警 —— pipeline 直接打印。"""
    msgs: list[str] = []
    cand, kept = stats.get("candidates", 0), stats.get("kept", 0)

    if cand == 0:
        msgs.append("列表页没有任何候选节点命中：item 选择器可能失配（站点改版），请更新 config.list_selectors")
        return msgs

    if kept == 0:
        if stats.get("no_date"):
            msgs.append(f"命中 {cand} 个条目，但 {stats['no_date']} 条全部因**日期无法解析**被丢弃，"
                        f"请检查 list_selectors.date，或确认列表页是否真的带日期")
        elif stats.get("no_link"):
            msgs.append(f"命中 {cand} 个条目，但都取不到有效链接，请检查 list_selectors.link")
        elif stats.get("short_title"):
            msgs.append(f"命中 {cand} 个条目，但标题都太短（疑似导航/分页），请检查 list_selectors.title")
    elif stats.get("no_date"):
        pct = stats["no_date"] / max(1, cand) * 100
        msgs.append(f"另有 {stats['no_date']} 条因日期无法解析被丢弃（占 {pct:.0f}%），"
                    f"若比例偏高请检查 date 选择器")
    return msgs


def _clean_content(node: Tag) -> str:
    """把一个正文节点的可读文本抽出来，保留段落结构。"""
    node = BeautifulSoup(str(node), "lxml")  # 复制一份，避免污染原 soup
    for tag in node.find_all(_DROP_TAGS):
        tag.decompose()
    for br in node.find_all("br"):
        br.replace_with("\n")

    lines = [clean_space(ln) for ln in node.get_text("\n", strip=True).split("\n")]
    out: list[str] = []
    for ln in lines:
        if not ln:
            continue
        # 合并被 HTML 硬换行切碎的句子：上一行不以句末标点结尾且很短时不空行
        out.append(ln)
    return "\n".join(out)


def _fallback_content(soup: BeautifulSoup) -> str:
    """正文选择器全失效时的兜底：找「文字量最大」的那个 div。"""
    best, best_len = None, 0
    for div in soup.find_all(["div", "article", "section"]):
        if div.find(["div", "article"], recursive=False) is not None:
            continue  # 跳过纯布局容器
        n = len(div.get_text(" ", strip=True))
        if n > best_len:
            best, best_len = div, n
    return _clean_content(best) if best is not None else ""


def parse_detail(html: str, selectors: dict, url: str) -> Detail:
    """解析详情页 → Detail（标题 / 发布日期 / 正文）。"""
    soup = _make_soup(html)
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()

    title = _text(_first(soup, selectors.get("title", ["h1"])))
    if len(title) < 4:  # h1 没命中时退回 <title>，再去掉站点后缀
        title = clean_space(soup.title.get_text(strip=True)) if soup.title else ""
        title = re.split(r"\s{2,}|\u3000|\s+[-_|—]+\s+", title)[0].strip()

    date_node = _first(soup, selectors.get("publish_date", []))
    publish_date = parse_date_to_iso(_text(date_node))
    if not publish_date:
        # 全页搜第一个日期，通常是发文日期
        publish_date = parse_date_to_iso(soup.get_text(" ", strip=True)[:800])

    content = ""
    for sel in selectors.get("content", []):
        node = _first(soup, [sel])
        if node is not None:
            text = _clean_content(node)
            if len(text) > 200:  # 太短的命中是「相关链接」之类的噪声，继续试下一个
                content = text
                break
    if not content:
        content = _fallback_content(soup)

    return Detail(
        title=title,
        url=url,
        publish_date=publish_date,
        content=content,
        raw_length=len(content),
    )
