"""通用工具：哈希、文本归一化、文件名清洗、原子写 JSON、日期解析。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

CST = timezone(timedelta(hours=8))  # 北京时间，日志时间统一用它

# 空白字符（含全角空格 U+3000、不换行空格 U+00A0）→ 单个半角空格
_WS_RE = re.compile(r"[\s\u3000\u00a0\u200b\ufeff]+")
# Windows 文件名非法字符
_BAD_FILENAME_RE = re.compile(r'[\\/:*?"<>|\r\n\t]')


def now_iso() -> str:
    """返回带时区的 ISO8601 时间串，例如 2026-10-26T01:20:00+08:00。"""
    return datetime.now(CST).isoformat(timespec="seconds")


def normalize_text(text: str) -> str:
    """正文归一化：**算哈希前必须调用**。

    HTML 里一个换行、一个全角空格、一次模板微调（比如多加了个 &nbsp;），
    都会让原始字节的 sha256 变化，从而把「没改内容」的政策误判成「内容更新」。
    所以先做语义无关的归一化，再算哈希：
        1. NFKC 统一全角/半角（"１" → "1"）
        2. 所有空白（含全角空格、换行、制表符）压成单个空格
        3. 去掉首尾空白

    ⚠ 注意：NFKC 会把中文全角标点（，、；）压成半角（, ;）。
      因此它**只用于哈希指纹**，落盘正文请用 clean_space()，保留全角标点。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def clean_space(text: str) -> str:
    """只压缩空白、保留中文全角标点 —— 用于落盘/展示的正文清洗。"""
    if not text:
        return ""
    return _WS_RE.sub(" ", text).strip()



def sha256_text(text: str) -> str:
    """对已归一化的文本算 sha256（十六进制小写）。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def url_key(url: str, length: int = 16) -> str:
    """把 URL 压成稳定的短主键。

    crawled_history.json 用这个值做 dict key：
      - 定长、无非法字符，JSON 里好读
      - 同一 URL 永远算出同一个 key，因此可以 O(1) 判断「抓过没有」
    """
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:length]


def slugify(title: str, max_len: int = 60) -> str:
    """政策标题 → 安全文件名：去非法字符、压缩空格、限长。"""
    name = _BAD_FILENAME_RE.sub("_", title or "untitled")
    name = _WS_RE.sub("_", name).strip("._")
    return name[:max_len] or "untitled"


def parse_date_to_iso(raw: str) -> str:
    """把页面上的各种日期写法统一成 YYYY-MM-DD，失败返回空串。

    覆盖：2026-10-20 / 2026/10/20 / 2026.10.20 / 2026年10月20日 / 20261020
    """
    if not raw:
        return ""
    s = normalize_text(raw)
    m = re.search(r"(\d{4})\s*[-/.年]\s*(\d{1,2})\s*[-/.月]\s*(\d{1,2})", s)
    if not m:
        m = re.search(r"\b(\d{4})(\d{2})(\d{2})\b", s)
    if not m:
        return ""
    y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    try:
        return datetime(y, mo, d).strftime("%Y-%m-%d")
    except ValueError:
        return ""


def atomic_write_json(path: str | Path, data: Any, indent: int = 2) -> None:
    """原子写 JSON —— 状态文件绝对不能写坏。

    直接 open(path, "w") 写盘时，如果进程被 Ctrl+C / 断电打断，
    会留下一个「半个大括号」的文件，下次运行 json.load 直接崩，历史状态全丢。
    做法：先写同目录的临时文件 → fsync 落盘 → os.replace 原子替换。
    os.replace 在同一文件系统内是原子操作，要么是旧文件，要么是新文件。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)  # 原子替换
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def resolve_source_path(raw: str, search_dirs: tuple[str | Path, ...] = ()) -> Path | None:
    """解析打标 JSON 里记录的原文路径。

    绝对路径失效（换机、迁移、备份目录改名）时，按文件名在给定目录里回退查找。
    返回 None 表示原始 txt 确实找不到，调用方应跳过该条而不是静默入库空正文。
    """
    if not raw:
        return None
    p = Path(raw)
    if p.exists():
        return p
    name = p.name
    if not name:
        return None
    for d in search_dirs:
        cand = Path(d) / name
        if cand.exists():
            return cand
    return None


def read_json(path: str | Path, default: Any = None) -> Any:
    """读 JSON；文件不存在或已损坏时返回 default（并保留一个 .broken 备份）。"""
    path = Path(path)
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        broken = path.with_suffix(path.suffix + ".broken")
        path.replace(broken)
        print(f"[WARN] {path} 不是合法 JSON，已备份为 {broken}，本次按空历史处理")
        return default


def load_env_file(path: str | Path) -> int:
    """极简 .env 读取：KEY=VALUE，支持 # 注释与引号，**不覆盖已存在的环境变量**。

    不引 python-dotenv：只为读几个 KEY 加一个依赖不值当，而且行为要可预测——
    系统环境变量优先级永远高于 .env。
    """
    path = Path(path)
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip("'\"")
        if k and k not in os.environ:
            os.environ[k] = v
            n += 1
    return n
