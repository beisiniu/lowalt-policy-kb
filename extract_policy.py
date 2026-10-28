#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段二 · 大模型自动打标与结构化抽取（DeepSeek / OpenAI 兼容协议）

用法：
    # 1) 先配置 Key（Windows PowerShell）
    #    $env:DEEPSEEK_API_KEY="sk-xxxxxxxx"
    python extract_policy.py "demo/data/policies/2026-03-15_深圳经济特区低空经济产业促进条例.txt"

    # 2) 批量处理整个目录
    python extract_policy.py data/policies/

    # 3) 没配 Key 也能跑通链路（规则版抽取，验证数据结构）
    python extract_policy.py <file> --offline

    # 4) 只要 JSON（方便管道给下一步）
    python extract_policy.py <file> --quiet
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policycrawler.extract_rules import extract_by_rules, parse_header
from policycrawler.llm import ChatClient, LLMError, provider_client
from policycrawler.utils import load_env_file, parse_date_to_iso, resolve_source_path

ROOT = Path(__file__).resolve().parent
FIELDS = ("policy_name", "issue_date", "effective_date", "expiry_date", "apply_region", "status", "summary")

# 正文太长会白烧 token：头部 6000 字（发文机关/文号/施行日期都在开头）+
# 尾部 1500 字（附则里的"本办法自…起施行""有效期 X 年"通常在最后）
HEAD_CHARS, TAIL_CHARS = 6000, 1500

SYSTEM_PROMPT = """你是一名中国低空经济与民航政策法规领域的信息抽取专家。
你的任务：从给定政策原文中抽取结构化元数据，**只输出一个 JSON 对象**，不要输出解释、Markdown 或代码块。

输出字段（全部必填）：
{
  "policy_name": "政策全称，去掉书名号和发文机关前缀，例：深圳经济特区低空经济产业促进条例",
  "issue_date": "发布日期，严格 YYYY-MM-DD；无法确定填 \"无\"",
  "effective_date": "生效/施行日期，严格 YYYY-MM-DD；原文写「自公布之日起施行」时取发布日期；无法确定填 \"无\"",
  "expiry_date": "失效日期，严格 YYYY-MM-DD；无有效期条款填 \"无\"",
  "apply_region": "适用地域，落到最小可判定行政区，如「全国」「广东省」「深圳市」；国家层面或跨省填「全国」",
  "status": "只能是「现行有效」或「已废止」",
  "summary": "一句话（60 字内）概括该政策的核心审批/合规要求：谁、什么场景、是否需要审批/备案、关键门槛（重量/高度/时限）"
}

判定规则：
1. 日期一律补零成 YYYY-MM-DD；中文数字日期要换算，如「二〇二六年五月一日」→ 2026-05-01。
2. 原文有「有效期 X 年」且能确定施行日时，expiry_date = 施行日 + X 年（对应的前一天）；
   原文有「有效期至 YYYY 年 MM 月 DD 日」直接使用。
   **规划期不等于失效期**：标题中的「2024—2026年」「三年行动方案」、目标完成时限、
   工作阶段截止年份都不能当作 expiry_date；没有明确有效期/失效条款必须填「无」。
3. status 判定依据：
   - 原文明确废止/停止执行该文件本身，或该文件已被新文件明确取代 → 已废止
   - 有效期已过期（相对"今天"）→ 已废止
   - 其余 → 现行有效
4. apply_region 不允许写「本市」「本省」，必须写具体行政区名。
5. summary 必须体现审批/合规口径，不要写成政策目的或意义。
6. 缺失信息一律填 "无"，禁止编造。输出必须是**单个 JSON 对象**。
"""


# ----------------------------------------------------------------------
def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def read_policy_text(path: Path) -> str:
    """支持 .txt 正文，或爬虫产出的 .json（自动跳转到对应 txt）。"""
    if path.suffix.lower() == ".json":
        meta = json.loads(path.read_text(encoding="utf-8"))
        search_dirs = (ROOT / "data" / "national_policies",
                       ROOT / "data" / "gd_policies",
                       ROOT / "data" / "beijing_policies")
        for key in ("txt_path", ("_meta", "source_file")):
            txt = meta.get(key) if isinstance(key, str) else (meta.get("_meta") or {}).get(key[1])
            src = resolve_source_path(str(txt), search_dirs)
            if src is not None:
                return src.read_text(encoding="utf-8", errors="replace")
        return json.dumps(meta, ensure_ascii=False)
    return path.read_text(encoding="utf-8", errors="replace")


def smart_truncate(text: str) -> str:
    if len(text) <= HEAD_CHARS + TAIL_CHARS:
        return text
    return (
        text[:HEAD_CHARS]
        + "\n……（此处省略中间正文，共 "
        + str(len(text))
        + " 字）……\n"
        + text[-TAIL_CHARS:]
    )


def build_user_prompt(text: str, header: dict) -> str:
    today = date.today().isoformat()
    hint = ""
    if header.get("政策名称") or header.get("发布日期"):
        hint = (
            "\n【爬虫已抓取的页面元数据，可作参考但不保证准确】\n"
            f"页面标题：{header.get('政策名称', '无')}\n页面发布日期：{header.get('发布日期', '无')}\n"
        )
    return f"今天是 {today}。{hint}\n【政策原文】\n{smart_truncate(text)}"


# ----------------------------------------------------------------------
def _norm_date(value, fallback: str = "无") -> str:
    if value is None:
        return fallback
    s = str(value).strip()
    if s in ("", "无", "None", "null", "未注明", "-", "N/A", "n/a"):
        return "无"
    return parse_date_to_iso(s) or fallback


def _norm_status(value) -> str:
    s = str(value or "").strip()
    if any(k in s for k in ("废止", "失效", "无效", "已被取代")):
        return "已废止"
    if any(k in s for k in ("现行", "有效", "生效", "执行中")):
        return "现行有效"
    return "现行有效"


def normalize_result(raw: dict, file_stem: str = "") -> dict:
    """把模型输出洗成严格一致的 schema —— 下游 Dify / 评估脚本都依赖它。"""
    if not isinstance(raw, dict):
        raise LLMError(f"模型输出不是 JSON 对象：{type(raw)}")

    name = str(raw.get("policy_name") or "").strip().strip("《》「」\"'")
    out = {
        "policy_name": name or file_stem or "未命名政策",
        "issue_date": _norm_date(raw.get("issue_date")),
        "effective_date": _norm_date(raw.get("effective_date")),
        "expiry_date": _norm_date(raw.get("expiry_date")),
        "apply_region": str(raw.get("apply_region") or "全国").strip().strip("「」\"'") or "全国",
        "status": _norm_status(raw.get("status")),
        "summary": " ".join(str(raw.get("summary") or "").split())[:180],
    }
    missing = [f for f in FIELDS if not out[f]]
    if missing:
        log(f"[WARN] 字段为空已兜底：{missing}")
    if out["effective_date"] == "无" and out["issue_date"] != "无":
        # 法律上「未写施行日期」通常等于「自公布之日起施行」，所以按发布日兜底；
        # 但这毕竟是推断，打个标记，别让下游以为是原文明确写的日子
        out["effective_date"] = out["issue_date"]
        out["_assumed_effective"] = True
    return out


# 只有这些原文表述才足以支持一个具体失效日。注意「政策有效期内每家企业…」
# 并没有说明期限，不能据此让模型猜一个日期。
_EXPIRY_EVIDENCE_RES = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"有效期\s*(?:至|到|为)\s*\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日",
    r"有效期\s*(?:至|到|为)\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}",
    r"有效期\s*[一二三四五六七八九十百\d]+\s*年",
    r"有效期\s*[一二三四五六七八九十百\d]+\s*个?月",
    r"(?:失效|废止)(?:日期|时间)?\s*(?:为|至|到|：|:)\s*\d{4}",
))


def audit_temporal_fields(result: dict, source_text: str) -> dict:
    """对模型的时间字段做原文证据审计，纠正后可追溯。

    真实数据中出现过：模型看到标题「行动方案（2024—2026年）」便填入
    expiry_date=2026-12-31。这个错误会让时效门禁过滤掉仍应检索的文件，
    因此不能只靠 Prompt 约束，必须在代码里再做一道确定性校验。
    """
    expiry = result.get("expiry_date", "无")
    if expiry and expiry != "无" and not any(p.search(source_text) for p in _EXPIRY_EVIDENCE_RES):
        result.setdefault("_audit", []).append({
            "field": "expiry_date", "old": expiry, "new": "无",
            "reason": "原文没有明确有效期或失效日期；规划期/目标年份不等于失效期",
        })
        result["expiry_date"] = "无"
    return result


# ----------------------------------------------------------------------
def extract_one(path: Path, client: ChatClient | None, out_dir: Path | None, offline: bool) -> dict:
    text = read_policy_text(path)
    if len(text) < 30:
        raise ValueError(f"{path} 内容过短（{len(text)} 字）")

    if offline or client is None:
        result = normalize_result(extract_by_rules(text), path.stem)
        result["_engine"] = "rules"
    else:
        raw = client.chat_json(SYSTEM_PROMPT, build_user_prompt(text, parse_header(text)))
        result = normalize_result(raw, path.stem)
        result["_engine"] = f"llm:{client.model}"

    audit_temporal_fields(result, text)
    result["_meta"] = {
        "source_file": str(path.resolve()),
        "char_count": len(text),
        "extracted_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{path.stem}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
        )
    return result


def collect_paths(targets: list[str]) -> list[Path]:
    """文件或目录都接受。目录优先取 *.txt（正文），没有 txt 时再取 *.json（爬虫/打标产物）。

    只取 txt 是为了避免 demo/data/policies 这类 txt+json 并存的目录被重复处理两遍。
    """
    out: list[Path] = []
    for t in targets:
        p = Path(t)
        if p.is_dir():
            batch = sorted(p.glob("*.txt")) or sorted(p.glob("*.json"))
            out.extend(batch)
        elif p.exists():
            out.append(p)
        else:
            log(f"[WARN] 路径不存在：{t}")
    return out


def main() -> int:
    load_env_file(ROOT / ".env")   # .env 与系统环境变量都支持
    ap = argparse.ArgumentParser(description="阶段二 · 政策文本结构化抽取（DeepSeek）")
    ap.add_argument("paths", nargs="+", help="政策 txt/json 路径，或目录")
    ap.add_argument("--offline", action="store_true", help="不调大模型，用规则抽取（无 Key 时验证链路）")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "extracted"), help="结果输出目录（传 '' 不落盘）")
    ap.add_argument("--api-key", default=None, help="覆盖 DEEPSEEK_API_KEY")
    ap.add_argument("--base-url", default=None, help="覆盖 base_url，默认 https://api.deepseek.com")
    ap.add_argument("--provider", default="deepseek", choices=["deepseek", "qwen", "glm"],
                    help="模型供应商（DeepSeek 限流时可切换）")
    ap.add_argument("--model", default=None, help="覆盖模型名，默认 deepseek-flash（即 DeepSeek-V4.1-Flash）")
    ap.add_argument("--quiet", action="store_true", help="stdout 只输出 JSON")
    args = ap.parse_args()

    files = collect_paths(args.paths)
    if not files:
        log("没有找到可处理的文件")
        return 1

    client = None
    if not args.offline:
        try:
            if args.api_key or args.base_url or args.model:
                client = ChatClient(api_key=args.api_key, base_url=args.base_url, model=args.model)
            else:
                client = provider_client(args.provider)
        except LLMError as e:
            log(f"[ERROR] {e}")
            return 3

    out_dir = Path(args.out_dir) if args.out_dir else None
    results, failed = [], 0

    for i, f in enumerate(files, 1):
        try:
            if not args.quiet:
                log(f"[{i}/{len(files)}] 抽取：{f.name}")
            results.append(extract_one(f, client, out_dir, args.offline))
        except (LLMError, ValueError, OSError) as e:
            failed += 1
            log(f"[FAIL] {f.name}: {e}")

    payload = results[0] if len(results) == 1 else results
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not args.quiet:
        log(f"\n完成 {len(results)} 个，失败 {failed} 个" + (f"；结果已保存到 {out_dir}" if out_dir else ""))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
