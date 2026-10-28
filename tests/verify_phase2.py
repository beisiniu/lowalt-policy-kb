#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段二自检：不花一分钱、不联网，验证抽取链路。

  A. --offline 规则模式：3 份样例政策的 7 个字段是否齐全、日期格式是否合规
  B. 大模型模式（打本地 Mock）：请求体是否带 response_format=json_object、
     model 是否为当前官方模型名 deepseek-flash，模型输出裹了 ```json 围栏能否正确剥离

运行：python tests/verify_phase2.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_servers import EXTRACT_JSON, openai_server  # noqa: E402
from extract_policy import audit_temporal_fields      # noqa: E402

FIELDS = ("policy_name", "issue_date", "effective_date", "expiry_date", "apply_region", "status", "summary")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
POLICY_DIR = ROOT / "demo" / "data" / "policies"


def run(args: list[str], env_extra: dict | None = None) -> tuple[int, str, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost", **(env_extra or {})}
    env.pop("DEEPSEEK_API_KEY", None)  # 保证不误用真实 Key
    p = subprocess.run([sys.executable, *args], cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    return p.returncode, p.stdout, p.stderr


def check_schema(obj: dict) -> list[str]:
    errs = []
    for f in FIELDS:
        if f not in obj:
            errs.append(f"缺少字段 {f}")
    for f in ("issue_date", "effective_date", "expiry_date"):
        v = str(obj.get(f, ""))
        if v != "无" and not DATE_RE.fullmatch(v):
            errs.append(f"{f}={v!r} 不是 YYYY-MM-DD 或 '无'")
    if obj.get("status") not in ("现行有效", "已废止"):
        errs.append(f"status={obj.get('status')!r} 不合法")
    if not obj.get("policy_name"):
        errs.append("policy_name 为空")
    if not obj.get("apply_region"):
        errs.append("apply_region 为空")
    return errs


def main() -> int:
    ok = True

    # ---------------- A. 规则模式 ----------------
    code, out, err = run(["extract_policy.py", str(POLICY_DIR), "--offline", "--out-dir", ""])
    try:
        payload = json.loads(out)
    except json.JSONDecodeError:
        print("[FAIL] A 规则模式：stdout 不是合法 JSON\n", out[:500], err[:500])
        return 1

    print(f"[{'PASS' if code == 0 and len(payload) == 4 else 'FAIL'}] A1 规则模式处理 4 个文件（exit={code}, 返回 {len(payload)} 条）")
    ok &= code == 0 and len(payload) == 4

    for item in payload:
        errs = check_schema(item)
        ok &= not errs
        print(f"[{'PASS' if not errs else 'FAIL'}] A2 schema 校验：{item['policy_name']}"
              + (f"  {errs}" if errs else f"  ({item['issue_date']} / {item['effective_date']} / {item['apply_region']})"))

    # ---------------- B. 大模型模式（Mock） ----------------
    with openai_server() as srv:
        with tempfile.TemporaryDirectory() as td:
            code, out, err = run([
                "extract_policy.py", str(POLICY_DIR / "2026-03-15_深圳经济特区低空经济产业促进条例.txt"),
                "--api-key", "sk-mock", "--base-url", srv.base_url, "--out-dir", td,
            ])
            saved = list(Path(td).glob("*.json"))
            saved_text = saved[0].read_text(encoding="utf-8") if saved else ""
            req = srv.log[0]["body"] if srv.log else {}

    good = code == 0 and len(saved) == 1
    print(f"[{'PASS' if good else 'FAIL'}] B1 调用 Mock 大模型成功（exit={code}，落盘 {len(saved)} 个 json）")
    ok &= good
    if not good:
        print(err[:800])

    body_ok = (req.get("model") == "deepseek-flash"
               and req.get("response_format") == {"type": "json_object"}
               and req.get("temperature") == 0.0)
    print(f"[{'PASS' if body_ok else 'FAIL'}] B2 请求体合规：model={req.get('model')}, "
          f"temperature={req.get('temperature')}, response_format={req.get('response_format')}")
    ok &= body_ok

    system = next((m["content"] for m in req.get("messages", []) if m["role"] == "system"), "")
    prompt_ok = "信息抽取专家" in system and all(f in system for f in FIELDS)
    print(f"[{'PASS' if prompt_ok else 'FAIL'}] B3 System Prompt 含角色设定与全部 7 个字段名")
    ok &= prompt_ok

    if saved:
        got = json.loads(saved_text)
        fence_ok = got["policy_name"] == EXTRACT_JSON["policy_name"] and got["status"] == EXTRACT_JSON["status"]
        print(f"[{'PASS' if fence_ok else 'FAIL'}] B4 剥掉 ```json 围栏并正确入库：{got['policy_name']} / {got['status']}")
        ok &= fence_ok

    # ---------------- B5. 模型名失效自动降级 ----------------
    # 官方 2026-07-24 起废弃 deepseek-chat，这里模拟"用户仍写了旧名字"的场景
    with openai_server(reject_models=("deepseek-chat",)) as srv:
        code, out, err = run([
            "extract_policy.py", str(POLICY_DIR / "2026-03-15_深圳经济特区低空经济产业促进条例.txt"),
            "--api-key", "sk-mock", "--base-url", srv.base_url, "--model", "deepseek-chat",
            "--out-dir", "", "--quiet",
        ])
        tried = [r["body"].get("model") for r in srv.log]
    b5 = code == 0 and tried[:2] == ["deepseek-chat", "deepseek-flash"]
    print(f"[{'PASS' if b5 else 'FAIL'}] B5 旧模型名失效时自动降级  →  尝试顺序 {tried}")
    ok &= b5

    # ---------------- B6. 模型返回散文 → 触发 JSON 修复 ----------------
    with openai_server(first_bad=True) as srv:
        code, out, err = run([
            "extract_policy.py", str(POLICY_DIR / "2026-03-15_深圳经济特区低空经济产业促进条例.txt"),
            "--api-key", "sk-mock", "--base-url", srv.base_url, "--out-dir", "", "--quiet",
        ])
        calls = len(srv.log)
    try:
        fixed = json.loads(out)
        payload_ok = fixed.get("policy_name") == EXTRACT_JSON["policy_name"]
    except json.JSONDecodeError:
        payload_ok = False
    b6 = code == 0 and calls == 2 and payload_ok
    print(f"[{'PASS' if b6 else 'FAIL'}] B6 模型不给 JSON 时自动修复（{calls} 次调用）")
    ok &= b6

    # ---------------- B7. 目录里只有 .json 也能处理 ----------------
    with tempfile.TemporaryDirectory() as td:
        only_json = Path(td) / "only_json"
        only_json.mkdir()
        for p in list(POLICY_DIR.glob("*.json"))[:1]:
            (only_json / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        code, out, err = run(["extract_policy.py", str(only_json), "--offline", "--out-dir", ""])
    b7 = code == 0 and '"policy_name"' in out
    print(f"[{'PASS' if b7 else 'FAIL'}] B7 只给 json 目录也能抽取（走 _meta.source_file 回跳正文）")
    ok &= b7

    # ---------------- B8. 规划期不能被编造为失效期 ----------------
    guessed = {"expiry_date": "2026-12-31"}
    audit_temporal_fields(guessed, "某市低空经济高质量发展行动方案（2024—2026年）\n到2026年产业规模突破100亿元。")
    explicit = {"expiry_date": "2026-12-31"}
    audit_temporal_fields(explicit, "本措施自印发之日起实施，有效期至2026年12月31日。")
    b8 = guessed["expiry_date"] == "无" and bool(guessed.get("_audit")) and explicit["expiry_date"] == "2026-12-31"
    print(f"[{'PASS' if b8 else 'FAIL'}] B8 失效日期原文审计：规划期纠正，明确有效期保留")
    ok &= b8

    print("=" * 60)
    print("阶段二自检：全部通过 ✅" if ok else "阶段二自检：存在失败项 ❌")
    print("=" * 60)
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
