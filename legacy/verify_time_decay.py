#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""阶段四自检：评估链路与出图（不需要真实 API Key）

  A. detect_status 判定 + test_set 结构
  B. **本地 LangGraph 后端**：抽取式回答 + 规则裁判，跑满 20 题（无任何外部服务）
  C. 大模型裁判链路：--judge llm 打 Mock，断言 Prompt 含四类标签
  D. --mock 出对比图 + 指标 CSV
  E. --compare 复用已有结果重新出图

运行：python tests/verify_phase4.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mock_servers import openai_server  # noqa: E402

PY = sys.executable
TMP = ROOT / "results" / "_selftest"
INDEX = ROOT / "kb" / "index"


def run(args: list[str], env_extra: dict | None = None) -> tuple[int, str, str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
           "NO_PROXY": "127.0.0.1,localhost", "no_proxy": "127.0.0.1,localhost"}
    env.pop("DEEPSEEK_API_KEY", None)
    env.update(env_extra or {})
    p = subprocess.run([PY, *args], cwd=str(ROOT), capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    return p.returncode, p.stdout, p.stderr


def ok_line(flag: bool, label: str, extra: str = "") -> bool:
    print(f"[{'PASS' if flag else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return flag


def main() -> int:
    ok = True
    TMP.mkdir(parents=True, exist_ok=True)

    # ---------------- A ----------------
    from evaluate import detect_status

    cases = [
        ("根据条例第九条，微型无人机在适飞空域内飞行无需提出飞行计划申请。", "无需审批"),
        ("应当提前 3 个工作日提出飞行计划申请，经批准后方可实施。", "需要审批"),
        ("微型无人机无需申请，但 25 千克以上需要提前申请。", "不确定"),
        ("建议咨询主管部门。", "不确定"),
    ]
    bad = [(t, detect_status(t), e) for t, e in cases if detect_status(t) != e]
    ok &= ok_line(not bad, "A1 detect_status 关键词判定", f"{len(cases) - len(bad)}/{len(cases)} 正确")

    ts = json.loads((ROOT / "test_set.json").read_text(encoding="utf-8"))
    need = {"id", "question", "ground_truth", "expected_status", "answer_type"}
    missing = [i.get("id") for i in ts if not need <= set(i)]
    typed = {i["answer_type"] for i in ts}
    ok &= ok_line(len(ts) == 20 and not missing and len(typed) == 2,
                  "A2 test_set.json 结构", f"{len(ts)} 题，类型 {typed}，缺字段 {missing or '无'}")

    # ---------------- B. 本地 LangGraph 后端（无外部服务） ----------------
    out_dir = TMP / "local"
    code, out, err = run([str(ROOT / "evaluate.py"), "--backend", "local", "--no-llm",
                          "--judge", "rules", "--index-dir", str(INDEX), "--limit", "8",
                          "--sleep", "0", "--out-dir", str(out_dir), "--tag", "selftest_local"])
    res_path = out_dir / "eval_selftest_local.json"
    b1 = code == 0 and res_path.exists()
    ok &= ok_line(b1, "B1 本地知识库评估跑通（LangGraph + 抽取式）", f"exit={code}")
    if b1:
        res = json.loads(res_path.read_text(encoding="utf-8"))
        m = res["metrics"]
        total_pct = m["accuracy"] + m["stale_rate"] + m["refusal_rate"] + m["incorrect_rate"]
        ok &= ok_line(abs(total_pct - 1.0) < 1e-6, "B2 四类占比之和 = 100%", f"{total_pct:.4f}")
        ok &= ok_line(m["total"] == 8 and res["answer_mode"] == "extractive", "B3 指标口径正确",
                      f"N={m['total']} mode={res['answer_mode']} judge={res['judge']}")
        with_route = [i for i in res["items"] if i.get("route")]
        ok &= ok_line(len(with_route) == 8, "B4 每题都记录了图路径（route）",
                      f"{with_route[0]['route'] if with_route else '-'}")
        cited = [i for i in res["items"] if i.get("cited")]
        ok &= ok_line(bool(cited), "B5 回答带可追溯的引用来源", f"{len(cited)} 题有引用")

        # 双口径：有效应答正确率应 ≥ 严格正确率，且分母 = 总题数 - 拒答数
        refusals = m["Refusal"]
        exp_eff = m["Correct"] / max(1, m["total"] - refusals)
        ok &= ok_line(abs(m["effective_accuracy"] - exp_eff) < 1e-3
                      and m["effective_accuracy"] >= m["accuracy"],
                      "B6 双口径自洽（拒答不计入分母）",
                      f"严格 {m['accuracy']:.1%} / 有效 {m['effective_accuracy']:.1%}")

        # 分类统计：各题型题数之和 = 总数，正确数之和 = 总正确数
        bt = m.get("by_type") or {}
        ok &= ok_line(sum(b["n"] for b in bt.values()) == m["total"]
                      and sum(b["correct"] for b in bt.values()) == m["Correct"]
                      and len(bt) == 2,
                      "B7 按题型拆分自洽",
                      " / ".join(f"{k} {b['correct']}/{b['n']}" for k, b in bt.items()))
    else:
        print(out[-1200:], err[-800:])

    # ---------------- C. 大模型裁判链路（Mock） ----------------
    with openai_server(default_answer="根据现行政策，在深圳使用 249 克无人机无需提出飞行计划申请。") as srv:
        cdir = TMP / "judge"
        code, out, err = run([str(ROOT / "evaluate.py"), "--backend", "local", "--no-llm",
                              "--judge", "llm", "--limit", "2", "--sleep", "0",
                              "--api-key", "sk-mock", "--base-url", srv.base_url,
                              "--out-dir", str(cdir), "--tag", "selftest_judge"])
        judge_req = next((r["body"] for r in srv.log if "评估专家" in str(r["body"])), {})
        calls = len(srv.log)

    c1 = code == 0 and (cdir / "eval_selftest_judge.json").exists()
    ok &= ok_line(c1, "C1 接入大模型裁判（Mock）", f"exit={code} 请求 {calls} 次")
    sys_msg = ""
    if judge_req:
        sys_msg = next((m["content"] for m in judge_req.get("messages", []) if m["role"] == "system"), "")
    ok &= ok_line(all(v in sys_msg for v in ("Correct", "Stale_Fact_Error", "Refusal", "Incorrect_Other")),
                  "C2 裁判 Prompt 含全部四类标签")

    # ---------------- C3. 裁判多次投票 ----------------
    with openai_server(default_answer="根据现行政策，在深圳使用 249 克无人机无需提出飞行计划申请。") as srv:
        rdir = TMP / "repeat"
        code, out, err = run([str(ROOT / "evaluate.py"), "--backend", "local", "--no-llm",
                              "--judge", "llm", "--judge-repeat", "3", "--limit", "2", "--sleep", "0",
                              "--api-key", "sk-mock", "--base-url", srv.base_url,
                              "--out-dir", str(rdir), "--tag", "selftest_repeat"])
        judge_calls = len(srv.log)
    rpath = rdir / "eval_selftest_repeat.json"
    if code == 0 and rpath.exists():
        rm = json.loads(rpath.read_text(encoding="utf-8"))["metrics"]
        c3 = judge_calls == 6 and rm.get("judge_agreement") == 1.0
        extra = f"{judge_calls} 次调用，一致性 {rm.get('judge_agreement')}"
    else:
        c3, extra = False, f"exit={code}"
    ok &= ok_line(c3, "C3 裁判 3 次投票取多数", extra)

    # ---------------- D. --mock 出图 ----------------
    fig = TMP / "fig_mock.png"
    code, out, err = run([str(ROOT / "evaluate.py"), "--mock", "--out-dir", str(TMP), "--figure", str(fig)])
    ok &= ok_line(code == 0 and fig.exists() and fig.stat().st_size > 30_000,
                  "D1 --mock 生成对比图（300dpi）",
                  f"{fig.stat().st_size // 1024} KB" if fig.exists() else "缺失")
    ok &= ok_line(fig.with_suffix(".csv").exists(), "D2 指标表 CSV 已生成")

    # ---------------- E. --compare ----------------
    fig2 = TMP / "fig_compare.png"
    code, out, err = run([str(ROOT / "evaluate.py"), "--compare",
                          str(TMP / "eval_before.json"), str(TMP / "eval_after.json"),
                          "--figure", str(fig2)])
    ok &= ok_line(code == 0 and fig2.exists() and fig2.stat().st_size > 30_000,
                  "E1 --compare 复用已有结果重新出图", f"exit={code}")

    print("=" * 60)
    print("阶段四自检：全部通过 ✅" if ok else "阶段四自检：存在失败项 ❌")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
