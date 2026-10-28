#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""时序对抗评估的自动化验收。

覆盖：
  T1 valid_to 语义（过期 / 未来到期 / null）
  T2 新旧条款配对
  T3 LLM 出题结果的反泄漏校验
  T4 Temporal_Check 真过滤，且过滤节点拿不到 benchmark 标签
  T5 指标计算口径（严格拯救率、毒化消除率）
  T6 [集成] Chroma + build_benchmark.py + evaluate.py + Mock LLM 全链路

运行：python tests/verify_temporal_benchmark.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_benchmark import build_candidate_pairs, generate_item          # noqa: E402
from evaluate import compute_metrics                                      # noqa: E402
from mock_servers import MockServer                                       # noqa: E402
from rag_eval.models import Clause                                        # noqa: E402
from rag_eval.temporal import clause_matches, parse_date, usable          # noqa: E402
from rag_eval.temporal_graph import TemporalRAGGraph                      # noqa: E402

PY = sys.executable
AS_OF = date(2026, 10, 26)


def ok(flag: bool, label: str, extra: str = "") -> bool:
    print(f"[{'PASS' if flag else 'FAIL'}] {label}" + (f"  {extra}" if extra else ""))
    return flag


def fixture_clauses() -> list[Clause]:
    raw = json.loads((ROOT / "demo" / "temporal_fixture.json").read_text(encoding="utf-8"))
    return [Clause(x["id"], x["text"], x["metadata"]) for x in raw]


def fake_judge_record(n: str, a: str) -> dict:
    return {
        "old_clause_penetrated": True,
        "naive_judge": {"status": n},
        "agent_judge": {"status": a},
    }


def test_t1() -> bool:
    old, _, new = fixture_clauses()
    # 状态没写废止，仅靠 valid_to 也必须在时间上被拦
    date_only = Clause("date_only", "旧条款", {
        "valid_from": "2025-01-01", "valid_to": "2026-08-31", "status": "现行有效",
    })
    future = Clause("future", "条款", {"valid_from": "2026-01-01", "valid_to": "2099-01-01", "status": "现行有效"})
    old_ok, _ = usable(old, AS_OF)
    date_ok, date_why = usable(date_only, AS_OF)
    new_ok, _ = usable(new, AS_OF)
    future_ok, _ = usable(future, AS_OF)
    parsed = parse_date("2026年8月31日")
    flag = (
        (not old_ok) and (not date_ok and "有效期" in date_why)
        and new_ok and future_ok and parsed == date(2026, 8, 31)
    )
    return ok(flag, "T1 valid_to 语义：过期被拦、未来到期可用、null 长期有效",
              f"date_only={date_why or '可用'} future={future_ok} date={parsed}")


def test_t2() -> bool:
    pairs, stats = build_candidate_pairs(fixture_clauses(), AS_OF, min_similarity=0.0,
                                         max_pairs_per_topic=5)
    flag = len(pairs) == 2 and all(p.region == "深圳市" and p.topic == "飞行计划审批" for p in pairs)
    return ok(flag, "T2 同一 region/topic 下 2 条旧条款配 1 条现行条款", f"{len(pairs)} 对")


class FakeJsonClient:
    model = "fake-json"

    def __init__(self, payload: dict):
        self.payload = payload
        self.calls = 0

    def chat_json(self, system, user, **kwargs):
        self.calls += 1
        return self.payload


class FakeChat:
    model = "fake-chat"

    def __init__(self, answer: str):
        self.answer = answer
        self.prompts: list[str] = []

    def chat(self, messages, **kwargs):
        self.prompts.append("\n".join(m.get("content", "") for m in messages))
        return self.answer


def test_t3() -> bool:
    pair = build_candidate_pairs(fixture_clauses(), AS_OF, min_similarity=0.0)[0][0]
    good = {
        "has_substantive_difference": True,
        "conflict_type": "approval_removed",
        "question": "现在在深圳适飞空域飞249克无人机，还要提前5个工作日申请吗？",
        "ground_truth": "无需提出飞行计划申请，完成实名登记和飞行前信息报备即可。",
        "reasoning": "微型轻型由申请改为免申请",
    }
    item = generate_item(FakeJsonClient(good), pair, AS_OF, index=1)
    bad_payload = dict(good, question="根据旧政策，现在还要提前5个工作日申请吗？")
    bad = generate_item(FakeJsonClient(bad_payload), pair, AS_OF, index=1)
    false_payload = {"has_substantive_difference": False}
    none_item = generate_item(FakeJsonClient(false_payload), pair, AS_OF, index=1)
    flag = bool(item) and bad is None and none_item is None and "5个工作日" in item["question"]
    return ok(flag, "T3 出题成功，且拒绝“旧政策”泄漏与无实质差异项")


def test_t4() -> bool:
    old, old2, new = fixture_clauses()
    fake = FakeChat("根据现行办法，249克无人机无需提出飞行计划申请。")
    out = TemporalRAGGraph(fake).ask(
        "现在在深圳飞249克无人机需要申请吗？", [old, new], AS_OF
    )
    prompt = fake.prompts[0] if fake.prompts else ""
    filtered_ok = len(out["filtered_docs"]) == 1 and len(out["dropped"]) == 1
    no_leak = (
        "无论机型大小" not in prompt
        and "demo-old-2024" not in prompt
        and "old_clause_id" not in prompt
    )

    # 全是旧条款：不需要调用 LLM，必须走安全拒答
    refuse = TemporalRAGGraph(None).ask("现在还要提前5个工作日吗？", [old, old2], AS_OF)
    flag = filtered_ok and no_leak and refuse["route"] == "refuse" and "均已失效" in refuse["answer"]
    return ok(flag, "T4 Temporal_Check 过滤旧条款；旧条款全文/ID 不进入生成上下文",
              f"keep={len(out['filtered_docs'])} drop={len(out['dropped'])} refuse={refuse['route']}")


def test_t5() -> bool:
    records = [
        fake_judge_record("POISONED", "CORRECT"),
        fake_judge_record("POISONED", "REFUSED"),
        fake_judge_record("CORRECT", "CORRECT"),
        fake_judge_record("OTHER_ERROR", "POISONED"),
    ]
    m = compute_metrics(records)
    flag = (
        m["penetration_rate"] == 1.0
        and m["naive_poison_rate"] == 0.5
        and m["agent_poison_rate"] == 0.25
        and m["rescue_rate_strict"] == 0.5
        and m["de_poison_rate"] == 1.0
    )
    return ok(flag, "T5 指标口径：严格拯救率只算答对，毒化消除含安全拒答",
              f"naive={m['naive_poison_rate']} agent={m['agent_poison_rate']} "
              f"rescue={m['rescue_rate_strict']} de-poison={m['de_poison_rate']}")


# ---------------------------------------------------------------- T6
BENCH_MOCK = {
    "has_substantive_difference": True,
    "conflict_type": "approval_removed",
    "question": "现在在深圳适飞空域飞249克无人机，还要提前5个工作日申请飞行计划吗？",
    "ground_truth": "无需提出飞行计划申请，实行实名登记与飞行前信息报备；小型及以上才需提前1个工作日申请。",
    "reasoning": "旧办法要求所有机型提前5日，新办法豁免微型轻型",
}


def temporal_mock_server() -> MockServer:
    """一个只服务本测试的 OpenAI 协议 Mock。

    - 出题系统提示 → BENCH_MOCK
    - 回答系统提示 → 上下文含旧条款则输出旧答案，否则输出新答案
    - 裁判系统提示 → 答案含“5个工作日”判 POISONED，否则 CORRECT
    """

    def route(body, headers):
        msgs = body.get("messages") or []
        system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
        user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")

        if "对抗测试集设计专家" in system:
            content = json.dumps(BENCH_MOCK, ensure_ascii=False)
        elif "质量审核员" in system:
            content = json.dumps({
                "valid": True, "same_specific_issue": True,
                "gt_supported_by_new": True, "reason": "同一补贴事项，新条款直接给出标准",
            }, ensure_ascii=False)
        elif "时序 RAG 的评估专家" in system:
            # 只检查【模型回答】段；旧条款正文本身也含关键词，不能整段搜
            answer_part = user.rsplit("【模型回答】", 1)[-1]
            poisoned = "5个工作日" in answer_part
            content = json.dumps({
                "status": "POISONED" if poisoned else "CORRECT",
                "uses_old_clause": poisoned,
                "reason": "沿用了旧办法5个工作日门槛" if poisoned else "符合新办法豁免与1个工作日规定",
            }, ensure_ascii=False)
        elif "低空政策问答助手" in system:
            if "无论机型大小" in user:
                content = "根据《深圳市低空飞行活动管理暂行办法（2024年）》第六条，无论机型大小均应提前5个工作日申请。"
            else:
                content = "根据《深圳市低空飞行活动管理办法（2026年修订）》第五条，249克微型无人机无需提出飞行计划申请，完成实名登记与飞行前信息报备即可。"
        else:
            content = "OK"
        return 200, {
            "id": "chatcmpl-temporal-mock",
            "object": "chat.completion",
            "created": 1750000000,
            "model": body.get("model", "deepseek-flash"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    return MockServer([("POST", r"/chat/completions", route)])


def make_chroma_fixture(path: Path, collection: str = "temporal_test") -> None:
    import chromadb
    from rag_eval.backends import _HashingEmbeddingFunction

    ef = _HashingEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(path))
    try:
        col = client.get_or_create_collection(
            name=collection,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        raw = json.loads((ROOT / "demo" / "temporal_fixture.json").read_text(encoding="utf-8"))
        metadatas = []
        for x in raw:
            md = {k: v for k, v in x["metadata"].items() if v is not None}
            metadatas.append(md)
        col.add(ids=[x["id"] for x in raw], documents=[x["text"] for x in raw], metadatas=metadatas)
    finally:
        # Windows 必须显式释放文件句柄，否则 TemporaryDirectory 清理会 WinError 32
        client.close()


def run_cli(args: list[str], base_url: str) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    env.pop("DEEPSEEK_API_KEY", None)
    return subprocess.run(
        [PY, *args], cwd=str(ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace", env=env,
    )


def test_t6() -> bool:
    try:
        import chromadb  # noqa: F401
    except ImportError:
        return ok(False, "T6 Chroma 全链路（未安装 chromadb，先 pip install -r requirements.txt）")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        chroma_dir = tmp / "chroma"
        make_chroma_fixture(chroma_dir)
        bench = tmp / "benchmark.json"
        out = tmp / "eval.json"
        fig = tmp / "eval.png"

        with temporal_mock_server() as srv:
            b = run_cli([
                "build_benchmark.py",
                "--backend", "chroma", "--chroma-dir", str(chroma_dir),
                "--collection", "temporal_test", "--embedding-function", "hash",
                "--as-of", AS_OF.isoformat(), "--max-pairs", "1", "--max-items", "1",
                "--min-similarity", "0", "--require-old-in-topk", "5",
                "--output", str(bench),
                "--api-key", "sk-mock", "--base-url", srv.base_url, "--sleep", "0",
            ], srv.base_url)
            e = run_cli([
                "evaluate.py",
                "--backend", "chroma", "--chroma-dir", str(chroma_dir),
                "--collection", "temporal_test", "--embedding-function", "hash",
                "--benchmark", str(bench), "--as-of", AS_OF.isoformat(),
                "--top-k", "5", "--out", str(out), "--figure", str(fig),
                "--api-key", "sk-mock", "--base-url", srv.base_url, "--sleep", "0",
            ], srv.base_url)
            calls_before_resume = len(srv.log)
            rr = run_cli([
                "evaluate.py",
                "--backend", "chroma", "--chroma-dir", str(chroma_dir),
                "--collection", "temporal_test", "--embedding-function", "hash",
                "--benchmark", str(bench), "--as-of", AS_OF.isoformat(),
                "--top-k", "5", "--out", str(out), "--figure", str(fig),
                "--api-key", "sk-mock", "--base-url", srv.base_url, "--sleep", "0",
                "--resume",
            ], srv.base_url)
            calls_after_resume = len(srv.log)

        if b.returncode != 0 or e.returncode != 0 or rr.returncode != 0:
            print("  build stderr:", b.stderr[-700:])
            print("  eval  stderr:", e.stderr[-700:])
            print("  resume stderr:", rr.stderr[-500:])
            return ok(False, "T6 Chroma 全链路", f"build={b.returncode} eval={e.returncode} resume={rr.returncode}")

        items = json.loads(bench.read_text(encoding="utf-8"))
        result = json.loads(out.read_text(encoding="utf-8"))
        m = result["metrics"]
        rec = result["items"][0]
        flag = (
            len(items) == 1
            and m["penetration_rate"] == 1.0
            and m["new_clause_penetration_rate"] == 1.0
            and m["naive_poison_rate"] == 1.0
            and m["agent_poison_rate"] == 0.0
            and m["rescue_rate_strict"] == 1.0
            and rec["old_clause_rank"] is not None
            and rec["naive_judge"]["status"] == "POISONED"
            and rec["agent_judge"]["status"] == "CORRECT"
            and rec["old_clause_filtered"] is True
            and calls_after_resume == calls_before_resume  # --resume 不得重复调模型
            and fig.exists() and fig.stat().st_size > 10_000
        )
        return ok(flag, "T6 Chroma 全链路 + 断点续跑：出题→同一Top5→双分支→裁判→图",
                  f"penetration={m['penetration_rate']} naive={m['naive_poison_rate']} "
                  f"agent={m['agent_poison_rate']} resume_calls={calls_after_resume - calls_before_resume} "
                  f"fig={fig.stat().st_size if fig.exists() else 0}B")


def main() -> int:
    results = [test_t1(), test_t2(), test_t3(), test_t4(), test_t5(), test_t6()]
    print("=" * 64)
    print("时序对抗评估自检：全部通过 ✅" if all(results) else "时序对抗评估自检：存在失败项 ❌")
    print("=" * 64)
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
