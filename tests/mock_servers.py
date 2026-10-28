# -*- coding: utf-8 -*-
"""本地 Mock 服务：模拟 OpenAI 协议（DeepSeek）与 Dify 知识库/对话 API。

用途：**不花钱、不联网**地把 phase2/3/4 的请求链路跑通并断言 payload 结构。
真实上线时把 base_url 换回官方地址即可，业务代码一行不用改。
"""
from __future__ import annotations

import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

Route = tuple[str, str, Callable[[dict, dict], tuple[int, dict]]]  # (method, 正则, fn(body, headers))


class MockServer:
    def __init__(self, routes: list[Route]):
        self.routes = routes
        self.log: list[dict] = []
        self.hits: list[str] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a, **k):
                return

            def _handle(self, method: str) -> None:
                # 某些环境装了 HTTP_PROXY，requests 会给 localhost 也发绝对 URI
                # （GET http://127.0.0.1:port/v1/... HTTP/1.1），这里统一还原成路径
                path = self.path
                if path.startswith(("http://", "https://")):
                    from urllib.parse import urlsplit

                    parts = urlsplit(path)
                    path = parts.path + (f"?{parts.query}" if parts.query else "")
                self.path = path

                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b"{}"
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError:
                    body = {"_raw": raw.decode("utf-8", "replace")}
                headers = dict(self.headers)
                outer.log.append({"method": method, "path": self.path, "body": body, "headers": headers})
                outer.hits.append(f"{method} {self.path}")

                for m, pattern, fn in outer.routes:
                    if m == method and re.fullmatch(pattern, self.path.split("?")[0]):
                        status, payload = fn(body, headers)
                        outer._send(self, status, payload)
                        return
                outer._send(self, 404, {"error": f"mock: no route for {method} {self.path}"})

            def do_POST(self):
                self._handle("POST")

            def do_GET(self):
                self._handle("GET")

        self._handler = Handler
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @staticmethod
    def _send(handler, status: int, payload: dict) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        handler.end_headers()
        handler.wfile.write(data)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "MockServer":
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


# ---------------------------------------------------------------- OpenAI 协议
EXTRACT_JSON = {
    "policy_name": "深圳经济特区低空经济产业促进条例",
    "issue_date": "2026-03-15",
    "effective_date": "2026-05-01",
    "expiry_date": "无",
    "apply_region": "深圳市",
    "status": "现行有效",
    "summary": "深圳市内使用25千克以上无人机飞行须提前3个工作日申请飞行计划并获批，微型轻型在适飞空域内免申请。",
}

JUDGE_JSON = {"verdict": "Correct", "reason": "回答与标准答案一致，引用了现行有效的条例。"}


def openai_route(default_answer: str = "根据现行政策，需要提前申请。",
                 reject_models: tuple[str, ...] = (),
                 first_bad: bool = False) -> Route:
    """模拟 DeepSeek /chat/completions，按 system prompt 分流返回。

    reject_models: 这些模型名会返回 400 Model Not Exist，用来验证模型名自动降级。
    first_bad: 第一次调用故意返回一段散文（不带 JSON），用来验证"JSON 修复"重试路径。
    """
    state = {"calls": 0}

    def fn(body: dict, headers: dict) -> tuple[int, dict]:
        if body.get("model") in reject_models:
            return 400, {"error": {"message": f"Model Not Exist: {body.get('model')}",
                                   "type": "invalid_request_error", "code": "model_not_found"}}
        state["calls"] += 1
        msgs = body.get("messages") or []
        system = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
        user = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")

        if "修复器" in system:  # 客户端发来的 JSON 修复请求
            content = json.dumps(EXTRACT_JSON, ensure_ascii=False)
        elif "信息抽取专家" in system:
            if first_bad and state["calls"] == 1:
                content = "抱歉，我无法直接给出结构化结果，但这份文件讲的是深圳的低空条例。"  # 故意不给 JSON
            else:
                content = "```json\n" + json.dumps(EXTRACT_JSON, ensure_ascii=False) + "\n```"  # 故意加围栏，测试剥离逻辑
        elif "评估专家" in system or "judge" in system.lower():
            content = json.dumps(JUDGE_JSON, ensure_ascii=False)
        else:
            content = default_answer

        return 200, {
            "id": "chatcmpl-mock",
            "object": "chat.completion",
            "created": 1750000000,
            "model": body.get("model", "deepseek-flash"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(system + user), "completion_tokens": len(content), "total_tokens": 0},
        }

    return ("POST", r"/chat/completions", fn)


def openai_server(**kwargs) -> MockServer:
    return MockServer([openai_route(**kwargs)])


# ---------------------------------------------------------------- Dify
DIFY_CREATE_RESPONSE = {
    "batch": "20261026012000001",
    "document": {
        "id": "doc-mock-0001",
        "name": "深圳经济特区低空经济产业促进条例",
        "indexing_status": "indexing",
        "doc_form": "text_model",
        "display_status": "indexing",
        "doc_metadata": [],
    },
}


def dify_server() -> MockServer:
    """模拟 Dify 知识库 5 个接口 + 对话接口。"""

    def create_doc(body, headers):
        assert headers.get("Authorization", "").startswith("Bearer "), "缺少 Bearer 鉴权"
        resp = json.loads(json.dumps(DIFY_CREATE_RESPONSE))
        resp["document"]["name"] = body.get("name", "")
        return 200, resp

    def list_metadata(body, headers):
        return 200, {
            "doc_metadata": [{"id": "meta-status-id", "type": "string", "name": "status", "use_count": 3}],
            "built_in_field_enabled": True,
        }

    def create_metadata(body, headers):
        return 201, {"id": f"meta-{body.get('name')}-id", "name": body.get("name"), "type": body.get("type", "string")}

    def update_doc_metadata(body, headers):
        return 200, {"result": "success"}

    def indexing_status(body, headers):
        return 200, {"data": [{"id": "doc-mock-0001", "indexing_status": "completed", "completed_segments": 12, "total_segments": 12}]}

    def chat_messages(body, headers):
        return 200, {
            "event": "message",
            "message_id": "msg-mock",
            "conversation_id": "conv-mock",
            "mode": "chat",
            "answer": "根据《深圳经济特区低空经济产业促进条例》第九条，微型、轻型无人机在适飞空域内飞行无需申请飞行计划。",
            "metadata": {"retriever_resources": [{"document_name": "深圳经济特区低空经济产业促进条例", "score": 0.91}]},
            "created_at": 1750000000,
        }

    return MockServer(
        [
            ("POST", r"/v1/datasets/[^/]+/document/create-by-text", create_doc),
            ("GET", r"/v1/datasets/[^/]+/metadata", list_metadata),
            ("POST", r"/v1/datasets/[^/]+/metadata", create_metadata),
            ("POST", r"/v1/datasets/[^/]+/documents/metadata", update_doc_metadata),
            ("GET", r"/v1/datasets/[^/]+/documents/[^/]+/indexing-status", indexing_status),
            ("POST", r"/v1/chat-messages", chat_messages),
        ]
    )
