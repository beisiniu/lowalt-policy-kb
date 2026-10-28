"""Dify 知识库 / 对话 API 客户端（对齐 2026 年官方文档）。

接口清单（Base URL 已含 /v1）：
    GET  /datasets                                   列出知识库（找 dataset_id 用）
    POST /datasets/{id}/document/create-by-text      通过文本创建文档
    GET  /datasets/{id}/metadata                     列出元数据字段定义
    POST /datasets/{id}/metadata                     新建元数据字段 {name, type}
    POST /datasets/{id}/documents/metadata           批量写入文档元数据值
    GET  /datasets/{id}/documents/{doc}/indexing-status  查询分段索引进度
    POST /chat-messages                              创建对话消息（阶段四）

⚠ 重要：`doc_metadata` 只是 create-by-text **响应里**的只读字段，
   创建文档时**不能**直接传值，必须再调一次 documents/metadata 写入。
   （老版本接口曾用 create_by_text 下划线风格，这里保留了 404 回退。）
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import requests

DEFAULT_BASE = os.getenv("DIFY_BASE_URL", "https://api.dify.ai/v1")


class DifyError(RuntimeError):
    pass


class DifyClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, timeout: float = 60):
        self.api_key = api_key or os.getenv("DIFY_API_KEY", "")
        self.base_url = (base_url or DEFAULT_BASE).rstrip("/")
        self.timeout = timeout
        if not self.api_key:
            raise DifyError(
                "缺少 Dify API Key。请在环境变量里设置 DIFY_API_KEY：\n"
                "  知识库 → 左侧「API 访问」/ 右上角「Service API」→ 创建密钥\n"
                "  Windows PowerShell :  $env:DIFY_API_KEY=\"dataset-xxxx\""
            )
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})

    # ---------------- 底层请求 ----------------
    def _request(self, method: str, path: str, payload: dict | None = None, retries: int = 3) -> dict:
        url = f"{self.base_url}{path}"
        last: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                resp = self.session.request(method, url, json=payload, timeout=self.timeout)
            except requests.RequestException as e:
                last = e
                if attempt == retries:
                    break
                time.sleep(2 ** (attempt - 1))
                continue

            if resp.status_code in (200, 201):
                try:
                    return resp.json()
                except json.JSONDecodeError:
                    return {"raw": resp.text}

            if resp.status_code == 401:
                raise DifyError("鉴权失败(401)：API Key 无效，或该 Key 没有这个知识库的权限。")
            if resp.status_code == 404:
                raise DifyError(f"接口不存在(404)：{method} {url}\n"
                                f"请检查 DIFY_BASE_URL 是否包含 /v1，以及 dataset_id 是否正确。\n{resp.text[:300]}")
            if resp.status_code == 429 or resp.status_code >= 500:
                last = DifyError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                if attempt == retries:
                    break
                time.sleep(2 ** (attempt - 1))
                continue
            raise DifyError(f"请求失败 {resp.status_code} {method} {url}\n{resp.text[:500]}")

        raise DifyError(f"请求重试耗尽：{method} {url} → {last}")

    # ---------------- 知识库 ----------------
    def list_datasets(self, page: int = 1, limit: int = 100) -> list[dict]:
        data = self._request("GET", f"/datasets?page={page}&limit={limit}")
        return data.get("data", [])

    def find_dataset_id(self, name_keyword: str) -> Optional[str]:
        for ds in self.list_datasets():
            if name_keyword in (ds.get("name") or ""):
                return ds.get("id")
        return None

    def create_document_by_text(
        self,
        dataset_id: str,
        name: str,
        text: str,
        indexing_technique: str = "high_quality",
        process_rule: Optional[dict] = None,
        doc_form: str = "text_model",
        doc_language: str = "Chinese",
    ) -> dict:
        payload = {
            "name": name,
            "text": text,
            "indexing_technique": indexing_technique,
            "doc_form": doc_form,
            "doc_language": doc_language,
            "process_rule": process_rule or {"mode": "automatic"},
        }
        try:
            return self._request("POST", f"/datasets/{dataset_id}/document/create-by-text", payload)
        except DifyError as e:
            # 兼容 0.x 老版本的 create_by_text 下划线命名
            if "404" not in str(e):
                raise
            return self._request("POST", f"/datasets/{dataset_id}/document/create_by_text", payload)

    def get_indexing_status(self, dataset_id: str, document_id: str) -> dict:
        return self._request("GET", f"/datasets/{dataset_id}/documents/{document_id}/indexing-status")

    def wait_for_indexing(self, dataset_id: str, document_id: str, timeout: float = 120) -> str:
        """轮询直到 completed / error，返回最终状态。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            data = self.get_indexing_status(dataset_id, document_id)
            items = data.get("data") or [data]
            status = (items[0] or {}).get("indexing_status", "")
            if status in ("completed", "error"):
                return status
            time.sleep(2)
        return "timeout"

    # ---------------- 元数据 ----------------
    def list_metadata_fields(self, dataset_id: str) -> dict[str, dict]:
        """返回 {字段名: {id, name, type}}，用于 value 写入时带 id。"""
        data = self._request("GET", f"/datasets/{dataset_id}/metadata")
        return {m["name"]: m for m in data.get("doc_metadata", [])}

    def create_metadata_field(self, dataset_id: str, name: str, type_: str = "string") -> dict:
        return self._request("POST", f"/datasets/{dataset_id}/metadata", {"name": name, "type": type_})

    def ensure_metadata_fields(self, dataset_id: str, names: list[str], type_: str = "string") -> dict[str, dict]:
        existing = self.list_metadata_fields(dataset_id)
        for n in names:
            if n not in existing:
                try:
                    created = self.create_metadata_field(dataset_id, n, type_)
                    existing[n] = created
                except DifyError as e:
                    print(f"  [WARN] 创建元数据字段 {n} 失败：{e}")
        return existing

    def update_document_metadata(
        self, dataset_id: str, document_id: str, values: dict[str, str],
        fields: dict[str, dict], partial: bool = True,
    ) -> dict:
        metadata_list = [
            {"id": fields[name]["id"], "name": name, "value": value}
            for name, value in values.items()
            if name in fields and value not in (None, "")
        ]
        payload = {
            "operation_data": [
                {"document_id": document_id, "metadata_list": metadata_list, "partial_update": partial}
            ]
        }
        return self._request("POST", f"/datasets/{dataset_id}/documents/metadata", payload)

    # ---------------- 对话（阶段四用） ----------------
    def chat_message(self, query: str, user: str = "eval-bot", inputs: dict | None = None,
                     conversation_id: str = "", timeout: float = 120) -> dict:
        payload = {
            "query": query,
            "inputs": inputs or {},
            "response_mode": "blocking",
            "conversation_id": conversation_id,
            "user": user,
            "auto_generate_name": False,
        }
        return self._request("POST", "/chat-messages", payload, retries=2)


def pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
