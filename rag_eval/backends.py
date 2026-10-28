"""可替换的检索后端。

生产路径：Chroma
    collection.query(query_texts=[question], n_results=k)

兼容路径：本地 BM25 索引
    仅用于没有 Chroma / 没有 embedding 服务时跑通与做消融对照。
    它不是向量检索，不能把两边的绝对分数混在一起比较。

新增向量库时，只要实现 RetrievalBackend 的四个方法即可：
    similarity_search / all_clauses / describe / close
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import os
import re
from pathlib import Path
from typing import Any, Optional, Protocol

from .models import Clause
from .temporal import stable_clause_id

# Chroma 的 metadata 不接受 None；这里只做读取侧兼容，不负责写库。
# 如果你的库字段名不同，在 Clause.region / temporal.clause_valid_* 里加别名即可。


class RetrievalBackend(Protocol):
    def similarity_search(self, query: str, k: int = 5,
                          where: Optional[dict] = None) -> list[Clause]:
        ...

    def all_clauses(self, limit: int = 0) -> list[Clause]:
        ...

    def describe(self) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


# ----------------------------------------------------------------------
class _HashingEmbeddingFunction:
    """仅供自动化测试的确定性 embedding。

    生产环境请使用建库时同一个 embedding function；这个类没有语义泛化能力，
    只是让 CI 能在没有模型下载、没有 API Key 的情况下验证 Chroma 适配层。
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    def name(self) -> str:
        return "lowalt_hashing"

    def __call__(self, input):  # noqa: A002 - Chroma 接口参数名就是 input
        out = []
        for text in input:
            vec = [0.0] * self.dim
            s = re.sub(r"\s+", "", text or "")
            grams = [s[i:i + 2] for i in range(max(1, len(s) - 1))]
            for g in grams:
                h = int(hashlib.blake2b(g.encode("utf-8"), digest_size=8).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out

    def embed_documents(self, input):  # Chroma 1.5 写入接口
        return self(input)

    def embed_query(self, input):  # Chroma 1.5 查询接口
        # 单条 query 也要返回二维列表，Chroma 期望 Embeddings
        return self([input]) if isinstance(input, str) else self(input)

    def get_config(self) -> dict:
        return {"dim": self.dim}

    @staticmethod
    def build_from_config(config: dict):
        return _HashingEmbeddingFunction(int((config or {}).get("dim", 256)))


def _resolve_embedding_function(spec: str):
    """解析 --embedding-function。

    auto    : 不传，交给 Chroma 读取 collection 建立时的配置（推荐）
    default : 强制使用 Chroma 内置 ONNX MiniLM（需要首次下载模型）
    hash    : 仅测试用确定性哈希 embedding
    module:attr : 自定义 EmbeddingFunction 工厂，例如 my_ef:build
    """
    spec = (spec or "auto").strip()
    if spec in ("", "auto", "none"):
        return None
    if spec == "hash":
        return _HashingEmbeddingFunction()
    if spec in ("bge", "bge-zh", "bge-small-zh"):
        from .embeddings import BGEEmbeddingFunction
        return BGEEmbeddingFunction()
    if spec == "default":
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
        return DefaultEmbeddingFunction()
    if ":" in spec:
        module_name, attr = spec.split(":", 1)
        fn = getattr(importlib.import_module(module_name), attr)
        return fn() if callable(fn) else fn
    raise ValueError(
        f"未知 embedding-function：{spec}。可用 auto / default / hash / module:attr"
    )


class ChromaBackend:
    """Chroma 适配器（通过原生 chromadb，不依赖 langchain-chroma）。"""

    def __init__(
        self,
        collection: Any,
        *,
        persist_dir: str = "",
        collection_name: str = "",
        embedding_spec: str = "auto",
        client: Any = None,
    ):
        self.collection = collection
        self.persist_dir = persist_dir
        self.collection_name = collection_name or getattr(collection, "name", "")
        self.embedding_spec = embedding_spec
        self.client = client

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ChromaBackend":
        try:
            import chromadb
        except ImportError as e:
            raise RuntimeError(
                "未安装 chromadb。请先执行：pip install chromadb\n"
                "如果只想用当前本地索引跑通，请加 --backend local。"
            ) from e

        embedding = _resolve_embedding_function(getattr(args, "embedding_function", "auto"))
        try:
            if getattr(args, "chroma_host", ""):
                client = chromadb.HttpClient(host=args.chroma_host, port=int(args.chroma_port))
            else:
                client = chromadb.PersistentClient(path=args.chroma_dir)

            kwargs: dict[str, Any] = {"name": args.collection}
            if embedding is not None:
                kwargs["embedding_function"] = embedding
            collection = client.get_collection(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"无法打开 Chroma collection「{args.collection}」（{args.chroma_dir}）：{e}\n"
                "请确认 --chroma-dir / --collection 指向你的真实库；"
                "如果 embedding 不是 Chroma 默认模型，用 --embedding-function module:attr 传入同一个函数。"
            ) from e

        return cls(
            collection,
            persist_dir=args.chroma_dir,
            collection_name=args.collection,
            embedding_spec=getattr(args, "embedding_function", "auto"),
            client=client,
        )

    def similarity_search(self, query: str, k: int = 5,
                          where: Optional[dict] = None) -> list[Clause]:
        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": max(1, k),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        res = self.collection.query(**kwargs)

        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        out: list[Clause] = []
        for i, cid in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            if not text:
                continue
            dist = float(dists[i]) if i < len(dists) and dists[i] is not None else None
            # 只做展示；不同距离度量之间不可比，因此不参与任何过滤
            score = None if dist is None else round(1.0 / (1.0 + max(0.0, dist)), 6)
            out.append(Clause(str(cid), str(text), dict(metas[i] or {}), score))
        return out

    def all_clauses(self, limit: int = 0) -> list[Clause]:
        total = int(self.collection.count())
        page = 500
        out: list[Clause] = []
        offset = 0
        while offset < total:
            res = self.collection.get(
                limit=min(page, total - offset),
                offset=offset,
                include=["documents", "metadatas"],
            )
            ids = res.get("ids") or []
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            for i, cid in enumerate(ids):
                text = docs[i] if i < len(docs) else ""
                if not text:
                    continue
                out.append(Clause(str(cid), str(text), dict(metas[i] or {})))
                if limit and len(out) >= limit:
                    return out
            if not ids:
                break
            offset += len(ids)
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "chroma",
            "collection": self.collection_name,
            "persist_dir": self.persist_dir,
            "embedding_function": self.embedding_spec,
            "count": int(self.collection.count()),
        }

    def close(self) -> None:
        # Chroma 会持有 SQLite/leveldb 文件句柄；Windows 下不关会导致临时目录无法清理
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None


class LocalIndexBackend:
    """本项目已有的 chunks.jsonl + BM25，兼容 effective_date/expiry_date 字段。"""

    def __init__(self, index_dir: str):
        from kb.retriever import PolicyRetriever
        from kb.store import load_index

        self.index_dir = str(index_dir)
        docs, tokens, manifest = load_index(index_dir)
        self.docs = docs
        self.manifest = manifest
        # 关键：Naive RAG 的检索不能偷偷做“新法优先”，否则对照组已经带了时序偏置
        self.retriever = PolicyRetriever(docs, tokens, newer_bias=0.0, enforce_freshness=False)
        self._clauses = [
            Clause(
                stable_clause_id(d.page_content, d.metadata),
                d.page_content,
                dict(d.metadata),
            )
            for d in docs
        ]

    def similarity_search(self, query: str, k: int = 5,
                          where: Optional[dict] = None) -> list[Clause]:
        docs = self.retriever.retrieve(
            query, k=k, region="", enforce_freshness=False, dedupe_by_policy=False
        )
        out: list[Clause] = []
        for d in docs:
            md = dict(d.metadata)
            out.append(Clause(
                stable_clause_id(d.page_content, md),
                d.page_content,
                md,
                float(md.get("_score")) if md.get("_score") is not None else None,
            ))
        return out

    def all_clauses(self, limit: int = 0) -> list[Clause]:
        return self._clauses[:limit] if limit else list(self._clauses)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "local_bm25",
            "index_dir": self.index_dir,
            "chunks": len(self.docs),
            "policies": self.manifest.get("policies"),
            "note": "BM25 兼容后端，不是向量检索；不用于和 Chroma 比较绝对分数",
        }

    def close(self) -> None:
        return


# ----------------------------------------------------------------------
def add_backend_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("检索后端")
    g.add_argument("--backend", choices=["chroma", "local"], default=os.getenv("RAG_BACKEND", "chroma"),
                   help="默认 chroma；没有 Chroma 时可用 local 跑通流程")
    g.add_argument("--chroma-dir", default=os.getenv("CHROMA_DIR", "./chroma_db"),
                   help="Chroma PersistentClient 目录")
    g.add_argument("--chroma-host", default=os.getenv("CHROMA_HOST", ""),
                   help="可选：连接远端 Chroma 服务")
    g.add_argument("--chroma-port", default=os.getenv("CHROMA_PORT", "8000"))
    g.add_argument("--collection", default=os.getenv("CHROMA_COLLECTION", "lowalt_policy"),
                   help="Chroma collection 名称")
    g.add_argument("--embedding-function", default=os.getenv("CHROMA_EMBEDDING_FUNCTION", "auto"),
                   help="auto（读库配置，推荐）/ default / hash（测试）/ module:attr")
    g.add_argument("--index-dir", default=str(Path(__file__).resolve().parent.parent / "kb" / "index"),
                   help="--backend local 时使用的本地索引目录")


def build_backend(args: argparse.Namespace) -> RetrievalBackend:
    if args.backend == "chroma":
        return ChromaBackend.from_args(args)
    return LocalIndexBackend(args.index_dir)
