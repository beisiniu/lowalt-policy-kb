"""中文 ONNX embedding：BAAI BGE-small-zh-v1.5。

为什么不用 Chroma 内置模型：
  内置 all-MiniLM-L6-v2 是英文模型。实测中文政策问题「中山市到2028年产业规模」
  的 top-1 会错排到广州文件，检索层不可用。

模型来源：
  ModelScope 的 Xenova/bge-small-zh-v1.5（Transformers.js 转换版，含 ONNX）。
  使用 int8 量化模型（约 24MB），CPU 推理，不依赖 torch / sentence-transformers。

实现细节：
  - BGE 用 CLS pooling，再 L2 归一化；
  - query 加官方中文检索指令，document 不加；
  - 这就是 Chroma 要求的 EmbeddingFunction 接口，可直接存入/查询 collection。
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "bge-small-zh-v1.5"
MODEL_REPO = "Xenova/bge-small-zh-v1.5"
MODEL_BASE = f"https://modelscope.cn/models/{MODEL_REPO}/resolve/master"
MODEL_FILES = {
    "onnx/model_quantized.onnx": 24_010_842,
    "tokenizer.json": 439_125,
    "tokenizer_config.json": 367,
    "special_tokens_map.json": 125,
    "config.json": 776,
}
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："
_DOWNLOAD_LOCK = threading.Lock()


def ensure_model(model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    """下载缺失模型文件；已存在且大小合理则直接用。"""
    import requests

    model_dir.mkdir(parents=True, exist_ok=True)
    with _DOWNLOAD_LOCK:
        for rel, expected in MODEL_FILES.items():
            dst = model_dir / rel
            if dst.exists() and dst.stat().st_size >= int(expected * 0.9):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            url = f"{MODEL_BASE}/{rel}"
            print(f"[model] 下载 {rel} → {dst}")
            tmp = dst.with_suffix(dst.suffix + ".part")
            with requests.get(url, stream=True, timeout=90, verify=False) as r:
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            f.write(chunk)
            if tmp.stat().st_size < 100:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"模型文件下载失败：{url}")
            os.replace(tmp, dst)
    return model_dir


class BGEEmbeddingFunction:
    """Chroma 兼容的中文 embedding function。"""

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, batch_size: int = 32,
                 max_length: int = 512, query_instruction: str = QUERY_INSTRUCTION):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        model_dir = ensure_model(Path(model_dir))
        self.model_dir = str(model_dir)
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_instruction = query_instruction

        tok_path = model_dir / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.tokenizer.enable_truncation(max_length=max_length)
        pad_id = 0
        try:
            cfg = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
            pad_token = cfg.get("pad_token") or "[PAD]"
            pad_id = int(cfg.get("pad_token_id") or 0)
        except Exception:
            pad_token = "[PAD]"
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token=pad_token)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(4, (os.cpu_count() or 4)))
        self.session = ort.InferenceSession(
            str(model_dir / "onnx" / "model_quantized.onnx"), sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {x.name for x in self.session.get_inputs()}
        self.output_names = [x.name for x in self.session.get_outputs()]

    def name(self) -> str:
        return "bge_small_zh_v1_5"

    def get_config(self) -> dict:
        return {"model_dir": self.model_dir, "max_length": self.max_length}

    @staticmethod
    def build_from_config(config: dict):
        return BGEEmbeddingFunction(config.get("model_dir") or DEFAULT_MODEL_DIR,
                                    max_length=int(config.get("max_length") or 512))

    def _embed(self, texts: Iterable[str]) -> list[list[float]]:
        texts = [str(t or "") for t in texts]
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encs = self.tokenizer.encode_batch(batch)
            input_ids = np.asarray([e.ids for e in encs], dtype=np.int64)
            attention = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
            feeds: dict[str, Any] = {"input_ids": input_ids, "attention_mask": attention}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.asarray([e.type_ids for e in encs], dtype=np.int64)
            outputs = self.session.run(self.output_names, feeds)
            named = dict(zip(self.output_names, outputs))
            # Transformers.js 导出的 BGE 通常直接输出 last_hidden_state；兼容 sentence_embedding
            if "sentence_embedding" in named:
                vec = np.asarray(named["sentence_embedding"], dtype=np.float32)
            else:
                hidden = np.asarray(
                    named.get("last_hidden_state", outputs[0]), dtype=np.float32
                )
                vec = hidden[:, 0, :]   # CLS pooling
            norm = np.linalg.norm(vec, axis=1, keepdims=True)
            vec = vec / np.clip(norm, 1e-12, None)
            out.extend(vec.astype(np.float32).tolist())
        return out

    def embed_documents(self, input):  # noqa: A002
        return self._embed(input)

    def embed_query(self, input):  # noqa: A002
        if isinstance(input, str):
            return self._embed([self.query_instruction + input])
        return self._embed([self.query_instruction + str(x) for x in input])

    def __call__(self, input):  # noqa: A002
        return self.embed_documents(input)
