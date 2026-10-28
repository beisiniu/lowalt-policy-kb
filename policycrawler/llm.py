"""大模型调用封装（兼容 OpenAI 协议的 DeepSeek）。

阶段二（打标）、阶段四（裁判/答题）共用，避免重复造轮子。
只做三件事：**鉴权配置、JSON 强约束、失败重试与 JSON 修复**。
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Optional

DEFAULT_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# ⚠ 模型名会随官方迭代变化，不要硬编码成"版本号"：
#   2026-07-24 起 deepseek-chat / deepseek-reasoner 已废弃；
#   当前对话模型名是 deepseek-flash（实际由 DeepSeek-V4.1-Flash 提供）。
#   可用环境变量 DEEPSEEK_MODEL 覆盖，或在调用时传 --model。
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
MODEL_FALLBACKS = ("deepseek-flash", "deepseek-v4-pro", "deepseek-chat")

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    """从模型输出里抠出 JSON 对象。

    模型即使开了 json mode，也偶尔会裹 ```json 代码块或前后加一句客套话，
    所以先剥代码块，再用「第一个 { 到最后一个 }」兜底。
    """
    if not text:
        raise LLMError("模型返回为空")
    s = _FENCE_RE.sub("", text.strip())
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start : end + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"模型返回不是合法 JSON：{e}\n原始输出前 300 字：{s[:300]}") from e
    raise LLMError(f"模型返回中找不到 JSON 对象：{s[:300]}")



# 三家兼容 OpenAI 协议的供应商。统一成一个 ChatClient，避免每个脚本各写一套。
PROVIDERS: dict[str, dict[str, str]] = {
    "deepseek": {
        "key_env": "DEEPSEEK_API_KEY", "base_env": "DEEPSEEK_BASE_URL",
        "base": "https://api.deepseek.com", "model_env": "DEEPSEEK_MODEL",
        "model": "deepseek-flash", "style": "deepseek",
    },
    "qwen": {
        "key_env": "QWEN_API_KEY", "base_env": "QWEN_BASE_URL",
        "base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model_env": "QWEN_MODEL",
        "model": "qwen3.8-flash", "style": "qwen",
    },
    "glm": {
        "key_env": "GLM_API_KEY", "base_env": "GLM_BASE_URL",
        "base": "https://open.bigmodel.cn/api/paas/v4", "model_env": "GLM_MODEL",
        "model": "glm-5.3-flash", "style": "glm",
    },
}


def provider_client(provider: str, **kwargs: Any) -> "ChatClient":
    """按供应商名构造客户端；未显式传 key/base/model 时从环境变量取。"""
    conf = PROVIDERS.get(provider)
    if not conf:
        raise LLMError(f"未知供应商 {provider}，可选：{', '.join(PROVIDERS)}")
    kwargs.setdefault("api_key", os.getenv(conf["key_env"], ""))
    kwargs.setdefault("base_url", os.getenv(conf["base_env"], conf["base"]))
    kwargs.setdefault("model", os.getenv(conf["model_env"], conf["model"]))
    kwargs["provider"] = provider
    return ChatClient(**kwargs)


class ChatClient:
    @staticmethod
    def _is_model_error(msg: str) -> bool:
        """识别「模型名不存在」类错误。

        官方改名后常见的返回：
          - Model Not Exist / model_not_found
          - The model `deepseek-chat` does not exist or you do not have access to it
        这类错误重试同一个模型名没有意义，必须换名字。
        """
        m = (msg or "").lower()
        return "model" in m and any(k in m for k in ("not exist", "not found", "invalid", "does not exist", "no such", "not support"))

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120,
        max_retries: int = 3,
        provider: str = "",
    ):
        conf = PROVIDERS.get(provider or "deepseek", PROVIDERS["deepseek"])
        self.provider = provider or "deepseek"
        self.style = conf["style"]
        self.api_key = api_key or os.getenv(conf["key_env"], "") or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = base_url or os.getenv(conf["base_env"], conf["base"])
        self.model = model or os.getenv(conf["model_env"], conf["model"])
        # 模型名兜底链：仅 DeepSeek 需要（官方改名/下线时自动降级）
        self._candidates = [self.model] if self.style != "deepseek" else ([self.model] + [m for m in MODEL_FALLBACKS if m != self.model])
        self._cursor = 0
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.api_key:
            raise LLMError(
                "缺少 API Key。请先设置环境变量 DEEPSEEK_API_KEY：\n"
                "  Windows PowerShell :  $env:DEEPSEEK_API_KEY=\"sk-xxxx\"\n"
                "  Windows CMD        :  set DEEPSEEK_API_KEY=sk-xxxx\n"
                "  Linux / macOS      :  export DEEPSEEK_API_KEY=sk-xxxx\n"
                "（也可以在 .env 里写，或用 --api-key 参数临时传入）"
            )
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise LLMError("缺少 openai 库，请执行：pip install -r requirements.txt") from e
        # DeepSeek 完全兼容 OpenAI 协议，只需换 base_url
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        json_mode: bool = True,
        thinking: bool = False,
    ) -> str:
        """单次对话，带指数退避重试（应对 429 / 5xx / 网络抖动）。

        thinking: DeepSeek-V4 默认会先输出推理过程（reasoning_content），
          这会**吃掉 max_tokens 额度**，抽取/判分类任务里纯属浪费且拖慢。
          - 结构化抽取、分类裁决 → False（关掉，稳定且便宜）
          - 需要跨条款推理的回答生成 → True（质量更好）
        """
        last: Optional[Exception] = None
        kwargs: dict[str, Any] = {"extra_body": self._thinking_body(thinking)}
        if self.style == "glm":
            # GLM-5.3-flash 始终思考且 reasoning_effort 压不住（实测 1200 预算
            # 会被思考吃满、正文为空），给足 3000 让正文有空间
            max_tokens = max(max_tokens, 5000)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # openai 的异常层级较深，统一兜住
                last = e
                msg = str(e)

                # 模型名失效（官方改名/下线）：换下一个候选，不当作失败重试
                if self._is_model_error(msg) and self._cursor + 1 < len(self._candidates):
                    old = self.model
                    self._cursor += 1
                    self.model = self._candidates[self._cursor]
                    print(f"[WARN] 模型 {old} 不可用（{msg[:80]}），自动切换到 {self.model}")
                    continue

                retryable = any(k in msg for k in ("429", "500", "502", "503", "504", "timeout", "Timeout", "Connection"))
                if attempt == self.max_retries or not retryable:
                    break
                time.sleep(2 ** (attempt - 1))
        raise LLMError(f"调用大模型失败（{self.model} @ {self.base_url}）：{last}")

    def _thinking_body(self, thinking: bool) -> dict[str, Any]:
        """各家控制思考的字段不同，这里统一成开关。"""
        if self.style == "qwen":
            return {"enable_thinking": bool(thinking)}
        if self.style == "glm":
            return {"thinking": {"type": "enabled", "reasoning_effort": "low" if not thinking else "high"}}
        return {"thinking": {"type": "enabled" if thinking else "disabled"}}

    def chat_json(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 2000,
                  thinking: bool = False) -> dict:
        """要求模型只输出 JSON，解析失败时发起一次「修复」请求。"""
        raw = self.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            thinking=thinking,
        )
        try:
            return extract_json(raw)
        except LLMError:
            fix = self.chat(
                [
                    {"role": "system", "content": "你是 JSON 修复器。只输出修正后的合法 JSON 对象，不要任何解释。"},
                    {"role": "user", "content": f"下面的内容应当是一个 JSON 对象，请修复格式后原样返回字段：\n{raw[:4000]}"},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
                json_mode=True,
            )
            return extract_json(fix)
