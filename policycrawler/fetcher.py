"""HTTP 抓取层：UA 伪装、超时、重试退避、编码自适应、限速。"""
from __future__ import annotations

import random
import re
import time
from typing import Optional

import requests
from requests import exceptions as rq_exc
from requests.adapters import HTTPAdapter

_META_CHARSET_RE = re.compile(
    rb"""<meta[^>]+charset\s*=\s*["']?\s*([a-zA-Z0-9_\-]+)""", re.IGNORECASE
)


class FetchError(RuntimeError):
    """抓取失败（重试耗尽）。"""


def decode_html(resp: requests.Response, forced: str = "") -> str:
    """政府站点编码经常是 gb2312/gbk 且 header 不声明，直接 resp.text 会乱码。

    优先级：配置强制 → HTTP header 声明 → HTML meta 声明 → charset_normalizer 猜测。
    """
    if forced:
        return resp.content.decode(forced, errors="replace")

    enc = (resp.encoding or "").lower()
    if enc and enc not in ("iso-8859-1", "ascii"):
        try:
            return resp.content.decode(enc, errors="replace")
        except (LookupError, UnicodeDecodeError):
            pass

    m = _META_CHARSET_RE.search(resp.content[:4096])
    if m:
        try:
            return resp.content.decode(m.group(1).decode("ascii", "ignore"), errors="replace")
        except (LookupError, UnicodeDecodeError):
            pass

    guessed = resp.apparent_encoding or "utf-8"
    return resp.content.decode(guessed, errors="replace")


class Fetcher:
    def __init__(
        self,
        timeout: float = 15,
        retries: int = 3,
        backoff: float = 2.0,
        delay: float = 1.0,
        user_agents: Optional[list[str]] = None,
        charset: str = "",
        verify_ssl: bool = True,
    ):
        self.timeout = timeout
        self.retries = max(1, retries)
        self.backoff = backoff
        self.delay = delay
        self.charset = charset
        self.verify_ssl = verify_ssl
        self.user_agents = user_agents or ["Mozilla/5.0 (compatible; LowAltPolicyBot/1.0)"]
        self._last_request_at = 0.0

        self.session = requests.Session()
        # 连接池复用，减少 TLS 握手；对同一站点批量抓取明显更快
        adapter = HTTPAdapter(pool_connections=8, pool_maxsize=8)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _headers(self, referer: str = "") -> dict:
        headers = {
            "User-Agent": random.choice(self.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
        }
        if referer:
            headers["Referer"] = referer
        return headers

    def _sleep_politely(self) -> None:
        """对目标站点限速：两次请求之间至少间隔 delay 秒（带 ±30% 抖动更自然）。"""
        if self.delay <= 0:
            return
        elapsed = time.time() - self._last_request_at
        wait = self.delay * random.uniform(0.7, 1.3) - elapsed
        if wait > 0:
            time.sleep(wait)

    def get_text(self, url: str, referer: str = "", _allow_downgrade: bool = True) -> str:
        """GET 一个 URL 并返回解码后的 HTML 文本；重试耗尽抛 FetchError。"""
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retries + 1):
            self._sleep_politely()
            try:
                resp = self.session.get(
                    url, headers=self._headers(referer), timeout=self.timeout,
                    allow_redirects=True, verify=self.verify_ssl,
                )
                self._last_request_at = time.time()

                # 4xx 不重试（除 429 限流）；5xx 与网络错误重试
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise rq_exc.HTTPError(f"HTTP {resp.status_code}", response=resp)
                resp.raise_for_status()
                return decode_html(resp, self.charset)

            # ⚠ 异常类必须从 requests.exceptions 取：
            #   requests.ChunkedEncodingError 在 2.x 顶层**不存在**，写在这里会让
            #   except 元组求值时就抛 AttributeError，重试逻辑整段失效。
            except rq_exc.SSLError as e:
                msg = str(e)
                # 证书本身的错（过期/链不全）→ 关掉校验就能解决
                if "CERTIFICATE_VERIFY" in msg.upper():
                    raise FetchError(
                        f"{url} SSL 证书校验失败。政府站点证书过期/链不全很常见，"
                        f"确认可信后在 config.json 里设 \"crawl\": {{\"verify_ssl\": false}}"
                    ) from e
                # 协议级 TLS 错误（如 BAD_ECPOINT：本地 OpenSSL 不支持对方的 ECC 证书）
                # 这类问题关校验也没用，但很多政府站点 http 仍可用 → 降级重试一次
                if _allow_downgrade and url.startswith("https://"):
                    http_url = "http://" + url[len("https://"):]
                    print(f"  [INFO] {url} TLS 握手失败（{msg.split(':')[-1].strip()[:40]}），降级为 http 重试")
                    return self.get_text(http_url, referer, _allow_downgrade=False)
                raise FetchError(
                    f"{url} TLS 握手失败（既不是证书过期，http 也不可用）：{msg[:120]}"
                ) from e

            except (rq_exc.Timeout, rq_exc.ConnectionError, rq_exc.HTTPError, rq_exc.ChunkedEncodingError) as e:
                self._last_request_at = time.time()
                last_err = e
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status is not None and 400 <= status < 500 and status != 429:
                    raise FetchError(f"{url} 返回 {status}，不重试") from e
                if attempt < self.retries:
                    sleep_s = self.backoff ** (attempt - 1)  # 1s → 2s → 4s 指数退避
                    print(f"  [retry {attempt}/{self.retries - 1}] {url} 失败({type(e).__name__})，{sleep_s:.1f}s 后重试")
                    time.sleep(sleep_s)
            except rq_exc.RequestException as e:  # 兜底，避免异常类型漏网
                last_err = e
                if attempt == self.retries:
                    break
                time.sleep(self.backoff ** (attempt - 1))

        raise FetchError(f"{url} 抓取失败，已重试 {self.retries} 次：{last_err}")
