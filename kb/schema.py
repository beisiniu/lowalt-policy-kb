"""时效性判定与检索过滤 —— 本项目"知识库能不能用对"的核心规则层。

三条规则，缺一条就会答错：

  1. **状态**：`status` 必须是「现行有效」。已废止的文件哪怕字面最像，也不能作为依据。
  2. **生效**：`effective_date <= today`。还没生效的政策不能拿来回答"现在要不要审批"。
  3. **失效**：`expiry_date == 无` 或 `expiry_date >= today`。有效期过了等同废止。

外加 **地域**：问"在深圳飞"，就不该召回上海的文件；但「全国」层级的文件对任何地方都适用。
"""
from __future__ import annotations

import re
from datetime import date, datetime
from typing import Optional

import jieba

# 广东 21 个地市全部列入，避免只覆盖深圳/广州后，中山、惠州等问题绕过地域过滤。
GD_CITIES = (
    "广州市", "深圳市", "珠海市", "汕头市", "佛山市", "韶关市", "湛江市",
    "肇庆市", "江门市", "茂名市", "惠州市", "梅州市", "汕尾市", "河源市",
    "阳江市", "清远市", "东莞市", "中山市", "潮州市", "揭阳市", "云浮市",
)
# 区县要排在市前面；detect_region 会按长度倒序匹配。
REGIONS = (
    "广州市黄埔区", *GD_CITIES,
    "上海市", "北京市", "杭州市", "成都市", "南京市", "武汉市", "西安市",
    "重庆市", "天津市", "苏州市", "合肥市", "长沙市",
    "广东省", "浙江省", "江苏省", "四川省", "湖北省", "陕西省", "安徽省", "湖南省",
)
NATIONAL = "全国"

# 城市 → 所属省。省级文件对省内城市**同样适用**（上位规定）。
PROVINCE_OF = {city: "广东省" for city in GD_CITIES}
PROVINCE_OF.update({
    "广州市黄埔区": "广东省",
    "杭州市": "浙江省", "南京市": "江苏省", "苏州市": "江苏省",
    "成都市": "四川省", "武汉市": "湖北省", "西安市": "陕西省",
    "合肥市": "安徽省", "长沙市": "湖南省",
    "北京市": None, "上海市": None, "天津市": None, "重庆市": None,
})
# 行政区直系上级：市级文件也适用于其下辖区县。
PARENT_REGION = {"广州市黄埔区": "广州市"}


def province_of(city: str) -> str | None:
    return PROVINCE_OF.get(city)

STOPWORDS = set(
    """的 了 和 与 及 或 在 是 有 对 为 我 你 他 她 它 们 这 那 哪些 什么 怎么 如何 是否 需要 要不要
    请问 一下 现在 目前 可以 能否 应该 规定 政策 要求 情况 相关 一般 具体 吗 呢 啊 吧
    一 个 条 款 第 号 年 月 日""".split()
)


def today_default() -> date:
    return date.today()


def to_date(s: str) -> Optional[date]:
    if not s or s in ("无", "None", "null", "-"):
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except (ValueError, AttributeError):
        return None


def detect_region(question: str) -> str:
    """从问题里抽适用地域；抽不到返回空串（= 不做地域过滤）。

    要同时认全称和口语简称："在深圳飞"和"在深圳市飞"是同一个问题，
    只匹配"深圳市"会把一半的真实提问漏掉（地域过滤就白做了）。
    """
    for r in sorted(REGIONS, key=len, reverse=True):
        if r in question:
            return r
        short = r[:-1] if r.endswith(("市", "省")) else ""
        if len(short) >= 2 and short in question:
            return r
    if NATIONAL in question or "国家" in question or "民航局" in question:
        return NATIONAL
    return ""


def region_matches(chunk_region: str, query_region: str) -> bool:
    """文件地域是否适用于提问地域。

    规则（从宽到严）：
      1. 全国性文件对任何地域都适用
      2. 同地域直接命中
      3. 省级文件适用于省内城市（深圳市 ← 广东省）
      4. 其余不命中（广东省的文件不能拿去回答上海的问题）
    """
    if not query_region:
        return True
    cr = (chunk_region or "").strip()
    if not cr or cr == NATIONAL:
        return True
    if cr == query_region:
        return True
    # 市级文件适用于其下辖区县，例如问「广州市黄埔区」时可用广州市文件。
    if PARENT_REGION.get(query_region) == cr:
        return True
    if cr.endswith("省"):
        return province_of(query_region) == cr
    return False


def freshness(meta: dict, today: Optional[date] = None) -> tuple[bool, str]:
    """返回 (是否可用, 原因)。原因为空串表示无异常。"""
    today = today or today_default()
    status = (meta.get("status") or "").strip()

    if status and status != "现行有效":
        return False, f"状态为「{status}」"

    eff = to_date(meta.get("effective_date", ""))
    if eff and eff > today:
        return False, f"尚未生效（{eff} 施行）"

    exp = to_date(meta.get("expiry_date", ""))
    if exp and exp < today:
        return False, f"已过有效期（{exp} 失效）"

    return True, ""


def is_usable(meta: dict, today: Optional[date] = None, query_region: str = "") -> bool:
    ok, _ = freshness(meta, today)
    if not ok:
        return False
    return region_matches(meta.get("apply_region", ""), query_region)


def tokenize(text: str) -> list[str]:
    """中文分词 → 检索词。BM25 靠它建倒排，分词质量直接决定召回。"""
    words = []
    for w in jieba.lcut(text or ""):
        w = w.strip()
        if not w or w in STOPWORDS:
            continue
        if len(w) == 1 and not re.match(r"[0-9a-zA-Z]", w):
            continue  # 丢掉无意义的单字
        words.append(w.lower())
    return words
