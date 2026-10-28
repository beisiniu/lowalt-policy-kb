"""回答生成：抽取式（无 LLM）与生成式（LLM）两条路。

抽取式不是玩具：它把检索到的最相关条文原样摘出来，
作为**检索质量的对照组** —— 如果连抽取式都答不对，说明是检索/数据的问题，
换更大的模型也没用。论文里这一栏很有用。
"""
from __future__ import annotations

import re

from langchain_core.documents import Document

_APPROVAL_RE = re.compile(r"(无需|不需要|免于|应当|须|需|不得|禁止|未经)")
_DATE_RE = re.compile(r"\d{4}\s*年|\d{1,2}\s*月\d{1,2}\s*日|有效期")
_TIME_ASK = ("什么时候", "何时", "时间", "日期", "几年", "有效期")
# 公文体套话：摘出来对回答问题零价值
_BOILERPLATE = ("现将", "印发给你们", "请遵照执行", "各有关单位", "特此通知", "抄送", "请认真贯彻")
_PREFIX_RE = re.compile(r"^(《[^》]*》\s*)?(第[一二三四五六七八九十百零\d]+条\s*)?")


def _strip_prefix(text: str) -> str:
    """去掉 chunk 前缀里的《政策名》与"第X条"，避免引用时重复两遍。"""
    return _PREFIX_RE.sub("", text, count=1).strip()


def _bigrams(s: str) -> set[str]:
    s = re.sub(r"\s+", "", s)
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _cite(meta: dict) -> str:
    name = meta.get("policy_name", "")
    art = meta.get("article", "")
    eff = meta.get("effective_date", "")
    tail = f"（{eff}）" if eff and eff != "无" else ""
    return f"《{name}》{art}{tail}"


def _strip_known_prefix(text: str, policy_name: str) -> str:
    """按**已知政策名**精确剥前缀。

    政策名里本身可能带书名号（《关于印发《XX办法》的通知》），
    用 `^《[^》]*》` 这种正则会停在第一个 》 上，剥出个残句。
    既然调用方已经知道 policy_name，直接按确定长度切最稳。
    """
    prefix = f"《{policy_name}》"
    if policy_name and text.startswith(prefix):
        return text[len(prefix):].strip()
    return _strip_prefix(text)


def _too_similar_to_title(sent: str, policy_name: str, threshold: float = 0.8) -> bool:
    """句子是不是基本就是政策标题本身（网页正文常以标题开头，摘出来毫无信息量）。"""
    if not policy_name or len(sent) < 8:
        return False
    overlap = len(set(sent) & set(policy_name)) / max(1, len(set(sent)))
    return overlap >= threshold


def _best_sentence(text: str, question: str, policy_name: str = "") -> str:
    """挑一句和问题最贴的。

    打分 = 与问题的二元词重合 + 含审批动作词 + （问时间时）含日期。
    只看字面重合会挑到"本条例施行前…"这类无效句；只看动作词又会挑到不相关的处罚条款。
    """
    body = _strip_prefix(text)
    sents = [s.strip() for s in re.split(r"(?<=[。；])", body) if len(s.strip()) > 8]
    # 跳过「正文里重复标题」的句子，否则答案一开头就是政策名念一遍
    sents = [x for x in sents if not _too_similar_to_title(x, policy_name)] or sents
    if not sents:
        return body[:160]

    q_grams = _bigrams(question)
    ask_time = any(w in question for w in _TIME_ASK)

    def score(s: str) -> float:
        v = len(q_grams & _bigrams(s))
        if _APPROVAL_RE.search(s):
            v += 2
        if ask_time and _DATE_RE.search(s):
            v += 3
        if any(w in s for w in _BOILERPLATE):
            v -= 4
        return v

    return max(sents, key=score)


def extractive_answer(question: str, hits: list[Document], max_cite: int = 2) -> str:
    """不调用任何模型，直接把最相关条文摘出来。"""
    if not hits:
        return "知识库中未检索到与该问题相关的政策条文。"

    parts = []
    for d in hits[:max_cite]:
        name = d.metadata.get("policy_name", "")
        sent = _best_sentence(_strip_known_prefix(d.page_content, name), question, name)
        parts.append(f"根据{_cite(d.metadata)}：{sent}")

    head = parts[0]
    tail = f"另可参考{_cite(hits[1].metadata)}。" if len(parts) > 1 else ""
    return head + ("\n" + tail if tail else "")


SYSTEM_PROMPT = """你是低空经济政策问答助手。**只能**依据下面提供的政策条文回答，不得使用条文以外的知识。
回答要求：
1. 审批/合规判断题：先给结论（需要审批 / 无需审批），再给依据。
   事实查询题（问金额、比例、日期、数量、名称）：直接回答所问事实，不要强行添加审批结论。
2. 引用时写明《政策名》+ 条款号（原文没有条号就只写政策名）+ 施行日期。
3. 若条文之间冲突，通常以「施行日期更晚」且「现行有效」的为准并说明冲突；
   但题目明确指明年份、版本或历史政策时，必须回答题目指定版本，不能盲目改用最新文件。
4. 数字、单位、适用条件必须完整保留；不得把“机库”当“起降点”、把“企业数”当“产业规模”。
5. 若提供的条文不足以回答，直接说"根据现有政策条文无法确定"，不要猜，也不要用其他城市的数值代替。
"""


def build_context(hits: list[Document]) -> str:
    lines = []
    for i, d in enumerate(hits, 1):
        md = d.metadata
        lines.append(
            f"[{i}] 《{md.get('policy_name')}》{md.get('article', '')}"
            f"｜适用地域：{md.get('apply_region')}｜施行：{md.get('effective_date')}"
            f"｜失效：{md.get('expiry_date')}｜状态：{md.get('status')}\n{d.page_content}"
        )
    return "\n\n".join(lines)


def llm_answer(client, question: str, hits: list[Document], region: str = "") -> str:
    """调用大模型生成（需要 DEEPSEEK_API_KEY）。"""
    user = (
        (f"提问者所在地域：{region}\n" if region else "")
        + f"【政策条文】\n{build_context(hits)}\n\n【问题】\n{question}"
    )
    return client.chat(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
        temperature=0.0,
        max_tokens=1200,
        json_mode=False,
        thinking=True,   # 回答要跨条款权衡，开推理质量明显更好
    )


def refusal_answer(question: str, stale: list[Document], region: str = "") -> str:
    """命中但全部失效时的拒答话术。

    这不是"我不知道"，而是"我知道的那条已经不作数了" —— 这两者对评估的意义完全不同：
    前者是检索失败，后者是时效性风险被正确拦截。
    """
    if not stale:
        return "知识库中没有检索到与该问题相关的现行政策条文，无法给出确定结论，建议咨询主管部门。"

    top = stale[0]
    md = top.metadata
    reason = md.get("_reason") or "已失效"
    return (
        f"知识库中与问题最相关的是《{md.get('policy_name')}》{md.get('article', '')}，"
        f"但该政策{reason}，不能作为当前审批依据。"
        f"在缺少现行有效条文的情况下，无法给出确定的审批结论，建议以主管部门最新公布的规定为准。"
    )
