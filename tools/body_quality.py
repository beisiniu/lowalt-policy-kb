"""正文质量检测与清洗。

背景：
  抓取环节有两种失败模式会把"非正文"混进政策文本：
    1. 网页源码残留——整页 HTML/JS/CSS 被当成正文（武汉、东莞等）；
    2. PDF 字形乱码——内嵌字体以 CID 编码输出（如 /G21/G22），汉字占比极低。
  这两类内容进入向量库后会变成大量无意义切片，既污染检索又虚增库规模。

本模块提供：
  · metrics(text)       —— 四个客观指标，用于判断正文是否可信；
  · clean_html_noise()  —— 从网页源码中剥离脚本/样式/标签/实体，回收正文；
  · judge(text)         —— 综合判定 ok / suspect / reject。

判据（经验阈值，可用 --threshold 调）：
  汉字占比 < 0.55 且清洗后无改善 → suspect/reject
  JS 关键词 > 20 处/万字       → suspect
  PDF 乱码片段 (/GXX) 占比高    → reject（无法用清洗修复，必须换提取方式）
"""
from __future__ import annotations

import re
from dataclasses import dataclass

CJK = re.compile(r"[\u4e00-\u9fff]")
JS_KW = re.compile(
    r"function\s*\(|var\s+\w+\s*=|let\s+\w+\s*=|document\.|window\.|addEventListener|"
    r"\.style\.|querySelector|getElementById|=>\s*\{|console\.log|\\u00[0-9a-f]{2}"
)
TAG = re.compile(r"<[^>]{1,200}>")
SCRIPT_BLOCK = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.S | re.I)
STYLE_BLOCK = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.S | re.I)
COMMENT_BLOCK = re.compile(r"<!--.*?-->", re.S)
CSS_BLOCK = re.compile(r"[\w\-\.\#\:\s,\>\*\[\]\(\)]{2,80}\{[^{}]{0,600}\}", re.S)
PDF_CID = re.compile(r"/G[0-9A-F]{2}")
ENTITY = re.compile(r"&(#\d+|#x[0-9a-fA-F]+|[a-zA-Z]{2,8});")
WS = re.compile(r"[ \t\u3000]+")
MULTI_NL = re.compile(r"\n{3,}")


def metrics(text: str) -> dict:
    """返回正文的客观质量指标。"""
    n = max(1, len(text))
    cjk = len(CJK.findall(text))
    return {
        "chars": len(text),
        "cjk": cjk,
        "cjk_ratio": round(cjk / n, 3),
        "js_kw": len(JS_KW.findall(text)),
        "js_per_10k": round(len(JS_KW.findall(text)) / n * 10000),
        "tags": len(TAG.findall(text)),
        "pdf_cid": len(PDF_CID.findall(text)),
    }


def _unescape(text: str) -> str:
    """还原 HTML 实体与 JSON 转义。"""
    simple = {"nbsp": " ", "amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'",
              "ldquo": "“", "rdquo": "”", "lsquo": "‘", "rsquo": "’", "mdash": "—",
              "ndash": "–", "hellip": "…", "middot": "·", "times": "×", "ensp": " ",
              "emsp": " ", "thinsp": " ", "yen": "￥"}

    def rep(m: re.Match) -> str:
        k = m.group(1)
        if k.startswith("#x") or k.startswith("#X"):
            try:
                return chr(int(k[2:], 16))
            except ValueError:
                return m.group(0)
        if k.startswith("#"):
            try:
                return chr(int(k[1:]))
            except ValueError:
                return m.group(0)
        return simple.get(k.lower(), m.group(0))

    text = ENTITY.sub(rep, text)
    # JSON 风格转义：\u0022 \u003C 等
    text = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), text)
    text = text.replace("\\n", "\n").replace("\\t", " ").replace('\\"', '"')
    return text


def clean_html_noise(text: str) -> str:
    """尽最大努力从网页源码里回收正文。"""
    t = text
    t = SCRIPT_BLOCK.sub("\n", t)
    t = STYLE_BLOCK.sub("\n", t)
    t = COMMENT_BLOCK.sub("\n", t)
    t = _unescape(t)
    t = TAG.sub("\n", t)
    # 纯 CSS 规则块（已无标签包裹的残留）
    for _ in range(3):
        t2 = CSS_BLOCK.sub("\n", t)
        if t2 == t:
            break
        t = t2
    # 去掉典型代码行
    kept = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            kept.append("")
            continue
        # 整行几乎不含中文、且含代码特征 → 丢弃
        if len(CJK.findall(s)) <= 1 and JS_KW.search(s):
            continue
        if re.fullmatch(r"[{}();,\.\s\w\"'=+\-*/\\:\[\]<>!&|%$#@~`^?]+", s) and not CJK.search(s):
            continue
        kept.append(s)
    t = "\n".join(kept)
    t = WS.sub(" ", t)
    t = MULTI_NL.sub("\n\n", t)
    return t.strip()


# 网页框架文本（导航 / 页脚），浏览器渲染取 innerText 时会一并带出
NAV_WORDS = {
    "首页", "要闻动态", "政策发布", "政务服务", "信息公开", "政民互动", "走进上海",
    "简体", "繁体", "繁体版", "关怀版", "无障碍", "登录", "注册", "长者助手",
    "网站地图", "返回首页", "打印本页", "关闭窗口", "扫一扫", "分享", "分享到",
    "EN", "网站支持IPV6", "工作机构", "营商环境", "互动交流", "政务公开",
    "天河动态", "走进天河", "通知公告", "其他公告", "新闻", "政务资讯",
}
# 页脚起点标志：出现即切断其后内容
FOOTER_MARKERS = (
    "相关附件", "相关稿件", "历史沿革", "主办单位", "承办单位", "版权所有",
    "网站标识码", "ICP备", "ICP证", "公网安备", "政务邮箱", "邮编：",
    "联系电话：", "地址：", "技术支持", "网站声明", "关于我们", "联系我们",
    "友情链接", "市政府部门网站", "区政府网站", "国内重要城市网站",
    "关于本网", "国务院部门网站", "地方政府网站", "驻港澳机构网站", "驻外机构",
    "最高人民法院", "最高人民检察院", "打开APP", "进入频道", "网络支持",
    "链接：", "全国人大", "全国政协", "国家监察委员会", "中国政府网",
    "责编", "编辑：", "来源：", "扫码关注", "微信公众号", "政务微博",
    "智能问答", "问答知识库", "回到顶部", "收起", "展开", "个人中心",
    "主办：", "承办：", "协办", "技术支持单位",
    "国务院文件", "网站地图", "简繁切换",
    "上一篇：", "下一篇：", "打印关闭", "关闭页面", "【打印】", "【关闭】",
)
# 强标志：带字段名/编号，出现在正文里的概率极低，长行也认
STRONG_MARKERS = (
    "ICP备", "ICP证", "公网安备", "网站标识码", "政府网站标识",
    "主办单位：", "承办单位：", "版权所有", "政务邮箱", "邮编：",
    "联系电话：", "联系方式：", "技术支持：", "网站声明", "友情链接",
    "责编", "编辑：", "扫码关注", "微信公众号", "政务微博", "个人中心",
)
TITLE_HINT = re.compile(r"关于|通告|通知|办法|规定|方案|意见|条例|细则|措施|公告|决定|规划")


def strip_page_furniture(text: str) -> str:
    """去掉网页导航栏与页脚，只保留政策正文区。

    浏览器渲染取 innerText 时，导航区和页脚会与正文连成一片：
      "首页 / 要闻动态 / … / 主办单位：… / 沪ICP备…"
    这里按两类信号切边：
      · 页脚：命中 FOOTER_MARKERS 的第一行起，其后全部丢弃；
      · 头部：跳过连续的导航短行，直到出现标题行或首个长段落。
    """
    lines = [l.rstrip() for l in text.split("\n")]

    # 1) 切页脚：只在文档后半段找页脚标志，避免正文中段的"地址/联系电话"误切
    #    页脚行通常很短；长行命中的多是正文句子（如"……技术支持等服务"），不能截断
    cut = len(lines)
    guard = max(20, int(len(lines) * 0.45))
    for i in range(guard, len(lines)):
        s = lines[i].strip()
        if not s:
            continue
        hit = next((m for m in FOOTER_MARKERS if m in s), None)
        if not hit:
            continue
        if len(s) <= 40 or hit in STRONG_MARKERS:
            cut = i
            break
    lines = lines[:cut]

    # 2) 切头部：跳过面包屑与导航短行，首个标题/长段落起算
    start = 0
    for i, l in enumerate(lines):
        s = l.strip()
        if not s:
            continue
        if "当前位置" in s or "所在的位置" in s:
            continue
        cjk = len(CJK.findall(s))
        is_nav = s in NAV_WORDS or (cjk <= 8 and not TITLE_HINT.search(s) and len(s) <= 12)
        is_title = cjk >= 10 and TITLE_HINT.search(s)
        is_para = cjk >= 40
        if is_title or is_para:
            start = i
            break
        if not is_nav and cjk >= 15:
            start = i
            break
    lines = lines[start:]

    out = "\n".join(l.strip() for l in lines)
    out = MULTI_NL.sub("\n\n", out).strip()
    # 防误切：正文被砍掉三分之二以上，多半是命中了正文里的假页脚，回退原文
    if len(text) > 400 and len(out) < len(text) * 0.35:
        return text.strip()
    return out


# 页面 UI 词：出现即整行丢弃（都是导航/工具条，与政策内容无关）
UI_TOKENS = (
    "字号", "打印", "纠错", "分享", "推荐", "扫一扫", "浏览", "收藏", "关闭",
    "上一篇", "下一篇", "返回", "首页", "无障碍", "长者助手", "关怀版",
    "建议使用", "极速模式", "浏览器", "网站地图", "联系我们", "关于我们",
)
JSON_FRAG = re.compile(r'"[A-Za-z_][A-Za-z0-9_]{1,24}"\s*:\s*("(?:[^"\\]|\\.)*"|-?\d+(?:\.\d+)?|\[[^\]]{0,200}\]|\{[^{}]{0,200}\}|null|true|false)\s*,?')
JS_CALL = re.compile(r"\b(?:alert|confirm|console\.\w+|document\.\w+|window\.\w+)\s*\([^)]{0,300}\)\s*;?")
ICP_LINE = re.compile(r"(ICP备|ICP证|公网安备|网站标识码|政府网站标识|粤公网安备|京公网安备)")


ARTICLE_RE = re.compile(r"第[一二三四五六七八九十百零〇\d]{1,4}条")
# 公文主送机关：正文从这里开始（"各区人民政府，市政府各部门："）
ADDRESSEE_RE = re.compile(r"^\s*(各[^。\n]{0,30}(人民政府|部门|单位|委员会|管委会|直属机构)[^。\n]{0,20}[：:]|各有关单位[：:])")
# 正文结束锚点
TAIL_ANCHOR_RE = re.compile(r"(有效期至\s*20\d{2}\s*年|自\s*20\d{2}\s*年[^。\n]{0,12}起施行|特此(通告|公告|通知)|本(规定|办法|条例|措施|细则|方案)自)")


def anchor_trim(text: str) -> str:
    """用公文自身结构确定正文边界，剥离首尾的网站框架内容。

    两类结构分别处理：
      · 有条文（"第X条"≥3 处）：首条条文是正文起点，末条条文（含"施行"）是终点，之前留标题；
      · 无条文（通告/通知/方案）：主送机关行是起点，施行/有效期句是终点。
    中间夹带的页面杂质交给 deep_clean 的逐行过滤。
    """
    t = text
    arts = list(ARTICLE_RE.finditer(t))
    if len(arts) >= 3:
        start = arts[0].start()
        head = [l for l in t[:start].split("\n") if l.strip()][-3:]
        end = arts[-1].end()
        # 末条之后通常是落款与施行说明，留 250 字
        tail = t[end:end + 250]
        stop = TAIL_ANCHOR_RE.search(tail)
        if stop:
            seg = tail[stop.start():]
            m = re.search(r"。", seg[:300])
            if m:
                tail = tail[: stop.start() + m.end()]
            else:
                tail = tail[: stop.start()] + "\n".join(seg.split("\n")[:3])
        else:
            tail = tail.split("\n\n")[0][:120]
        return ("\n".join(head) + "\n" + t[start:end] + tail).strip()

    lines = t.split("\n")
    start_i = None
    for i, l in enumerate(lines):
        if ADDRESSEE_RE.match(l):
            start_i = i
            break
    if start_i is None:
        for i, l in enumerate(lines):
            if TAIL_ANCHOR_RE.search(l) or ARTICLE_RE.search(l):
                start_i = max(0, i - 12)
                break
    if start_i is None:
        return t
    body = "\n".join(lines[start_i:])
    stop = TAIL_ANCHOR_RE.search(body)
    if stop:
        seg = body[stop.start():]
        m = re.search(r"。", seg[:300])
        if m:
            body = body[: stop.start() + m.end()]
        else:
            body = body[: stop.start()] + "\n".join(seg.split("\n")[:3])
    return body.strip()


def deep_clean(text: str) -> str:
    """针对"汉字占比正常但混入页面框架"的深度清洗。

    与 clean_html_noise 的分工：
      · clean_html_noise 处理"整页是源码"（汉字占比极低）；
      · deep_clean 处理"正文是中文、但夹着导航栏/页脚/JSON 元数据"——
        这类文本汉字占比往往 60% 以上，靠比例指标发现不了。
    """
    t = text

    # 1) JSON 片段（政务站常把整条记录元数据塞进正文）
    for _ in range(3):
        t2 = JSON_FRAG.sub(" ", t)
        if t2 == t:
            break
        t = t2

    # 2) JS 调用与脚本残留
    t = JS_CALL.sub(" ", t)

    # 3) 头部导航：连续短行（无标点、<20 字）视为导航条，遇到标题/正文即停
    lines = t.split("\n")
    i = 0
    while i < len(lines):
        sk = lines[i].strip()
        if not sk:
            i += 1
            continue
        site_hint = ("_" in sk and len(sk) < 60) or ".gov" in sk or "中国政府网" in sk
        no_punct = not re.search(r"[，。：；（）《》、]", sk)
        if (len(sk) <= 20 and no_punct) or (site_hint and len(sk) < 60):
            i += 1
            continue
        break
    t = "\n".join(lines[i:])

    # 4) 逐行过滤
    kept = []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            kept.append("")
            continue
        if ICP_LINE.search(s):
            continue
        if any(mk in s for mk in STRONG_MARKERS):
            continue
        if len(s) <= 20 and (any(tok in s for tok in UI_TOKENS) or s in NAV_WORDS):
            continue
        # 纯符号/数字/英文的短行（面包屑分隔、编号）
        if len(s) <= 6 and not CJK.search(s):
            continue
        kept.append(s)
    t = "\n".join(kept)

    # 5) 去导航与页脚
    t = strip_page_furniture(t)

    # 6) 折叠重复标题（同一行连出两次以上只留一次）
    out, prev = [], None
    for line in t.split("\n"):
        s = line.strip()
        if s and s == prev:
            continue
        out.append(line)
        prev = s
    t = "\n".join(out)

    return MULTI_NL.sub("\n\n", t).strip()


def judge(text: str) -> tuple[str, dict]:
    """判定正文可用性：ok / suspect / reject。"""
    m = metrics(text)
    if m["pdf_cid"] > 50 or (m["cjk_ratio"] < 0.15 and m["pdf_cid"] > 5):
        return "reject", {**m, "reason": "PDF 字形乱码（CID 编码未解码），无法用清洗修复"}
    if m["cjk_ratio"] < 0.30:
        return "reject", {**m, "reason": "汉字占比过低，正文实质为源码"}
    if m["cjk_ratio"] < 0.60 or m["js_per_10k"] > 20:
        return "suspect", {**m, "reason": "汉字占比偏低或 JS 关键词密集"}
    return "ok", m


def auto_repair(text: str, min_cjk_ratio: float = 0.60) -> tuple[str, bool]:
    """尝试清洗；若清洗后质量达标则返回清洗结果。"""
    before = metrics(text)
    if before["cjk_ratio"] >= min_cjk_ratio and before["js_per_10k"] <= 20:
        return text, False
    cleaned = clean_html_noise(text)
    after = metrics(cleaned)
    if after["cjk_ratio"] > before["cjk_ratio"] + 0.15 and after["cjk"] >= 800:
        return cleaned, True
    return cleaned, False


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    import sys

    ap = argparse.ArgumentParser(description="正文质量检测/清洗")
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--repair", action="store_true", help="输出清洗后正文（不写盘）")
    a = ap.parse_args()
    for p in a.paths:
        raw = Path(p).read_text(encoding="utf-8", errors="replace")
        body = raw.split("-" * 60, 1)[-1]
        verdict, m = judge(body)
        print(json.dumps({"file": Path(p).name[:60], "verdict": verdict, **m}, ensure_ascii=False))
        if a.repair:
            cleaned, ok = auto_repair(body)
            print("  清洗后:", json.dumps(metrics(cleaned), ensure_ascii=False), "可修复:" , ok)
