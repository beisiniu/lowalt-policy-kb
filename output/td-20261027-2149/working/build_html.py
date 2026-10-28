import re
import sys
from pathlib import Path

SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
SUB = sys.argv[3] if len(sys.argv) > 3 else "给非本专业读者的完整说明与整改方案"
CAT = sys.argv[4] if len(sys.argv) > 4 else "技术评审 · TECHNICAL REVIEW"
KICK = sys.argv[5] if len(sys.argv) > 5 else "内部交接文档"
COVER_META = sys.argv[6] if len(sys.argv) > 6 else "低空政策 RAG 项目组　|　2026 年 10 月 27 日　|　版本 v1.0"
WARN = sys.argv[7] if len(sys.argv) > 7 else "核对基准：语料库 154 篇 / 6106 个切片　·　题库 96 题　·　v9 评估结果"
md = SRC.read_text(encoding="utf-8")

# ---------- helpers ----------
def esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def inline(t: str) -> str:
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`([^`]+)`", r'<code>\1</code>', t)
    return t

lines = md.split("\n")
i = 0
blocks = []          # (kind, payload)
title = ""
while i < len(lines):
    ln = lines[i]
    s = ln.strip()
    if s.startswith("# ") and not title:
        title = s[2:].strip()
        i += 1
        continue
    if s.startswith("## "):
        txt = s[3:].strip()
        if txt.startswith("——"):
            i += 1
            continue
        blocks.append(("h1", txt))
        i += 1
        continue
    if s.startswith("### "):
        blocks.append(("h2", s[4:].strip()))
        i += 1
        continue
    if s.startswith("```"):
        i += 1
        buf = []
        while i < len(lines) and not lines[i].strip().startswith("```"):
            buf.append(lines[i])
            i += 1
        i += 1
        blocks.append(("code", "\n".join(buf)))
        continue
    if s.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            rows.append(lines[i].strip())
            i += 1
        blocks.append(("table", rows))
        continue
    if s.startswith("- "):
        items = []
        while i < len(lines) and lines[i].strip().startswith("- "):
            items.append(lines[i].strip()[2:].strip())
            i += 1
        blocks.append(("ul", items))
        continue
    if re.match(r"^\d+\.\s", s):
        items = []
        while i < len(lines) and re.match(r"^\d+\.\s", lines[i].strip()):
            items.append(re.sub(r"^\d+\.\s+", "", lines[i].strip()))
            i += 1
        blocks.append(("ol", items))
        continue
    if s.startswith("> "):
        buf = []
        while i < len(lines) and lines[i].strip().startswith(">"):
            buf.append(lines[i].strip().lstrip(">").strip())
            i += 1
        blocks.append(("quote", " ".join(buf)))
        continue
    if s in ("---", "***"):
        blocks.append(("divider", ""))
        i += 1
        continue
    if not s:
        i += 1
        continue
    # paragraph: gather until blank / structural
    buf = []
    while i < len(lines):
        cur = lines[i].strip()
        if not cur or cur.startswith(("#", "|", "- ", "```", "> ")) or re.match(r"^\d+\.\s", cur) or cur in ("---", "***"):
            break
        buf.append(cur)
        i += 1
    blocks.append(("p", " ".join(buf)))

# ---------- split executive summary ----------
exec_blocks, main_blocks = [], []
META = re.compile(r"^\*\*(文档版本|写作时间|核对基准|适用范围)\*\*")
mode = "pre"
for b in blocks:
    if b[0] == "h1" and b[1].startswith("阅读提示"):
        mode = "exec"
        continue
    if b[0] == "h1" and mode == "exec":
        mode = "main"
    if b[0] == "p" and META.match(b[1]):
        continue
    (exec_blocks if mode == "exec" else main_blocks).append(b)

# ---------- TOC ----------
toc = []
sec = 0
sub = 0
for b in main_blocks:
    if b[0] == "h1":
        sec += 1
        sub = 0
        toc.append((1, sec, None, b[1]))
    elif b[0] == "h2":
        sub += 1
        toc.append((2, sec, sub, b[1]))

# ---------- render ----------
def render_blocks(bs, start_after=None):
    out = []
    sec = 0
    sub = 0
    for kind, payload in bs:
        if kind == "h1":
            sec += 1
            sub = 0
            out.append(f'<h1 id="section-{sec}" class="chapter">{inline(payload)}</h1>')
        elif kind == "h2":
            sub += 1
            out.append(f'<h2 id="section-{sec}-{sub}" class="section">{inline(payload)}</h2>')
        elif kind == "p":
            out.append(f"<p>{inline(payload)}</p>")
        elif kind == "table":
            rows = payload
            head = [c.strip() for c in rows[0].strip("|").split("|")]
            body = []
            for r in rows[1:]:
                if re.match(r"^\|[\s:\-|]+\|$", r):
                    continue
                body.append([c.strip() for c in r.strip("|").split("|")])
            t = ['<table class="data-table">', "<thead><tr>"]
            t += [f"<th>{inline(c)}</th>" for c in head]
            t.append("</tr></thead><tbody>")
            for r in body:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t))
        elif kind == "ul":
            out.append("<ul class=\"bullet\">" + "".join(f"<li>{inline(x)}</li>" for x in payload) + "</ul>")
        elif kind == "ol":
            out.append("<ol class=\"num\">" + "".join(f"<li>{inline(x)}</li>" for x in payload) + "</ol>")
        elif kind == "quote":
            out.append(f'<p class="quote">&nbsp;&nbsp;{inline(payload)}</p>')
        elif kind == "code":
            out.append(f'<pre class="code">{esc(payload)}</pre>')
        elif kind == "divider":
            out.append('<p class="divider"></p>')
    return "\n".join(out)

# TOC html
toc_html = []
k = 0
while k < len(toc):
    lvl, a, b, txt = toc[k]
    if lvl == 1:
        inner = []
        k += 1
        while k < len(toc) and toc[k][0] == 2:
            inner.append(f'<li class="toc-sub"><a href="#section-{toc[k][1]}-{toc[k][2]}">{inline(toc[k][3])}</a></li>')
            k += 1
        li = f'<li><a href="#section-{a}">{inline(txt)}</a>'
        if inner:
            li += '<ol class="toc-sublist">' + "".join(inner) + "</ol>"
        li += "</li>"
        toc_html.append(li)
    else:
        k += 1

exec_html = render_blocks(exec_blocks)
main_html = render_blocks(main_blocks)

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="docx-page-size" content="A4">
<title>{esc(title)}</title>
<style>
  @page {{ @bottom-center {{ content: "第 " counter(page) " 页"; }} }}
  @page cover {{ @bottom-center {{ content: none; }} }}
  section[role="cover"] {{ page: cover; }}

  :root {{
    --fs-body: 12pt;
    --fs-small: 10.5pt;
    --fs-h1: 16pt;
    --fs-h2: 13pt;
    --ff-body: "SimSun", "宋体", serif;
    --ff-head: "SimHei", "黑体", sans-serif;
    --ff-code: "Consolas", "Courier New", monospace;
    --c-primary: #1f3a5f;
    --c-accent: #b3261e;
    --c-text: #1a1a1a;
    --c-muted: #5c6670;
    --c-border: #c9d1d9;
    --c-bg: #f4f6f9;
    --page-content-width: 15.6cm;
    --sp-paragraph: 0.55em;
    --sp-section: 1.2em;
  }}

  body {{
    font-family: var(--ff-body);
    font-size: var(--fs-body);
    line-height: 1.7;
    color: var(--c-text);
    max-width: var(--page-content-width);
    margin-left: auto;
    margin-right: auto;
  }}

  section[role="cover"] {{ text-align: center; }}
  .cover-category {{ font-family: var(--ff-head); font-size: 11pt; color: var(--c-muted); letter-spacing: 0.18em; text-align: center; margin-top: 5.2cm; margin-bottom: 0.6em; }}
  .cover-rule-text {{ font-family: var(--ff-head); font-size: 9pt; color: var(--c-muted); text-align: center; margin-bottom: 2.2em; }}
  .cover-title {{ font-family: var(--ff-head); font-size: 26pt; font-weight: bold; color: var(--c-primary); text-align: center; line-height: 1.35; margin-bottom: 0.5em; }}
  .cover-subtitle {{ font-family: var(--ff-head); font-size: 14pt; color: var(--c-muted); text-align: center; margin-bottom: 2.6em; }}
  .cover-meta {{ font-size: var(--fs-small); color: var(--c-muted); text-align: center; line-height: 2.0; }}
  .cover-warn {{ font-size: 9pt; color: var(--c-muted); text-align: center; margin-top: 3.4em; }}

  .toc-title {{ font-family: var(--ff-head); font-size: 15pt; font-weight: bold; color: var(--c-primary); text-align: center; margin-bottom: 1.0em; letter-spacing: 0.1em; }}
  .toc-list, .toc-list ol, .toc-list ul {{ list-style: none; list-style-type: none; padding-left: 0; }}
  .toc-list > li {{ margin-bottom: 0.35em; text-indent: 0; font-family: var(--ff-head); font-size: 11.5pt; }}
  .toc-list a {{ color: var(--c-text); text-decoration: none; }}
  .toc-sublist {{ padding-left: 1.6em; margin-top: 0.2em; }}
  .toc-sublist li.toc-sub {{ font-family: var(--ff-body); font-size: 10.5pt; color: var(--c-muted); margin-bottom: 0.1em; text-indent: 0; }}
  .toc-sublist a {{ color: var(--c-muted); }}

  h1.chapter {{
    font-family: var(--ff-head); font-size: var(--fs-h1); font-weight: bold;
    color: var(--c-primary); margin-top: var(--sp-section); margin-bottom: 0.6em;
    padding-bottom: 0.25em; border-bottom: 2px solid var(--c-primary);
    text-indent: 0; text-align: left;
  }}
  h2.section {{
    font-family: var(--ff-head); font-size: var(--fs-h2); font-weight: bold;
    color: var(--c-primary); margin-top: 1.0em; margin-bottom: 0.45em;
    text-indent: 0; text-align: left;
  }}
  p {{ text-indent: 2em; margin-bottom: var(--sp-paragraph); text-align: justify; }}
  p.quote {{ border-left: 3px solid var(--c-accent); background: #fbf3f2; padding: 0.5em 0.7em; text-indent: 0; color: #2b2b2b; }}
  p.divider {{ border-bottom: 1px solid var(--c-border); text-indent: 0; margin: 0.6em 0; }}

  ul.bullet, ol.num {{ margin: 0.2em 0 0.7em 1.6em; padding-left: 1.0em; }}
  ul.bullet li, ol.num li {{ margin-bottom: 0.18em; text-indent: 0; text-align: justify; }}

  table.data-table {{
    width: 100%; border-collapse: collapse; margin: 0.5em 0 0.9em 0;
    font-size: var(--fs-small); font-family: var(--ff-body);
  }}
  table.data-table th {{
    background: var(--c-bg); font-family: var(--ff-head); font-weight: bold;
    border: 1px solid var(--c-border); padding: 0.35em 0.5em; text-align: left;
  }}
  table.data-table td {{ border: 1px solid var(--c-border); padding: 0.32em 0.5em; text-indent: 0; vertical-align: top; }}
  table.data-table tr:nth-child(even) td {{ background: #fafbfc; }}

  pre.code {{
    font-family: var(--ff-code); font-size: 8.5pt; line-height: 1.35;
    background: #f6f8fa; border: 1px solid var(--c-border); padding: 0.5em 0.7em;
    margin: 0.4em 0 0.8em 0; white-space: pre-wrap; text-indent: 0;
  }}
  code {{ font-family: var(--ff-code); font-size: 9.5pt; }}

  .executive-summary {{ border-left: 4px solid var(--c-primary); background: var(--c-bg); padding: 0.7em 0.9em; margin-bottom: 1.2em; }}
  .summary-title {{ font-family: var(--ff-head); font-size: 13pt; font-weight: bold; color: var(--c-primary); text-indent: 0; margin-bottom: 0.4em; }}
  .summary-badge {{ font-size: 9pt; color: var(--c-muted); text-indent: 0; margin-bottom: 0.5em; }}
</style>
</head>
<body>

<section role="cover">
  <p class="cover-category">{CAT}</p>
  <p class="cover-rule-text">{KICK}</p>
  <p class="cover-title">{esc(title)}</p>
  <p class="cover-subtitle">{SUB}</p>
  <p class="cover-meta">{COVER_META}</p>
  <p class="cover-warn">{WARN}</p>
</section>

<section role="cover">
  <p class="toc-title">目　录</p>
  <ol class="toc-list">
    {''.join(toc_html)}
  </ol>
</section>

<section role="body" data-page-restart="1">
  <div class="executive-summary">
    <p class="summary-title">内容提要</p>
    <p class="summary-badge">Executive Summary</p>
    <p>本文拆解 49 道 TQ 测试题的生产过程：一个脚本从向量库中取出全部切片，按地区与主题分桶，在同桶内把失效条款与现行条款两两配对，再由语言模型判断是否构成实质冲突并生成问题与答案，最后经第二个语言模型审核后写入题集。</p>
    <p>这套流程产生四个结构性偏差：题目集中在四个地区、八成集中在补贴与门槛两类、部分题目的“新条款”其实是废止通告、与另一类事实题混算分母导致 38.5% 这个混合数字。文中给出每一步的真实参数与中间结果，并在第五节对照四类同类研究，第六节列出八条可直接落地的改造动作。</p>
    {exec_html}
  </div>
  {main_html}
</section>

</body>
</html>
"""

OUT.write_text(HTML, encoding="utf-8")
print("written:", OUT, len(HTML), "chars")
print("chapters:", sum(1 for b in main_blocks if b[0] == "h1"), "sections:", sum(1 for b in main_blocks if b[0] == "h2"))
