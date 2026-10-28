#!/usr/bin/env python3
"""把 Stage 1 的 Markdown 转成 doc-typeset business-report 风格的完整 HTML。

只处理本项目用到的 Markdown 子集：H1/H2/H3、段落、GFM 表格、引用、无序/有序列表、加粗。
不引入 grid/flex（docx 转换不支持），所有关键样式写在元素或 class 上。
"""
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "stage1" / "final_draft.md"
OUT = ROOT / "stage2" / "formatted-低空政策RAG系统技术说明.html"


def esc(t: str) -> str:
    return html.escape(t, quote=False)


def inline(t: str) -> str:
    """行内标记：**加粗**、`代码`。"""
    t = esc(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    return t


def split_row(line: str) -> list[str]:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def convert(md: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = md.splitlines()
    body: list[str] = []
    toc: list[tuple[int, str, str]] = []
    title = ""
    subtitle = ""
    cover_note = ""
    i = 0
    chapter_no = 0
    sec_no = 0
    para: list[str] = []
    list_buf: list[str] = []
    list_type = ""
    quote_buf: list[str] = []

    def flush_para() -> None:
        nonlocal para
        if para:
            text = " ".join(x.strip() for x in para).strip()
            if text and text != "---":
                body.append(f'      <p class="body-p">{inline(text)}</p>')
            para = []

    def flush_list() -> None:
        nonlocal list_buf, list_type
        if list_buf:
            tag = "ol" if list_type == "ol" else "ul"
            body.append(f'      <{tag} class="doc-list">')
            for it in list_buf:
                body.append(f"        <li>{inline(it)}</li>")
            body.append(f"      </{tag}>")
            list_buf = []
            list_type = ""

    def flush_quote() -> None:
        nonlocal quote_buf
        if quote_buf:
            text = " ".join(x.strip() for x in quote_buf).strip()
            body.append(f'      <p class="callout">&nbsp;&nbsp;{inline(text)}</p>')
            quote_buf = []

    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        if not line.strip():
            flush_para(); flush_list(); flush_quote()
            i += 1
            continue

        if line.strip() == "---":
            flush_para(); flush_list(); flush_quote()
            i += 1
            continue

        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            flush_para(); flush_list(); flush_quote()
            level, text = len(m.group(1)), m.group(2).strip()
            if level == 1:
                title = text
            elif level == 2:
                if chapter_no == 0 and "零基础" in text:
                    subtitle = text          # 副标题只上封面，不占正文章节
                    i += 1
                    continue
                chapter_no += 1
                sec_no = 0
                cid = f"section-{chapter_no}"
                toc.append((1, text, cid))
                body.append(f'      <h1 id="{cid}" class="chapter">{inline(text)}</h1>')
            elif level == 3:
                if not chapter_no:      # 标题下的 TL;DR 等，作摘要小标题
                    body.append(f'      <h2 class="sub">{inline(text)}</h2>')
                else:
                    sec_no += 1
                    sid = f"section-{chapter_no}-{sec_no}"
                    toc.append((2, text, sid))
                    body.append(f'      <h2 id="{sid}" class="sub">{inline(text)}</h2>')
            i += 1
            continue

        # 表格
        if line.lstrip().startswith("|"):
            flush_para(); flush_list(); flush_quote()
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i].rstrip())
                i += 1
            rows = [r for r in block if not re.match(r"^\|[\s:\-|]+\|?$", r.strip())]
            if rows:
                head = "".join(f"<th>{inline(c)}</th>" for c in split_row(rows[0]))
                body.append('      <table class="doc-table">')
                body.append(f"        <thead><tr>{head}</tr></thead>")
                body.append("        <tbody>")
                for r in rows[1:]:
                    tds = "".join(f"<td>{inline(c)}</td>" for c in split_row(r))
                    body.append(f"          <tr>{tds}</tr>")
                body.append("        </tbody>")
                body.append("      </table>")
            continue

        # 引用
        if line.lstrip().startswith(">"):
            quote_buf.append(line.lstrip()[1:].strip())
            i += 1
            continue

        # 列表
        m = re.match(r"^\s*[-*]\s+(.*)$", line)
        if m:
            flush_para(); flush_quote()
            if list_type and list_type != "ul":
                flush_list()
            list_type = "ul"
            list_buf.append(m.group(1).strip())
            i += 1
            continue

        m = re.match(r"^\s*\d+\.\s+(.*)$", line)
        if m:
            flush_para(); flush_quote()
            if list_type and list_type != "ol":
                flush_list()
            list_type = "ol"
            list_buf.append(m.group(1).strip())
            i += 1
            continue

        para.append(line)
        i += 1

    flush_para(); flush_list(); flush_quote()

    # 封面副标题与说明：取 H1 后的第一个引用
    for ln in lines:
        if ln.startswith("> ") and not cover_note:
            cover_note = ln[2:].strip()
            break
    # 二级标题“一份写给零基础同事的流程说明”作副标题
    for ln in lines:
        if ln.startswith("## ") and not subtitle:
            subtitle = ln[3:].strip()
            break

    toc_html = []
    for level, text, cid in toc:
        cls = "toc-l1" if level == 1 else "toc-l2"
        toc_html.append(f'        <li class="{cls}"><a href="#{cid}">{inline(text)}</a></li>')

    return "\n".join(body), toc_html, title, subtitle, cover_note


CSS = """
  @page { @bottom-center { content: counter(page); } }
  @page cover { @bottom-center { content: none; } }
  section[role="cover"] { page: cover; }

  :root {
    --fs-cover-eyebrow: 11pt;
    --fs-cover-title: 26pt;
    --fs-cover-subtitle: 13pt;
    --fs-cover-note: 10.5pt;
    --fs-toc-title: 15pt;
    --fs-toc-1: 11pt;
    --fs-toc-2: 10.5pt;
    --fs-h1: 18pt;
    --fs-h2: 14pt;
    --fs-body: 12pt;
    --fs-table: 10.5pt;
    --fs-code: 10.5pt;
    --fs-small: 10pt;
    --ff-heading: "SimHei", "黑体", sans-serif;
    --ff-body: "SimSun", "宋体", serif;
    --ff-code: "Consolas", monospace;
    --spacing-xs: 0.28em;
    --spacing-sm: 0.45em;
    --spacing-md: 0.62em;
    --spacing-lg: 1em;
    --spacing-xl: 1.5em;
    --spacing-cover-top: 4.5cm;
    --spacing-cover-note: 0 1.2cm 2.4em 1.2cm;
    --spacing-cell: 0.36em 0.5em;
    --spacing-callout: 0.5em 0.8em;
    --spacing-callout-box: 0.6em 0 1em 0;
    --spacing-list: 0.2em 0 0.8em 2.2em;
    --spacing-toc-item: 0.28em;
    --spacing-toc-indent: 1.6em;
    --spacing-list-indent: 2.2em;
    --color-primary: #1F4E79;
    --color-text: #1a1a1a;
    --color-muted: #666666;
    --color-border: #d0d7de;
    --color-highlight: #f3f6fa;
    --color-subheading: #14395c;
    --color-code-bg: #f2f2f2;
    --color-toc-2: #333333;
    --border-chapter: 2px solid var(--color-primary);
    --border-callout: 3px solid var(--color-primary);
    --border-cell: 1px solid var(--color-border);
    --margin-page: 2.4cm;
    --page-content-width: 15.6cm;
  }

  body { font-family: var(--ff-body); font-size: var(--fs-body); line-height: 1.7;
         color: var(--color-text); max-width: var(--page-content-width);
         margin-left: auto; margin-right: auto; }

  section[role="cover"] { text-align: center; padding-top: var(--spacing-cover-top); }
  .cover-eyebrow { text-align: center; font-family: var(--ff-heading); font-size: var(--fs-cover-eyebrow);
                   letter-spacing: 0.16em; color: var(--color-primary); margin-bottom: var(--spacing-lg); }
  .cover-title { text-align: center; font-family: var(--ff-heading); font-size: var(--fs-cover-title);
                 font-weight: 700; color: var(--color-primary); line-height: 1.4;
                 margin-bottom: var(--spacing-md); }
  .cover-subtitle { text-align: center; font-size: var(--fs-cover-subtitle); color: var(--color-muted);
                    margin-bottom: var(--spacing-xl); }
  .cover-note { text-align: center; font-size: var(--fs-cover-note); color: var(--color-muted);
                line-height: 1.9; margin: var(--spacing-cover-note); }
  .cover-meta { text-align: center; font-size: var(--fs-small); color: var(--color-muted); }

  nav.doc-toc { padding: var(--spacing-sm) 0; }
  .toc-title { font-family: var(--ff-heading); font-size: var(--fs-toc-title); font-weight: 700;
               color: var(--color-primary); text-align: center; margin-bottom: var(--spacing-lg); }
  .toc-list, .toc-list ol, .toc-list ul { list-style: none; padding-left: 0; margin: 0; }
  .toc-list li { margin-bottom: var(--spacing-toc-item); text-indent: 0; font-size: var(--fs-toc-1); }
  .toc-l1 { font-weight: 700; margin-top: var(--spacing-xs); }
  .toc-l2 { padding-left: var(--spacing-toc-indent); font-weight: 400; font-size: var(--fs-toc-2); color: var(--color-toc-2); }
  .toc-list a { color: var(--color-text); text-decoration: none; }

  h1.chapter { font-family: var(--ff-heading); font-size: var(--fs-h1); font-weight: 700;
               color: var(--color-primary); margin: var(--spacing-xl) 0 var(--spacing-md) 0;
               padding-bottom: var(--spacing-xs); border-bottom: var(--border-chapter); text-indent: 0; }
  h2.sub { font-family: var(--ff-heading); font-size: var(--fs-h2); font-weight: 700;
           color: var(--color-subheading); margin: var(--spacing-lg) 0 var(--spacing-sm) 0; text-indent: 0; }
  p.body-p { text-indent: 2em; margin: 0 0 var(--spacing-md) 0; text-align: justify; }
  ul.doc-list, ol.doc-list { margin: var(--spacing-list); padding-left: var(--spacing-list-indent); }
  ul.doc-list li, ol.doc-list li { margin-bottom: var(--spacing-toc-item); text-align: justify; }

  p.callout { text-indent: 0; margin: var(--spacing-callout-box); padding: var(--spacing-callout);
              border-left: var(--border-callout); background: var(--color-highlight);
              font-size: var(--fs-small); }

  table.doc-table { width: 100%; border-collapse: collapse; margin: var(--spacing-callout-box);
                    font-size: var(--fs-table); }
  table.doc-table th, table.doc-table td { border: var(--border-cell);
              padding: var(--spacing-cell); text-align: left; vertical-align: top; }
  table.doc-table th { background: var(--color-highlight); font-family: var(--ff-heading);
              font-weight: 700; }
  code { font-family: var(--ff-code); font-size: var(--fs-code); background: var(--color-code-bg); }
  strong { font-weight: 700; }
"""


def main() -> int:
    md = SRC.read_text(encoding="utf-8")
    body, toc_items, title, subtitle, cover_note = convert(md)
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="docx-page-size" content="A4">
  <title>{esc(title)}</title>
  <style>{CSS}</style>
</head>
<body>
  <section role="cover">
    <p class="cover-eyebrow">技术说明 · 内部汇报材料</p>
    <h1 class="cover-title">{esc(title)}</h1>
    <p class="cover-subtitle">{esc(subtitle)}</p>
    <p class="cover-note">{esc(cover_note)}</p>
    <p class="cover-meta">低空政策知识库项目 · 2026 年 10 月</p>
  </section>

  <section role="cover">
    <nav class="doc-toc" aria-label="文档目录">
      <p class="toc-title">目 录</p>
      <ol class="toc-list">
{chr(10).join(toc_items)}
      </ol>
    </nav>
  </section>

  <section role="body" data-page-restart="1">
{body}
  </section>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(f"written {OUT} ({len(doc)} chars, chapters={sum(1 for t in toc_items if 'toc-l1' in t)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
