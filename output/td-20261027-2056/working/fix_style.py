import re
import sys
from pathlib import Path

p = Path(sys.argv[1])
html = p.read_text(encoding="utf-8")

CSS = """
  @page { @bottom-center { content: "第 " counter(page) " 页"; } }
  @page cover { @bottom-center { content: none; } }
  section[role="cover"] { page: cover; }

  :root {
    --fs-cover-category: 11pt;
    --fs-cover-title: 25pt;
    --fs-cover-subtitle: 14pt;
    --fs-cover-meta: 10.5pt;
    --fs-toc-title: 15pt;
    --fs-toc-1: 11.5pt;
    --fs-toc-2: 10.5pt;
    --fs-h1: 16pt;
    --fs-h2: 13pt;
    --fs-body: 12pt;
    --fs-small: 10.5pt;
    --fs-code-block: 8.5pt;
    --fs-code-inline: 9.5pt;
    --fs-summary-title: 13pt;
    --fs-badge: 9pt;

    --ff-body: "SimSun", "宋体", serif;
    --ff-head: "SimHei", "黑体", sans-serif;
    --ff-code: "Consolas", "Courier New", monospace;

    --color-text: #1a1a1a;
    --color-primary: #1f3a5f;
    --color-accent: #b3261e;
    --color-muted: #5c6670;
    --color-border: #c9d1d9;
    --color-bg: #f4f6f9;
    --color-surface: #fafbfc;
    --color-quote-bg: #fbf3f2;
    --color-code-bg: #f6f8fa;

    --spacing-paragraph: 0.55em;
    --spacing-section: 1.2em;
    --spacing-none: 0;
    --spacing-hair: 0.1em;
    --spacing-xs: 0.2em;
    --spacing-sm: 0.35em;
    --spacing-md: 0.5em;
    --spacing-lg: 0.7em;
    --spacing-xl: 1.0em;
    --spacing-2xl: 1.6em;
    --spacing-3xl: 2.2em;
    --spacing-4xl: 2.6em;
    --spacing-5xl: 3.4em;
    --spacing-6xl: 5.2em;
    --spacing-indent: 1.6em;
    --spacing-indent-sm: 1.0em;
    --spacing-tight: 0.18em;
    --spacing-rule: 0.25em;
    --spacing-card: 0.9em;
    --spacing-list: 0.7em;

    --page-content-width: 15.6cm;
  }

  body {
    font-family: var(--ff-body);
    font-size: var(--fs-body);
    line-height: 1.7;
    color: var(--color-text);
    max-width: var(--page-content-width);
    margin-left: auto;
    margin-right: auto;
  }

  section[role="cover"] { text-align: center; }
  .cover-category { font-family: var(--ff-head); font-size: var(--fs-cover-category); color: var(--color-muted); letter-spacing: 0.18em; text-align: center; margin-top: var(--spacing-6xl); margin-bottom: var(--spacing-md); }
  .cover-rule-text { font-family: var(--ff-head); font-size: var(--fs-badge); color: var(--color-muted); text-align: center; margin-bottom: var(--spacing-3xl); }
  .cover-title { font-family: var(--ff-head); font-size: var(--fs-cover-title); font-weight: bold; color: var(--color-primary); text-align: center; line-height: 1.35; margin-bottom: var(--spacing-md); }
  .cover-subtitle { font-family: var(--ff-head); font-size: var(--fs-cover-subtitle); color: var(--color-muted); text-align: center; margin-bottom: var(--spacing-4xl); }
  .cover-meta { font-size: var(--fs-cover-meta); color: var(--color-muted); text-align: center; line-height: 2.0; }
  .cover-warn { font-size: var(--fs-badge); color: var(--color-muted); text-align: center; margin-top: var(--spacing-5xl); }

  .toc-title { font-family: var(--ff-head); font-size: var(--fs-toc-title); font-weight: bold; color: var(--color-primary); text-align: center; margin-bottom: var(--spacing-xl); letter-spacing: 0.1em; }
  .toc-list, .toc-list ol, .toc-list ul { list-style: none; list-style-type: none; padding-left: var(--spacing-none); }
  .toc-list > li { margin-bottom: var(--spacing-sm); text-indent: var(--spacing-none); font-family: var(--ff-head); font-size: var(--fs-toc-1); }
  .toc-list a { color: var(--color-text); text-decoration: none; }
  .toc-sublist { padding-left: var(--spacing-indent); margin-top: var(--spacing-xs); }
  .toc-sublist li.toc-sub { font-family: var(--ff-body); font-size: var(--fs-toc-2); color: var(--color-muted); margin-bottom: var(--spacing-hair); text-indent: var(--spacing-none); }
  .toc-sublist a { color: var(--color-muted); }

  h1.chapter {
    font-family: var(--ff-head); font-size: var(--fs-h1); font-weight: bold;
    color: var(--color-primary); margin-top: var(--spacing-section); margin-bottom: var(--spacing-lg);
    padding-bottom: var(--spacing-rule); border-bottom: 2px solid var(--color-primary);
    text-indent: var(--spacing-none); text-align: left;
  }
  h2.section {
    font-family: var(--ff-head); font-size: var(--fs-h2); font-weight: bold;
    color: var(--color-primary); margin-top: var(--spacing-xl); margin-bottom: var(--spacing-md);
    text-indent: var(--spacing-none); text-align: left;
  }
  p { text-indent: 2em; margin-bottom: var(--spacing-paragraph); text-align: justify; }
  p.quote { border-left: 3px solid var(--color-accent); background: var(--color-quote-bg); padding: var(--spacing-md) var(--spacing-lg); text-indent: var(--spacing-none); color: var(--color-text); }
  p.divider { border-bottom: 1px solid var(--color-border); text-indent: var(--spacing-none); margin-top: var(--spacing-md); margin-bottom: var(--spacing-md); }

  ul.bullet, ol.num { margin-top: var(--spacing-xs); margin-bottom: var(--spacing-list); margin-left: var(--spacing-indent); padding-left: var(--spacing-indent-sm); }
  ul.bullet li, ol.num li { margin-bottom: var(--spacing-tight); text-indent: var(--spacing-none); text-align: justify; }

  table.data-table {
    width: 100%; border-collapse: collapse; margin-top: var(--spacing-md); margin-bottom: var(--spacing-card);
    font-size: var(--fs-small); font-family: var(--ff-body);
  }
  table.data-table th {
    background: var(--color-bg); font-family: var(--ff-head); font-weight: bold;
    border: 1px solid var(--color-border); padding: var(--spacing-sm) var(--spacing-md); text-align: left;
  }
  table.data-table td { border: 1px solid var(--color-border); padding: var(--spacing-sm) var(--spacing-md); text-indent: var(--spacing-none); vertical-align: top; }
  table.data-table tr:nth-child(even) td { background: var(--color-surface); }

  pre.code {
    font-family: var(--ff-code); font-size: var(--fs-code-block); line-height: 1.35;
    background: var(--color-code-bg); border: 1px solid var(--color-border);
    padding: var(--spacing-md) var(--spacing-lg);
    margin-top: var(--spacing-sm); margin-bottom: var(--spacing-card);
    white-space: pre-wrap; text-indent: var(--spacing-none);
  }
  code { font-family: var(--ff-code); font-size: var(--fs-code-inline); }

  .executive-summary { border-left: 4px solid var(--color-primary); background: var(--color-bg); padding: var(--spacing-lg) var(--spacing-card); margin-bottom: var(--spacing-section); }
  .summary-title { font-family: var(--ff-head); font-size: var(--fs-summary-title); font-weight: bold; color: var(--color-primary); text-indent: var(--spacing-none); margin-bottom: var(--spacing-md); }
  .summary-badge { font-size: var(--fs-badge); color: var(--color-muted); text-indent: var(--spacing-none); margin-bottom: var(--spacing-md); }
"""

new = re.sub(r"<style>.*?</style>", "<style>" + CSS + "</style>", html, flags=re.S)
p.write_text(new, encoding="utf-8")
print("style replaced, len:", len(new))
