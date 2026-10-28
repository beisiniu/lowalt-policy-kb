from docx import Document

p = r"output/td-20261027-2149/stage3/TQ题库出题流程说明.docx"
d = Document(p)
txt = "\n".join(x.text for x in d.paragraphs)
for t in d.tables:
    for r in t.rows:
        txt += "\n" + " ".join(c.text for c in r.cells)

print("段落:", len(d.paragraphs), "| 表格:", len(d.tables), "| 字符:", len(txt), "| 节:", len(d.sections))
for bad in ["**", "## ", " | ---", "```"]:
    print("  残留", repr(bad), ":", txt.count(bad))
for kw in ["Jaccard", "Top-K", "切片", "53.1%", "56.1%", "38.5%", "CRAG", "ARES", "FreshQA", "BSARD", "KILT"]:
    print("  含", kw, ":", kw in txt)
print("--- 前 8 段 ---")
for x in d.paragraphs[:8]:
    print("  ", x.text[:50])
