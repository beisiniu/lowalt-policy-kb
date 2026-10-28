from docx import Document

p = r"output/td-20261027-2149/stage3/TQ题库出题流程说明.docx"
d = Document(p)
print("=== 9-26 段（目录区）===")
for i, x in enumerate(d.paragraphs[8:26], start=8):
    print(f"{i:3} {x.text[:56]}")
print()
print("=== 表格 1（元数据）===")
t = d.tables[0]
for r in t.rows[:4]:
    print("  ", " | ".join(c.text[:20] for c in r.cells))
print()
print("=== 表格 4（改造动作）行数 ===", len(d.tables[3].rows))
for r in d.tables[3].rows[:3]:
    print("  ", " | ".join(c.text[:18] for c in r.cells))
