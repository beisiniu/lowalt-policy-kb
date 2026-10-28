#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时：抽取几篇政策里含数字/条件的条款，用来出题。用完即删。"""
import glob
import re
import sys

keys = sys.argv[1:]
for f in sorted(glob.glob("data/gd_policies/*.txt")):
    name = f.replace("\\", "/").split("/")[-1]
    if not any(k in name for k in keys):
        continue
    body = open(f, encoding="utf-8").read().split("-" * 20, 1)[-1]
    print("=" * 78)
    print(name)
    hits = [s.strip() for s in re.split(r"[。\n]", body) if re.search(r"\d", s) and len(s) > 18]
    for s in hits[:10]:
        print("  -", s[:110])
