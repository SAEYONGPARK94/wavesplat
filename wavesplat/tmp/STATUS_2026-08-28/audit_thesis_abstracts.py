#!/usr/bin/env python3
"""논문 정합 감사 #6 — **국문 초록 대 영문 초록**의 주장 원자 대조.

런 0회. 두 초록에서 계수(a/b)·퍼센트·p 를 뽑아 한쪽에만 있는 것을 보고한다.
논문의 주석이 반복해서 *"so the two abstracts do not diverge"* 라 적었으므로
그 약속을 기계로 검사한다.
사용: python3 tmp/STATUS_2026-08-28/audit_thesis_abstracts.py
"""
import re, hashlib, pathlib, collections

TEX = pathlib.Path("/home/vexit/wavesplat/wavesplat/thesis/thesis_main.tex")
lines = TEX.read_text(encoding="utf-8").splitlines()
print(f"# tex md5 = {hashlib.md5(TEX.read_bytes()).hexdigest()}")

def span(startpat, endpat):
    s = e = None
    for i, l in enumerate(lines, 1):
        if s is None and re.search(startpat, l): s = i
        elif s is not None and re.search(endpat, l): e = i; break
    return s, e

KS, KE = span(r'\\chapter\*\{초\\quad 록\}', r'\\chapter\*\{Abstract\}')
ES, EE = span(r'\\chapter\*\{Abstract\}',   r'^\\newpage\s*$|tableofcontents')
print(f"# 국문초록 {KS}-{KE} · 영문초록 {ES}-{EE}")

PAT = {"계수": re.compile(r'(?<![\d.])(\d{1,3})\s*/\s*(\d{1,3})(?![\d.])'),
       "퍼센트": re.compile(r'([+-]?\d{1,3}(?:\.\d{1,2})?)\s*\\%'),
       "p값":  re.compile(r'(\d(?:\.\d+)?)\s*\\times\s*10\^\{(-?\d+)\}')}

def atoms(a, b):
    out = collections.defaultdict(set)
    for i in range(a, b):
        l = lines[i-1]
        if l.lstrip().startswith('%'): continue
        l = re.sub(r'(?<!\\)%.*$', '', l)
        for name, p in PAT.items():
            for m in p.finditer(l):
                g = m.groups()
                if name == "계수":
                    if int(g[1]) == 0 or int(g[0]) > int(g[1]): continue
                    out[name].add(f"{g[0]}/{g[1]}")
                elif name == "퍼센트":
                    out[name].add(f"{float(g[0]):+g}%")
                else:
                    out[name].add(f"{g[0]}e{g[1]}")
    return out

K, E = atoms(KS, KE), atoms(ES, EE)
for name in PAT:
    ko, en = K[name], E[name]
    print(f"\n== {name}  국문 {len(ko)} · 영문 {len(en)}")
    only_k, only_e = sorted(ko - en), sorted(en - ko)
    print(f"   국문에만: {only_k}")
    print(f"   영문에만: {only_e}")
