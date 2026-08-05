"""Mine the EVALUATION sea states out of the wave-domain papers.

Emits one excerpt file per paper containing every sentence that carries a
sea-state parameter, so the values can be read and tabulated by hand.
No inference here -- this step only locates, it does not interpret.
"""
import os, re, json

BASE = '/home/vexit/wavesplat/wavesplat/tmp/litproj_2026-08-05'
TXT = os.path.join(BASE, 'text')
EXC = os.path.join(BASE, 'excerpts')
os.makedirs(EXC, exist_ok=True)

idx = json.load(open(os.path.join(BASE, 'index.json')))
idx += json.load(open(os.path.join(BASE, 'index_root.json')))
papers = [r for r in idx if r['wave_score'] >= 5]
# drop md5 duplicates by (chars, wave_score)
seen, uniq = set(), []
for r in sorted(papers, key=lambda r: -r['wave_score']):
    k = (r['chars'], r['wave_score'])
    if k in seen:
        print(f"  [dup skipped] {r['rel']}")
        continue
    seen.add(k); uniq.append(r)

PATTERNS = [
    r'significant wave height', r'\bH_?s\b', r'\bHs\s*=', r'\bH1/3\b',
    r'peak period', r'\bT_?p\b', r'peak frequency', r'\bf_?p\b',
    r'JONSWAP', r'Pierson', r'Bretschneider', r'\bgamma\b', r'γ\s*=',
    r'spreading', r'directional spread', r'cos\^?2?s', r'short-crested',
    r'long-crested', r'unidirectional', r'uni-directional', r'multidirectional',
    r'water depth', r'deep water', r'\bdepth of\b',
    r'steepness', r'\bk_?p\b', r'wave age', r'sea state', r'Beaufort',
]
RX = re.compile('|'.join(PATTERNS), re.I)
NUM = re.compile(r'\d')

summary = []
for r in uniq:
    txt = open(os.path.join(TXT, r['txt']), encoding='utf-8').read()
    txt = re.sub(r'-\n', '', txt)              # de-hyphenate line breaks
    flat = re.sub(r'\s+', ' ', txt)
    sents = re.split(r'(?<=[.;:])\s+', flat)
    hits = [s.strip() for s in sents if RX.search(s) and NUM.search(s) and len(s) < 700]
    # collapse near-duplicates
    out, seen_s = [], set()
    for s in hits:
        k = s[:120]
        if k in seen_s:
            continue
        seen_s.add(k); out.append(s)
    dst = os.path.join(EXC, r['txt'].replace('.txt', '.excerpt.txt'))
    with open(dst, 'w', encoding='utf-8') as f:
        f.write(f"### {r['rel']}  (chars={r['chars']}, wave_score={r['wave_score']})\n\n")
        f.write('\n\n'.join(out))
    summary.append((r['rel'], len(out)))
    print(f"{len(out):>4} excerpts  {r['rel']}")

print(f"\n{len(uniq)} unique wave-domain papers")
