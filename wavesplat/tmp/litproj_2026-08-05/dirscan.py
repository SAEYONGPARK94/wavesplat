import os, re
EXC = 'tmp/litproj_2026-08-05/excerpts'
KEYS = ['long-crested', 'short-crested', 'unidirectional', 'uni-directional',
        'multidirectional', 'multi-directional', 'directional spread',
        'spreading function', 'spreading parameter', 'directional distribution',
        'cos2s', 'cos^2s', 'cosine-power', 'irregular waves', '2D wave', '3D wave']
for fn in sorted(os.listdir(EXC)):
    txt = open(os.path.join(EXC, fn), encoding='utf-8').read()
    low = txt.lower()
    found = []
    for k in KEYS:
        start = 0
        while True:
            i = low.find(k, start)
            if i < 0:
                break
            found.append(txt[max(0, i-130):i+150].replace('\n', ' '))
            start = i + len(k)
            if len(found) > 8:
                break
        if len(found) > 8:
            break
    if found:
        print(f"════════ {fn}")
        for s in found[:6]:
            print("   …" + s + "…")
