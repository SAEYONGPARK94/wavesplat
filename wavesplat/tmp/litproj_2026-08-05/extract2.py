"""Add the repo-root ref_paper/ tree (scope of search = scope of claim)."""
import os, json, fitz
ROOT = '/home/vexit/wavesplat/ref_paper'
BASE = '/home/vexit/wavesplat/wavesplat/tmp/litproj_2026-08-05'
OUT = os.path.join(BASE, 'text')
WAVE_MARKERS = ['significant wave height', 'jonswap', 'pierson', 'wave spectrum',
                'peak period', 'directional spread', 'sea state', 'wave field',
                'surface elevation', 'wave elevation']
rows = []
for dp, dn, fns in os.walk(ROOT):
    for fn in sorted(fns):
        if not fn.lower().endswith('.pdf'):
            continue
        p = os.path.join(dp, fn)
        rel = 'ROOT__' + os.path.relpath(p, ROOT)
        key = rel.replace('/', '__').replace(' ', '_').replace('.pdf', '.txt')
        dst = os.path.join(OUT, key)
        if os.path.exists(dst):
            txt = open(dst, encoding='utf-8').read()
        else:
            try:
                doc = fitz.open(p); txt = '\n'.join(pg.get_text() for pg in doc); doc.close()
            except Exception as e:
                txt = ''; print('FAIL', rel, e)
            open(dst, 'w', encoding='utf-8').write(txt)
        low = txt.lower()
        rows.append({'rel': rel, 'txt': key, 'chars': len(txt),
                     'wave_score': sum(low.count(m) for m in WAVE_MARKERS)})
rows.sort(key=lambda r: -r['wave_score'])
json.dump(rows, open(os.path.join(BASE, 'index_root.json'), 'w'), indent=2)
for r in rows:
    print(f"{r['wave_score']:>6}  {r['chars']:>7}  {r['rel']}")
