"""Full-text extraction of the 73-paper corpus + wave-domain screen.

R5 next-step #2: find the SEA STATES the literature actually tests on, so the
CDIP support projection becomes a FIELD audit rather than a self-audit.
"""
import os, sys, json
import fitz

ROOT = '/home/vexit/wavesplat/wavesplat/ref_paper'
BASE = '/home/vexit/wavesplat/wavesplat/tmp/litproj_2026-08-05'
OUT = os.path.join(BASE, 'text')
os.makedirs(OUT, exist_ok=True)

pdfs = []
for dirpath, dirnames, filenames in os.walk(ROOT):
    if 'graphify-out' in dirpath:
        continue
    for fn in filenames:
        if fn.lower().endswith('.pdf'):
            pdfs.append(os.path.join(dirpath, fn))
pdfs.sort()

WAVE_MARKERS = ['significant wave height', 'jonswap', 'pierson', 'wave spectrum',
                'peak period', 'directional spread', 'sea state', 'wave field',
                'surface elevation', 'wave elevation']

index = []
for p in pdfs:
    rel = os.path.relpath(p, ROOT)
    key = rel.replace('/', '__').replace('.pdf', '.txt')
    dst = os.path.join(OUT, key)
    if os.path.exists(dst):
        txt = open(dst, encoding='utf-8').read()
    else:
        try:
            doc = fitz.open(p)
            txt = '\n'.join(page.get_text() for page in doc)
            doc.close()
        except Exception as e:
            txt = ''
            print(f'FAIL {rel}: {e}', file=sys.stderr)
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(txt)
    low = txt.lower()
    hits = {m: low.count(m) for m in WAVE_MARKERS}
    index.append({'rel': rel, 'txt': key, 'chars': len(txt),
                  'wave_score': sum(hits.values()), 'hits': hits})

index.sort(key=lambda r: -r['wave_score'])
with open(os.path.join(BASE, 'index.json'), 'w') as f:
    json.dump(index, f, indent=2)

print(f'{len(index)} pdfs extracted')
print(f'{"score":>6}  {"chars":>7}  path')
for r in index:
    if r['wave_score'] > 0:
        print(f'{r["wave_score"]:>6}  {r["chars"]:>7}  {r["rel"]}')
print('--- zero-score (non wave-domain) ---')
print(', '.join(r['rel'] for r in index if r['wave_score'] == 0))
