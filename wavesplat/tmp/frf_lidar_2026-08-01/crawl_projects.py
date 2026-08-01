"""Focused, fail-loud crawl of frf/projects/** — the subtree the first crawl silently
dropped. Differences from crawl_thredds.py:
  - errors are RAISED into the output, never swallowed
  - polite delay between requests (the .mil host appears to throttle)
  - expands every year folder's first entry so file naming is visible
"""
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = 'https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/catalog.xml'
YEAR = re.compile(r'^(19|20)\d\d$')
DELAY = 0.6
MAX_DEPTH = 7

seen = set()
lines = []
errors = []


def get(url):
    last = None
    for i in range(4):
        try:
            time.sleep(DELAY)
            with urllib.request.urlopen(url, timeout=90) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            last = f'{type(e).__name__}: {e}'
            time.sleep(1.5 * (i + 1))
    errors.append((url, last))
    return None


def refs_of(xml):
    out = []
    for m in re.finditer(r'<catalogRef\b([^>]*)>', xml):
        a = m.group(1)
        h = re.search(r'xlink:href="([^"]+)"', a)
        t = re.search(r'xlink:title="([^"]+)"', a)
        if h and t:
            out.append((h.group(1), t.group(1)))
    return out


def files_of(xml):
    return re.findall(r'<dataset[^>]*\bname="([^"]+)"[^>]*\burlPath="([^"]+)"', xml)


def walk(url, path, depth):
    if depth > MAX_DEPTH or url in seen:
        return
    seen.add(url)
    xml = get(url)
    if xml is None:
        lines.append(f'{"  "*depth}{path.split("/")[-1]:<32} [!! FETCH FAILED]')
        return
    fl = files_of(xml)
    if fl:
        lines.append(f'{"  "*depth}{path.split("/")[-1]:<32} [FILES] {len(fl)} — e.g. {fl[0][0]}')
    rs = refs_of(xml)
    years = [t for _, t in rs if YEAR.match(t)]
    if years:
        lines.append(f'{"  "*depth}{path.split("/")[-1]:<32} [YEARS] {len(years)}: {min(years)}-{max(years)}')
    base = url.rsplit('/', 1)[0] + '/'
    for href, title in rs:
        if YEAR.match(title):
            continue
        lines.append(f'{"  "*(depth+1)}{title:<32} [DIR]')
        walk(urllib.parse.urljoin(base, href), f'{path}/{title}', depth + 1)
    for href, title in rs:                     # one year sample only
        if YEAR.match(title):
            walk(urllib.parse.urljoin(base, href), f'{path}/{title}', depth + 1)
            break


walk(ROOT, 'frf/projects', 0)
txt = '\n'.join(lines)
open('projects_tree.txt', 'w', encoding='utf-8').write(txt + '\n')
print(txt)
print(f'\nnodes visited: {len(seen)}')
if errors:
    print(f'\n!! {len(errors)} FETCH FAILURES:')
    for u, e in errors[:20]:
        print(f'   {u}\n     {e}')
else:
    print('\nno fetch failures')
hits = [l for l in lines if re.search(r'lidar|scan', l, re.I)]
print('\n=== LIDAR / SCAN under projects ===')
print('\n'.join(hits) if hits else '(none)')
