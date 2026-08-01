"""Recursive crawl of the FRF THREDDS catalog — full tree, no manual clicking.

THREDDS serves machine-readable catalog.xml alongside catalog.html. We walk
catalogRef nodes depth-first and record every dataset leaf. Year folders are
sampled (not fully expanded) to keep the crawl bounded.
"""
import re
import sys
import time
import urllib.request
from collections import defaultdict

BASE = 'https://chldata.erdc.dren.mil/thredds/catalog/frf/catalog.xml'
ROOT = 'https://chldata.erdc.dren.mil/thredds/catalog/'
MAX_DEPTH = 8
YEAR = re.compile(r'^(19|20)\d\d$')

seen = set()
tree = []           # (depth, path, kind, extra)
leaf_files = defaultdict(list)


def get(url, tries=3):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception as e:
            if i == tries - 1:
                return f'<!--ERR {type(e).__name__}: {e}-->'
            time.sleep(2)


def walk(url, path, depth):
    if depth > MAX_DEPTH or url in seen:
        return
    seen.add(url)
    xml = get(url)
    # catalogRef -> subfolder;  dataset with urlPath -> actual file
    refs = re.findall(r'<catalogRef[^>]*xlink:href="([^"]+)"[^>]*xlink:title="([^"]+)"', xml)
    if not refs:
        refs = [(h, t) for t, h in re.findall(
            r'<catalogRef[^>]*xlink:title="([^"]+)"[^>]*xlink:href="([^"]+)"', xml)]
    files = re.findall(r'<dataset[^>]*name="([^"]+)"[^>]*urlPath="([^"]+)"', xml)
    if files:
        leaf_files[path] = [f[0] for f in files[:3]]
        tree.append((depth, path, 'FILES', f'{len(files)} datasets, e.g. {files[0][0]}'))
    yearlike = [t for _, t in refs if YEAR.match(t)]
    if yearlike:
        tree.append((depth, path, 'YEARS', f'{len(yearlike)} year folders: '
                                           f'{min(yearlike)}-{max(yearlike)}'))
    for href, title in refs:
        if YEAR.match(title):
            continue                      # don't expand every year
        base = url.rsplit('/', 1)[0] + '/'
        child = urllib.parse.urljoin(base, href)
        tree.append((depth + 1, f'{path}/{title}', 'DIR', ''))
        walk(child, f'{path}/{title}', depth + 1)
    # expand exactly one year folder per node so we can see file naming
    for href, title in refs:
        if YEAR.match(title):
            base = url.rsplit('/', 1)[0] + '/'
            child = urllib.parse.urljoin(base, href)
            walk(child, f'{path}/{title}', depth + 1)
            break


import urllib.parse
walk(BASE, 'frf', 0)

with open('thredds_tree.txt', 'w', encoding='utf-8') as fh:
    for depth, path, kind, extra in tree:
        line = f'{"  " * depth}{path.split("/")[-1]:<34} [{kind}] {extra}'
        fh.write(line + '\n')
        print(line)

lid = [t for t in tree if 'lidar' in t[1].lower() or 'scan' in t[1].lower()]
print('\n\n=== LIDAR / SCAN NODES ===')
with open('thredds_lidar_nodes.txt', 'w', encoding='utf-8') as fh:
    for depth, path, kind, extra in lid:
        line = f'{path}  [{kind}] {extra}'
        fh.write(line + '\n')
        print(line)
print(f'\ntotal nodes visited: {len(seen)}')
