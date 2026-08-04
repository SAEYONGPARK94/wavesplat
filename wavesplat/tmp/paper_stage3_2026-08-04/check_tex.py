import re, collections
src = open('paper/wavesplat.tex').read()
labels = set(re.findall(r'\\label\{([^}]+)\}', src))
refs = set(re.findall(r'\\(?:ref|eqref|autoref)\{([^}]+)\}', src))
missing = sorted(refs - labels)
unused = sorted(labels - refs)
print('MISSING label for \\ref:', missing or 'none')
print('labels never referenced:', unused or 'none')

# citation keys vs bib
cites = set()
for m in re.findall(r'\\cite[a-z]*\{([^}]+)\}', src):
    cites |= {c.strip() for c in m.split(',')}
bib = set(re.findall(r'@\w+\{([^,\s]+)\s*,', open('paper/references.bib').read()))
print('cited but not in bib:', sorted(cites - bib) or 'none')

# environment balance
opens = collections.Counter(re.findall(r'\\begin\{(\w+\*?)\}', src))
closes = collections.Counter(re.findall(r'\\end\{(\w+\*?)\}', src))
bad = {k: (opens[k], closes[k]) for k in set(opens) | set(closes) if opens[k] != closes[k]}
print('unbalanced envs:', bad or 'none')

# tabular column count check
for m in re.finditer(r'\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}', src, re.S):
    spec, body = m.group(1), m.group(2)
    ncol = len(re.findall(r'[lcrp]', re.sub(r'\{[^}]*\}', '', spec)))
    line_no = src[:m.start()].count('\n') + 1
    for row in body.split('\\\\'):
        row = re.sub(r'%.*', '', row)
        if '&' not in row: continue
        if any(t in row for t in ('multicolumn', 'cmidrule', 'toprule', 'midrule', 'bottomrule')): continue
        cells = row.count('&') + 1
        if cells != ncol:
            print(f'  tabular@{line_no} spec={spec!r} ncol={ncol} got {cells}: {row.strip()[:70]}')
