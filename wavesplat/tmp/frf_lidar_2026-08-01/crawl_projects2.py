
import re, sys, time, urllib.parse, urllib.request
ROOT='https://chldata.erdc.dren.mil/thredds/catalog/frf/projects/catalog.xml'
YEAR=re.compile(r'^(19|20)\d\d$')
seen=set(); OUT=open('projects_tree.txt','w',encoding='utf-8',buffering=1)
def emit(s): OUT.write(s+'\n'); print(s,flush=True)
def get(u):
    last=None
    for i in range(4):
        try:
            with urllib.request.urlopen(u,timeout=90) as r: return r.read().decode('utf-8','replace')
        except Exception as e:
            last=f'{type(e).__name__}: {e}'; time.sleep(1.0*(i+1))
    emit(f'!! FETCH FAILED {u}  {last}'); return None
def refs(x):
    o=[]
    for m in re.finditer(r'<catalogRef\b([^>]*)>', x):
        a=m.group(1); h=re.search(r'xlink:href="([^"]+)"',a); t=re.search(r'xlink:title="([^"]+)"',a)
        if h and t: o.append((h.group(1),t.group(1)))
    return o
def walk(u,p,d):
    if d>7 or u in seen: return
    seen.add(u); x=get(u)
    if x is None: return
    fl=re.findall(r'<dataset[^>]*\bname="([^"]+)"[^>]*\burlPath="([^"]+)"',x)
    if fl: emit(f'{"  "*d}{p.split("/")[-1]:<34}[FILES] {len(fl)} e.g. {fl[0][0]}')
    rs=refs(x); ys=[t for _,t in rs if YEAR.match(t)]
    if ys: emit(f'{"  "*d}{p.split("/")[-1]:<34}[YEARS] {len(ys)}: {min(ys)}-{max(ys)}')
    base=u.rsplit('/',1)[0]+'/'
    for h,t in rs:
        if YEAR.match(t): continue
        emit(f'{"  "*(d+1)}{t:<34}[DIR]')
        walk(urllib.parse.urljoin(base,h), p+'/'+t, d+1)
    for h,t in rs:
        if YEAR.match(t):
            walk(urllib.parse.urljoin(base,h), p+'/'+t, d+1); break
walk(ROOT,'frf/projects',0)
emit(f'nodes visited: {len(seen)}')
OUT.close()

