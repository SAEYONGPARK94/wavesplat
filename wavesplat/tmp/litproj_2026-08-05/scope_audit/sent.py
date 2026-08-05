import re,sys,os
T='/home/vexit/wavesplat/wavesplat/tmp/litproj_2026-08-05/text'
pat=re.compile(r'long.?crested|short.?crested|uni.?directional|multi.?directional|directional spread\w*|spreading function|directional spectrum|directionality',re.I)
for f in sys.argv[1:]:
    s=open(os.path.join(T,f+'.txt'),encoding='utf-8',errors='replace').read()
    s=re.sub(r'-\s*\n\s*','',s); s=re.sub(r'\s+',' ',s)
    n=len(s)
    print('#'*20, f, 'len', n)
    for m in pat.finditer(s):
        a=max(0,m.start()-330); b=min(n,m.end()+330)
        print('  [@%.2f] ...%s...' % (m.start()/n, s[a:b].strip()))
        print()
