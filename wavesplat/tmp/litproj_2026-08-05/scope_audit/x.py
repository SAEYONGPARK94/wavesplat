import re, os, sys, json
T='/home/vexit/wavesplat/wavesplat/tmp/litproj_2026-08-05/text'
E='/home/vexit/wavesplat/wavesplat/tmp/litproj_2026-08-05/excerpts'
keys=[f[:-len('.excerpt.txt')] for f in sorted(os.listdir(E))]
out='/tmp/claude-1000/-home-vexit-wavesplat/ccc6698b-ddc9-46f1-a5ba-a4f1b1f46a0b/scratchpad/aud'
for k in keys:
    p=os.path.join(T,k+'.txt')
    s=open(p,encoding='utf-8',errors='replace').read()
    s2=re.sub(r'[ \t]+',' ',s)
    head=s2[:5000]
    # find conclusion-ish sections
    concl=''
    for m in re.finditer(r'\n\s*(?:\d+\.?\s*)?(Conclusion[s]?|Concluding remarks|Summary and conclusion[s]?|Discussion and conclusion[s]?|CONCLUSION[S]?)\b', s2):
        concl+='\n\n<<<@%d %s>>>\n'%(m.start(), m.group(1))+s2[m.start():m.start()+4500]
    open(os.path.join(out,k+'.head.txt'),'w').write(head)
    open(os.path.join(out,k+'.concl.txt'),'w').write(concl if concl else '[NO CONCLUSION HEADER FOUND] tail:\n'+s2[-6000:])
    print(k, len(s2), 'concl_len', len(concl))
