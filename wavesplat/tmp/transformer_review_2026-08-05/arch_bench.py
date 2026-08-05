"""Standalone architectural microbenchmark. No wavesplat import, no physics, no training.
Reconstructs the two shipped/orphaned set-predictor architectures at their exact dims and
times a forward pass. Purpose: cost of 500-step diffusion sampling vs a single-forward
DETR-style set decoder."""
import time, torch, torch.nn as nn
torch.set_num_threads(1)
print('cuda', torch.cuda.is_available())
DEV='cpu'

class CondEncoder(nn.Module):           # d3_beamform_cond_diffusion.py:44
    def __init__(s,d=128):
        super().__init__()
        blk=lambda i,o,st=1: nn.Sequential(nn.Conv2d(i,o,3,st,1),nn.GroupNorm(8,o),nn.GELU())
        s.net=nn.Sequential(blk(1,d),blk(d,d,2),blk(d,d),blk(d,d,2),blk(d,d))
        s.pos=nn.Parameter(torch.randn(1,16*16,d)*0.02)
    def forward(s,B): return s.net(B[:,None]).flatten(2).transpose(1,2)+s.pos

class CondSetDenoiser(nn.Module):       # d3_beamform_cond_diffusion.py:60
    def __init__(s,d=128,heads=4,layers=4):
        super().__init__(); s.d=d; s.enc=CondEncoder(d)
        s.inp=nn.Linear(3,d); s.temb=nn.Sequential(nn.Linear(d,d),nn.SiLU(),nn.Linear(d,d))
        s.blocks=nn.ModuleList([nn.ModuleDict(dict(
            ln1=nn.LayerNorm(d),attn=nn.MultiheadAttention(d,heads,batch_first=True),
            lnc=nn.LayerNorm(d),cross=nn.MultiheadAttention(d,heads,batch_first=True),
            ln2=nn.LayerNorm(d),
            ff=nn.Sequential(nn.Linear(d,2*d),nn.SiLU(),nn.Linear(2*d,d)))) for _ in range(layers)])
        s.out=nn.Linear(d,3)
    def forward(s,x,temb,cond):
        h=s.inp(x)+temb[:,None,:]
        for b in s.blocks:
            hn=b['ln1'](h); h=h+b['attn'](hn,hn,hn)[0]
            hc=b['lnc'](h); h=h+b['cross'](hc,cond,cond)[0]
            h=h+b['ff'](b['ln2'](h))
        return s.out(h)

class Encoder96(nn.Module):             # d3_beamform_detr.py:37
    def __init__(s,d=96):
        super().__init__()
        blk=lambda i,o,st=1: nn.Sequential(nn.Conv2d(i,o,3,st,1),nn.GroupNorm(8,o),nn.GELU())
        s.net=nn.Sequential(blk(1,d),blk(d,d),blk(d,d,2),blk(d,d),blk(d,d,2),blk(d,d))
        s.pos=nn.Parameter(torch.randn(1,16*16,d)*0.02)
    def forward(s,B): return s.net(B[:,None]).flatten(2).transpose(1,2)+s.pos

class DETRSet(nn.Module):               # d3_beamform_detr.py:53
    def __init__(s,d=96,nq=48,layers=4,heads=4):
        super().__init__(); s.enc=Encoder96(d)
        s.query=nn.Parameter(torch.randn(nq,d)*0.02)
        dl=nn.TransformerDecoderLayer(d,heads,4*d,batch_first=True,activation='gelu')
        s.dec=nn.TransformerDecoder(dl,layers); s.head=nn.Linear(d,4)
    def forward(s,B):
        m=s.enc(B); q=s.query[None].expand(B.shape[0],-1,-1)
        return s.head(s.dec(q,m))

def npar(m): return sum(p.numel() for p in m.parameters())
def timeit(fn,rep=20):
    for _ in range(3): fn()
    ts=[]
    for _ in range(rep):
        t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    ts.sort(); return ts[len(ts)//2]

T, NSAMP, M = 500, 4, 32
Bmap=torch.randn(1,64,64)
cd_=CondSetDenoiser().eval(); dt_=DETRSet().eval()
print('params CondSetDenoiser', npar(cd_), ' DETRSet', npar(dt_))
with torch.no_grad():
    cond=cd_.enc(Bmap).expand(NSAMP,-1,-1)
    x=torch.randn(NSAMP,M,3); temb=torch.randn(NSAMP,128)
    t_enc=timeit(lambda: cd_.enc(Bmap))
    t_step=timeit(lambda: cd_(x,temb,cond), rep=30)
    t_detr=timeit(lambda: dt_(Bmap), rep=30)
print(f'cond encoder            : {t_enc*1e3:8.3f} ms  (once)')
print(f'denoiser 1 reverse step : {t_step*1e3:8.3f} ms  (batch n={NSAMP})')
print(f'-> full sample T={T}    : {(t_enc+T*t_step):8.3f} s')
print(f'DETR single forward     : {t_detr*1e3:8.3f} ms  (enc+dec, one shot)')
print(f'ratio diffusion/DETR    : {(t_enc+T*t_step)/t_detr:8.1f} x')
