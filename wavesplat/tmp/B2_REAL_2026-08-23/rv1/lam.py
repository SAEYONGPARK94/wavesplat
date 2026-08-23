"""RV-1: lam_p 가 상자 기본파수에 고정돼 있는가 + APERTURE_AXIS 상관 재현. 추정 런 0회."""
import os,sys,json,numpy as np
ROOT='/home/vexit/wavesplat/wavesplat'; os.chdir(ROOT)
for p in (ROOT,os.path.join(ROOT,'scripts'),os.path.join(ROOT,'tmp','B2_REAL_2026-08-23')): sys.path.insert(0,p)
zf=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'zf.json')))
rows=[json.loads(l) for l in open('tmp/B2_REAL_2026-08-23/results.jsonl')]
d={(r['cell'],r['record']):r['nowcast'] for r in rows}
recs=list(zf)
print(f"{'rec':8s}{'lam_p':>8s}{'Lx':>7s}{'Ly':>7s}{'diag_fund':>10s}{'Lmin_fund':>10s}{'lam/Lmin':>9s}")
for r in recs:
    z=zf[r]; Lx,Ly=z['box_x'],z['box_y']
    kdiag=np.hypot(2*np.pi/Lx,2*np.pi/Ly); ldiag=2*np.pi/kdiag
    print(f"{r:8s}{z['lam_p']:8.2f}{Lx:7.1f}{Ly:7.1f}{ldiag:10.2f}{min(Lx,Ly):10.2f}{z['lam_p']/min(Lx,Ly):9.3f}")
from scipy.stats import spearmanr
lam=np.array([zf[r]['lam_p'] for r in recs]); sig=np.array([zf[r]['sg'] for r in recs])
P=np.array([d[('PROP',r)] for r in recs]); S=np.array([d[('SHIP',r)] for r in recs])
Lmin=np.array([min(zf[r]['box_x'],zf[r]['box_y']) for r in recs])
truezf=np.array([zf[r]['zf_nowcast'] for r in recs])
Pn=P/truezf; Sn=S/truezf
print()
for nm,x in (('lam_p',lam),('sigma',sig),('noise/sig',0.02/sig),('boxmin',Lmin),
             ('boxmin/lam',Lmin/lam),('trueZF',truezf)):
    a=spearmanr(x,P); b=spearmanr(x,S); c=spearmanr(x,Pn)
    print(f'{nm:12s} vs PROP rho={a.statistic:+.3f} p={a.pvalue:.4f} | SHIP rho={b.statistic:+.3f} p={b.pvalue:.4f} | PROP/실제ZF rho={c.statistic:+.3f} p={c.pvalue:.4f}')
print()
print('lam_p 순위 == boxmin 순위?', spearmanr(lam,Lmin))
