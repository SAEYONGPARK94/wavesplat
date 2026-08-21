import sys, math, statistics as st
sys.path.insert(0, "/home/vexit/wavesplat/wavesplat/tmp/Rx6_channel_curve_2026-08-07")
from audit_con_load import *

TWOPI = 2 * math.pi
wrap = lambda x: (x + math.pi) % TWOPI - math.pi
rms = lambda v: math.sqrt(sum(x * x for x in v) / len(v))
AUTOb = block("AUTO", "nRMSE")
EPS_PRIM = [e for e in EPS if AUTOb[e] < NULLS[e]["zero_null_tracking_nRMSE_mean"]]
CH = [f"{c}-S{i}" for c in "WP" for i in "12345"]


def Dcell(c, ep, seed):
    r = IDX[(c, ep, seed)]; b = IDX[("AUTO-NOTRACK", ep, seed)]
    n = len(b["w"])
    if c.startswith("W"):
        return rms([r["w"][i] - b["w"][i] for i in range(n)])
    return rms([b["w"][i] * wrap(r["phi"][i] - b["phi"][i]) for i in range(n)])


print("K1. reconciling the reported D numbers (which aggregation gives 0.111 / 0.107 ?)")
for c in ["W-S1", "P-S1"]:
    allv = [Dcell(c, e, s) for e in EPS for s in SEEDS]
    medsd = [st.median([Dcell(c, e, s) for s in SEEDS]) for e in EPS]
    print(f"  {c}: mean-all={sum(allv)/len(allv):.4f}  median-all={st.median(allv):.4f}  "
          f"blockmed-of-seedmed={st.median(medsd):.4f}  mean-of-seedmed={sum(medsd)/len(medsd):.4f}")

print()
print("K2. SATURATION: lr multiplier vs realised displacement multiplier (direct-field proxy = D_lin)")
for ch, lrs in [("W", [5e-6, 5e-5, 2.205e-4, 5e-4, 5e-3]), ("P", [1.1339e-4, 1.1339e-3, 5.0005e-3, 1.13391e-2, 1.13391e-1])]:
    order = ["S5", "S4", "S3", "S2", "S1"]
    Ds = [st.median([st.median([Dcell(f"{ch}-{o}", e, s) for s in SEEDS]) for e in EPS]) for o in order]
    print(f"  {ch}: ", end="")
    for j in range(1, 5):
        print(f"{order[j-1]}->{order[j]} lr x{lrs[j]/lrs[j-1]:.2f} D x{Ds[j]/Ds[j-1]:.2f}   ", end="")
    print()

print()
print("K3. CONFIG FOLD (n=8 each) for NOTRACK - AUTO   [PREREG 4.6: no significance, no effect size]")
for bf in (4, 5):
    for nc in (15, 30):
        sub = [e for e in EPS_PRIM if f"BF{bf}_{nc}comp" in e]
        d = paired("AUTO-NOTRACK", "AUTO", eps=sub)
        n, pos, p = sign_test(d)
        print(f"   BF{bf}_{nc}comp  n={len(sub)}  medD={med(d):+.5f}  neg(NOTRACK better)={n-pos}/{n}")
print()
print("K4. CONFIG FOLD for C(S1) and C(S4)")
for lab in ["1", "4"]:
    print(f"  C(S{lab}):", end="")
    for bf in (4, 5):
        for nc in (15, 30):
            sub = [e for e in EPS_PRIM if f"BF{bf}_{nc}comp" in e]
            d = paired(f"W-S{lab}", f"P-S{lab}", eps=sub)
            n, pos, p = sign_test(d)
            print(f"  BF{bf}_{nc}: {med(d):+.5f} ({pos}/{n})", end="")
    print()

print()
print("K5. magnitude structure of NOTRACK-AUTO (is the sign test throwing away signal?)")
d = paired("AUTO-NOTRACK", "AUTO", eps=EPS_PRIM)
fav = [x for x in d if x < 0]; unf = [x for x in d if x > 0]
print(f"   favourable {len(fav)} sum={sum(fav):+.4f}   unfavourable {len(unf)} sum={sum(unf):+.4f}"
      f"   ratio={abs(sum(fav)/sum(unf)):.1f}x")
print(f"   mean={sum(d)/len(d):+.5f}  median={med(d):+.5f}")

print()
print("K6. same for C(S1) -- does the magnitude structure agree with the sign test?")
d1 = paired("W-S1", "P-S1", eps=EPS_PRIM)
fav = [x for x in d1 if x > 0]; unf = [x for x in d1 if x < 0]
print(f"   'amplitude worse' {len(fav)} sum={sum(fav):+.4f}   'phase worse' {len(unf)} sum={sum(unf):+.4f}"
      f"   ratio={abs(sum(fav)/sum(unf)):.2f}x")
print(f"   mean={sum(d1)/len(d1):+.5f}  median={med(d1):+.5f}")
d4 = paired("W-S4", "P-S4", eps=EPS_PRIM)
fav = [x for x in d4 if x > 0]; unf = [x for x in d4 if x < 0]
print(f"   C(S4): 'amplitude worse' {len(fav)} sum={sum(fav):+.4f}  'phase worse' {len(unf)} sum={sum(unf):+.4f}")
print(f"   mean={sum(d4)/len(d4):+.5f}  median={med(d4):+.5f}")

print()
print("K7. THE SHIPPED POINT IS A DIAGONAL, NOT A ROW")
print("   shipped tracking_mode='w_phi', tracking_lr=0.005 -> amplitude arm sits at S1, phase arm at S3.")
print("   S1 row therefore contrasts the SHIPPED amplitude lr against a 22.7x-overdriven phase lr;")
print("   S3 row contrasts the SHIPPED phase lr against a 22.7x-underdriven amplitude lr.")
r = report("W-S1", "P-S3", eps=EPS_PRIM, label="W-S1 - P-S3 (both at shipped lr=0.005)")
print("   " + fmt(r))

print()
print("K8. equivalence framing: what does the data EXCLUDE at S1? (paired bootstrap CI on the median)")
import random
random.seed(7)
bs = []
for _ in range(20000):
    samp = [d1[random.randrange(len(d1))] for _ in d1]
    bs.append(st.median(samp))
bs.sort()
print(f"   C(S1) median = {med(d1):+.5f}  bootstrap 95% CI [{bs[int(.025*len(bs))]:+.5f}, {bs[int(.975*len(bs))]:+.5f}]")
bs4 = []
for _ in range(20000):
    samp = [d4[random.randrange(len(d4))] for _ in d4]
    bs4.append(st.median(samp))
bs4.sort()
print(f"   C(S4) median = {med(d4):+.5f}  bootstrap 95% CI [{bs4[int(.025*len(bs4))]:+.5f}, {bs4[int(.975*len(bs4))]:+.5f}]")
dn = paired("AUTO-NOTRACK", "AUTO", eps=EPS_PRIM)
bsn = []
for _ in range(20000):
    samp = [dn[random.randrange(len(dn))] for _ in dn]
    bsn.append(st.median(samp))
bsn.sort()
print(f"   NOTRACK-AUTO median = {med(dn):+.5f}  bootstrap 95% CI [{bsn[int(.025*len(bsn))]:+.5f}, {bsn[int(.975*len(bsn))]:+.5f}]")
