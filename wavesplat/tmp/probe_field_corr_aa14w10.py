"""Diagnose the unreproducible field-corr claim (AA14 w10 hand +0.72 vs head +0.66).
Compute per-lead field corr, pooled variants, amplitude ratios."""
import sys, os
sys.path.insert(0, 'scripts')
import numpy as np
from d3_external_figs import Window

w = Window('AA14', 10)
show = [l_ for l_ in w.leads if l_ <= 10.0]
gx, gy, tr, fh, fn = w.fields(show, stride=2)
trs, fhs, fns = (np.array(a) for a in (tr, fh, fn))
m = np.isfinite(trs)

def corr(a, b, mm):
    return float(np.corrcoef(a[mm], b[mm])[0, 1])

print("per-lead field corr (hand / net+head):")
for i, l_ in enumerate(show):
    mi = np.isfinite(trs[i])
    print(f"  lead {l_:4.1f}s  hand {corr(fhs[i],trs[i],mi):+.3f}  "
          f"net {corr(fns[i],trs[i],mi):+.3f}")
print(f"pooled 0-10s: hand {corr(fhs,trs,m):+.3f}  net {corr(fns,trs,m):+.3f}")
# mean of per-lead corrs
ch=[corr(fhs[i],trs[i],np.isfinite(trs[i])) for i in range(len(show))]
cn=[corr(fns[i],trs[i],np.isfinite(trs[i])) for i in range(len(show))]
print(f"mean per-lead: hand {np.mean(ch):+.3f}  net {np.mean(cn):+.3f}")
# amplitude preservation (why hand LOOKS better)
sd_t=float(np.nanstd(trs[m])); sd_h=float(np.std(fhs[m])); sd_n=float(np.std(fns[m]))
print(f"field std: truth {sd_t:.3f}  hand {sd_h:.3f} ({sd_h/sd_t:.2f}x)  "
      f"net {sd_n:.3f} ({sd_n/sd_t:.2f}x)")
# obs-window fit (lead 0 field = 'nowcast' appearance)
