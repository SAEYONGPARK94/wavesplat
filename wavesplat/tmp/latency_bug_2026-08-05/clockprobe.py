"""Clock forensics: is CLOCK_REALTIME (time.time) non-monotonic / drifting vs CLOCK_MONOTONIC?"""
import time
import numpy as np

DUR = 60.0
SAMPLE_DT = 0.002
steps = []          # (monotonic_offset, backward_jump_seconds)
m0 = time.monotonic(); w0 = time.time()
mp = m0; wp = w0
n = 0
while True:
    m = time.monotonic(); w = time.time()
    dm = m - mp; dw = w - wp
    if dw < dm - 0.005:            # realtime fell behind monotonic by >5 ms in one sample gap
        steps.append((m - m0, dw - dm))
    mp, wp = m, w
    n += 1
    if m - m0 >= DUR:
        break
    time.sleep(SAMPLE_DT)
m1 = time.monotonic(); w1 = time.time()
M = m1 - m0; W = w1 - w0
print(f"samples={n}  monotonic_elapsed={M:.4f}s  realtime_elapsed={W:.4f}s")
print(f"TOTAL DRIFT realtime-monotonic = {W-M:+.4f}s  ({(W-M)/M*1e6:+.0f} ppm, {(W-M)/M*100:+.3f}%)")
print(f"discrete backward events (>5ms behind in one 2ms gap): {len(steps)}")
for t, d in steps[:20]:
    print(f"   at t={t:7.3f}s   realtime lost {-d*1000:9.3f} ms")
if steps:
    tot = -sum(d for _, d in steps)
    print(f"   total lost in discrete events = {tot:.4f}s of the {-(W-M):.4f}s drift "
          f"({tot/max(-(W-M),1e-9)*100:.1f}%)")
    gaps = np.diff([t for t, _ in steps])
    if len(gaps):
        print(f"   inter-event gap: mean={gaps.mean():.2f}s min={gaps.min():.2f}s max={gaps.max():.2f}s")
