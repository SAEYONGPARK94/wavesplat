"""Direct reproduction of the defect shape: time a ~6.5 ms unit of work with
time.time() exactly the way SequentialUpdater.update() does, back to back."""
import time
import numpy as np
A = np.random.randn(320, 320)
vals = []
m0 = time.monotonic()
while time.monotonic() - m0 < 70.0:
    t0 = time.time()          # <-- sequential_update.py:840
    A @ A                     # ~6-7 ms of work, stands in for the tracking gradient steps
    vals.append((time.time() - t0) * 1000.0)   # <-- sequential_update.py:875
v = np.array(vals)
neg = v[v < 0]
print(f"calls={len(v)}  median={np.median(v):.3f} ms  min={v.min():.1f} ms  n_negative={len(neg)}")
print(f"negative values (ms): {[round(float(x),1) for x in neg]}")
print(f"if these {len(neg)} calls were frames of a 160-frame episode with true 6.6 ms/frame,")
for x in neg:
    print(f"   avg_inference_ms would be {(6.6*159 + x)/160:+.3f} ms")
