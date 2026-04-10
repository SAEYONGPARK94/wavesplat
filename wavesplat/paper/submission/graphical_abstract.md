# Graphical Abstract — Plan

## Ocean Engineering specification
- Size: **max 9 cm (width) × 5 cm (height)** landscape, but we target the
  recommended 12 cm × 6 cm single-figure layout.
- Resolution: **300 dpi** minimum.
- Format: EPS, PDF, TIFF, or PNG.
- No multi-panel labels ("(a) (b) ..."); must be visually self-contained
  and understandable without the paper.
- File name (Elsevier convention): `graphical_abstract.pdf`.

## Design (single row, three panels without (a)/(b) labels)

```
+------------------+   +------------------+   +------------------+
|  SPARSE LIDAR    |   |   WaveSplat      |   |   METRIC BARS    |
|  INPUT           |   |   RECONSTRUCTION |   |                  |
|                  |   |                  |   |  nRMSE  Speed    |
|  scattered dots  |-->|  continuous wave |-->|  WS 0.22 13 ms  |
|  on empty sea    |   |  field, arbitrary|   |  ETKF 0.32 132  |
|  (BF4, ~50 pts)  |   |  spatial eval    |   |  GPR 0.30 200+  |
+------------------+   +------------------+   +------------------+
```

- Left panel: BF4 15-component scene, LiDAR returns as scatter over
  near-ship sector. Axes in meters, no grid.
- Center panel: the same frame reconstructed by WaveSplat, plotted as a
  2D colormap (e.g., cmocean `balance`), over the full 200×150 m domain.
  Overlay a small equation `z(x,y,t) = Σ wᵢ cos(kᵢ·r − ωᵢt + φᵢ)` with
  `ωᵢ = √(g|kᵢ|tanh(|kᵢ|d))` annotation.
- Right panel: paired bar chart of nRMSE and per-frame latency for the
  three methods. Use the same color per method across bars. Include the
  one-line headline:
  **"Grid-free, dispersion-constrained, 10× faster, only method with
  positive Skill Score."**

## Source assets
- Left + center: reuse the BF4 15-component row of
  `figures/fig8_qualitative.pdf` (ground truth and WS columns).
- Right: new bar plot generated from `results/p1_tuned_etkf.json` (nRMSE)
  and Table III latencies (13.4 / 132.6 / ~200 ms).

## Generation script (to add as `scripts/generate_graphical_abstract.py`)
- Load qualitative BF4-15c frame from the existing NPZ/HDF5 used by
  `scripts/generate_paper_figures.py` (`make_fig8_qualitative`).
- Render 3-panel figure at `figsize=(12/2.54, 6/2.54)`, `dpi=300`.
- Save to `paper/submission/graphical_abstract.pdf` and `.png`.
- Keep fonts at ≥7 pt after the 9-cm shrink; use `tight_layout`.

## Status
Design locked; generation deferred to a short follow-up commit that
wires the script to the existing `generate_paper_figures.py` helpers so
we do not duplicate data-loading logic.
