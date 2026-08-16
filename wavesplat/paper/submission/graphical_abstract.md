# Graphical Abstract — Plan

> ⛔**RETRACTED (2026-08-14) — 이 계획서의 수치 패널은 그대로 생성하면 안 된다.**
> 아래 우측 패널이 쓰는 `WS 0.22 / ETKF 0.32 / GPR 0.30` 과 지연 `13.4 / 132.6 / ~200 ms`,
> 그리고 헤드라인의 **"10× faster"** 는 전부 인용금지 수치다. 사유(출처 부재가 아니다):
> **(i) 프레임 불일치** — 근거 파일 `results/b1_full_comparison.json` 에서 WaveSplat 은
> 에피소드당 평균 167 프레임(120–220), ETKF·GPR-1 은 301 프레임에서 채점됐고
> **100/100 에피소드 전부**에서 WS 가 더 적다(직접 재도출 2026-08-14).
> **(ii) 기준선 선택** — 같은 WS 열을 튜닝 LETKF 와 대조하면 `results/100ep_ws_letkf.json` 에서
> 0.2229 vs 0.2278, 51/100, p=0.9204 = **널**.
> **(iii) 지연은 프로세스 간 비교** — 동일 프로세스·1스레드 정직값은 **1.5–2.6×**,
> CPU 는 **AMD Ryzen 7 7800X3D**.
> ⟹ **검증된 대체값이 없으므로 숫자를 새로 만들지 않는다.** 생성 스크립트를 쓰기 전에
> `memory/reference_citation_bans.md` 를 먼저 확인할 것.
> (참고: 2026-08-13 감사가 `scripts/generate_graphical_abstract.py` 를 금지수치 블록
> 진입점에서 `RuntimeError` 로 차단해 두었다.)
> 근거: `memory/reference_citation_bans.md` (08-13 3라운드) ·
> `memory/project_audit_recheck_2026-08-13.md` §6, §6.5

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
|  scattered dots  |-->|  continuous wave |-->|  WS   ⛔RETRACTED |
|  on empty sea    |   |  field, arbitrary|   |  ETKF ⛔RETRACTED |
|  (BF4, ~50 pts)  |   |  spatial eval    |   |  GPR  ⛔RETRACTED |
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
  ⛔**RETRACTED (2026-08-14)** — 원문 헤드라인
  ~~"Grid-free, dispersion-constrained, 10× faster, only method with
  positive Skill Score."~~ 는 인용금지 수치 두 개("10× faster", Skill Score 우위)에
  의존한다. 검증된 대체 문구가 없으므로 **헤드라인을 비워 둔다**.

## Source assets
- Left + center: reuse the BF4 15-component row of
  `figures/fig8_qualitative.pdf` (ground truth and WS columns).
- Right: ⛔**RETRACTED (2026-08-14)** — 원문은
  ~~`results/p1_tuned_etkf.json` (nRMSE) + Table III latencies (13.4 / 132.6 / ~200 ms)~~
  로 막대그래프를 그리라고 지시했다. 두 자료 모두 인용금지 수치로 이어지므로
  **이 패널은 생성하지 말 것.**

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
