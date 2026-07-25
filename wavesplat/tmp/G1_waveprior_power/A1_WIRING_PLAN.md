# A1 execution plan — learned proposal vs the paper's WaveSplat, fair test

## Key finding that makes A1 tractable
`scripts/d3_beamform_cond_diffusion.py::evaluate()` already loads the PAPER's episodes via
the paper's own `wavesplat.data_loader.EpisodeDataLoader` (`d3_beamform_cond_diffusion.py:182`),
gathers the paper's LiDAR frames (`gather_frames`), and scores nRMSE against `fstar.gt_field`
(the paper's GT), normalized by Hs, at 3 frames. So the learned proposal is NOT on a foreign
protocol — the obs and GT are the paper's. The 0.110 (cond+polish) was on the OLD coarse-mesh
episodes (eval json dated Jul 3, pre-regeneration).

## Remaining gaps to a fair comparison vs the paper's coarse_to_fine WaveSplat
1. **Data**: re-run cond-diffusion eval on `data_finemesh_2026-07-23/` (not old mesh).
   `evaluate()` hardcodes `'data/' + tag_to_file(tag)` → need a data-dir override or symlink.
2. **Metric parity**: cond uses `/Hs`, 3 frames, nowcast (best-of-4 sample avg + polish).
   Paper WS uses `/gt_std`, per-frame over the sequence, with the sequential tracker.
   For apples-to-apples: score BOTH on the SAME frames + SAME normalizer. Two options:
   (a) add a WS arm inside the cond eval loop (run `run_estimation` per episode, extract
       nowcast at the same `sel` frames, normalize identically); or
   (b) re-score cond's rendered field with the paper's `evaluation/metrics` normalizer.
3. **OOD check**: the cond net was trained on direction3 synthetic (32-comp JONSWAP) obs;
   the paper episodes are sparse 1/5/15/30-comp. Report per-complexity so the sparse regime
   is visible (this is where a learned proposal must prove itself — unlike the continuous
   bank prior, a discrete set-proposal CAN represent sparse peaks, so this is the real test).

## Kill-test
Does cond-diffusion (proposal+polish), scored identically on the finemesh episodes, beat the
paper's coarse_to_fine WaveSplat (≈0.199 nowcast on finemesh)? If yes on the sparse episodes
too → strong case for (A). If it only wins on dense/high-comp → learned proposal is a
partial win; position accordingly.

## Then (if kill-test passes)
Wire the proposal as `init_method='learned'` feeding (kx,ky) atoms into
`coarse_to_fine`/`initialize_from_frames`, so the paper's sequential tracker runs on top —
full integration, benchmarked vs WS/ETKF/GPR on the 5-metric panel.
