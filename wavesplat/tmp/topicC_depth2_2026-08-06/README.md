# Topic C — depth-2 (mechanism / confound / power) re-derivation, 2026-08-06

All numbers in the depth-2 verdict were computed from raw jsonl/h5 by the blocks below.
Run from /home/vexit/wavesplat/wavesplat.

- flatnull.py   -> flat-sea (z=0) null per episode: nRMSE_flat(f) = rms(gt_field_f)/Hs_true
                   outputs topicC_depth2_flatnull.json / topicC_depth2_flatframes.json
- D2-A  m vs 1            (m = MSE/Var(gt) = (1-rho^2) + (sqrt(v)-rho)^2 ; m>1 == worse than z=0)
- D2-B  gap decomposition (quality term vs scoring-domain drift term)
- D2-C  wavenumber error as the covariate of the frozen-tracker floor
- D2-D  identifiability of init-quality vs k-error
- D2-E  convention sweep on the AUTO gap
- D2-F/G 2x2 factorial structure; cancelled interaction with component count
- D2-H  seed-split IV test of regression dilution on the shrinkage slope b
- D2-J/K power under a cancelled interaction; minimum feasible design
- D2-L/M/N/O  E_remeasure recomputed on the COMPLETED file (520/520)
- D2-P  m>1 across two independent seed-triplets of the same cfg

Nothing was written outside tmp/. No configs, no paper, no checkpoints, no commits.
