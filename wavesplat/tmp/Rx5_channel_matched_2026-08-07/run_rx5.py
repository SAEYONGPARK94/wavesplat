"""Rx-4 -- tracker-vs-initialisation axis: WHICH PARAMETER CHANNEL does Phase 2 damage?

VERBATIM COPY of tmp/B3_rx3_2026-08-06/run_b3.py (md5 14b5644e1273c6af11ed1d71e9880596)
with exactly five edits, all listed here so the copy can be audited against the original:
  (F1) this docstring + OUT default -> tmp/Rx4_tracker_init_2026-08-06/ + SEEDS default
       1524,1525,1526 (RX2_SEEDS still overrides, so the B3 reproduction lane can ask for
       1521,1522,1523).
  (F2) CROWDED16 / SPARSE8 episode lists + RX2_EPISODES values 'c16' and 's8'.  The
       pre-existing 'p1' and the crowded-4 default are untouched, so the B3 reproduction
       lane selects exactly B3's episodes.
  (F3) seven Rx-4 cells appended to CELLS (nothing existing is touched).
  (F4) _rx4_install(): two class-method patches installed once per worker.  BOTH ARE
       NO-OPS unless the per-run config carries the triggering key, so they cannot leak
       between cells the way a mutable module attribute can:
         _build_tracking_optimizer -> adds a 'w_only' branch (freeze kx, ky, phi; train w).
             The shipped modes 'phi_only' / 'w_phi' / 'all' fall through to the original.
         update                    -> if cfg['rx4_track_frames']=E is present, zeroes every
             tracking-optimizer lr from frame E on (the same lr=0 mechanism AUTO-NOTRACK
             uses).  Absent key == original function, byte for byte.
  (F5) 'w_only' / 'phi_only' / 'rx4_track_frames' recorded in the per-run 'cfg' dump
       (already dumped by the original; no new code).
The AUTO cell run on seeds 1521/1522/1523 over the crowded-4 must therefore be
BIT-IDENTICAL to tmp/B3_rx3_2026-08-06/results.jsonl.  That reproduction control plus the
Phase-1 invariance check are mandatory before any Rx-4 number is read (PREREG sec.6).

Pre-registered in tmp/Rx4_tracker_init_2026-08-06/PREREG.md (frozen BEFORE any result was
seen).  Cell definitions are frozen; do not edit them after the first run.

--- original header follows ---
B3 / Rx-3 -- tracker ON/OFF at the ONLY shipping arm that beats the flat-sea null.

VERBATIM COPY of tmp/rx2_init_2026-08-05/run_rx2b.py (md5 cd44d093f161bd04673e131618324e79)
with exactly four edits, all listed here so the copy can be audited against the original:
  (E1) this docstring + OUT default -> tmp/B3_rx3_2026-08-06/
  (E2) six AUTO-* cells appended to CELLS (nothing existing is touched)
  (E3) RX1_* env keys added to the per-cell reset list.  `_rx1_gate` treats '0' and unset
       identically ("Unset or '0' == shipped behaviour, bit-for-bit", sequential_update.py:87),
       so this cannot change any pre-existing cell's numbers.
  (E4) four extra router fields recorded per run (auto_selected / auto_n_selected /
       ensemble_k / final_n).  Recording only; no computation is changed.
The AUTO cell run on seeds 1518/1519/1520 must therefore be BIT-IDENTICAL to
tmp/rx2_init_2026-08-05/results.jsonl and tmp/B2_g1p1_ext_2026-08-06/results.jsonl.
That reproduction control is mandatory before any B3 number is read (PREREG sec.5).

Pre-registered in tmp/B3_rx3_2026-08-06/PREREG.md (frozen BEFORE any result was seen).
Cell definitions are frozen; do not edit them after the first run.

--- original header follows ---
Rx-2 -- initialisation-gap decomposition driver (PRO arm).

Pre-registered in tmp/rx2_init_2026-08-05/PREREG.md + PREREG_ADDENDUM_PRO.md.
Cell definitions are frozen by the addendum; do not edit them after the first run.

Protocol inherited VERBATIM from tmp/s1s2_2026-08-05/run_s1s2.py (panel, spectral_accuracy,
resumable jsonl, spawn pool, 1 BLAS thread per worker) so cell X-BASE pairs bit-for-bit with
arm S1-A0 there when run on seeds {42,123,456}.

PRODUCTION DEFAULTS ARE NOT CHANGED. Every factor is applied inside the worker process by
(a) environment gates read at CALL time (initializer._rx2_gate), or
(b) module-attribute monkey-patches that are undone in a finally block.

TIMING IS RECORDED BUT NOT CITABLE (this box's clock is under separate re-measurement).
"""

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA = os.path.join(REPO, os.environ.get('K6_DATA', 'data_indep2_2026-08-07'))
OUT = os.path.join(HERE, 'results.jsonl')   # (E1) HERE is now tmp/B3_rx3_2026-08-06/

# Seed ledger [1518-1818]; this experiment consumes 1518/1519/1520 (median-of-3).
# The bit-identity batch alone uses the stored {42,123,456} so it can be compared to
# tmp/s1s2_2026-08-05/results.jsonl arm S1-A0 value-for-value.
SEEDS = [int(s) for s in os.environ.get('RX2_SEEDS', '1524,1525,1526').split(',')]  # (F1)

CROWDED4 = [
    'episode_BF4_15comp_seed0_uniform.h5',
    'episode_BF4_30comp_seed0_uniform.h5',
    'episode_BF5_15comp_seed0_uniform.h5',
    'episode_BF5_30comp_seed0_uniform.h5',
]
# PREREG AMENDMENT 1 item 6 ("widen the observation unit beyond n=4"): RX2_EPISODES=p1
# swaps in the full 12-episode P1 set (scripts/run_p1_comparison.py::EPISODES verbatim).
# Crowded-4 stays the default so nothing already run is affected.
P1_12 = [
    'episode_BF2_1comp_seed0_uniform.h5', 'episode_BF3_1comp_seed0_uniform.h5',
    'episode_BF3_2comp_seed0_uniform.h5', 'episode_BF3_2comp_seed0_slope.h5',
    'episode_BF3_5comp_seed0_uniform.h5', 'episode_BF3_5comp_seed0_slope.h5',
    'episode_BF4_1comp_seed0_uniform.h5', 'episode_BF4_5comp_seed0_uniform.h5',
] + CROWDED4
# (F2) Rx-4 blocks.  CROWDED16 = the 4 crowded configs x 4 INDEPENDENT episode-seeds
# (uniform only).  `slope` variants are deliberately EXCLUDED: they carry a bit-identical
# gt_wave_field to their `uniform` twin (verified: allclose == True, and the flat-sea nulls
# agree to 1e-6), so a uniform/slope pair is one wave field observed twice, not two
# observation units.  SPARSE8 = the 8 non-crowded P1 episodes, unchanged.
CROWDED16 = [f'episode_BF{bf}_{nc}comp_seed{es}_uniform.h5'
             for bf in (4, 5) for nc in (15, 30) for es in (0, 1, 2, 3)]
# (K6, 2026-08-07) The INDEPENDENT rebuild of the same 16 cells.  data_indep_2026-08-07/
# was generated with random_amplitudes=True + decorrelate_stream=True, so unlike the
# 2026-07-23 set (4 distinct amplitude vectors, 8 distinct phase vectors, 5 distinct
# direction draws) every episode is its own realisation.  Verified by
# tmp/audit_independence_2026-08-07/ before this batch was run.
# v2 (2026-08-07): the first rebuild had random_amplitudes+decorrelate_stream but omega
# was STILL 4 draws -- omega is the bin centre, a pure function of (Tp, n_components), and
# |k| follows from omega, so the radial wavenumber structure did not move.  The pre-run
# gate in PREREG sec.3 caught this and FAILED the batch.  v2 adds random_frequencies.
CROWDED16_INDEP = [e.replace('.h5', '_ra_rf_ds.h5') for e in CROWDED16]
SPARSE8 = [e for e in P1_12 if e not in CROWDED4]
_EPSEL = os.environ.get('RX2_EPISODES')
EPISODES = ({'p1': P1_12, 'c16': CROWDED16, 's8': SPARSE8,
             'c16s8': CROWDED16 + SPARSE8,
             'indep16': CROWDED16_INDEP}.get(_EPSEL) or CROWDED4)
PANEL = ['nRMSE', 'Variance_Ratio', 'Pearson_rho', 'Skill_Score',
         'Directional_R2', 'Hs_error']
N_GT = {'episode_BF4_15comp_seed0_uniform.h5': 15,
        'episode_BF4_30comp_seed0_uniform.h5': 30,
        'episode_BF5_15comp_seed0_uniform.h5': 15,
        'episode_BF5_30comp_seed0_uniform.h5': 30}

WIDE = (-4.0, 4.0)
NARROW = (-0.8, 0.8)

# WS-P1 deployment operating point == tmp/s1s2_2026-08-05 arm S1-A0. sum(max_iters) == 900.
BASE_CFG = dict(init_method='v2_konly', use_dispersion=True, use_confidence=True,
                adaptive_n=True, tracking_window=3, prune_threshold=0.10, max_n=30,
                n_gaussians=[3, 8, 15], loss_thresholds=[0.1, 0.05, None],
                max_iters_per_stage=[200, 200, 500])

# cell -> dict(env=..., mod=..., krange=..., cfg=...)
#   env    : RX2_* gates, read at call time by initializer._rx2_gate
#   mod    : module attributes to override (undone in finally)
#   krange : coarse beamform (kx,ky) range at ALL THREE placement sites
#   cfg    : run_estimation config overrides on top of BASE_CFG
CELLS = {
    # --- controls -----------------------------------------------------------------
    'X-BASE':  dict(),                       # shipped, PRIMARY control
    # byte-duplicates of X-BASE: one per batch, so every batch carries a worker-leak probe
    'X-BASE2': dict(),
    'X-BASE3': dict(),
    'X-BASE4': dict(),
    # --- axis P: the residual-site greedy (12 of 15 atoms on the deployment ladder) --
    'P1': dict(env={'RX2_GROW_SUBTRACT': '1'}),
    'P2': dict(env={'RX2_GROW_RECOMPUTE': '1'}),
    'P3': dict(env={'RX2_GROW_RECOMPUTE': '1', 'RX2_GROW_TIGHT_SUPPRESS': '1'}),
    # --- axis W: deterministic (w, phi) seeding -------------------------------------
    'W1': dict(mod={'I.KONLY_RANDOM_WPHI': False}),
    'W2': dict(env={'RX2_GROW_FITPHI': '1'}),
    'W3': dict(mod={'I.KONLY_RANDOM_WPHI': False}, env={'RX2_GROW_FITPHI': '1'}),
    # --- axis S: schedule (same final N=15, same 900 total gradient steps) -----------
    'S1': dict(cfg=dict(n_gaussians=[3, 6, 9, 12, 15],
                        loss_thresholds=[0.1, 0.05, 0.05, 0.05, None],
                        max_iters_per_stage=[180, 180, 180, 180, 180])),
    'S2': dict(mod={'C.SCALE_INVARIANT_ADVANCE': True}),
    # --- anchor: best previously-measured placement intervention (E1 cell B) ---------
    'G1': dict(krange=NARROW),
    # --- STACKS (bundles, named as such; interaction reported explicitly) ------------
    'G1P1': dict(krange=NARROW, env={'RX2_GROW_SUBTRACT': '1'}),
    'G1P3': dict(krange=NARROW, env={'RX2_GROW_RECOMPUTE': '1',
                                     'RX2_GROW_TIGHT_SUPPRESS': '1'}),
    # --- upper-bound markers (NOT deployable) ---------------------------------------
    # ORACLE   == s1s2 arm S2-T1f : oracle placement AND k frozen in Phase 1.
    # ORACLE-K == s1s2 arm S2-T1  : oracle placement, k TRAINABLE.  PREREG AMENDMENT 1
    #   (neutral audit) established that k-freezing is an OPTIMIZER setting no deployable
    #   initializer can have, so ORACLE-K -- not ORACLE -- is the REACHABLE ceiling and the
    #   PRIMARY gap-closure denominator.
    'ORACLE':   dict(cfg='__oracle__'),
    'ORACLE-K': dict(cfg='__oracle_k__'),
    # --- the ACTUAL shipped default (PREREG AMENDMENT 1 A4): init_method='auto' routes
    #   through the learned proposal and MedianEnsembleUpdater(median-of-3). Explicit
    #   v2_konly gets neither, so X-BASE is the SHIPPING FALLBACK, not "deployment".
    'AUTO': dict(cfg=dict(init_method='auto', analytic_ensemble_k=3)),
    'X-BASE5': dict(),
}

# ===================== (E2) B3 / Rx-3 CELLS -- appended, nothing above is touched =========
# All six share the SHIPPED operating point `init_method='auto', analytic_ensemble_k=3`
# (== run.py DEFAULT_CONFIG) and differ from cell AUTO by EXACTLY ONE knob each.
# `init_method` is explicit in every one of them (W9 rule).
#
# Tracker-off mechanism, inherited verbatim from tmp/rx1_tracker_2026-08-05/run_rx1.py:97-98
# ("budget 0: Adam lr 0 -> tracker is a no-op").  With tracking_lr=0 every Adam param group
# in _build_tracking_optimizer has lr 0, and TRACK_PHASE_PREDICT is False by default, so
# Phase 2 leaves the model bit-frozen; only predict_field's own -w*t term moves the field.
# MedianEnsembleUpdater hands `dict(self.config)` to every member, so the knob reaches all
# three ensemble members, not just member 0.
CELLS.update({
    # byte-duplicate of AUTO: per-batch worker-leak / env-residue probe (rx2 used X-BASE2-5)
    'AUTO-DUP':     dict(cfg=dict(init_method='auto', analytic_ensemble_k=3)),
    # --- axis: displacement budget --------------------------------------------------
    'AUTO-NOTRACK': dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                                  tracking_lr=0.0)),        # budget 0 == tracker frozen
    'AUTO-K6':      dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                                  tracking_steps=1)),       # 10 -> 1 step/frame
    'AUTO-K7':      dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                                  tracking_lr=0.0005)),     # 0.005 -> 0.0005 (x0.1)
    # --- axis: tracking objective (regularisation channel), rx1 K4 -------------------
    # RX1_PHASE2_NOREG makes the TRACKING objective the data term alone; Phase 1 keeps its
    # penalties (sequential_update.py:917).  This is the intervention that was -0.1046 (4/4)
    # at oracle placement and +0.0007 at the v2_konly arm that never beat the null.
    'AUTO-K4':      dict(cfg=dict(init_method='auto', analytic_ensemble_k=3),
                         env={'RX1_PHASE2_NOREG': '1'}),
})
B3_CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK', 'AUTO-K6', 'AUTO-K7', 'AUTO-K4']
# =========================================================================================

# ===================== (F3) Rx-4 CELLS -- appended, nothing above is touched ==============
# All share the SHIPPED operating point `init_method='auto', analytic_ensemble_k=3`
# (== run.py DEFAULT_CONFIG) and differ from cell AUTO by EXACTLY ONE knob each.
# `init_method` is explicit in every one of them (W9 rule).  No bundle cells.
#
# Structural facts these cells rest on, MEASURED in tmp/Rx4_tracker_init_2026-08-06/
# diag_analysis.txt over 3,384 frames before any cell was designed:
#   * omega is NOT a free parameter.  spectral_gaussian._omega() recomputes
#     sqrt(g|k| tanh(|k|d)) from (kx, ky) at EVERY forward call, and tracking_mode='w_phi'
#     freezes kx/ky, so the measured dispersion residual is 4.8e-07 (float32 noise) and
#     ||dk||_2 is EXACTLY 0 at every frame.  => a "re-impose dispersion during tracking"
#     cell (T-DISP) would be a no-op and is NOT run.
#   * `w` is the AMPLITUDE vector [m] (spectral_gaussian.py:38), not angular frequency.
#     So shipped tracking_mode='w_phi' == "amplitude + phase", and the two single-channel
#     cells below are the amplitude channel and the phase channel.
CELLS.update({
    # --- axis: WHICH parameter channel tracks (tracking_mode, one shipped enum) ---------
    # phi_only is a first-class shipped value (sequential_update.py:153, configs/
    # estimation.yaml:15).  Freezes the AMPLITUDE, keeps phase tracking.
    'T-PHIONLY': dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                               tracking_mode='phi_only')),
    # w_only is the SYMMETRIC control; it does not exist in the shipped enum, so it is
    # supplied by the _rx4_install patch below.  Freezes the PHASE, keeps amplitude
    # tracking.  Without it, "phi_only wins" could not be attributed to a channel.
    'T-WONLY':   dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                               tracking_mode='w_only')),
    # --- axis: for HOW LONG the tracker runs (damage measured as monotone accumulation) --
    'T-EARLY20': dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                               rx4_track_frames=20)),
    # --- in-batch reference: best partial intervention known from B3 (72.2% of the floor)
    'T-K4REF':   dict(cfg=dict(init_method='auto', analytic_ensemble_k=3),
                      env={'RX1_PHASE2_NOREG': '1'}),
})
RX4_CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK',
             'T-PHIONLY', 'T-WONLY', 'T-EARLY20', 'T-K4REF']

# ===================== (K6) 2026-08-07 CELLS -- appended, nothing above is touched ========
# Motivated by tmp/K6_scope_2026-08-07/ (PRO/NEUTRAL/CON, three independent re-derivations).
# Two facts those three established BEFORE these cells were written:
#   (1) `ORACLE-K` is NOT a k-placement arm.  run_rx2b.py:174-177 changes THREE things at
#       once: oracle (kx,ky) AND n_gaussians=[N_gt] AND the 3-stage ladder -> one 900-step
#       stage.  So the published -0.0105 has never been attributed to k.  `K6-NSCHED`
#       below is the missing control: N and schedule oracle, k BLIND.
#   (2) The budget split on the crowded block is
#         X-BASE .2849 -> AUTO .2214 -> ORACLE-K .1967 -> ORACLE(k frozen) .1626
#       i.e. handing the optimiser PERFECT k and then letting it move k loses -0.0341 (4/4),
#       which is LARGER than the -0.0247 of placement still on the table.  `init_k_lr_scale`
#       (sequential_update.py:248) is the knob for that, and only its endpoints have ever
#       been run: 0.0 (oracle cells) and 1.0 (shipped).  K6-KLR* sweeps the interior.
# Every cell below carries init_method explicitly (W9 rule) and differs from AUTO by
# EXACTLY ONE knob.  No bundle cells.
CELLS.update({
    # --- axis: Phase-1 k trust region, ON THE SHIPPED ARM ---------------------------
    'K6-KLR03': dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                              init_k_lr_scale=0.3)),
    'K6-KLR01': dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                              init_k_lr_scale=0.1)),
    'K6-KLR00': dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                              init_k_lr_scale=0.0)),
    # --- the missing de-confounding control for ORACLE-K ----------------------------
    # N and schedule handed over exactly as ORACLE-K gets them, but the placement is the
    # ordinary blind beamformer.  ORACLE-K minus this cell is the part attributable to k.
    'K6-NSCHED': dict(cfg='__nsched__'),
})
K6_CELLS = ['AUTO', 'AUTO-DUP', 'ORACLE', 'ORACLE-K', 'K6-NSCHED',
            'K6-KLR03', 'K6-KLR01', 'K6-KLR00']

# ===================== (Rx-5 / E-sweep) 2026-08-07 -- appended, nothing above touched ====
# Rx-4 tried to ask "which parameter channel does Phase 2 damage?" by switching
# tracking_mode between phi_only and w_only at the SHIPPED tracking_lr=0.005.  Three
# reviewers plus a raw-data check killed that contrast: Adam takes an ~lr-sized ABSOLUTE
# step, and the two channels have wildly different natural scales --
#   amplitude: median |A| = 0.0604 m  ->  0.005 is  8.3 % of it (689 % of the smallest)
#   phase    : range 2*pi rad         ->  0.005 is 0.080 % of it
# a ~100x asymmetry in RELATIVE drive.  So `tracking_mode` was one config key but a
# BUNDLE, and Rx-4's D1 verdict is UNDECIDED, not REFUTED.
#
# The fix is a DOSE-RESPONSE curve on the amplitude channel's lr, with the decisive
# comparison at MATCHED relative drive against the phase channel at its shipped lr.
# RX5_LR_W_MATCHED is filled in from a measurement of the model's own |w| at the start
# of tracking (measure_wscale.py), taken BEFORE any Rx-5 result is scored.
RX5_LR_W_MATCHED = float(os.environ.get('RX5_LR_W_MATCHED', '0.0'))  # 0.0 == not yet set

CELLS.update({
    # --- amplitude channel, lr dose-response (tracking_mode fixed at w_only) ----------
    'W-5e3':  dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                            tracking_mode='w_only', tracking_lr=0.005)),
    'W-5e4':  dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                            tracking_mode='w_only', tracking_lr=5e-4)),
    'W-MATCH': dict(cfg='__w_match__'),
    'W-5e6':  dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                            tracking_mode='w_only', tracking_lr=5e-6)),
    # --- phase channel at its shipped lr: the reference the matched cell is compared to
    'PHI-5e3': dict(cfg=dict(init_method='auto', analytic_ensemble_k=3,
                             tracking_mode='phi_only', tracking_lr=0.005)),
    # --- (E-sweep) tracking DURATION.  Rx-4 ran only the single point 20 and lost the
    # mechanism claim for it because one pre-chosen point is not a sweep.
    'E-2':   dict(cfg=dict(init_method='auto', analytic_ensemble_k=3, rx4_track_frames=2)),
    'E-5':   dict(cfg=dict(init_method='auto', analytic_ensemble_k=3, rx4_track_frames=5)),
    'E-10':  dict(cfg=dict(init_method='auto', analytic_ensemble_k=3, rx4_track_frames=10)),
    'E-20':  dict(cfg=dict(init_method='auto', analytic_ensemble_k=3, rx4_track_frames=20)),
    'E-40':  dict(cfg=dict(init_method='auto', analytic_ensemble_k=3, rx4_track_frames=40)),
    'E-80':  dict(cfg=dict(init_method='auto', analytic_ensemble_k=3, rx4_track_frames=80)),
})
RX5_CELLS = ['AUTO', 'AUTO-DUP', 'AUTO-NOTRACK',
             'W-5e3', 'W-5e4', 'W-MATCH', 'W-5e6', 'PHI-5e3']
ESWEEP_CELLS = ['AUTO', 'AUTO-NOTRACK', 'E-2', 'E-5', 'E-10', 'E-20', 'E-40', 'E-80']

_RX4_ORIG = {}


def _rx4_install():
    """(F4) Install the two Rx-4 class patches. Idempotent, worker-local, LEAK-PROOF:
    both branches are gated on a per-RUN config key, never on module or env state, so a
    worker that has just run T-WONLY behaves byte-for-byte like a fresh one on the next
    cell.  (AUTO-DUP is still carried as the empirical leak probe.)"""
    if _RX4_ORIG:
        return
    import torch
    import wavesplat.optimizer.sequential_update as S
    _RX4_ORIG['bto'] = S.SequentialUpdater._build_tracking_optimizer
    _RX4_ORIG['upd'] = S.SequentialUpdater.update

    def bto(self):
        if getattr(self, 'tracking_mode', None) != 'w_only':
            return _RX4_ORIG['bto'](self)          # shipped path, untouched
        # mirror image of the shipped 'phi_only' branch (sequential_update.py:266-271)
        self.model.kx.requires_grad_(False)
        self.model.ky.requires_grad_(False)
        self.model.phi.requires_grad_(False)
        self.model.w.requires_grad_(True)
        params = [{'params': [self.model.w], 'lr': self.tracking_lr}]
        if not self.use_dispersion:                # unreachable here (use_dispersion=True)
            self.model.omega_free.requires_grad_(True)
            params.append({'params': [self.model.omega_free], 'lr': self.tracking_lr})
        self.tracking_optimizer = torch.optim.Adam(params)

    def upd(self, frame):
        lim = self.config.get('rx4_track_frames')
        if lim is not None:
            n = getattr(self, '_rx4_n', 0)
            if n >= int(lim) and self.tracking_optimizer is not None:
                # same mechanism as AUTO-NOTRACK: every Adam group lr -> 0 => bit-frozen
                for g in self.tracking_optimizer.param_groups:
                    g['lr'] = 0.0
            self._rx4_n = n + 1
        return _RX4_ORIG['upd'](self, frame)

    S.SequentialUpdater._build_tracking_optimizer = bto
    S.SequentialUpdater.update = upd
# =========================================================================================

ORACLE_CELLS = {'ORACLE', 'ORACLE-K'}
# (E3) RX1_* appended.  sequential_update._rx1_gate: "Unset or '0' == shipped behaviour,
# bit-for-bit", so writing '0' here is a no-op for every pre-existing cell.
RX2_ENV_KEYS = ('RX2_GROW_SUBTRACT', 'RX2_GROW_RECOMPUTE',
                'RX2_GROW_TIGHT_SUPPRESS', 'RX2_GROW_FITPHI',
                'RX1_CONF_NORM', 'RX1_PHASE2_NOREG')


def apply_cell(cell):
    """Install the cell's factors. Returns an undo callable. Worker-local."""
    import wavesplat.optimizer.initializer as I
    import wavesplat.optimizer.coarse_to_fine as C
    import wavesplat.optimizer.sequential_update as S
    spec = CELLS[cell]
    mods = {'I': I, 'C': C, 'S': S}

    # 1. environment gates -- always fully rewritten so a previous cell cannot leak
    for k in RX2_ENV_KEYS:
        os.environ[k] = '0'
    for k, v in (spec.get('env') or {}).items():
        os.environ[k] = v

    # 2. module attributes
    saved = {}
    for key, val in (spec.get('mod') or {}).items():
        mod_name, attr = key.split('.')
        m = mods[mod_name]
        saved[key] = getattr(m, attr)
        setattr(m, attr, val)

    # 3. beamform search range at all three placement sites
    orig_bf = I._beamform_coarse
    krange = spec.get('krange')
    if krange is not None:
        def bf(x, y, z, t, depth, kx_range=WIDE, ky_range=WIDE, n_kx=81, n_ky=81):
            # range overridden; grid COUNT stays whatever the call site asked for (81/61).
            return orig_bf(x, y, z, t, depth, krange, krange, n_kx, n_ky)
        I._beamform_coarse = bf
        C._beamform_coarse = bf   # coarse_to_fine imported the symbol at module load

    def undo():
        for k in RX2_ENV_KEYS:
            os.environ[k] = '0'
        for key, val in saved.items():
            mod_name, attr = key.split('.')
            setattr(mods[mod_name], attr, val)
        I._beamform_coarse = orig_bf
        C._beamform_coarse = orig_bf
    return undo


def _n_gt(episode):
    if episode in N_GT:
        return N_GT[episode]
    import h5py                       # only reached on the 12-episode extension
    with h5py.File(os.path.join(DATA, episode), 'r') as f:
        return int(np.asarray(f['config/wave_components']).shape[0])


def cell_config(cell, episode):
    cfg = dict(BASE_CFG)
    spec = CELLS[cell].get('cfg')
    if spec == '__oracle__':
        cfg.update(init_method='oracle', n_gaussians=[_n_gt(episode)],
                   loss_thresholds=[None], max_iters_per_stage=[900],
                   init_k_lr_scale=0.0)
    elif spec == '__oracle_k__':
        cfg.update(init_method='oracle', n_gaussians=[_n_gt(episode)],
                   loss_thresholds=[None], max_iters_per_stage=[900])
    elif spec == '__w_match__':
        # amplitude channel driven at the SAME relative step as the phase channel
        assert RX5_LR_W_MATCHED > 0.0, 'RX5_LR_W_MATCHED not set (see measure_wscale.py)'
        cfg.update(init_method='auto', analytic_ensemble_k=3,
                   tracking_mode='w_only', tracking_lr=RX5_LR_W_MATCHED)
    elif spec == '__nsched__':
        # (K6) identical to __oracle_k__ EXCEPT init_method: the blind beamformer places
        # the atoms.  ORACLE-K - K6-NSCHED isolates the placement factor.
        cfg.update(init_method='v2_konly', n_gaussians=[_n_gt(episode)],
                   loss_thresholds=[None], max_iters_per_stage=[900])
    elif spec:
        cfg.update(spec)
    assert 'init_method' in cfg          # W9 discipline: never implicit
    return cfg


# ---- spectral matching, verbatim from tmp/s1s2_2026-08-05/run_s1s2.py ----
def match_components(est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A):
    N_est = len(est_kx); N_gt = len(gt_kx)
    if N_est == 0 or N_gt == 0:
        return [], list(range(N_gt)), list(range(N_est))
    k_scale = max(np.max(np.sqrt(gt_kx ** 2 + gt_ky ** 2)), 0.01)
    cost = np.zeros((N_est, N_gt))
    for i in range(N_est):
        for j in range(N_gt):
            dk = np.sqrt((est_kx[i] - gt_kx[j]) ** 2 + (est_ky[i] - gt_ky[j]) ** 2)
            cost[i, j] = dk / k_scale
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)
    matched = [(r, c, cost[r, c]) for r, c in zip(row_ind, col_ind) if cost[r, c] < 0.5]
    matched_gt = {m[1] for m in matched}; matched_est = {m[0] for m in matched}
    unmatched_gt = [j for j in range(N_gt) if j not in matched_gt]
    unmatched_est = [i for i in range(N_est) if i not in matched_est]
    return matched, unmatched_gt, unmatched_est


def spectral_accuracy(est_params, gt_components):
    gt_A = gt_components[:, 0]; gt_omega = gt_components[:, 1]
    gt_kx = gt_components[:, 2]; gt_ky = gt_components[:, 3]
    est_kx = np.asarray(est_params['kx']); est_ky = np.asarray(est_params['ky'])
    est_w = np.abs(np.asarray(est_params['w'])); est_omega = np.asarray(est_params['omega'])
    matched, unmatched_gt, unmatched_est = match_components(
        est_kx, est_ky, est_w, gt_kx, gt_ky, gt_A)
    result = {'n_gt': int(len(gt_A)), 'n_est': int(len(est_kx)),
              'n_matched': int(len(matched)),
              'detection_rate': len(matched) / max(len(gt_A), 1)}
    if matched:
        k_mag_errors = []; omega_errors = []; amp_errors = []; direction_errors = []
        for est_idx, gt_idx, cost in matched:
            gt_kmag = np.sqrt(gt_kx[gt_idx] ** 2 + gt_ky[gt_idx] ** 2)
            est_kmag = np.sqrt(est_kx[est_idx] ** 2 + est_ky[est_idx] ** 2)
            k_mag_errors.append((est_kmag - gt_kmag) / max(gt_kmag, 1e-6))
            omega_errors.append((est_omega[est_idx] - gt_omega[gt_idx]) / max(gt_omega[gt_idx], 1e-6))
            amp_errors.append((est_w[est_idx] - gt_A[gt_idx]) / max(gt_A[gt_idx], 1e-6))
            gt_dir = np.degrees(np.arctan2(gt_ky[gt_idx], gt_kx[gt_idx]))
            est_dir = np.degrees(np.arctan2(est_ky[est_idx], est_kx[est_idx]))
            d_dir = abs(est_dir - gt_dir); d_dir = 360 - d_dir if d_dir > 180 else d_dir
            direction_errors.append(d_dir)
        result['k_mag_rel_error_mean'] = float(np.mean(np.abs(k_mag_errors)))
        result['omega_rel_error_mean'] = float(np.mean(np.abs(omega_errors)))
        result['amp_rel_error_mean'] = float(np.mean(np.abs(amp_errors)))
        result['direction_error_deg_mean'] = float(np.mean(direction_errors))
        matched_gt_indices = [m[1] for m in matched]
        matched_energy = float(np.sum(gt_A[matched_gt_indices] ** 2))
        total_energy = float(np.sum(gt_A ** 2))
        result['energy_detection_rate'] = matched_energy / max(total_energy, 1e-12)
    else:
        for k in ('k_mag_rel_error_mean', 'omega_rel_error_mean',
                  'amp_rel_error_mean', 'direction_error_deg_mean'):
            result[k] = float('nan')
        result['energy_detection_rate'] = 0.0
    return result


def build_jobs(cells=None):
    cells = cells or list(CELLS)
    return [{'job_id': f'{c}|{ep}|{s}', 'cell': c, 'episode': ep, 'seed': s}
            for c in cells for ep in EPISODES for s in SEEDS]


def run_one(job):
    sys.path.insert(0, REPO)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    from wavesplat.run import run_estimation
    from wavesplat.data_loader import EpisodeDataLoader
    import wavesplat.optimizer.initializer as _init
    _rx4_install()                                   # (F4) no-op for every non-Rx-4 cell

    base = {k: job[k] for k in ('job_id', 'cell', 'episode', 'seed')}
    t0 = time.time()
    path = os.path.join(DATA, job['episode'])
    undo = None
    try:
        cfg = cell_config(job['cell'], job['episode'])
        cfg['seed'] = job['seed']
        gt_comp = np.asarray(EpisodeDataLoader(path).wave_components)
        undo = apply_cell(job['cell'])
        _init.ORACLE_COMPONENTS = gt_comp if job['cell'] in ORACLE_CELLS else None
        try:
            r = run_estimation(path, cfg)
        finally:
            _init.ORACLE_COMPONENTS = None

        frames = r['frame_results']
        avg = {k: float(np.nanmean([f[k] for f in frames if k in f])) for k in PANEL}
        vr = np.array([f['Variance_Ratio'] for f in frames if 'Variance_Ratio' in f], float)
        rho = np.array([f['Pearson_rho'] for f in frames if 'Pearson_rho' in f], float)
        # PREREG: |VR-1| is formed PER FRAME and then averaged (never mean-then-transform).
        abs_vr = float(np.nanmean(np.abs(vr - 1.0))) if vr.size else float('nan')
        # frame-wise orthogonal decomposition m = (1-rho^2) + (sqrt(v)-rho)^2
        with np.errstate(invalid='ignore'):
            decorr = 1.0 - rho ** 2
            shrink = (np.sqrt(np.clip(vr, 0, None)) - rho) ** 2
        ru = [f['RMSE_unobserved'] for f in frames
              if 'RMSE_unobserved' in f and np.isfinite(f.get('RMSE_unobserved', np.nan))]
        try:
            spec = spectral_accuracy(r['spectrum_params'], gt_comp)
        except Exception as se:
            spec = {'spec_error': repr(se)}
        sp = r.get('spectrum_params') or {}
        return {**base, **avg,
                'absVR_framewise': abs_vr,
                'decorr_mean': float(np.nanmean(decorr)) if rho.size else float('nan'),
                'shrink_mean': float(np.nanmean(shrink)) if rho.size else float('nan'),
                'm_mean': float(np.nanmean(decorr + shrink)) if rho.size else float('nan'),
                'avg_RMSE_unobserved': float(np.mean(ru)) if ru else float('nan'),
                'elapsed_ms': float(np.mean([f['elapsed_ms'] for f in frames]))
                if frames else float('nan'),
                'N': r.get('n_gaussians', -1),
                'init_nrmse': float(r.get('init_nrmse', float('nan'))),
                'init_rmse': float(r.get('init_rmse', float('nan'))),
                # (E4) recording only -- AUTO is a ROUTER, so which arm it picked (and
                # whether the median-of-3 analytic ensemble was built at all) has to be in
                # the record or "AUTO" is an unnamed bundle.  No computation changes.
                'auto_selected': r.get('auto_selected'),
                'auto_n_selected': r.get('auto_n_selected'),
                'ensemble_k': r.get('ensemble_k'),
                'final_n': r.get('final_n'),
                'spec': spec,
                # discipline #2 from tmp/placement_field_debate_2026-08-05/VERDICT.md:
                # dump atom coordinates so the matcher radius can be re-chosen post hoc.
                'kx': [float(v) for v in np.asarray(sp.get('kx', []))],
                'ky': [float(v) for v in np.asarray(sp.get('ky', []))],
                'w': [float(v) for v in np.asarray(sp.get('w', []))],
                'vr_frames': [float(v) for v in vr],
                'rho_frames': [float(v) for v in rho],
                'cfg': {k: v for k, v in cfg.items() if k != 'seed'},
                'wall_s': time.time() - t0}
    except Exception as e:
        import traceback
        return {**base, 'error': repr(e), 'tb': traceback.format_exc()[-1500:],
                'wall_s': time.time() - t0}
    finally:
        if undo is not None:
            undo()


def _init_worker():
    sys.path.insert(0, REPO)
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ.setdefault(v, '1')


def main():
    import multiprocessing as mp
    global OUT
    cells = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    OUT = os.environ.get('RX2_OUT', OUT)
    n_workers = int(os.environ.get('WORKERS', '8'))
    jobs = build_jobs(cells)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if 'error' not in r:
                done.add(r['job_id'])
    todo = [j for j in jobs if j['job_id'] not in done]
    print(f'total {len(jobs)} | done {len(done)} | todo {len(todo)} | workers {n_workers} '
          f'| seeds {SEEDS}', flush=True)
    if not todo:
        print('nothing to do'); return
    ctx = mp.get_context('spawn')
    t0 = time.time()
    with ctx.Pool(n_workers, initializer=_init_worker) as pool, open(OUT, 'a') as fh:
        for i, rec in enumerate(pool.imap_unordered(run_one, todo, chunksize=1), 1):
            fh.write(json.dumps(rec) + '\n'); fh.flush()
            el = time.time() - t0
            tag = rec.get('nRMSE', rec.get('error', '?'))
            if isinstance(tag, float):
                tag = f"{tag:.4f} init={rec.get('init_nrmse', float('nan')):.4f}"
            print(f'[{i}/{len(todo)}] {rec["job_id"]:<46s} {tag} '
                  f'ETA {el/i*(len(todo)-i)/60:.0f}min', flush=True)
    print(f'DONE {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
