"""W4 · arms `L-SHIP` / `L-ITER0` / `L-MERGE` + intervention counters (PREREG section 2 and 4).

`wavesplat/**` is NOT modified.  Everything here is config injection plus monkey-patches
installed from the batch directory, following the S16/S17 "package untouched" precedent and
the W3 `install_probe` precedent (`tmp/W3_SOLVER_2026-08-15/run_w3.py:181-253`).

WHAT THE COUNTERS COUNT
-----------------------
Not flags.  Events.  W3 burned 96 runs on `UNDECIDED-NO-INTERVENTION` because it read the
gate instead of the effect, so every counter here is a hook on the *statement that does the
work*:

  `adam_steps_phase1`  <- real `torch.optim.Adam.step()` calls inside the
                          `learned_init_iters` loop (`sequential_update.py:645-653`).
                          NOT `init_result['n_iters']`, which is the config value echoed
                          back (`sequential_update.py:673` returns `max_iters`, and
                          `run.py:461` prints it as `init_iters`) -- an intent table.
  `adam_steps_grow`    <- the SAME hook, attributed to the 50-step fine-tune inside
                          `_adaptive_prune_grow` (`sequential_update.py:1202-1207`).  This
                          separation is mandatory: without it `L-ITER0` would report
                          `adam_steps_phase1 == 50` on every episode where growth fires
                          (PREREG section 0.2: `add_gaussians` fires 17/96 on SHIP crowded)
                          and gate G-2 ("exactly 0") would fail for the wrong reason.
  `merge_removed`      <- real `len(kx_in) - len(kx_out)` across `learned_init.merge_atoms`
                          (`learned_init.py:190`), plus the `merge_frac` value the callee
                          actually received at `learned_init.py:369`.
  `clamp_*`            <- see the CO-FACTOR note below.

CO-FACTOR NOTE (PREREG section 2 labels `L-ITER0` "단일" -- read `recon/C1_arms.md`)
------------------------------------------------------------------------------------
`learned_init_iters=0` empties `for iteration in range(max_iters)`
(`sequential_update.py:645`).  The loop body holds FOUR statements, and the descent is only
three of them; the fourth is `self.model.clamp_steepness(self.max_steepness)`
(`sequential_update.py:650`), a projection onto the physical steepness constraint that
`L-SHIP` applies 500 times and `L-ITER0` applies ZERO times.  Nothing else re-applies it
before tracking, so `L-ITER0` can hand an unclamped `w` to `_adaptive_prune_grow` -- whose
prune rule is `w**2` energy (`sequential_update.py:998`).  The `clamp_*` counters measure
whether that second factor ever bites; if `clamp_hits_p1 == 0` across `L-SHIP` the treatment
is single-factor in effect, and the PREREG label stands.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ── arm definitions (PREREG section 2) ────────────────────────────────────────
# Every knob a W4 arm depends on is written out even when it equals the shipped default, so
# `effective_config` readback is unambiguous and a future default flip cannot silently
# redefine an arm (the CLAUDE.md lambda_l1 lesson: "기준선 팔이 λ 를 명시하는지 확인").
SHIPPED_LEARNED_INIT_ITERS = 500        # sequential_update.py:643 fallback
MERGE_FRAC = 0.5                        # PREREG section 2, frozen before any run


def base_config():
    """Config common to the whole A-family.  `lambda_l1` is imported, never retyped."""
    from wavesplat.optimizer.sequential_update import LAMBDA_L1_DEFAULT
    return {
        # G-1: the router is REMOVED, not measured.  `init_method != 'auto'` means
        # `_initialize_inner` skips the whole auto/real-gate block
        # (`sequential_update.py:445`) and dispatches straight to `_initialize_learned`
        # (`sequential_update.py:516-517`), so `route` is 100% learned by construction.
        'init_method': 'learned',
        'lambda_l1': LAMBDA_L1_DEFAULT,             # single source, sequential_update.py:40
        'learned_init_iters': SHIPPED_LEARNED_INIT_ITERS,
        'learned_merge_frac': None,                 # learned_init.py:367-369 no-op default
        'learned_konly': False,                     # -> ls_fit_wphi, learned_init.py:386
        'learned_peak_anchor': False,               # learned_init.py:360 no-op default
        'learned_energy_thresh': 0.03,
        'learned_dedup_dk': 0.03,
        'max_n': 30,
        'adaptive_n': True,
        'tracking_mode': 'w_phi',
        # inert here (make_updater reads it only when init_method=='auto',
        # sequential_update.py:1586) -- pinned so the readback cannot be misread.
        'analytic_ensemble_k': 1,
        'init_k_lr_scale': 1.0,                     # sequential_update.py:262
    }


#: arm name -> the delta applied on top of `base_config()`.
ARM_DELTAS = {
    'L-SHIP':     {},                                     # control, no intervention
    'L-SHIP-DUP': {},                                     # G-5 bit-determinism replicate
    'L-ITER0':    {'learned_init_iters': 0},              # PREREG section 2
    'L-MERGE':    {'learned_merge_frac': MERGE_FRAC},     # PREREG section 2
    # AMENDMENT_1 A-4: separates "400 more steps" from "a better parameterisation".
    # Without it, L-VP's gain is confounded with simply spending a longer optimisation
    # budget, and PREREG section 7 could not attribute the effect to VARPRO at all.
    'L-STEP400':  {'learned_init_iters': 900},            # 500 shipped + 400
}

#: gate G-2 / G-4 expectations, keyed by arm (PREREG section 4).
ARM_GATES = {
    'L-SHIP':     {'adam_steps_phase1': ('>', 0)},
    'L-SHIP-DUP': {'adam_steps_phase1': ('>', 0)},
    'L-ITER0':    {'adam_steps_phase1': ('==', 0)},
    'L-MERGE':    {'adam_steps_phase1': ('>', 0), 'merge_removed': ('>', 0)},
    'L-STEP400':  {'adam_steps_phase1': ('>', 0)},
}


def arm_config(arm, episode_seed=None, est_seed=None, extra=None):
    """Full config dict for `arm`.  Raises on an unknown arm -- never silently ships control."""
    if arm not in ARM_DELTAS:
        raise KeyError('unknown W4 arm %r (known: %s)' % (arm, sorted(ARM_DELTAS)))
    cfg = base_config()
    cfg.update(ARM_DELTAS[arm])
    if est_seed is not None:
        cfg['seed'] = int(est_seed)
        # learned_init.py:~330 defaults the diffusion seed to `seed`; pin it explicitly so
        # the proposal is provably identical across arms at a fixed est_seed (paired design).
        cfg['learned_diffusion_seed'] = int(est_seed)
    if extra:
        cfg.update(extra)
    return cfg


# ── intervention counters ─────────────────────────────────────────────────────
_PHASE = []          # phase stack; innermost wins
_C = {}


def _blank():
    return {
        'learned_init_calls': 0,        # G-1: `_initialize_learned` actually entered
        'adam_steps_phase1': 0,         # G-2
        'adam_steps_grow': 0,           # attributed away from phase1 on purpose
        'adam_steps_track': 0,
        'adam_steps_other': 0,
        'merge_calls': 0,               # G-4
        'merge_removed': 0,             # G-4
        'merge_frac_seen': [],          # the value the CALLEE received
        'n_atoms_pooled': None,         # pre-merge (pool_and_select output)
        'n_atoms_after_merge': None,
        'clamp_calls_p1': 0,            # co-factor probe, see module docstring
        'clamp_hits_p1': 0,
        'clamp_calls_grow': 0,
        'clamp_hits_grow': 0,
        'steepness_max_post_p1': None,  # measured at prune/grow entry = end of the P1 loop
        'steepness_max_post_init': None,
        'prune_grow_calls': 0,
        'n_atoms_pre_prune': None,
        'n_atoms_post_init': None,
    }


def counters_reset():
    """Call once per (episode, arm, est_seed) row, BEFORE `run_estimation`."""
    _C.clear()
    _C.update(_blank())
    del _PHASE[:]
    return _C


def counters_snapshot():
    """Plain-JSON copy of the counters for the result row."""
    out = dict(_C)
    out['merge_frac_seen'] = list(out.get('merge_frac_seen', []))
    return out


class _Phase(object):
    """Phase stack that survives exceptions -- a raised episode must not poison the next row."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        _PHASE.append(self.name)
        return self

    def __exit__(self, *exc):
        if _PHASE and _PHASE[-1] == self.name:
            _PHASE.pop()
        return False


def _cur():
    return _PHASE[-1] if _PHASE else None


def _max_steepness(model):
    """max |k|*|w| over the live model -- passive, never mutates."""
    import torch
    if model is None:
        return None
    try:
        with torch.no_grad():
            km = torch.sqrt(model.kx ** 2 + model.ky ** 2 + 1e-12)
            return float((km * torch.abs(model.w)).max().item())
    except Exception:
        return None


def install_counters():
    """Patch the five call sites that DO the work.  Returns (counters, undo)."""
    import torch
    import wavesplat.optimizer.learned_init as L
    from wavesplat.optimizer.sequential_update import SequentialUpdater as S
    from wavesplat.model.spectral_gaussian import SpectralGaussianModel as M

    counters_reset()

    o_step = torch.optim.Adam.step
    o_learned = S._initialize_learned
    o_grow = S._adaptive_prune_grow
    o_clamp = M.clamp_steepness
    o_merge = L.merge_atoms
    o_pool = L.pool_and_select

    _KEY = {'p1': 'adam_steps_phase1', 'grow': 'adam_steps_grow',
            'track': 'adam_steps_track'}

    def step(self, *a, **k):
        _C[_KEY.get(_cur(), 'adam_steps_other')] += 1
        return o_step(self, *a, **k)

    def initialize_learned(self, frames, *a, **k):
        _C['learned_init_calls'] += 1
        with _Phase('p1'):
            out = o_learned(self, frames, *a, **k)
        _C['steepness_max_post_init'] = _max_steepness(getattr(self, 'model', None))
        _C['n_atoms_post_init'] = (self.model.n_gaussians
                                   if getattr(self, 'model', None) is not None else None)
        return out

    def prune_grow(self, frames, *a, **k):
        # Entry here is exactly the end of the `learned_init_iters` loop
        # (`sequential_update.py:658-660`), so this is the post-P1 / pre-prune state.
        _C['prune_grow_calls'] += 1
        _C['steepness_max_post_p1'] = _max_steepness(getattr(self, 'model', None))
        _C['n_atoms_pre_prune'] = (self.model.n_gaussians
                                   if getattr(self, 'model', None) is not None else None)
        with _Phase('grow'):
            return o_grow(self, frames, *a, **k)

    def clamp(self, max_steepness=0.15):
        ph = _cur()
        if ph in ('p1', 'grow'):
            _C['clamp_calls_%s' % ('p1' if ph == 'p1' else 'grow')] += 1
            with torch.no_grad():
                km = torch.sqrt(self.kx ** 2 + self.ky ** 2 + 1e-12)
                hit = bool((km * torch.abs(self.w) > max_steepness).any().item())
            if hit:
                _C['clamp_hits_%s' % ('p1' if ph == 'p1' else 'grow')] += 1
        return o_clamp(self, max_steepness)

    def pool_and_select(kx, ky, energy, *a, **k):
        out = o_pool(kx, ky, energy, *a, **k)
        _C['n_atoms_pooled'] = int(len(out[0]))
        return out

    def merge_atoms(kx, ky, energy, merge_frac):
        n_in = int(len(kx))
        out = o_merge(kx, ky, energy, merge_frac)
        n_out = int(len(out[0]))
        _C['merge_calls'] += 1
        _C['merge_removed'] += max(0, n_in - n_out)
        _C['merge_frac_seen'].append(None if merge_frac is None else float(merge_frac))
        _C['n_atoms_after_merge'] = n_out
        return out

    torch.optim.Adam.step = step
    S._initialize_learned = initialize_learned
    S._adaptive_prune_grow = prune_grow
    M.clamp_steepness = clamp
    L.pool_and_select = pool_and_select
    L.merge_atoms = merge_atoms

    def undo():
        torch.optim.Adam.step = o_step
        S._initialize_learned = o_learned
        S._adaptive_prune_grow = o_grow
        M.clamp_steepness = o_clamp
        L.pool_and_select = o_pool
        L.merge_atoms = o_merge
        del _PHASE[:]

    return _C, undo


def install_tracking_phase():
    """Attribute tracker Adam steps to 'track' so `adam_steps_other` stays a real alarm."""
    from wavesplat.optimizer.sequential_update import SequentialUpdater as S
    o_update = S.update

    def update(self, frame, *a, **k):
        with _Phase('track'):
            return o_update(self, frame, *a, **k)

    S.update = update

    def undo():
        S.update = o_update

    return undo


# ── effective_config: READ THE LIVE OBJECT, never the input dict ──────────────
_UPDATER = {}

#: live attributes set in `SequentialUpdater.__init__` (sequential_update.py:171-229).
_ATTR_KEYS = ('lambda_l1', 'lambda_smooth', 'lr', 'tracking_lr', 'tracking_mode',
              'tracking_steps', 'init_method', 'adaptive_n', 'prune_threshold',
              'grow_residual_threshold', 'max_n', 'stages', 'use_dispersion',
              'use_confidence', 'depth', 'seed', 'device', 'tracking_window',
              'max_steepness', 'real_estimator')

#: keys with NO attribute -- read at call time straight out of `self.config`
#: (`learned_init_iters` at sequential_update.py:643, the rest inside
#: `learned_init.learned_init_params`).  Reading `u.config` reads the LIVE dict the
#: updater will consult, which is `run.py`'s `cfg`, not the dict we passed in.
_CFG_KEYS = ('learned_init_iters', 'learned_merge_frac', 'learned_konly',
             'learned_peak_anchor', 'learned_energy_thresh', 'learned_dedup_dk',
             'learned_nsamp', 'learned_diffusion_seed', 'learned_ckpt',
             'analytic_ensemble_k', 'init_k_lr_scale', 'auto_switch_thresh',
             'real_domain_gate')


def install_updater_capture():
    """`run_estimation` never returns the updater, so grab it at construction (W3 precedent)."""
    import wavesplat.run as R
    orig = R.make_updater

    def cap(config):
        u = orig(config)
        _UPDATER['u'] = u
        return u

    R.make_updater = cap

    def undo():
        R.make_updater = orig
        _UPDATER.pop('u', None)

    return undo


def _scalar(v):
    import numpy as np
    if isinstance(v, (list, tuple)):
        return [_scalar(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    return str(v)


def effective_config(input_cfg=None):
    """Object readback of the captured updater.  S13b discipline: never an intent copy."""
    u = _UPDATER.get('u')
    if u is None:
        return {'_absent': True}
    members = getattr(u, 'members', None)
    core = members[0] if members else u
    live = getattr(core, 'config', None)
    out = {'updater_class': type(u).__name__, 'core_class': type(core).__name__,
           'n_members': len(members) if members else 1,
           # proof this is not the intent table we handed in
           'config_is_input_dict': (live is input_cfg) if input_cfg is not None else None,
           'config_id_matches_wrapper': (live is getattr(u, 'config', None))}
    for k in _ATTR_KEYS:
        out[k] = _scalar(getattr(core, k, '<absent>'))
    for k in _CFG_KEYS:
        out['cfg:' + k] = _scalar(live.get(k, '<absent>')) if isinstance(live, dict) \
            else '<no-config>'
    out['auto_selected'] = _scalar(getattr(core, '_auto_selected', None))
    out['auto_real_gate'] = _scalar(getattr(core, '_auto_real_gate', None))
    if members:
        out['member_lambda_l1'] = [_scalar(getattr(m, 'lambda_l1', None)) for m in members]
    return out


# ── gate evaluation (PREREG section 4, G-1/G-2/G-4) ───────────────────────────
_OPS = {'>': lambda a, b: a > b, '==': lambda a, b: a == b, '>=': lambda a, b: a >= b}


def check_gates(arm, counters=None):
    """Per-row gate verdict.  Batch-level G-4 (median > 0) is the runner's job."""
    c = counters if counters is not None else _C
    fails = []
    if c.get('learned_init_calls', 0) != 1:
        fails.append('G-1 learned_init_calls=%r != 1' % c.get('learned_init_calls'))
    for key, (op, want) in ARM_GATES.get(arm, {}).items():
        got = c.get(key)
        if got is None or not _OPS[op](got, want):
            fails.append('G-2/G-4 %s: got %r, need %s %r' % (key, got, op, want))
    if c.get('adam_steps_other', 0):
        fails.append('unattributed Adam steps: %d' % c['adam_steps_other'])
    return {'gates_pass': not fails, 'gate_fails': fails}


# ── dry-run: does the injection REACH the object / the callee?  No episode. ────
def _dryrun():
    import numpy as np
    import wavesplat.optimizer.learned_init as L
    from wavesplat.optimizer.sequential_update import SequentialUpdater

    ok = True

    def say(tag, cond, detail=''):
        nonlocal ok
        ok = ok and bool(cond)
        print('%-4s %-34s %s' % ('PASS' if cond else 'FAIL', tag, detail))

    counters, undo = install_counters()
    try:
        # --- 1. arm configs are distinct in exactly the intended key -------------------
        base = arm_config('L-SHIP', est_seed=42)
        for arm in ('L-ITER0', 'L-MERGE'):
            cfg = arm_config(arm, est_seed=42)
            diff = {k for k in set(base) | set(cfg) if base.get(k, object()) != cfg.get(k)}
            say('%s single-key delta' % arm, diff == set(ARM_DELTAS[arm]), sorted(diff))

        # --- 2. config reaches the LIVE object (no whitelist anywhere on the path) -----
        # run.py:139-141 does `DEFAULT_CONFIG.copy()` + `update(config)` with no key
        # filter, and sequential_update.py:171 stores the reference.  Construct the
        # updater the same way run.py would and read the object back.
        from wavesplat.run import DEFAULT_CONFIG, make_updater
        for arm in ('L-SHIP', 'L-ITER0', 'L-MERGE'):
            inj = arm_config(arm, est_seed=42)
            cfg = DEFAULT_CONFIG.copy()
            cfg.update(inj)
            u = make_updater(cfg)
            _UPDATER['u'] = u
            eff = effective_config(input_cfg=inj)
            say('%s -> SequentialUpdater' % arm,
                eff['updater_class'] == 'SequentialUpdater'
                and eff['init_method'] == 'learned',
                eff['updater_class'])
            say('%s cfg:learned_init_iters' % arm,
                eff['cfg:learned_init_iters'] == inj['learned_init_iters'],
                eff['cfg:learned_init_iters'])
            say('%s cfg:learned_merge_frac' % arm,
                eff['cfg:learned_merge_frac'] == inj['learned_merge_frac'],
                eff['cfg:learned_merge_frac'])
            say('%s readback is not intent copy' % arm,
                eff['config_is_input_dict'] is False, 'live dict is run.py cfg')
        _UPDATER.pop('u', None)

        # --- 3. merge path end-to-end, WITHOUT an episode -----------------------------
        # `learned_atoms` short-circuits the diffusion proposal (learned_init.py:~348) and
        # `frames=[]` makes ls_fit_wphi return zeros via _gather_init_obs -> None
        # (learned_init.py:89-90).  So this exercises pool_and_select -> merge_atoms ->
        # install with zero episode I/O, zero GPU and zero estimator steps.
        # Atoms chosen so pool_and_select keeps all 5 (every pair differs by >= dedup_dk
        # 0.03 in kx, learned_init.py:181) while merge_atoms collapses them to 2 at
        # r = 0.5 * kp = 0.15 (learned_init.py:206-208).  That separation is the whole
        # point of L-MERGE: duplicates INSIDE one main lobe that dedup cannot see.
        atoms = dict(kx=np.array([0.30, 0.35, 0.40, 0.80, 0.86]),
                     ky=np.array([0.00, 0.05, 0.10, 0.00, 0.06]),
                     energy=np.array([1.0, 0.9, 0.8, 0.7, 0.6]))

        counters_reset()
        cfg_ship = arm_config('L-SHIP', est_seed=42)
        cfg_ship['learned_atoms'] = atoms
        _, _, _, _, n_ship = L.learned_init_params([], 50.0, cfg_ship, seed=42)
        say('L-SHIP merge is a no-op',
            counters['merge_calls'] == 0 and counters['merge_removed'] == 0,
            'n=%d pooled=%s' % (n_ship, counters['n_atoms_pooled']))

        counters_reset()
        cfg_merge = arm_config('L-MERGE', est_seed=42)
        cfg_merge['learned_atoms'] = atoms
        _, _, _, _, n_merge = L.learned_init_params([], 50.0, cfg_merge, seed=42)
        say('L-MERGE calls merge_atoms', counters['merge_calls'] == 1,
            'frac_seen=%s' % counters['merge_frac_seen'])
        say('L-MERGE frac reached callee',
            counters['merge_frac_seen'] == [MERGE_FRAC], counters['merge_frac_seen'])
        say('merge_removed counts real removals',
            counters['merge_removed'] == n_ship - n_merge and counters['merge_removed'] > 0,
            '%d -> %d (removed %d)' % (n_ship, n_merge, counters['merge_removed']))
        say('G-4 gate reads the counter',
            check_gates('L-MERGE', dict(counters, learned_init_calls=1,
                                        adam_steps_phase1=500))['gates_pass'])

        # --- 4. Adam hook fires on the REAL torch symbol ------------------------------
        import torch
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.Adam([p])
        p.grad = torch.ones(2)
        counters_reset()
        with _Phase('p1'):
            opt.step()
            opt.step()
        with _Phase('grow'):
            opt.step()
        with _Phase('track'):
            opt.step()
        opt.step()
        say('Adam step attribution',
            (counters['adam_steps_phase1'], counters['adam_steps_grow'],
             counters['adam_steps_track'], counters['adam_steps_other']) == (2, 1, 1, 1),
            str([counters['adam_steps_phase1'], counters['adam_steps_grow'],
                 counters['adam_steps_track'], counters['adam_steps_other']]))

        # --- 5. the phase wrappers are actually installed on the real methods ---------
        say('_initialize_learned wrapped',
            SequentialUpdater._initialize_learned.__name__ == 'initialize_learned')
        say('_adaptive_prune_grow wrapped',
            SequentialUpdater._adaptive_prune_grow.__name__ == 'prune_grow')
        # grow-phase nesting is what keeps G-2 honest: prove the stack, not the estimator
        counters_reset()
        with _Phase('p1'):
            with _Phase('grow'):
                opt.step()
            opt.step()
        say('grow nests inside p1 (G-2 safe)',
            (counters['adam_steps_phase1'], counters['adam_steps_grow']) == (1, 1))

        # --- 6. clamp counter distinguishes call from hit -----------------------------
        from wavesplat.model.spectral_gaussian import SpectralGaussianModel
        m = SpectralGaussianModel(2, depth=50.0)
        with torch.no_grad():
            m.kx.copy_(torch.tensor([1.0, 1.0]))
            m.ky.copy_(torch.tensor([0.0, 0.0]))
            m.w.copy_(torch.tensor([0.01, 0.01]))
        counters_reset()
        with _Phase('p1'):
            m.clamp_steepness(0.15)
        say('clamp: call without hit',
            (counters['clamp_calls_p1'], counters['clamp_hits_p1']) == (1, 0))
        with torch.no_grad():
            m.w.copy_(torch.tensor([0.90, 0.01]))
        with _Phase('p1'):
            m.clamp_steepness(0.15)
        say('clamp: hit detected',
            (counters['clamp_calls_p1'], counters['clamp_hits_p1']) == (2, 1),
            'max steep now %.4f' % _max_steepness(m))

        # --- 7. undo restores every symbol --------------------------------------------
        undo()
        say('undo restores symbols',
            SequentialUpdater._initialize_learned.__name__ == '_initialize_learned'
            and L.merge_atoms.__name__ == 'merge_atoms'
            and torch.optim.Adam.step.__name__ == 'step')
    finally:
        try:
            undo()
        except Exception:
            pass
    print('\nDRYRUN %s' % ('OK' if ok else 'FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(_dryrun())
