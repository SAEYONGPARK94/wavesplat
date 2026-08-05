"""e39 STEP 1 — settle the l1s2bw TRAINING-GAP question bit-exactly.

Question: did the incumbent's training data (fan0 -> l1s2 head stage;
bwtask -> l1s2bw final stage; fan1/fan2 -> v6fan/v7fan pretrain) contain
cross-Tp crossing seas (gen_sample's bimodal_frac augmentation)?
Behavioural detectors failed calibration (2026-07-17); this script instead
replays the generator's RNG STREAM sample-by-sample and matches the stored
npz meta exactly (float32 of continuous draws => a match is ~2^-24 per
field per sample; over thousands of samples a full match is proof).

Method: gen_sample(rng, fan_frac, bimodal_frac, bw_inject) consumes draws
from ONE shared numpy Generator in a fixed order (verified against HEAD =
f1f85c8 lineage; fan0/fan1/fan2 generated at a4b6df5 which is stream-
identical to HEAD for bw_inject=0; bwtask generated exactly at f1f85c8):

  per sample: Tp,Hs,gam,spread,mdir ~5 uniforms; seed ~1 integers;
    u_b ~1 uniform  [bimodal gate — ALWAYS drawn]
    if u_b < bimodal_frac: r2, tp2f, diroff ~3 uniforms
    uc_mag, uc_ang ~2 uniforms
    if bw_inject > 0: bw_lvl ~1 uniform   [dataset-level flag]
    rsel ~1 uniform
    fan   : U_abs ~1;  rr ~400; aa ~400
    fixed : sample_obs ~800 (th 400 + rr 400)
    moving: ucp ~1; sample_obs ~800
    nfrac ~1 uniform                       [stored in meta col 4]
    noise ~normal(400)
    fan   : tq_e ~128; rrq ~128; aaq ~128
    else  : qx ~256; qy ~256

Per sample we know from the stored npz: fan flag (Q time column non-
constant), ucp (meta col 3; 0 => 'fixed'), Tp (col 2), Hs (col 1),
nfrac (col 4). We fork the stream at the bimodal gate (save/restore
bit_generator state), try {unimodal, bimodal}, and accept the branch that
reproduces (Tp, Hs, ucp, nfrac) to float32 exactly. Afterwards:
  bimodal_frac interval = (max u_b over accepted-bimodal,
                           min u_b over accepted-unimodal]
  fan_frac     interval = analogous over rsel with 3-way branch structure.
A dataset where every sample resolves 'unimodal' with a consistent gate
=> NO crossing seas, bit-exact.

Finally, for bwtask, the bw_inject VALUE is identified by regenerating
full samples through the real gen_sample and matching stored z (float32)
over candidate values.

Usage: python3 tmp/e39_provenance.py            # all datasets, stream match
       python3 tmp/e39_provenance.py --full_z   # + full-z regen check (slow)
Output: results/direction3/e39_provenance.json
"""
import os
import sys
import glob
import json
import argparse

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'scripts')

N_OBS = 400
OUT = 'results/direction3/e39_provenance.json'

DATASETS = {
    # stem: shards to test (None = all)
    'fan0': None,
    'bwtask': None,
    'fan1': 8,          # first 8 shards (4000 samples) is plenty of proof
    'fan2': 8,
    'cleanmatch': None,
}


def load_meta_q(stem, max_shards=None):
    metas, fans = [], []
    files = sorted(glob.glob(f'tmp/prior_train/{stem}_s*.npz'))
    if max_shards:
        files = files[:max_shards]
    for p in files:
        d = np.load(p)
        metas.append(d['meta'])
        # fan queries: tq_e = T + U(0,10) for 128 pts (std ~ 2.9 s);
        # nowcast: tq == T constant (float32 rounding std ~ 1e-5) -> 0.01
        fans.append(d['Q'][:, :, 2].std(axis=1) > 0.01)
    return np.concatenate(metas), np.concatenate(fans), len(files)


def f32(x):
    return np.float32(x)


def consume_mode_and_check(rng, mode, meta_row, Tp, Hs, cp):
    """Consume mode-branch + nfrac + noise + query draws; return
    (ok, rsel, nfrac). meta_row = (kp_e, Hs, Tp, ucp, nfrac) float32."""
    rsel = rng.uniform()
    if mode == 'fan':
        U_abs = rng.uniform(1.0, 12.0)
        ucp = U_abs / cp
        rng.uniform(20.0, 100.0, N_OBS)
        rng.uniform(-np.pi / 3, np.pi / 3, N_OBS)
    elif mode == 'fixed':
        ucp = 0.0
        rng.uniform(0, 2 * np.pi, N_OBS)      # sample_obs th
        rng.uniform(0, 1, N_OBS)              # sample_obs rr
    else:                                     # moving
        ucp = rng.uniform(0.1, 1.2)
        rng.uniform(0, 2 * np.pi, N_OBS)
        rng.uniform(0, 1, N_OBS)
    nfrac = rng.uniform(0.02, 0.10)
    ok = (f32(ucp) == meta_row[3]) and (f32(nfrac) == meta_row[4])
    if not ok:
        return False, rsel, nfrac
    rng.normal(0, nfrac * Hs, N_OBS)          # z noise
    if mode == 'fan':
        rng.uniform(0, 10.0, 128)             # tq_e
        rng.uniform(10.0, 110.0, 128)         # rrq
        rng.uniform(-np.pi / 3, np.pi / 3, 128)  # aaq
    else:
        rng.uniform(0, 1, 256)                # qx (bounds irrelevant)
        rng.uniform(0, 1, 256)                # qy
    return True, rsel, nfrac


def dispersion_lam_local(Tp, depth=50.0):
    from d3_sa_achievability_gap import dispersion_lam
    return dispersion_lam(Tp, depth)


def match_stream(stem, seed, bw_flag, max_shards=None, verbose=False):
    """Replay the stream; adaptively resolve each sample's bimodal branch.
    Returns dict with per-sample branch labels + parameter intervals, or
    None on failure."""
    meta, fanflag, nsh = load_meta_q(stem, max_shards)
    n = len(meta)
    rng = np.random.default_rng(seed)
    ub_uni, ub_bi = [], []                 # bimodal-gate draws by outcome
    rsel_by_mode = {'fan': [], 'fixed': [], 'moving': []}
    n_bi = 0
    fails = []
    for i in range(n):
        Tp = rng.uniform(5.0, 11.0)
        Hs = rng.uniform(1.0, 3.0)
        if not (f32(Tp) == meta[i][2] and f32(Hs) == meta[i][1]):
            fails.append((i, 'TpHs'))
            if len(fails) > 3:
                return None
            continue
        rng.uniform(1.5, 6.0)                 # gam_j
        rng.uniform(15.0, 45.0)               # spread
        rng.uniform(0.0, 360.0)               # mdir
        rng.integers(0, 2 ** 31 - 1)          # jonswap seed
        u_b = rng.uniform()                   # bimodal gate
        lam = dispersion_lam_local(Tp)
        cp = lam / Tp
        mode = ('fan' if fanflag[i]
                else ('fixed' if meta[i][3] == 0.0 else 'moving'))

        state = rng.bit_generator.state       # fork point
        # ---- hypothesis U: unimodal
        rng.uniform(0.0, 0.4)                 # uc_mag
        rng.uniform(0, 2 * np.pi)             # uc_ang
        if bw_flag:
            rng.uniform()                     # bw_lvl (0..bw_inject scale-free)
        okU, rselU, _ = consume_mode_and_check(rng, mode, meta[i], Tp, Hs, cp)
        if okU:
            ub_uni.append(u_b)
            rsel_by_mode[mode].append(rselU)
            continue
        # ---- hypothesis B: bimodal (+3 draws before uc)
        rng.bit_generator.state = state
        rng.uniform(0.15, 0.5)                # r2
        rng.uniform(0.6, 1.6)                 # tp2 factor
        rng.uniform(40, 140)                  # direction offset
        rng.uniform(0.0, 0.4)
        rng.uniform(0, 2 * np.pi)
        if bw_flag:
            rng.uniform()
        okB, rselB, _ = consume_mode_and_check(rng, mode, meta[i], Tp, Hs, cp)
        if okB:
            n_bi += 1
            ub_bi.append(u_b)
            rsel_by_mode[mode].append(rselB)
            continue
        fails.append((i, 'branch'))
        if verbose:
            print(f"    sample {i}: both branches fail "
                  f"(mode {mode}, meta {meta[i]})")
        if len(fails) > 3:
            return None
    if fails:
        return None
    # parameter intervals
    bf_lo = max(ub_bi) if ub_bi else 0.0
    bf_hi = min(ub_uni) if ub_uni else 1.0
    ff_lo = max(rsel_by_mode['fan']) if rsel_by_mode['fan'] else 0.0
    other = rsel_by_mode['fixed'] + rsel_by_mode['moving']
    ff_hi = min(other) if other else 1.0
    return dict(stem=stem, seed=seed, bw_flag=bool(bw_flag), n=n,
                n_shards=nsh, n_bimodal=int(n_bi),
                n_fan=int(sum(len(rsel_by_mode['fan']) for _ in [0])),
                bimodal_frac_interval=[float(bf_lo), float(bf_hi)],
                fan_frac_interval=[float(ff_lo), float(ff_hi)],
                fan_count=len(rsel_by_mode['fan']),
                fixed_count=len(rsel_by_mode['fixed']),
                moving_count=len(rsel_by_mode['moving']))


def find_seed(stem, max_seed=2100):
    """First sample's (Tp,Hs) pins the seed instantly."""
    meta, _, _ = load_meta_q(stem, 1)
    for s in range(max_seed):
        r = np.random.default_rng(s)
        Tp = r.uniform(5.0, 11.0)
        Hs = r.uniform(1.0, 3.0)
        if f32(Tp) == meta[0][2] and f32(Hs) == meta[0][1]:
            return s
    return None


def full_z_check(stem, seed, fan_frac, bimodal_frac, bw_candidates,
                 n_check=25):
    """Regenerate the first n_check samples through the REAL gen_sample and
    compare stored z bit-exactly, per candidate bw_inject."""
    from d3_prior_net_data import gen_sample
    d = np.load(sorted(glob.glob(f'tmp/prior_train/{stem}_s*.npz'))[0])
    zs, metas = d['z'], d['meta']
    out = {}
    for bw in bw_candidates:
        rng = np.random.default_rng(seed)
        n_exact, worst = 0, 0.0
        for i in range(n_check):
            s = gen_sample(rng, fan_frac, bimodal_frac, bw)
            dz = float(np.max(np.abs(s['z'].astype(np.float32) - zs[i])))
            worst = max(worst, dz)
            n_exact += int(dz == 0.0)
        out[str(bw)] = dict(n_exact=n_exact, n_check=n_check,
                            worst_abs_dz=worst)
        print(f"    bw_inject={bw}: exact z {n_exact}/{n_check} "
              f"(worst |dz| {worst:.2e})", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--full_z', action='store_true')
    a = ap.parse_args()
    results = {}
    for stem, msh in DATASETS.items():
        print(f"\n=== {stem} ===", flush=True)
        seed = find_seed(stem)
        if seed is None:
            print("  seed not in 0..49 — SKIP (report unknown)")
            results[stem] = dict(seed=None)
            continue
        print(f"  seed = {seed}")
        got = None
        for bwf in (False, True):
            r = match_stream(stem, seed, bwf, msh)
            if r is not None:
                got = r
                break
        if got is None:
            print("  ** NO stream match (code drift or exotic params) **")
            results[stem] = dict(seed=seed, matched=False)
            continue
        got['matched'] = True
        print(f"  MATCH: n={got['n']} samples, bw_draw={got['bw_flag']}, "
              f"bimodal samples = {got['n_bimodal']} "
              f"(bimodal_frac in ({got['bimodal_frac_interval'][0]:.4f}, "
              f"{got['bimodal_frac_interval'][1]:.4f}]), "
              f"fan/fixed/moving = {got['fan_count']}/{got['fixed_count']}"
              f"/{got['moving_count']} "
              f"(fan_frac in ({got['fan_frac_interval'][0]:.4f}, "
              f"{got['fan_frac_interval'][1]:.4f}])", flush=True)
        results[stem] = got

    if a.full_z:
        print("\n=== full-z regen check (gold standard) ===", flush=True)
        for stem, bw_cands in [('bwtask', [0.05, 0.1, 0.15, 0.2, 0.3]),
                               ('fan0', [0.0])]:
            r = results.get(stem, {})
            if not r.get('matched'):
                continue
            # a point value inside the interval; conventional values first
            ffi = r['fan_frac_interval']
            bfi = r['bimodal_frac_interval']
            ff = next((v for v in (0.5, 0.57, 0.6, 2 / 3, 0.75, 0.8, 0.875,
                                   0.9, 1.0) if ffi[0] < v <= ffi[1]),
                      0.5 * (ffi[0] + ffi[1]))
            bf = next((v for v in (0.0, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5)
                       if bfi[0] < v <= bfi[1]), 0.5 * (bfi[0] + bfi[1]))
            print(f"  {stem}: testing fan_frac={ff} bimodal_frac={bf}",
                  flush=True)
            results[stem]['z_check'] = full_z_check(
                stem, r['seed'], ff, bf,
                bw_cands if r['bw_flag'] else [0.0])
            results[stem]['z_check_params'] = dict(fan_frac=ff,
                                                   bimodal_frac=bf)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, 'w'), indent=1, default=float)
    print(f"\nsaved -> {OUT}")


if __name__ == '__main__':
    main()
