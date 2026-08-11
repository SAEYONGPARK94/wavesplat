"""S9 사전등록용 MDE + **근거리 등가한계 δ** 산출 (결과 관측 **전** 실행·동결).

## 무엇이 S8 과 다른가
1. ★**상수를 상속하지 않는다.** `s9_constants.py` 가 (n, k) 에서 **유도**한다.
   S8 은 `MDE_SIGN80_C = 0.7234` 를 파일에 적어 두었고 그것은 **(n=32, k=23) 전용**이다.
   S9 희소는 n=64·k=41 이라 계수가 **0.4720** — 상속하면 근거리 문턱이 **1.533배 느슨**해진다.
2. ★**근거리 등가한계 `delta_shell` 을 새로 낸다.** S9 는 근거리를 **부재 증명이 아니라
   이동 널 부호검정**으로 판정한다(PREREG §4). 그 δ 가 여기서 나온다.
3. **분산 출처 = S8 `results.jsonl`** — S9 가 쓸 바로 그 팔(`AUTO`, `AUTO-L1-1e4`)의 산포를
   담은 가장 최신 배치다.

## 순환논리가 아닌 이유
여기서 읽는 것은 **σ(산포)** 와 **기준선 수준**뿐이고, **처치 효과 크기(중앙 Δ)는 쓰지 않는다.**
σ 는 S8 의 에피소드(시드 32-39)에서 나오고 S9 는 **신선 에피소드(시드 48-63)** 에서 잰다
⟹ 설계용 **사전 추정치**다. ⚠S9 종료 후 실제 산포로 재도출해 **병기**한다(사후 재선택이 아니라 점검).
이 스크립트는 **S9 결과 파일을 읽지 않는다**(존재하지도 않는다).
"""
import json
import os
import statistics as st

import numpy as np

import s9_constants as C

HERE = os.path.dirname(os.path.abspath(__file__))
S8 = os.path.abspath(os.path.join(HERE, '..', 'S8_deploy_confirm_2026-08-10', 'results.jsonl'))
OUT = os.path.join(HERE, 'MDE_prespec_s9.json')

METRIC = 'nRMSE'
BASE = 'AUTO'                      # 출하 기본값 = S9 의 모든 대조 기준선
TREAT = 'AUTO-L1-1e4'              # S8 에서의 같은 처치 (산포 출처. ⛔효과 크기 미사용)
NEAR_SHELLS = ('0_5', '5_10', '10_20')
ALL_SHELLS = ('0_5', '5_10', '10_20', '20_30', '30_50', '50_75', '75_100', '100_150', '150_inf')


def load():
    recs = []
    for line in open(S8):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if 'error' not in r and r.get('cell') in (BASE, TREAT):
            recs.append(r)
    return recs


def per_episode(recs, cell, regime, getter):
    """에피소드당 = 3 추정기 시드의 중앙 (승계 규약)."""
    by_ep = {}
    for r in recs:
        if r['cell'] != cell or r['regime'] != regime:
            continue
        v = getter(r)
        if v is None or not np.isfinite(v):
            continue
        by_ep.setdefault(r['episode'], []).append(float(v))
    return {ep: st.median(v) for ep, v in by_ep.items() if v}


def robust_sigma(vals):
    """MAD 기반 robust 산포. ⛔효과 크기가 아니라 **퍼짐**만 쓴다."""
    a = np.asarray(vals, float)
    return float(1.4826 * np.median(np.abs(a - np.median(a)))) if a.size else float('nan')


def paired_spread(recs, regime, getter):
    """짝 Δ 의 robust 산포와 기준선 수준. ⚠중앙 Δ(효과 크기)는 **반환하지 않는다**."""
    t = per_episode(recs, TREAT, regime, getter)
    b = per_episode(recs, BASE, regime, getter)
    eps = sorted(set(t) & set(b))
    d = [t[e] - b[e] for e in eps]
    base_level = st.median([b[e] for e in eps]) if eps else float('nan')
    return robust_sigma(d), base_level, len(eps)


def main():
    recs = load()
    out = {
        'batch': 'S9',
        'source': os.path.relpath(S8, os.path.dirname(HERE)),
        'source_note': (
            'sigma + baseline levels from S8 episodes (seed 32-39); S9 scores FRESH episodes '
            '(seed 48-63). Design-time prior only. Effect sizes are NOT read. '
            'obs/MDE must be compared WITHIN regime, never across.'),
        'n_records': len(recs),
        'metric': METRIC,
        'constants_note': ('all n-dependent constants DERIVED by s9_constants.py; '
                           'S8 hardcoded MDE_SIGN80_C=0.7234 is (n=32,k=23)-only and is NOT inherited'),
        'by_regime': {},
    }

    for rg in ('crowded', 'sparse'):
        n, k = C.N[rg], C.primary_k(C.N[rg])
        c80 = C.mde_sign80_c(n, k)

        sig_d, base_level, n_pairs = paired_spread(recs, rg, lambda r: r.get(METRIC))
        mde_sign80 = c80 * sig_d
        # mde_agg: 시드잡음 기반 집계 MDE (승계식, N 만 레짐별로)
        ranges = []
        for cell in (BASE, TREAT):
            by_ep = {}
            for r in recs:
                if r['cell'] == cell and r['regime'] == rg and np.isfinite(r.get(METRIC, np.nan)):
                    by_ep.setdefault(r['episode'], []).append(float(r[METRIC]))
            ranges += [max(v) - min(v) for v in by_ep.values() if len(v) >= 2]
        sigma_hat = float(np.median(ranges)) / C.D2_N3 if ranges else float('nan')
        sigma_d = sigma_hat * np.sqrt(2.0) * 1.16 / np.sqrt(3.0)
        mde_agg = C.MDE_MULT * C.MEDIAN_SE * sigma_d / np.sqrt(n)

        # ★근거리 등가한계 δ (PREREG §4) — 셸별 상대 스케일
        shells = {}
        for s in ALL_SHELLS:
            sig_s, base_s, np_s = paired_spread(recs, rg, lambda r, _s=s: r['shell_rmse'].get(_s))
            delta_abs = c80 * sig_s
            shells[s] = dict(
                sigma_delta_robust=sig_s, baseline_median=base_s, n_pairs=np_s,
                delta_abs=float(delta_abs),
                delta_rel=float(delta_abs / base_s) if base_s else float('nan'),
                is_near_field=s in NEAR_SHELLS,
            )

        sig_a, base_a, np_a = paired_spread(recs, rg, lambda r: r.get('all_resid'))

        out['by_regime'][rg] = dict(
            n=n, primary_k=k, primary_p=C.p_two_sided(k, n),
            mde_sign80_c_DERIVED=c80,
            mde_sign80_c_S8_hardcoded=0.7234,
            inherit_error_factor=0.7234 / c80,
            harm_k=C.harm_k(n), block_n=C.BLOCK_N[rg], block_worse_k=C.block_worse_k(C.BLOCK_N[rg]),
            sigma_delta_episode_robust=sig_d, baseline_median=base_level, n_pairs=n_pairs,
            mde_sign80=float(mde_sign80),
            mde_sign80_rel=float(mde_sign80 / base_level) if base_level else float('nan'),
            sigma_hat=sigma_hat, sigma_d=float(sigma_d), mde_agg=float(mde_agg),
            mde_agg_rel=float(mde_agg / base_level) if base_level else float('nan'),
            all_resid=dict(sigma_delta_robust=sig_a, baseline_median=base_a,
                           delta_abs=float(c80 * sig_a),
                           delta_rel=float(c80 * sig_a / base_a) if base_a else float('nan')),
            shells=shells,
        )

    json.dump(out, open(OUT, 'w'), indent=1)

    # ── 사람이 읽는 요약 ────────────────────────────────────────────────────
    print('=== S9 사전등록 MDE / 근거리 등가한계 δ (결과 관측 전 동결) ===')
    for rg, d in out['by_regime'].items():
        print(f'\n--- {rg} (n={d["n"]}, PRIMARY k>={d["primary_k"]}) ---')
        print(f'  mde_sign80_c  유도 {d["mde_sign80_c_DERIVED"]:.4f}  '
              f'(S8 하드코딩 {d["mde_sign80_c_S8_hardcoded"]} = '
              f'{d["inherit_error_factor"]:.3f}배 과대 — 상속 금지)')
        print(f'  mde_sign80    {d["mde_sign80"]:.5f}  ({d["mde_sign80_rel"]*100:+.2f}% of baseline)')
        print(f'  mde_agg       {d["mde_agg"]:.5f}  ({d["mde_agg_rel"]*100:+.2f}%)')
        print(f'  안전게이트    HARM-SIGN {d["harm_k"]}/{d["n"]} · '
              f'블록 {d["block_worse_k"]}/{d["block_n"]}')
        print(f'  all_resid δ   {d["all_resid"]["delta_rel"]*100:+.2f}%')
        print('  ★근거리 등가한계 δ (이동 널 부호검정용):')
        for s in NEAR_SHELLS:
            sh = d['shells'][s]
            print(f'      {s:8s} δ = {sh["delta_rel"]*100:+6.2f}%  (절대 {sh["delta_abs"]:.5f})')
    print(f'\n동결: {OUT}')


if __name__ == '__main__':
    main()
