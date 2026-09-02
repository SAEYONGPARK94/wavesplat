"""`R2b-STEREONOISE` 분석기 — ⛔런 0회. 규칙은 `gates_r2b.py` 단일출처."""
import json, os, statistics as st, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r2b as G                                               # noqa: E402
ROWS = os.path.join(HERE, 'results_r2b.jsonl'); OUT = os.path.join(HERE, 'verdict_r2b.json')


def rng(vals):
    return dict(min=min(vals), median=st.median(vals), max=max(vals)) if vals else None


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    ok = [r for r in rows if 'error' not in r]
    gts = G.gates(rows); blocked = G.blocked_by(gts)
    w = sum(1 for r in ok if r['stereo_gt_gauge'])
    p = G.sign_test_one_sided(w, G.N) if len(ok) == G.N else None
    lab, msg = (G.label(w) if not blocked else (None, 'BLOCKED'))
    exc = [r['hf_excess_rms_mm'] for r in ok if r['stereo_gt_gauge']]

    v = dict(batch='R2b-STEREONOISE', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R2b-STEREONOISE :: {lab}' if lab else None), label_msg=msg,
             primary=dict(better=w, n=G.N, p_one_sided=p, threshold=G.THRESH),
             noise_floor=dict(hf_cut_hz=G.HF_CUT_HZ, rms_mm=rng(exc),
                              note='⛔**상한**이다 — 게이지 자신의 잡음과 공간 비균질이 섞인다'),
             detector=dict(
                 control_rmax=rng([r['ctrl_rmax'] for r in ok]),
                 control_null_p95=rng([r['ctrl_null_p95'] for r in ok]),
                 control_pass=sum(1 for r in ok if r['ctrl_exceeds_null']),
                 lag_injected_s=G.CTRL_LAG_S,
                 lag_recovered=rng([r['ctrl_lag_s'] for r in ok])),
             position=dict(
                 obs_rmax=rng([r['obs_rmax'] for r in ok]),
                 obs_null_p95=rng([r['obs_null_p95'] for r in ok]),
                 obs_exceeds=sum(1 for r in ok if r['obs_exceeds_null']),
                 peak_coverage=rng([r['obs_peak_cov'] for r in ok])),
             surrogate_check=dict(
                 kind='FT 위상무작위화 (⛔IAAFT 아님 — PREREG §1)',
                 gauge_skew=rng([r['gauge_skew'] for r in ok]),
                 gauge_kurt=rng([r['gauge_kurt'] for r in ok]),
                 note='정규(왜도 0·첨도 3)에 가까울수록 FT 와 IAAFT 가 수렴한다'),
             spectra=dict(
                 Tp_rel_median=(st.median([abs(r['Tp_stereo'] - r['Tp_gauge']) / r['Tp_gauge']
                                           for r in ok]) if ok else None),
                 Hm0_ratio=rng([r['Hm0_stereo'] / r['Hm0_gauge']
                                for r in ok if r['Hm0_gauge'] > 0])),
             disclosure=[
                 '⛔blind 아님 — 선행 R2 에서 PRIMARY 7/7 과 hf_excess 12.8–129.0 mm 를 보았다.',
                 '⛔CONFIRMATORY 아님 — 기능은 발견이 아니라 **기록**이다.',
                 '⚠초과분산은 **상한**이다(게이지 자신의 잡음·공간 비균질이 섞인다).',
                 '⛔Hm0 산포를 스테레오 오차로 귀속 금지 — 공위치가 아니다.',
                 '⛔격자점별 오차장·면 참값 주장 금지 — 6점(반경 0.25 m)은 면이 아니다.',
                 '⛔BS 7 레코드 · 4 사이트 중 1 — 집계 실측 판정의 재채점이 아니다.',
                 '⛔"스테레오가 나쁘다" 아님 — 재는 것은 1 Hz 초과 초과분산 상한 하나다.',
                 '⚠원장의 "재구성면 오차는 도달 불가" 는 **부분적으로만** 풀린다(점 대조).',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"PRIMARY {w}/{G.N}" + (f' p={p:.4g}' if p else ''))
    print(f"잡음 바닥 {v['noise_floor']['rms_mm']}")
    print(f"탐지기 대조 통과 {v['detector']['control_pass']}/{len(ok)} · 위치 초과 {v['position']['obs_exceeds']}/{len(ok)}")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
