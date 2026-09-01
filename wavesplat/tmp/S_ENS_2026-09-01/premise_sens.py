#!/usr/bin/env python3
"""`S-ENS` 전제 검증 — ⛔런 0회 · 추정기 0회. 실패하면 `AssertionError`."""
import hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
W15 = os.path.abspath(os.path.join(HERE, '..', 'W15_DEPLOY_VERIFY_2026-08-22'))
for p in (HERE, W15, ROOT):
    sys.path.insert(0, p)
F = []


def chk(name, ok, d=''):
    print(f'  [{"OK " if ok else "FAIL"}] {name}   {d}')
    if not ok:
        F.append(name)


def main():
    import gates_sens as G
    su = open(os.path.join(ROOT, 'wavesplat/optimizer/sequential_update.py')).read()
    out = {'batch': 'S-ENS', 'runs': 0, 'estimator_calls': 0, 'premises': {}}
    print('── PREREG §1 전제 (런 0회) ──────────────────────────────────')

    # P1 — W-15 의 비교자만 앙상블을 받았다
    rows = [json.loads(l) for l in open(os.path.join(W15, 'results.jsonl')) if l.strip()]
    ek = {c: sorted({r.get('ensemble_k') for r in rows if r.get('cell') == c})
          for c in ('POST', 'ALL-ANALYTIC')}
    p1 = ek['POST'] == [1] and ek['ALL-ANALYTIC'] == [3]
    chk('P1 W-15: 학습 팔 k=1 · 비교자 k=3 (비대칭)', p1, str(ek))
    out['premises']['P1_w15_asymmetry'] = bool(p1)
    out['w15_ensemble_k'] = ek

    # P4 — 패키지가 경로를 소유한다 · P5 기본값 OFF
    p4 = 'DEFAULT_LEARNED_ENSEMBLE_K' in su and '_effective_k' in su
    chk('P4 패키지가 학습 앙상블 경로를 소유', p4)
    from wavesplat.optimizer.sequential_update import (DEFAULT_LEARNED_ENSEMBLE_K,
                                                       MedianEnsembleUpdater)
    from wavesplat.run import DEFAULT_CONFIG
    k_def = MedianEnsembleUpdater(dict(DEFAULT_CONFIG))._effective_k('learned', False)
    chk('P5 배포 기본값에서 학습 앙상블 OFF', DEFAULT_LEARNED_ENSEMBLE_K == 1 and k_def == 1,
        f'default={DEFAULT_LEARNED_ENSEMBLE_K} effective={k_def}')
    out['premises']['P4_package_owns_path'] = bool(p4)
    out['premises']['P5_default_off'] = bool(DEFAULT_LEARNED_ENSEMBLE_K == 1 and k_def == 1)

    # 앵커 가용성
    anc = G._anchor_w15()
    chk('앵커: W-15 POST(seed 고정) 가 전 에피소드에 있다',
        len(anc) == len(G.EPISODES), f'{len(anc)}/{len(G.EPISODES)}')
    out['anchor_cells'] = len(anc)

    # 문턱 재도출
    t, m = G.thresholds(), G.mde()
    chk('NOMINAL == 40', t['NOMINAL'] == 40, str(t['NOMINAL']))
    chk('Holm 사다리 == [41, 40]', t['HOLM_MIN_WINS'] == [41, 40], str(t['HOLM_MIN_WINS']))
    chk('NOT_EARN == 23', t['NOT_EARN'] == 23, str(t['NOT_EARN']))
    chk('MDE q80 == 0.682', m['q80'] == 0.682, str(m['q80']))
    chk('기대 행/런 == 256/512', (G.expected_rows(), G.expected_runs()) == (256, 512),
        f'{G.expected_rows()}/{G.expected_runs()}')
    out['thresholds'], out['mde'] = t, m
    out['power'] = {q: round(G.power_at(q), 4) for q in (0.75, 0.70, 0.65, 0.60, 0.50)}
    out['code_md5'] = {b: hashlib.md5(open(os.path.join(ROOT, r), 'rb').read()).hexdigest()[:12]
                       for b, r in (('sequential_update.py', 'wavesplat/optimizer/sequential_update.py'),
                                    ('run.py', 'wavesplat/run.py'))}
    json.dump(out, open(os.path.join(HERE, 'premise_sens.json'), 'w'),
              ensure_ascii=False, indent=1, sort_keys=True)
    print('──────────────────────────────────────────────────────────────')
    if F:
        raise AssertionError(f'전제 검증 실패: {F}')
    print('✅전제 전건 통과 · 런 0회 → premise_sens.json')
    return 0


if __name__ == '__main__':
    sys.exit(main())
