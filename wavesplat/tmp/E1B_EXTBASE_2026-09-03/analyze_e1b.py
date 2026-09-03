"""`E1b-EXTBASE` 분석기 — ⛔**결과 전에 작성·동결**. 판정은 여기서만 나온다."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gates_e1b as G                                                 # noqa: E402

RES = os.path.join(HERE, 'results_e1b.jsonl')
META = os.path.join(HERE, 'meta_e1b.json')
OUT = os.path.join(HERE, 'verdict_e1b.json')


def _med(a):
    a = sorted(x for x in a if x == x)
    if not a:
        return float('nan')
    n = len(a)
    return a[n // 2] if n % 2 else 0.5 * (a[n // 2 - 1] + a[n // 2])


def main():
    rows = [json.loads(l) for l in open(RES, encoding='utf-8') if l.strip()]
    meta = json.load(open(META, encoding='utf-8'))
    ok = [r for r in rows if 'error' not in r]

    # 사다리 칸별 자기검정: 제3자 팔이 레코드별 참 제로필드를 이기는 레코드 수
    rungs = {}
    for m in G.LADDER + (G.PRIMARY_PER_FRAME,):
        key = f'cnn_{m}'
        w = sum(1 for r in ok if r.get(key) is not None and r[key] < r['zero'])
        n = sum(1 for r in ok if r.get(key) is not None)
        rungs[m] = dict(better_than_zero=w, n=n,
                        med=_med([r[key] for r in ok if r.get(key) is not None]))
    passing = sorted([m for m, d in rungs.items()
                      if d['n'] == G.N and d['better_than_zero'] >= G.SELF_MIN])
    self_ok = bool(passing)
    rung = passing[0] if self_ok else None

    w_ship = n_pair = 0
    if self_ok:
        key = f'cnn_{rung}'
        pair = [(r['ship'], r[key]) for r in ok if r.get(key) is not None]
        n_pair = len(pair)
        w_ship = sum(1 for a, b in pair if a < b)

    meta = dict(meta)
    meta['self_ok'] = self_ok
    meta['self_detail'] = ('통과 칸 ' + ','.join(str(x) for x in passing) if passing
                           else '전 칸 미달')
    # ★G-ONELINE 채움 — E1 의 동결 verdict 과 SHIP 열을 비교한다(런 0회)
    _e1 = json.load(open(os.path.join(G.ROOT, 'tmp', 'E1_EXTBASE_2026-09-03',
                                      'verdict_e1.json'), encoding='utf-8'))['per_record']
    for r in ok:
        ref = _e1.get(r['record'], {}).get('ship')
        r['ship_bit_identical'] = (ref is not None and r['ship'] == ref)
    gate_rows = G.gates(rows, meta)
    blocked = G.blocked_by(gate_rows)
    if blocked:
        lab, msg = 'BLOCKED', '차단 게이트 실패: ' + ','.join(blocked)
    else:
        lab, msg = G.label(self_ok, rung, w_ship)
    lab0 = lab
    # ★E1b 의 PRIMARY 는 우열이 아니라 **문턱 이동**이다(PREREG §2)
    bias_lab, bias_msg = G.label_bias(rung)
    lab = 'E1b-EXTBASE :: ' + bias_lab
    msg = bias_msg + f' | (참고: E1 규약 라벨은 {lab0})' if False else bias_msg

    v = dict(batch='E1b-EXTBASE', n_rows=len(rows), grade='EXPLORATORY',
             external=dict(repo='gitlab.com/fibe/wassfast', commit=G.EXT_COMMIT,
                           method='Pistellato et al. 2021 Remote Sensing 13(18):3780',
                           license='GPL-3', code_modified=False),
             gates=[dict(name=n, pass_=bool(o), detail=str(d)) for n, o, d in gate_rows],
             gates_pass=f'{sum(1 for _, o, _ in gate_rows if o)}/{len(gate_rows)}',
             blocked_by=blocked,
             self_test=dict(rule=f'제3자 팔이 레코드별 참 제로필드를 ≥ {G.SELF_MIN}/{G.N}',
                            passing_rungs=passing, chosen_rung=rung, per_rung=rungs),
             primary=dict(stat='SHIP < WFCNN (paired, nowcast nRMSE)', w=w_ship,
                          n=n_pair, rung=rung,
                          p_one_sided=(G.sign_test_one_sided(max(w_ship, G.N - w_ship), G.N)
                                       if self_ok else None),
                          budget_convention='총 관측 예산 고정 (600) — PREREG §1 (b)'),
             medians=dict(ship=_med([r['ship'] for r in ok]),
                          zero=_med([r['zero'] for r in ok]),
                          **{f'cnn_{m}': rungs[m]['med'] for m in rungs}),
             per_record={r['record']: {k: r[k] for k in r
                                       if k in ('ship', 'zero', 'dir_sign', 'n_obs_ship',
                                                'n_obs_cnn_primary')
                                       or k.startswith('cnn_')} for r in ok},
             prediction=dict(written_before_results='NO-VALID-COMPETITOR-AT-ANY-DENSITY 0.45 · '
                                                    'EXTERNAL-BEATS-SHIP 0.35 · '
                                                    'SHIP-BEATS-EXTERNAL 0.10 · UNDECIDED 0.10'),
             e1_rung=G.E1_RUNG, e1_convention_label=lab0,
             mu_rms={r['record']: r.get('mu_rms_over_sigma') for r in ok},
             label=lab, label_msg=msg,
             disclosures=[
                 '★제3자 코드(GPL-3) + 제3자 사전학습 가중치 — 우리 코드도 저들 코드도 무수정.',
                 '★E1b 의 유일한 개입: 제3자 입력에서도 mu_mem 을 뺐다(E1 에서는 그 정적장이 제3자 오차에만 들어갔다).',
                 '⛔E1 판정 갱신 아님 — 편향이 반대 방향이라 그 라벨은 이 수리로 약해지지 않는다.',
                 '⛔blind 아님(E1 의 수를 안다) ⟹ EXPLORATORY 상한.',
                 '⛔예산 규약 병기 없이 인용 금지 — 두 방법은 관측을 구조적으로 다르게 소비한다 '
                 '(우리 12점×40 s 누적 vs 저들 연속 3 프레임).',
                 '⛔비인과성: 저들 입력은 t-1,t,t+1 이라 미래 프레임을 본다 = 기준선에 유리한 방향.',
                 '⛔되샘플: 저들 256² 출력을 우리 질의 격자로 최근접 이웃 보간했다.',
                 '⛔저들 체크포인트의 학습 분포는 미확인(MDPI 403) — 추론이며 우리 사이트와의 관계 미상.',
                 '⛔사이트 4개 — 사이트 수준 일반화 금지.',
                 '⛔R1b·B-2·B-5·B2_WANGFIX 판정 갱신 아님 · ⛔합성으로 옮기지 마라.',
             ])
    json.dump(v, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'{lab}\n  {msg}\n  gates {v["gates_pass"]}')
    for n, o, d in gate_rows:
        print(f"   [{'OK ' if o else 'FAIL'}] {n:52s} {d}")
    print('  자기검정(제3자 vs 참 제로필드):')
    for m in sorted(rungs):
        d = rungs[m]
        print(f"    {m:5d} 점/프레임 → {d['better_than_zero']}/{d['n']}  중앙 {d['med']:.3f}")
    print(f"  선택된 칸 {rung} · PRIMARY SHIP<CNN {w_ship}/{n_pair}")
    print(f"  중앙: SHIP {v['medians']['ship']:.3f} · ZERO {v['medians']['zero']:.3f}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
