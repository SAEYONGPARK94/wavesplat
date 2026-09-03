"""`T11-EXTACC` 분석기 — ⛔**러너보다 먼저 쓰였다**(2026-09-04 규율).

⟹ 이 파일이 **행 스키마를 정의**하고 러너가 거기에 맞춘다. 그 반대가 아니다.
(오늘 `analyzer<results` 가 세 번 `--` 였던 것에 대한 처방.)

## 러너가 채워야 할 행 스키마 (여기서 정의한다)
  arm            : 'SHIP' | 'BANK' | 'ZERO'
  block          : 아카이브 안 .mat 경로
  hs             : 파일명에서 읽은 유의파고 (float)
  density        : 사다리 칸 (12|50|200|800)
  n_obs          : ★러너가 **실제로 센** 관측 수 (⛔리터럴 금지 --- G-DENSITY)
  nrmse          : 저들 eta 전 격자 대비 (lead 0)
  gt_attached_std: 붙인 gt_field 의 표준편차 (★상수면 0 --- G-NOLEAK)
  gt_attached_corr: 붙인 gt_field 와 채점 참값의 상관 (★상수면 정의 불가 ⟹ 0 기록)
  prov           : upd.arm_provenance() 통째로 (G-ARM)
  zero_sigma     : ZERO 채점에 쓴 σ (G-ZERO --- 팔 간 동일해야 한다)
  frame_index    : 채점 프레임 (G-ZERO --- 팔 간 동일해야 한다)
  seed, wall_s
"""
import hashlib, json, os, sys
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_t11 as G                                                # noqa: E402

ROWS = os.path.join(HERE, 'results_t11.jsonl'); META = os.path.join(HERE, 'meta_t11.json')


def sign_p(k, n):
    """단측 짝 부호검정 (귀무 0.5). ⛔전수 계산 --- 산문 신뢰 금지."""
    return sum(comb(n, i) * 0.5 ** n for i in range(k, n + 1))


def main():
    rows = [json.loads(x) for x in open(ROWS, encoding='utf-8') if x.strip()]
    meta = json.load(open(META, encoding='utf-8'))

    def sel(arm, dens):
        return {r['block']: r for r in rows if r['arm'] == arm and r['density'] == dens}

    ship = sel('SHIP', G.PRIMARY_DENSITY); zero = sel('ZERO', G.PRIMARY_DENSITY)
    common = sorted(set(ship) & set(zero))
    w = sum(1 for b in common if ship[b]['nrmse'] < zero[b]['nrmse'])
    n = len(common)

    A = []
    A.append((f'★G-SIZE 크기 == {G.WANT_BYTES}', meta.get('size_ok'), str(meta.get('zip_size'))))
    # ★★G-NOLEAK --- 붙인 장이 상수이고 참값과 무상관
    leak = [r for r in rows if r['arm'] == 'SHIP'
            and (abs(r.get('gt_attached_std', 1)) > 1e-12
                 or abs(r.get('gt_attached_corr', 1)) > 1e-12)]
    A.append(('★★G-NOLEAK 붙인 gt 가 상수(std 0)·참값과 상관 0', not leak,
              f'SHIP {len([r for r in rows if r["arm"]=="SHIP"])}행 중 위반 {len(leak)}'))
    # ★★G-ARM --- 팔 정체를 되읽기로
    bad_arm = []
    for r in rows:
        p = r.get('prov') or {}
        if r['arm'] == 'SHIP' and not (p.get('resolved_init_method') == 'learned'
                                       and p.get('real_domain_gate_fired') is False):
            bad_arm.append((r['arm'], r['block']))
        if r['arm'] == 'BANK' and not (p.get('resolved_init_method') == 'bank_mmse'
                                       and p.get('real_domain_gate_fired') is True):
            bad_arm.append((r['arm'], r['block']))
    A.append(('★★G-ARM arm_provenance 되읽기가 의도와 맞는다', not bad_arm,
              f'위반 {len(bad_arm)}'))
    A.append(('★G-DISJOINT T1·T1b·T10 과 교집합 0', not meta.get('disjoint'),
              f"교집합 {len(meta.get('disjoint', []))}"))
    dens_bad = [r for r in rows if r['arm'] != 'ZERO'
                and abs(r['n_obs'] - r['density'] * int(G.WIN_S)) > r['density']]
    A.append(('★G-DENSITY 행이 **센** 관측 수가 사다리 칸과 일치', not dens_bad,
              f'위반 {len(dens_bad)}'))
    zbad = []
    for b in common:
        for arm in ('SHIP', 'BANK'):
            r = sel(arm, G.PRIMARY_DENSITY).get(b)
            if r and (r.get('frame_index') != zero[b].get('frame_index')
                      or abs(r.get('zero_sigma', 0) - zero[b].get('zero_sigma', 1)) > 1e-12):
                zbad.append((arm, b))
    A.append(('★G-ZERO 같은 프레임·같은 σ 로 채점', not zbad, f'위반 {len(zbad)}'))
    A.append((f'G-N 파일 {G.N_FILE} · 사다리 {len(G.LADDER)} · 팔 {len(G.ARMS)}',
              n == G.N_FILE and len({r['density'] for r in rows}) == len(G.LADDER),
              f"짝 {n} · 밀도 {sorted({r['density'] for r in rows})}"))
    A.append(('G-ENV 스레드 1 · 시드 · code_md5 기록', bool(meta.get('code_md5')),
              f"seed={meta.get('seed')} threads={meta.get('threads')}"))
    probe = G.blocked_by([('★★G-NOLEAK 합성', False, ''), ('G-N 합성', True, '')])
    A.append(('G-BLOCK blocked_by 자가진단 (긴 이름 우선)', probe == ['G-NOLEAK'],
              f'합성 → {probe}'))
    A.append(('G-COV 게이트가 자료를 봤다', len(rows) > 0, f'rows={len(rows)}'))

    blocked = G.blocked_by(A)
    lab = G.label(w) if not blocked else None
    ladder = {}
    for d in G.LADDER:
        s, z = sel('SHIP', d), sel('ZERO', d)
        cm = sorted(set(s) & set(z))
        ladder[str(d)] = {'w': sum(1 for b in cm if s[b]['nrmse'] < z[b]['nrmse']),
                          'n': len(cm)}
    strata = {}
    for nm, pred in (('hs_low', lambda h: h <= G.HS_SPLIT),
                     ('hs_high', lambda h: h > G.HS_SPLIT)):
        cm = [b for b in common if pred(ship[b]['hs'])]
        strata[nm] = {'w': sum(1 for b in cm if ship[b]['nrmse'] < zero[b]['nrmse']),
                      'n': len(cm)}
    bank = sel('BANK', G.PRIMARY_DENSITY)
    bank_w = sum(1 for b in common if b in bank and bank[b]['nrmse'] < zero[b]['nrmse'])

    v = {'batch': 'T11-EXTACC', 'kind': 'CONFIRMATORY', 'grade': 'CONFIRMATORY',
         'new_runs': meta.get('n_runs'), 'n_rows': len(rows),
         'gates': [{'name': nm, 'pass_': bool(o), 'detail': d} for nm, o, d in A],
         'gates_pass': f'{sum(1 for _, o, _ in A if o)}/{len(A)}', 'blocked_by': blocked,
         'primary': {'stat': f'SHIP < ZERO at density {G.PRIMARY_DENSITY}',
                     'w': w, 'n': n, 'p_one_sided': sign_p(w, n) if n else None},
         'ladder': ladder, 'strata': strata,
         'bank_ruler': {'w': bank_w, 'n': len(common)},
         'prediction': {'label': 'EXTACC-SKILL-AT-200', 'p': 0.55},
         'label': lab, 'label_msg': lab or 'BLOCKED',
         'disclosures': [
             '⛔E2 와 대각선 비교 금지 --- 읽는 것은 "각자 남의 자료에서 어느 밀도부터"',
             '⛔저들 논문의 수치와 비교 금지(채점·격자·기준이 다르다)',
             '⛔solver 미명시 --- "HOS"·"완전 비선형" 이라 쓰지 않는다',
             '⚠예산 비대칭: 우리 코퍼스는 4,800 관측 중 600 을 캡해 쓰고 저들 40 s 창은 '
             '480 이라 캡이 안 걸린다 --- 같은 수라도 "고른 600" 과 "있는 것 전부" 는 다르다',
             '⚠BANK 는 **눈금**이다(2026-08-30 결정) --- 기여로 세지 않는다',
             '⚠완전 blind 아님 --- 같은 아카이브의 파일 1-36 을 앞선 세 배치가 읽었다',
         ]}
    json.dump(v, open(os.path.join(HERE, 'verdict_t11.json'), 'w'),
              ensure_ascii=False, indent=1)
    print(f"T11-EXTACC :: {v['label_msg']}")
    print(f"  PRIMARY {w}/{n}  p={v['primary']['p_one_sided']}  "
          f"(문턱 ≥{G.THRESH_SKILL})")
    print(f"  사다리 {ladder}")
    print(f"  Hs 층 {strata}   BANK 눈금 {bank_w}/{len(common)}")
    print(f"  gates {v['gates_pass']}")
    for nm, o, d in A:
        print(f"   [{'OK ' if o else 'FAIL'}] {nm:48s} {d}")


if __name__ == '__main__':
    main()
