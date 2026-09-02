"""`R1b-SCORINGFRAME` 분석기 — ⛔런 0회. 규칙은 `gates_r1b.py` 단일출처."""
import json, os, statistics as st, sys
from math import comb
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gates_r1b as G                                                # noqa: E402
ROWS = os.path.join(HERE, 'results_r1b.jsonl')
PROBE = os.path.join(HERE, 'probes_r1b.jsonl')
OUT = os.path.join(HERE, 'verdict_r1b.json')


def power(q, n, thr):
    return sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(thr, n + 1))


def mde(n, thr, target=0.80):
    """★이 설계가 80% 검정력으로 잡을 수 있는 최소 승률 `q`."""
    lo, hi = 0.5, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if power(mid, n, thr) >= target: hi = mid
        else: lo = mid
    return hi


def count(rows, probes, arm, pass_, conv):
    key = 'nowcast_new' if conv == 'new' else 'nowcast_old'
    ref = 'zero_new' if conv == 'new' else 'zero_old'
    sel = [r for r in rows if r['cell'] == arm and r['pass_'] == pass_]
    wins = [r['record'] for r in sel if r[key] < probes[r['record']][ref]]
    return len(wins), len(sel), sorted(wins)


def main():
    rows = [json.loads(l) for l in open(ROWS) if l.strip()]
    probes = {json.loads(l)['record']: json.loads(l) for l in open(PROBE) if l.strip()}
    ok = [r for r in rows if 'error' not in r]
    gts = G.gates(rows, list(probes.values())); blocked = G.blocked_by(gts)

    w, n, wins = count(ok, probes, 'SHIP', 'judge', 'new')
    p = G.sign_test_one_sided(w, n) if n == G.N else None
    lab, msg = (G.label(w) if not blocked else (None, 'BLOCKED'))

    sec = {}
    for arm in G.ARM_ORDER:
        for conv in ('old', 'new'):
            c, m, ws = count(ok, probes, arm, 'judge', conv)
            sec[f'{arm}_{conv}'] = dict(w=c, n=m, p=G.sign_test_one_sided(c, m) if m else None,
                                        wins=ws)
    zr = {k: dict(zero_old=v['zero_old'], zero_new=v['zero_new'],
                  rel_shift_pct=round((v['zero_new'] - v['zero_old']) / v['zero_old'] * 100, 2))
          for k, v in probes.items()}

    v = dict(batch='R1b-SCORINGFRAME', n_rows=len(rows),
             gates=[[a, bool(b), c] for a, b, c in gts],
             gates_pass=f'{sum(1 for _, b, _ in gts if b)}/{len(gts)}',
             blocked_by=blocked, grade=('BLOCKED' if blocked else 'EXPLORATORY'),
             label=(f'R1b-SCORINGFRAME :: {lab}' if lab else None), label_msg=msg,
             primary=dict(arm='SHIP', pass_='judge', convention='fixed (mu_mem)',
                          w=w, n=n, wins=wins, p_one_sided=p,
                          threshold=G.THRESH_SURVIVES,
                          mde_q80=round(mde(G.N, G.THRESH_SURVIVES), 4)),
             secondary=sec,
             zero_reference_shift=zr,
             zero_shift_summary=dict(
                 median_pct=round(st.median([x['rel_shift_pct'] for x in zr.values()]), 3),
                 min_pct=min(x['rel_shift_pct'] for x in zr.values()),
                 max_pct=max(x['rel_shift_pct'] for x in zr.values())),
             disclosure=[
                 '⛔blind 아님 — 옛 규약·수심 17 과 30 의 계수(SHIP 10/12 · PROP 7/12)를 읽었다.',
                 '⛔CONFIRMATORY 아님 — 자기 예측 하 ≥10 도달 확률 68.8%(PREREG §5).',
                 '⛔배포 기본값 변경 아님 — 채점 규약만 다룬다.',
                 '⛔SECONDARY 옛 규약 계수를 B-2 공표값과 직접 비교 금지(수심이 다르다).',
                 '⛔"바닥이 실측 널의 원인" 금지 — 두 패배 중 하나만 설명된다.',
                 '⛔"기존 배치가 오염됐다" 금지 — 결함 B 는 잠재이고 동결 전건 lead_max=14.0.',
                 '⛔with_future_obs=True 배치(B-3·B-5·B-7)는 범위 밖.',
                 '⛔사이트 일반화 금지(4 사이트).',
             ])
    json.dump(v, open(OUT, 'w'), ensure_ascii=False, indent=1)
    print(f"게이트 {v['gates_pass']} · 차단 {blocked}")
    for a, b, c in gts:
        if not b: print(f'  FAIL {a}  {c}')
    print(f"PRIMARY SHIP judge/fixed {w}/{n}" + (f' p={p:.4g}' if p else '')
          + f"  MDE q80={v['primary']['mde_q80']}")
    for k, d in sec.items(): print(f"  SEC {k:10s} {d['w']}/{d['n']}")
    print(f"제로 기준 이동 중앙 {v['zero_shift_summary']['median_pct']}% "
          f"({v['zero_shift_summary']['min_pct']}–{v['zero_shift_summary']['max_pct']}%)")
    print(f"라벨: {v['label']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
