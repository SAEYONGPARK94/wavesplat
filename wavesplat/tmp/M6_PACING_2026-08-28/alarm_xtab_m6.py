"""M-6 사후 보고표 (2026-08-29, ⚠판정 불참 — 판정은 동결된 analyze_m6.py/verdict_m6.json 이 전부다).

RESULT.md §3(G-8)·§4(G-2) 의 경보 교차표·kick 분포를 **동결 jsonl 에서 재유도**하고,
RESULT.md 에 실은 수와 다르면 AssertionError 로 죽는다 (선례 = tmp/B3_REFIT_2026-08-23/final6.py).
런 0회 — results_m6.jsonl 만 읽는다.
"""
import collections, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in open(os.path.join(HERE, 'results_m6.jsonl'))]
assert len(rows) == 360 and not any('error' in r for r in rows)
asy = [r for r in rows if r['cell'].startswith('ASYNC5_')]
ORDER = ['ASYNC5_P0.5', 'ASYNC5_P1', 'ASYNC5_P2', 'ASYNC5_P3',
         'ASYNC5_P4', 'ASYNC5_P6', 'ASYNC5_P8', 'ASYNC5_Pinf']

print(f"{'pace':<12} {'n':>3} {'warn':>4} {'full_noop':>9} {'silent_noop':>11} {'false_fire':>10} {'kicks=1':>7}")
tab = {}
for c in ORDER:
    cell = [r for r in asy if r['cell'] == c]
    t = dict(n=len(cell),
             warn=sum(1 for r in cell if r['noop_warning_fired']),
             noop=sum(1 for r in cell if r['applied_pre_wait'] == 0),
             silent=sum(1 for r in cell if r['applied_pre_wait'] == 0 and not r['noop_warning_fired']),
             ffire=sum(1 for r in cell if r['applied_pre_wait'] > 0 and r['noop_warning_fired']),
             k1=sum(1 for r in cell if r['async_kicks'] == 1))
    tab[c] = t
    print(f"{c:<12} {t['n']:>3} {t['warn']:>4} {t['noop']:>9} {t['silent']:>11} {t['ffire']:>10} {t['k1']:>7}")

tot_noop = sum(t['noop'] for t in tab.values())
tot_silent = sum(t['silent'] for t in tab.values())
tot_warn = sum(t['warn'] for t in tab.values())
tot_ffire = sum(t['ffire'] for t in tab.values())
print(f"TOTAL full_noop={tot_noop} silent={tot_silent} warn={tot_warn} false_fire={tot_ffire}")

# ── RESULT.md §3 의 문장 수치 (다르면 여기서 죽는다) ─────────────────────────
assert (tot_noop, tot_silent, tot_warn, tot_ffire) == (131, 116, 18, 3)
assert tab['ASYNC5_P6']['noop'] == tab['ASYNC5_P8']['noop'] == tab['ASYNC5_Pinf']['noop'] == 33
assert all(tab[c]['warn'] == 0 for c in ('ASYNC5_P4', 'ASYNC5_P6', 'ASYNC5_P8', 'ASYNC5_Pinf'))
assert tab['ASYNC5_P2']['warn'] == 10 and tab['ASYNC5_P2']['ffire'] == 3
assert tab['ASYNC5_P3']['warn'] == 8 and tab['ASYNC5_P3']['silent'] == 2
ff = sorted((r['record'], r['win'], r['first_apply_lead']) for r in asy
            if r['cell'] == 'ASYNC5_P2' and r['noop_warning_fired'] and r['applied_pre_wait'] > 0)
assert ff == [('AA14', 3, 14.0), ('BS11b', 1, 14.0), ('BS13b', 3, 14.0)], ff
p3s = [(r['async_kicks'], r['async_skipped']) for r in asy if r['cell'] == 'ASYNC5_P3'
       and r['applied_pre_wait'] == 0 and not r['noop_warning_fired']]
assert p3s == [(1, 1), (1, 1)], p3s          # 무경고 전면 no-op = 전부 kicks=1 (skip 소거)

# ── RESULT.md §4 의 kick 분포 ────────────────────────────────────────────────
kdist = {c: dict(collections.Counter(r['async_kicks'] for r in asy if r['cell'] == c)) for c in ORDER}
print('kick dist:', kdist)
assert kdist['ASYNC5_P0.5'] == {2: 36} and kdist['ASYNC5_P1'] == {2: 36}
assert kdist['ASYNC5_P2'] == {2: 26, 1: 10} and kdist['ASYNC5_P3'] == {2: 11, 1: 25}
assert kdist['ASYNC5_P4'] == kdist['ASYNC5_P6'] == kdist['ASYNC5_P8'] == kdist['ASYNC5_Pinf'] == {1: 33, 2: 3}
surv = sorted({(r['record'], r['win']) for r in asy if r['cell'] in
               ('ASYNC5_P6', 'ASYNC5_P8', 'ASYNC5_Pinf') and r['async_kicks'] == 2})
assert {rec for rec, _ in surv} == {'YS17'}, surv   # 6·8·∞× 의 kicks=2 생존 = 단일 레코드

# ── 공시: 행별 임포트 md5 균일성 + git_head 분할 ────────────────────────────
md5s = collections.Counter(r['md5_sequential_update'] for r in rows)
assert md5s == {'1b473e066cb02e43f39a9887a090c7d4': 360}, md5s
assert collections.Counter(r['md5_bank_mmse'] for r in rows) == {'d0e45d2217d969183e2e0ed1d6f38e2b': 360}
heads = collections.Counter(r['git_head'] for r in rows)
assert heads == {'0266deb9': 190, '19cc1ec5': 170}, heads
print('OK — RESULT.md 의 §3·§4·§6 수치 전건 재현')
