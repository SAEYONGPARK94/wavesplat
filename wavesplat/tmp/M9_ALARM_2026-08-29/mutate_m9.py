#!/usr/bin/env python3
"""M-9 변형 하네스 — 게이트가 아니라 **러너(재생)를 모형**한다 (2026-08-30 규율).

행 정체는 `gates_m9.expected_jobs()`/`row_identity()` 단일출처에서 오고, 변형은
①술어 변형(제품 함수 자리에 오염 술어를 꽂는다 — ★off-by-one 함정 포함)과
②자료/러너 변형(행 탈락·중복·되읽기 열 오염·0행)이다. **모든 변형이 최소 한
게이트를 죽여야** 하네스가 통과한다(안 죽으면 그 게이트는 공허). 런 0회 —
동결 jsonl 은 읽기만 하고 변형은 전부 메모리 사본에서 한다.

    python3 tmp/M9_ALARM_2026-08-29/mutate_m9.py            # 전체 (~수 초)
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import json    # noqa: E402
import sys     # noqa: E402
import time    # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gates_m9 as G                                       # noqa: E402


def _fails(rep):
    return [x['name'].split()[0] for x in rep['gates'] if x['ok'] is False]


def _run(rows, md5, pred=None):
    return G.rg1_run(rows, md5, predicate=pred,
                     predicate_src='변형' if pred else None)


# 술어 변형 — post 모드 발화 열 자리에 꽂는다. 이름: 무엇을 모형하나.
#  ⚠주의: rg1_run 은 predicate 가 있으면 M-3 형태 검사(모듈·시그니처)도 하므로
#  변형 술어는 그것만으로도 걸린다 — 하네스는 "동결 수치 게이트(M-4~M-7)로도
#  걸리는가" 를 따로 단언한다(형태 검사에만 기대면 프로브가 공허해진다).
PRED_MUTANTS = {
    'off_by_one_pre_increment': (          # ★PREREG §2-① 의 함정 그 자체
        lambda n, a: (n - 1) >= 2 and a == 0,
        ['M-3', 'M-4', 'M-6']),            # (2,0) 15런 포함 전 침묵 → 포착 0/131
    'tautology_always_fire': (             # 필터 항진 금지 — 신호 죽음 감지
        lambda n, a: True,
        ['M-3', 'M-5', 'M-6']),
    'never_fire': (
        lambda n, a: False,
        ['M-3', 'M-4', 'M-6']),
    'applied_leq_1': (                     # 문턱 오염 — 건강 대역 발화로 걸려야
        lambda n, a: n >= 2 and a <= 1,
        ['M-3', 'M-5']),
    'fires_on_first_trigger': (            # 사건1 발화 — 건강 대역 72 전체 발화
        lambda n, a: a == 0,
        ['M-3', 'M-5', 'M-6']),
}


def data_mutants(rows):
    """자료/러너 변형 생성기 — (이름, rows', md5', 기대 게이트 접두어들)."""
    md5 = G.M6_JSONL_MD5
    r = G.deepcopy_rows(rows)
    yield ('drop_last_row', r[:-1], md5, ['M-0b'])
    r = G.deepcopy_rows(rows)
    yield ('duplicate_first_row', r + [G.deepcopy_rows([r[0]])[0]], md5, ['M-0b'])
    r = G.deepcopy_rows(rows)
    h = next(x for x in r if x['cell'] == 'ASYNC5_P0.5')     # 건강 행의 진실 열 오염
    h['applied_after_lead']['9.0'] = 0
    yield ('healthy_row_truth_column_zeroed', r, md5, ['M-2', 'M-5', 'M-6'])
    r = G.deepcopy_rows(rows)
    f = next(x for x in r if x['noop_warning_fired'])        # 동결 발화열 오염
    f['noop_warning_fired'] = False
    yield ('flip_one_frozen_fire_flag', r, md5, ['M-2'])
    r = G.deepcopy_rows(rows)
    r[0]['async_kicks'], r[0]['async_skipped'] = 0, 2        # 재구성 전제 위반
    yield ('kicks_zero', r, md5, ['M-1'])
    r = G.deepcopy_rows(rows)
    a = next(x for x in r if x['cell'].startswith('ASYNC5_'))
    del a['applied_after_lead']['9.0']                       # 되읽기 키 결손
    yield ('missing_applied_key_9', r, md5, ['M-1'])
    yield ('md5_pin_mismatch', G.deepcopy_rows(rows), 'deadbeef' * 4, ['M-0a'])
    r = [x for x in rows if x['cell'] not in ('ASYNC5_P0.5', 'ASYNC5_P1')]
    yield ('healthy_band_absent(커버리지0→None)', G.deepcopy_rows(r), md5,
           ['M-0b', 'M-5'])                                  # 계수 None ⟹ 0 으로 통과 금지


def main():
    rows, md5 = G.load_rows()
    print(G.selftest())

    base = _run(rows, md5)                                   # pre 모드 기준선
    base_bad = _fails(base)
    print(f'기준선(pre): {"전건 통과" if not base_bad else "⛔FAIL " + str(base_bad)}')

    report = dict(baseline_pass=not base_bad, mutants={}, ts=time.strftime('%F %T'))
    ok_all = not base_bad

    for name, (fn, want) in PRED_MUTANTS.items():
        rep = _run(rows, md5, pred=fn)
        got = _fails(rep)
        # 형태 게이트(M-3)만이 아니라 동결 수치 게이트로도 걸리는가
        numeric = [g for g in got if not g.startswith('M-3')]
        caught = bool(got) and all(any(g.startswith(w) for g in got) for w in want) \
            and (bool(numeric) or want == ['M-3'])
        report['mutants']['pred:' + name] = dict(caught=caught, failing=got)
        ok_all &= caught
        print(f"  [{'잡힘' if caught else '⛔탈출'}] pred:{name:<28} → {got}")

    for name, rr, mm, want in data_mutants(rows):
        try:
            rep = _run(rr, mm)
            got = _fails(rep)
        except (ValueError, SystemExit, KeyError) as e:
            got = [f'요란한 실패({type(e).__name__})']
        caught = bool(got) and all(
            any(g.startswith(w) for g in got) or g.startswith('요란한') for w in want)
        report['mutants']['data:' + name] = dict(caught=caught, failing=got)
        ok_all &= caught
        print(f"  [{'잡힘' if caught else '⛔탈출'}] data:{name:<34} → {got}")

    # 0행 가드 — 요란하게 죽어야 한다 (조용한 전건 통과 금지)
    try:
        _run([], md5)
        caught0 = False
    except SystemExit:
        caught0 = True
    report['mutants']['data:empty_rows(0행 가드)'] = dict(caught=caught0,
                                                          failing=['SystemExit'])
    ok_all &= caught0
    print(f"  [{'잡힘' if caught0 else '⛔탈출'}] data:empty_rows(0행 가드)")

    report['all_caught'] = ok_all
    json.dump(report, open(G.MUT_REPORT, 'w'), ensure_ascii=False, indent=1)
    print(('변형 하네스 전건 통과 — 게이트는 공허하지 않다'
           if ok_all else '⛔변형 탈출 있음 — 게이트 보강 전 수리 착지 금지'),
          '→', G.MUT_REPORT)
    return 0 if ok_all else 1


if __name__ == '__main__':
    sys.exit(main())
