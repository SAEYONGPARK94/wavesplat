"""M-8b 앵커 재현 자료 생성 — `G-ANCHOR-B1` 이 읽는 24행을 만든다.

    python3 tmp/M8b_LADDER_2026-08-28/anchor_b2depth.py     # → anchor_b2depth.jsonl

⛔**신규 *판정* 런 0회다.** 여기서 도는 것은 **양성대조(앵커) 런**이고, `M8` 의 432행은
읽지도 건드리지도 않는다. 팔은 `BANK` 하나, 시드는 `ANCHOR_SEED` 하나, 레코드 12개를
**수심 2종**(B-2 시대 상수 · 현행 정정 상수)으로 돌려 **24 런**을 낸다.

★**왜 다시 도는가** — 진단서(`../M8_LADDER_2026-08-28/AMENDMENT_1.md §3.3`)가 같은 재현을
**BS 7 레코드**에서 이미 했지만 그 산출물이 scratchpad 로 나갔다. **게이트는 산문이 아니라
자료를 읽어야 한다.** 그리고 진단은 7 레코드였는데 여기서는 **12 전건**을 덮는다 —
`M8` 이 앵커 성립성을 **`AA14` 한 레코드**로만 확인해서 탐지에 실패한 바로 그 결함
(`AMENDMENT_1.md §5.1-2`)의 처방이다: ***앵커 검증은 층을 갈라 표본을 잡아라.***

★**하네스 동일성** — 추정은 `run_m8.run_one` 을 **그대로** 부른다(재구현 0). 러너가
`CUDA_VISIBLE_DEVICES=''` 를 torch 임포트 전에 세우고 계측 shim 을 설치하며, 전제 방벽
(`assert_shipping_defaults` · `assert_bank_patch` · `regress_pass`)도 러너의 것을 쓴다.
바꾸는 것은 **장면 dict 의 `depth` 한 값**뿐이다(`run_m8.py:265` 가 읽는 그 자리).

⛔**이 스크립트는 판정 수치를 화면에 찍지 않는다** — 비트 일치 여부(참/거짓)만 찍는다.
"""
import os
import sys

# ⛔torch 임포트 **전에**. `run_m8` 도 같은 일을 하지만 여기서 먼저 못박는다(이중 방어).
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'

import hashlib      # noqa: E402
import json         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
SRC = os.path.join(ROOT, 'tmp', 'M8_LADDER_2026-08-28')
for _p in (HERE, SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gates_m8b as GB                                  # noqa: E402
import run_m8 as R                                      # noqa: E402  (env·chdir·shim)
import torch                                            # noqa: E402

OUT = os.path.join(HERE, 'anchor_b2depth.jsonl')


def md5file(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def main():
    if torch.cuda.is_available():
        raise SystemExit('⛔GPU 가 보인다 — 중단.')
    # 러너와 **같은** 방벽을 통과해야 같은 하네스라 말할 수 있다
    R.G.assert_shipping_defaults()
    R.G.assert_bank_patch()
    reg_ok, reg_msg = R.G.regress_pass()
    if not reg_ok:
        raise SystemExit(f'⛔패치 회귀 미통과: {reg_msg}')
    R.install_shims()

    d_b2, d_now = GB.b2era_depths(), GB.current_depths()
    prov_ok, prov_msg = GB.b2era_provenance()
    if not (prov_ok and len(d_b2) >= len(GB.RECORDS) and len(d_now) == len(GB.RECORDS)):
        raise SystemExit(f'⛔수심 상수 출처 불량: git={prov_msg} '
                         f'b2={len(d_b2)} now={len(d_now)}')
    prov = dict(
        harness_run_md5=md5file(os.path.join(SRC, 'run_m8.py')),
        harness_gates_md5=md5file(os.path.join(SRC, 'gates_m8.py')),
        scene_station_md5=md5file(os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23',
                                               'scene_station.py')),
        b2era_src_md5=md5file(GB.B2ERA_SRC), b2era_git=prov_msg,
    )
    print(f'  방벽 OK (회귀 {reg_msg[:40]}) · cuda={torch.cuda.is_available()} · '
          f'수심 정정 레코드={[r for r in GB.RECORDS if d_b2[r] != d_now[r]]}')

    rows = []
    with open(OUT, 'w') as fh:
        for rec in GB.RECORDS:
            sc = R.scene_for(rec)          # ★장면은 한 번만 만든다 (수심은 장면에 안 들어간다)
            orig = float(sc['depth'])
            for tag, d in (('b2', float(d_b2[rec])), ('m8', float(d_now[rec]))):
                sc['depth'] = d
                r = R.run_one('BANK', rec, GB.ANCHOR_SEED)
                r.update(prov, batch=GB.BATCH, anchor_tag=tag, depth_declared=d,
                         depth_source=('scene_station_b2era.py(git %s)' % GB.B2ERA_GIT_REV
                                       if tag == 'b2' else 'scene_station.RECORDS_12'))
                fh.write(json.dumps(r) + '\n')
                fh.flush()
                rows.append(r)
                print(f'    {rec:7s} depth={d:>4.1f} tag={tag}  '
                      f'B={r["bank_B"]} members={r["bank_members"]} '
                      f'{r["wall_s"]:.1f}s', flush=True)
            sc['depth'] = orig

    # ── 검증 요약 — ⛔값은 찍지 않는다. **비트 일치 여부만.** ────────────────
    ref2 = R.G.b2_ship_ref()
    bat = {}
    for r in GB.source_rows():
        if (r.get('arm') == 'BANK' and 'error' not in r
                and int(r.get('est_seed', -1)) == GB.ANCHOR_SEED):
            bat.setdefault(r['record'], r)
    by = {(r['anchor_tag'], r['record']): r for r in rows}
    a_ok = sum(1 for rec in GB.RECORDS
               if GB._same_bits(by[('b2', rec)], ref2[rec]))
    b_ok = sum(1 for rec in GB.RECORDS
               if GB._same_bits(by[('m8', rec)], bat[rec]))
    print(f'\n  (a) B-2 수심 복원 → B-2 동결과 비트동일: {a_ok}/{len(GB.RECORDS)}')
    print(f'  (b) 정정 수심     → M8 배치와 비트동일:   {b_ok}/{len(GB.RECORDS)}')
    print(f'-> {OUT}  ({len(rows)}행)')
    return 0 if (a_ok == len(GB.RECORDS) and b_ok == len(GB.RECORDS)) else 1


if __name__ == '__main__':
    sys.exit(main())
