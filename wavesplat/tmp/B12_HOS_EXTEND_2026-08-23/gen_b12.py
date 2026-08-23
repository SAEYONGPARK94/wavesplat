"""B-12 생성기 — `B-11`(혼잡 HOS, n=16)을 **n=48 로 늘린다**.

## 왜
`B-11` 이 헤드라인의 단서 둘(선형 GT · 학습 분포 겹침)을 동시에 떨어뜨렸고 **가족단위 문턱도
통과**했다(14/16, p=0.00209). 그러나 **n=16 은 `B-1`(n=64)보다 작다** — R3 변호인의 원 요구는
**n=64** 였다. ⟹ **신선 시드를 더 붙여 n 을 3배로 늘린다.**

## 설계 — `HOSGATE` 생성기를 **그대로 import** 한다
⛔**생성 파라미터를 손으로 옮기지 않는다**(HOS M·격자·기간·수심·점수 전부 `gen_hos` 의 것).
바꾸는 것은 **시드 목록 하나**다: `1633–1640`(시드 원장 [1633–1818] 에서 앞의 8개).
`BF4/BF5 × 15/30comp × 8 시드 = 32 쌍`(LIN+HOS).
⚠**LIN 을 먼저 만들어야 HOS 가 같은 성분을 쓴다**(`gen_hos` 가 `_LAST` 로 넘긴다) ⟹ 쌍으로 만든다.
⚠**희소는 만들지 않는다** — `B-11b` 가 이미 n=16 으로 답을 냈고 그 층은 판정 무게가 낮다.

## 비용 (실측 기반)
`gen.log` 중앙 **225 s/에피소드** ⟹ 32 쌍 = 64 에피소드 ≈ **4.0 시간**.

## 이 파일은 **생성만** 한다
추정은 `B-12` 러너가 `B-11` 의 동결 게이트(`gates_b11.py`)를 **그대로 상속**해서 돌린다.
⟹ 판정 규칙은 **이미 동결돼 있고 새로 고르지 않는다**(문턱만 새 `n` 으로 재계산).
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
HG = os.path.abspath(os.path.join(HERE, '..', 'HOSGATE_2026-08-13'))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HG); sys.path.insert(0, ROOT)
os.chdir(ROOT)
import gen_hos as G                                   # noqa: E402

NEW_SEEDS = (1633, 1634, 1635, 1636, 1637, 1638, 1639, 1640)
JOBS = [(bf, nc, s) for bf in ('BF4', 'BF5') for nc in (15, 30) for s in NEW_SEEDS]
# ★출력은 HOSGATE 의 data_lin/data_hos 에 **그대로** 넣는다 — 같은 코퍼스를 키우는 것이므로
#   따로 두면 나중에 "어느 디렉터리를 썼는가" 가 또 하나의 함정이 된다.


def main():
    import json
    mpath = os.path.join(HG, 'gen_manifest.json')
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    fh = open(os.path.join(HERE, 'gen_b12.log'), 'a')
    G.log(f'=== B-12 확장: {len(JOBS)} 쌍  시드 {NEW_SEEDS}  HOS M={G.HOS_M}', fh)
    t0 = time.time()
    for i, (bf, nc, seed) in enumerate(JOBS, 1):
        key = f'{bf}_{nc}comp_seed{seed}'
        if f'HOS/{G.fname(bf, nc, seed)}' in manifest:
            G.log(f'[{i}/{len(JOBS)}] {key} 이미 있음 — 건너뜀', fh); continue
        G.log(f'[{i}/{len(JOBS)}] {key}', fh)
        for arm in ('LIN', 'HOS'):          # ★LIN 먼저(같은 성분을 HOS 가 물려받는다)
            try:
                rec = G.gen_one(arm, bf, nc, seed, fh)
                if rec is not None:
                    manifest[f'{arm}/{G.fname(bf, nc, seed)}'] = rec
            except Exception as e:
                import traceback
                G.log(f'  FAIL {arm} {key}: {e!r}', fh)
                G.log(traceback.format_exc()[-600:], fh)
        json.dump(manifest, open(mpath, 'w'), indent=1)
    G.log(f'=== B-12 완료 {time.time()-t0:.0f}s', fh)


if __name__ == '__main__':
    main()
