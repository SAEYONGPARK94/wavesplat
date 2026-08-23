"""B-13 — ★**`B-1` 헤드라인의 프레임 정합을 직접 잰다** (2026-08-23 심야, B2 심사 지적).

## 왜
`b1_full_comparison.json` 의 옛 헤드라인이 죽은 **1순위 사유**가 *"두 팔이 같은 프레임에서
채점된 적이 없다"* 였다(WS 120–220 프레임 vs ETKF 301, **100/100 에피소드 전부**).
그리고 지금 논문이 인용하는 새 헤드라인 `B-1`(혼잡 62/64 · −38.58%)은
- `SHIP` = `W16`(2026-08-22) · 기준선 = `S16`(2026-08-14) ⟹ **8일 떨어진 두 배치**
- ⛔**두 결과 파일 모두 프레임 수를 기록하지 않는다** (`tracking_frames` 가 키 목록에서 빠졌다
  — `RMSE_near`·`avg_shell_rmse`·`loss_history` 에 이은 같은 결함 부류)
⟹ ***같은 결함이 반복됐는지 아닌지를 기록만으로는 판정할 수 없다.***

## 무엇을 재나 (결과 전 확정)
에피소드 **1개**(혼잡 대표)에 대해 두 팔을 **지금 코드로** 한 번씩 돌려
`n_frames_scored` 를 각각 뽑는다.

## 판정 (결과 전)
- 두 팔의 채점 프레임 수가 **같다** ⟹ ✅`FRAMES-MATCH` — 옛 결함이 **재발하지 않았다**는
  **직접 증거**(⚠단 표본 1이고 *"이 에피소드에서"* 다).
- **다르다** ⟹ ⛔`FRAMES-DIFFER` — `B-1` 헤드라인에 **옛 헤드라인과 같은 단서**를 붙여야 한다.

⛔**이 스크립트는 `B-1` 을 재분석하지 않는다.** 프레임 수 하나만 본다.
⚠**표본 1** — 통과해도 *"전 에피소드에서 정합한다"* 로 쓰면 안 된다.
"""
import glob, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'tmp', 'S16_BASELINE_2026-08-14')):
    if p not in sys.path:
        sys.path.insert(0, p)

EP = sorted(glob.glob(os.path.join(ROOT, 'data', 'episode_BF4_15comp_seed*_uniform.h5')))
if not EP:
    EP = sorted(glob.glob(os.path.join(ROOT, 'data', 'episode_BF4_*comp_seed0_uniform.h5')))
ep = EP[0]
print('에피소드:', os.path.basename(ep))

out = {'episode': os.path.basename(ep)}

from wavesplat.run import run_estimation, DEFAULT_CONFIG          # noqa: E402
cfg = dict(DEFAULT_CONFIG); cfg.update(device='cpu', seed=42)
t0 = time.time()
r = run_estimation(ep, config=cfg, verbose=False)
out['ship'] = {'tracking_frames': r.get('tracking_frames'),
               'avg_nRMSE': r.get('avg_nRMSE'), 'wall_s': round(time.time() - t0, 1)}
print('SHIP  ', out['ship'])

import run_s16                                                     # noqa: E402
t0 = time.time()
b = run_s16.run_letkf(ep, 'SPEC25', filter_seed=42)
out['spec25'] = {k: b.get(k) for k in ('n_frames', 'tracking_frames', 'avg_nRMSE')}
out['spec25']['wall_s'] = round(time.time() - t0, 1)
out['spec25']['all_keys'] = sorted(b.keys())
print('SPEC25', {k: v for k, v in out['spec25'].items() if k != 'all_keys'})

a, c = out['ship']['tracking_frames'], (out['spec25'].get('n_frames')
                                        or out['spec25'].get('tracking_frames'))
out['label'] = ('FRAMES-MATCH' if (a is not None and a == c)
                else 'FRAMES-UNKNOWN' if (a is None or c is None) else 'FRAMES-DIFFER')
out['note'] = {'FRAMES-MATCH': '✅옛 결함 재발 없음 (⚠표본 1)',
               'FRAMES-DIFFER': '⛔B-1 헤드라인에 프레임 불일치 단서 필요',
               'FRAMES-UNKNOWN': '⚠한쪽이 프레임 수를 내놓지 않는다 — 계측부터 고쳐야 한다'}[out['label']]
json.dump(out, open(os.path.join(HERE, 'verdict_b13.json'), 'w'), indent=1, ensure_ascii=False)
print('\n', out['label'], '—', out['note'])
