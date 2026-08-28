"""M-8 전제 — **추정기 런 0회.**  `PREREG.md` 의 모든 수는 이 파일이 낸다(손복사 0건).

    python3 tmp/M8_LADDER_2026-08-28/premise_m8.py     # -> premise_m8.json

무엇을 확인하는가
  P1  배포 상수 되읽기 (bank·learned 양쪽)
  P2  ★사다리 기수(基數)의 기하 — `dk_c = frac*kp` 격자의 후보 수 `B(frac)` 는
      **레코드 무관**이다(축 자체가 kp 로 정규화돼 있다).  ⟹ B 를 사전등록에 못 박을 수 있다.
  P3  ★양성대조(앵커) 가용성 — `M4 SHIP-K-INL`(제안 끝) · `B-2 SHIP`(bank 끝) ·
      `M5b ANCHOR`(CPU 고정 끝점 대비) 의 행이 **실제로 존재하는지** 파일에 대고 센다.
      (⛔`M2 G-13` 의 교훈: 외부 참조 게이트는 동결 전에 실제 파일로 돌려라.)
  P4  MDE — 부호검정 p 표 · Holm 문턱 · 80% 검정력이 요구하는 승률 q
  P5  ★관측 신뢰도가 실측에서 **항등적으로 1** 인가 (VARPRO 의 confidence 가중이 무연산인가)
  P6  참 제로필드(`zero_ref`) 12 레코드 캐시 존재
  P7  ★디바이스 누출 — `propose_atoms` 가 `config['device']` 를 보지 않는다는 코드 사실
"""
import hashlib
import inspect
import json
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
for p in (ROOT, os.path.join(ROOT, 'scripts'),
          os.path.join(ROOT, 'tmp', 'B2_REAL_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'B3_REFIT_2026-08-23'),
          os.path.join(ROOT, 'tmp', 'bankmmse_port_2026-07-29')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.chdir(ROOT)

import numpy as np  # noqa: E402

OUT = {}

# ── P1 배포 상수 되읽기 ──────────────────────────────────────────────────────
from wavesplat.run import DEFAULT_CONFIG                       # noqa: E402
from wavesplat.optimizer import bank_mmse as BM                # noqa: E402
from wavesplat.optimizer import learned_init as LI             # noqa: E402
from wavesplat.optimizer import sequential_update as SU        # noqa: E402

_fit_sig = inspect.signature(BM.fit_bank_mmse).parameters
OUT['P1'] = {
    'shipped': {k: DEFAULT_CONFIG[k] for k in (
        'init_method', 'varpro_steps', 'learned_init_iters', 'auto_switch_thresh',
        'lambda_l1', 'analytic_ensemble_k', 'real_estimator', 'real_bank_ensemble_k',
        'bank_refit_every_s', 'bank_refit_async', 'adaptive_n', 'max_n',
        'prune_threshold', 'grow_residual_threshold', 'k_snap_dk',
        'use_confidence', 'tracking_mode', 'device', 'seed')},
    'bank_max_obs_default': 600,          # sequential_update._bank_fit_window
    'fit_bank_mmse_defaults': {k: (None if _fit_sig[k].default is inspect._empty
                                   else _fit_sig[k].default)
                               for k in ('kp_hint', 'dk_c', 'val_frac', 'seed',
                                         'max_em', 'band')},
    'learned_M_OBS_CAP': LI.M_OBS_CAP,
    'ckpt_sha256_ok': LI.ckpt_sha256() == LI.DEFAULT_CKPT_SHA256,
    'pool_defaults': {k: (None if v.default is inspect._empty else v.default)
                      for k, v in inspect.signature(LI.pool_and_select).parameters.items()},
}

# ── P2 사다리 기수의 기하 ────────────────────────────────────────────────────
# ⚠**정규화 근사 금지.**  `fit_bank_mmse` 는 `np.arange(-b_hi*kp, b_hi*kp+dk_c/2, dk_c)`
#   를 **절대 단위**로 만든다 ⟹ 부동소수 끝점 효과 때문에 B 는 kp 에 **약하게** 의존한다
#   (정규화 계산 1908 vs 실측 1918).  ⟹ ★레코드별 **실제 kp** 로 축자 재현한다.
def bank_B_exact(kp, frac, band=(0.35, 2.5)):
    """`bank_mmse.fit_bank_mmse:154-159` 축자."""
    b_lo, b_hi = band
    dk_c = frac * kp
    ax = np.arange(-b_hi * kp, b_hi * kp + dk_c / 2, dk_c)
    KX, KY = np.meshgrid(ax, ax)
    kk = np.hypot(KX, KY)
    return int(((kk > b_lo * kp) & (kk < b_hi * kp)).sum())


def bank_kp(P, z, depth):
    """`fit_bank_mmse` 의 2단 beamform zoom 축자 (GT-free, 추정기 런 아님)."""
    bx, by = BM._beamform_top(P.astype(np.float32), z.astype(np.float32),
                              depth, klim=2.0, n=120)
    kp = max(np.hypot(bx, by), 1e-3)
    bx, by = BM._beamform_top(P.astype(np.float32), z.astype(np.float32),
                              depth, klim=3.5 * kp, n=120)
    return float(max(np.hypot(bx, by), 1e-3))


FRACS = [0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]

# ── P3 앵커(양성대조) 가용성 ────────────────────────────────────────────────
def _jsonl(path):
    out = []
    if os.path.exists(path):
        for l in open(path):
            try:
                out.append(json.loads(l))
            except json.JSONDecodeError:
                pass
    return out

import glob                                                   # noqa: E402
m4 = []
for f in sorted(glob.glob(os.path.join(ROOT, 'tmp/M4_CAPACITY_2026-08-28/results_m4_*.jsonl'))):
    m4 += _jsonl(f)
m4_inl = [r for r in m4 if r.get('stratum') == 'real' and r.get('arm') == 'SHIP-K-INL']
b2 = _jsonl(os.path.join(ROOT, 'tmp/B2_REAL_2026-08-23/results.jsonl'))
m5b = []
for f in sorted(glob.glob(os.path.join(ROOT, 'tmp/M5b_WINDOW_2026-08-28/results_m5b_*.jsonl'))):
    m5b += _jsonl(f)
m5b_anchor = [r for r in m5b if r.get('level') == 'ANCHOR' and r.get('win') == 3
              and 'error' not in r]
from gates_b2 import RECORDS                                   # noqa: E402
OUT['P3'] = {
    'records': RECORDS,
    'M4_SHIP_K_INL': {'n_rows': len(m4_inl),
                      'records': sorted({r['unit'] for r in m4_inl}),
                      'seeds': sorted({r['est_seed'] for r in m4_inl}),
                      'cuda_available': sorted({r.get('cuda_available') for r in m4_inl}),
                      'nowcast': {r['unit']: r['nowcast'] for r in m4_inl}},
    'B2_SHIP': {'n_rows': len([r for r in b2 if r['cell'] == 'SHIP']),
                'nowcast': {r['record']: r['nowcast'] for r in b2 if r['cell'] == 'SHIP'}},
    'B2_PROP': {'n_rows': len([r for r in b2 if r['cell'] == 'PROP']),
                'nowcast': {r['record']: r['nowcast'] for r in b2 if r['cell'] == 'PROP'}},
    'M5b_ANCHOR': {'n_rows': len(m5b_anchor),
                   'cap_mode': sorted({r.get('cap_mode_used') for r in m5b_anchor})},
}
# ★앵커가 비어 있으면 전제 실패 (공허 참조 방어)
assert len(m4_inl) == 12 and sorted({r['est_seed'] for r in m4_inl}) == [42], 'M4 앵커 부재'
assert OUT['P3']['B2_SHIP']['n_rows'] == 12, 'B-2 SHIP 앵커 부재'

# ── P4 MDE ──────────────────────────────────────────────────────────────────
def sign_p(w, n):
    return sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n

def power(q, n, thr):
    return sum(comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(thr, n + 1))

def q80(n, thr):
    lo, hi = 0.5, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if power(mid, n, thr) < 0.80:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)

N = 12
K_FAMILY = 6                       # 사다리 고리 수 (부차 가족)
holm = [round(0.05 / (K_FAMILY - i), 6) for i in range(K_FAMILY)]
OUT['P4'] = {
    'n': N,
    'sign_p': {str(w): round(sign_p(w, N), 8) for w in range(6, N + 1)},
    'primary_threshold': 10,
    'primary_alpha': round(sign_p(10, N), 6),
    'primary_q80': q80(N, 10),
    'holm_alphas_family6': holm,
    'holm_min_wins': [min([w for w in range(N + 1) if sign_p(w, N) <= a] or [None])
                      for a in holm],
    'holm_q80': [q80(N, min([w for w in range(N + 1) if sign_p(w, N) <= a] or [N]))
                 for a in holm],
    'floor_p': round(sign_p(12, 12), 8),
}

# ── P5 실측 신뢰도가 항등적으로 1 인가 (VARPRO confidence 가중의 무연산 여부) ──
import torch                                                    # noqa: E402
from scene_station import scene_station                          # noqa: E402
import make_gif_bank_deployed as MK                              # noqa: E402
rec0 = RECORDS[0]
sc = scene_station(rec0, win=3)
t_shift = float(np.asarray(sc['P'])[:, 2].min())
fr = MK.frames_from_scene(sc, t_shift, dtype=torch.float32)
OUT['P5'] = {
    'record': rec0, 'n_frames': len(fr), 'n_obs': int(len(sc['z'])),
    'confidence_all_one': bool(all(torch.all(f.confidence == 1) for f in fr)),
    'incidence_all_zero': bool(all(torch.all(f.incidence_angles == 0) for f in fr)),
    'builder': 'tmp/bankmmse_port_2026-07-29/make_gif_bank_deployed.py:78',
    'consequence': ("confidence 가 항등적으로 1 이면 varpro weighting='confidence' 는 "
                    "'none' 과 **비트 동일**하다 (varpro.py:127-130 sqrt(clamp(conf))) "
                    "⟹ 신뢰가중 축은 실측에서 **도달 불가**."),
}

# ── P2(계속) 레코드별 kp 와 B(frac) ─────────────────────────────────────────
_kp, _B = {}, {}
for _r in RECORDS:
    _sc = sc if _r == rec0 else scene_station(_r, win=3)
    _kp[_r] = bank_kp(np.asarray(_sc['P'], float), np.asarray(_sc['z'], float),
                      float(_sc['depth']))
    _B[_r] = {str(f): bank_B_exact(_kp[_r], f) for f in FRACS}
OUT['P2'] = {'shipped_frac': 0.1, 'kp': {k: round(v, 6) for k, v in _kp.items()},
             'B_of_frac': _B,
             'B_range': {str(f): [min(_B[r][str(f)] for r in RECORDS),
                                  max(_B[r][str(f)] for r in RECORDS)] for f in FRACS},
             'note': ('B 는 frac 이 정하고 kp 는 부동소수 끝점으로만 ±소량 흔든다. '
                      '러너는 B 를 **행에 싣고** 게이트가 이 표와 대조한다.')}

# ── P6 참 제로필드 ───────────────────────────────────────────────────────────
from zero_ref import zero_ref_nowcast                            # noqa: E402
OUT['P6'] = {r: round(float(zero_ref_nowcast(r, win=3)), 6) for r in RECORDS}
assert len(OUT['P6']) == 12

# ── P7 디바이스 누출 (코드 사실) ─────────────────────────────────────────────
src = inspect.getsource(LI.propose_atoms)
OUT['P7'] = {
    'propose_atoms_device_line': [l.strip() for l in src.splitlines()
                                  if 'cuda' in l or 'device is None' in l],
    'reads_config_device': ('config' in inspect.signature(LI.propose_atoms).parameters),
    'sample_seeds_globally': True,   # d3_beamform_cond_diffusion.sample:160 torch.manual_seed
    'consequence': ("`propose_atoms` 는 `config['device']` 를 받지 않는다. 디바이스는 "
                    "`torch.cuda.is_available()` 로 정해지고, `sample()` 은 "
                    "`torch.randn(..., device=DEVICE)` 를 쓴다 ⟹ 같은 시드라도 "
                    "CPU/CUDA 에서 **다른 원자**가 나온다. ⟹ 러너는 반드시 "
                    "`CUDA_VISIBLE_DEVICES=''` 를 torch 임포트 전에 세운다."),
}

# ── 임포트 경로 md5 (동결 대상 후보) ─────────────────────────────────────────
def md5(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()

OUT['MD5'] = {p: md5(os.path.join(ROOT, p)) for p in (
    'wavesplat/optimizer/sequential_update.py',
    'wavesplat/optimizer/bank_mmse.py',
    'wavesplat/optimizer/learned_init.py',
    'wavesplat/optimizer/varpro.py',
    'wavesplat/run.py',
    'tmp/B2_REAL_2026-08-23/scene_station.py',
    'tmp/B2_REAL_2026-08-23/zero_ref.py',
    'tmp/B2_REAL_2026-08-23/gates_b2.py',
    'tmp/bankmmse_port_2026-07-29/make_gif_bank_deployed.py',
    'scripts/d3_beamform_cond_diffusion.py',
)}

with open(os.path.join(HERE, 'premise_m8.json'), 'w') as fh:
    json.dump(OUT, fh, indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in OUT.items() if k in ('P2', 'P4', 'P5', 'P7')},
                 indent=1, ensure_ascii=False))
print('\nP3 anchors: M4_SHIP_K_INL rows =', OUT['P3']['M4_SHIP_K_INL']['n_rows'],
      '| B2_SHIP rows =', OUT['P3']['B2_SHIP']['n_rows'],
      '| M5b ANCHOR rows =', OUT['P3']['M5b_ANCHOR']['n_rows'],
      OUT['P3']['M5b_ANCHOR']['cap_mode'])
print('OK -> premise_m8.json')
