"""M7 `train_m7.py` — Phase A(사전학습 3회) + Phase B(미세조정 5회).

⛔**배포 경로 무수정.** `scripts/d3_beamform_cond_diffusion.py`(모델·스케줄·ε-MSE·옵티마이저)와
`tmp/wp_e2e_2026-07-25/e2e_finetune.py`(장 오차 손실)는 **임포트**한다 — 한 줄도 고치지 않고
한 줄도 베끼지 않는다.  베끼면 이 저장소의 반복 결함 1위(*"같은 기능이 두 파일에 구현돼
한쪽만 계측"*)를 새로 만드는 것이다.

  Phase A = `tmp/mixed_retrain_2026-07-26/train_mixed.py` 의 레시피
            (40k · ε-MSE · AdamW 2e-4/1e-4 · cosine · bsz 32 · from scratch),
            **장면 공급원만** `gen_mixed` → `gen_m7`(BASE 모드에서 비트 동일, `G-10`).
  Phase B = `e2e_finetune.finetune` 의 레시피(4k · lr 5e-5 · bsz 32 · t_lo=T/4 · clip 5.0),
            **손실 항 `field_loss_from_x0hat` 을 그 파일에서 그대로 임포트**해 쓰고
            장면만 `gen_m7` 위로 이식한다.  λ_field ∈ {0.0, 1.0}.

⛔⛔**`e2e_finetune.py` 의 `__main__` 은 절대 실행하지 마라** — 그 블록이 곡선을
`tmp/wp_e2e_2026-07-25/train_curve_{A,B}.json` 에 **하드코딩**해서 쓴다(= 다른 배치의 **동결
산출물**을 덮는다).  이 파일은 그 모듈을 **라이브러리로만** 부르고 곡선은 배치 안에 쓴다.

⛔**`CUDA_VISIBLE_DEVICES` 를 세우지 않는다**(`AMENDMENT_1 §5.2`) — `learned_init.py:133-134` 가
스스로 cuda 를 고르고 `config['device']` 를 읽지 않으므로, 감추면 **측정된 적 없는 팔**이 된다.
그리고 이건 학습 배치다.

★**결정론**: 시드는 인자로 받고 **곡선·매니페스트·로그에 전부 적힌다**.  `BASE-B` 는
**사전학습 시드만** 다르다(Phase B 시드는 다섯 칸에서 같다 ⟹ 대비가 정확히 두 요인뿐).

★**손복사 상수 0건**: 학습 상수는 `gates_m7` 이 단일출처이고, 이 파일은 착수 전에 그 값들이
**동결 원본과 일치하는지 되읽어 대조**한다(`_constant_parity`).  어긋나면 학습 0스텝에서 멈춘다.

사용:
  python3 tmp/M7_SUPERVISION_2026-08-28/train_m7.py --phase A --data HARD --smoke 200 --out /tmp/x.pt
  python3 tmp/M7_SUPERVISION_2026-08-28/train_m7.py --phase A --data BASE --seed 20260828 \
      --steps 40000 --out results/direction3/m7_PRE-BASE-1.pt \
      --curve tmp/M7_SUPERVISION_2026-08-28/curve_PRE-BASE-1.json
  python3 tmp/M7_SUPERVISION_2026-08-28/train_m7.py --phase B --cell LOSS \
      --init results/direction3/m7_PRE-BASE-1.pt --lambda_field 1.0 --steps 4000 \
      --out results/direction3/m7_LOSS.pt --curve tmp/M7_SUPERVISION_2026-08-28/curve_LOSS.json
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (ROOT, os.path.join(ROOT, 'scripts'),
           os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26'),
           os.path.join(ROOT, 'tmp/wp_e2e_2026-07-25'), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(ROOT)

import torch
import torch.nn.functional as F

import d3_beamform_cond_diffusion as cd     # 배포 아키텍처/스케줄/손실 — ⛔무수정
import e2e_finetune as e2e                  # 장 오차 손실 — ⛔무수정, `__main__` 미실행
import gates_m7 as G
import gen_m7 as g7
import gen_mixed as gm

DEVICE = cd.DEVICE
SHIPPED_CKPT = os.path.join(ROOT, 'results/direction3/beamform_cond_diffusion_mixed_v1.pt')
LEGACY_CKPT = os.path.join(ROOT, 'results/direction3/beamform_cond_diffusion.pt')
CENSUS = os.path.join(HERE, 'stream_census.json')          # 누적 원장(중첩) — 이 파일이 원본
TRAIN_MANIFEST = os.path.join(HERE, 'train_manifest_m7.json')
# ★임무 B(`analyze_m7.load_train_manifest`)가 읽는 **평평한** 투영.  키 이름은 그쪽 규약이고,
#   값은 전부 원장에서 파생된다(⛔두 번 세지 않는다 — 투영이지 두 번째 구현이 아니다).
VAL_SEED = 987654321          # train_mixed.py:33 (유보 검증 장면 — 체크포인트를 고르지 않는다)
VAL_N = 256                   # train_mixed.make_val 기본값
FT_SEED = 0                   # e2e_finetune.CONFIGS['A']['seed'] — 다섯 칸에서 **같다**
V1_CURVE = os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26/train_curve.json')


def _stop(msg, code=1):
    print(f"\n[TRAIN-STOP] {msg}", flush=True)
    sys.exit(code)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


def md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()


# ── 착수 전 대조 (학습 0스텝에서 돈다) ───────────────────────────────────────
def _constant_parity():
    """★`gates_m7` 의 학습 상수가 **동결 원본**과 같은가.  손복사 상수 사냥의 자기 적용.

    출처: Phase A = `mixed_retrain/train_curve.json['args']`(v1 을 실제로 낸 실행의 인자),
          Phase B = `e2e_finetune.CONFIGS['A']`(그 파일의 동결 dict).
    ⛔어긋나면 학습을 시작하지 않는다."""
    v1 = json.load(open(V1_CURVE))['args']
    A = e2e.CONFIGS['A']
    checks = [
        ('PRETRAIN_STEPS', G.PRETRAIN_STEPS, v1['steps'], 'mixed_retrain/train_curve.json:args.steps'),
        ('PRETRAIN_LR', G.PRETRAIN_LR, v1['lr'], '〃 args.lr'),
        ('BSZ(pretrain)', G.BSZ, v1['bsz'], '〃 args.bsz'),
        ('FINETUNE_STEPS', G.FINETUNE_STEPS, A['steps'], "e2e_finetune.CONFIGS['A'].steps"),
        ('FINETUNE_LR', G.FINETUNE_LR, A['lr'], '〃 lr'),
        ('LAMBDA_FIELD_ON', G.LAMBDA_FIELD_ON, A['lambda_field'], '〃 lambda_field'),
        ('BSZ(finetune)', G.BSZ, A['bsz'], '〃 bsz'),
        ('T_LO', int(G.T_LO_FRAC * cd.T), A['t_lo'], '〃 t_lo'),
        ('NGRID', G.NGRID, e2e.NGRID, 'e2e_finetune.NGRID'),
        ('M_atoms', gm.M, cd.M, 'gen_mixed.M vs cd.M'),
        ('M_OBS', gm.M_OBS, cd.M_OBS, 'gen_mixed.M_OBS vs cd.M_OBS'),
        ('DEPTH', gm.DEPTH, cd.DEPTH, 'gen_mixed.DEPTH vs cd.DEPTH'),
    ]
    bad = [c for c in checks if c[1] != c[2]]
    for nm, a, b, src in checks:
        print(f"  [const] {nm:16s} gates={a!r:12s} == {b!r:12s}  ({src})"
              f"{'   ⛔불일치' if a != b else ''}", flush=True)
    if bad:
        _stop(f"학습 상수가 동결 원본과 다르다: {[(c[0], c[1], c[2]) for c in bad]}")
    return [dict(name=c[0], value=c[1], source=c[3]) for c in checks]


def _shipped_intact():
    """★G-3: 배포 체크포인트 무손상.  ⛔학습 전에, 그리고 저장 후에 다시 잰다."""
    s = sha256(SHIPPED_CKPT)
    if s != '636c72a797ee6d0b725449293d5cd230dbd64c8eb03e345da4561b0c265239d9':
        _stop(f"배포 체크포인트 sha256 가 바뀌었다: {s}")
    return s


def _git_head():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return None


def _census_is_stale(census):
    """★원장이 **현재 파일계와 일치하는가** — 지난 실행의 유령 행을 매니페스트에 흘리지 않는다.

    이 저장소가 반복해서 대가를 치른 부류(*"의도표"*)의 방어다: 원장이 *주장하는* 체크포인트를
    **되읽어** sha256 을 대조하고, 하나라도 사라졌거나 달라졌으면 **착수 전에 멈춘다**
    (시험 실행·중단된 실행의 행이 남아 `G-2`/`G-11` 을 오염시키는 것을 막는다)."""
    bad = []
    for tag, r in (census.get('runs') or {}).items():
        pth = os.path.join(ROOT, r.get('out', ''))
        if not os.path.exists(pth):
            bad.append((tag, 'ckpt 없음', r.get('out')))
        elif sha256(pth) != r.get('ckpt_sha256'):
            bad.append((tag, 'sha256 불일치', r.get('out')))
    return bad


def _merge_census(update):
    """★러너의 스탬프를 한 곳에 모은다 — `run_m7`/`analyze_m7` 이 매니페스트를 여기서 만든다.
    ⛔러너가 같은 수를 다시 세지 않는다(단일출처)."""
    cur = json.load(open(CENSUS)) if os.path.exists(CENSUS) else {}
    for k, v in update.items():
        if isinstance(v, dict) and isinstance(cur.get(k), dict):
            cur[k].update(v)
        else:
            cur[k] = v
    json.dump(cur, open(CENSUS, 'w'), indent=1, ensure_ascii=False)
    return cur


def _project_manifest(census):
    """원장 → `analyze_m7` 이 읽는 평평한 학습-측 매니페스트.

    ⛔**커버리지 0 금지**: 아직 재료가 없는 계수기는 `0` 이 아니라 **키를 두지 않는다**
    (0 을 두면 게이트가 자료 없이 통과한다 = `M4 G-13` 부류).
    ⛔`eval_records`·`ckpt_sha256`·`shipped_sha256` 은 **적지 않는다** — 그것들은
    `analyze_m7` 이 파일/행 되읽기로 만드는 것이 옳고, 매니페스트가 주장하면 안 된다."""
    st = census.get('stream', {})
    man = {}
    if census.get('train_scene_seeds_sample'):
        man['train_scene_seeds_sample'] = census['train_scene_seeds_sample']
    if 'HARD' in st:
        man['hard_scenes_n'] = st['HARD']['n_scenes']
        man['hard_scenes_below_repr'] = st['HARD']['below_repr']
        man['hard_kHs_median'] = st['HARD']['kHs_median']
        man['anchors_excluded'] = st['HARD']['anchors_excluded']
    if 'BASE' in st:
        man['base_kHs_median'] = st['BASE']['kHs_median']
    if 'lam0_finetune_steps' in census:
        man['lam0_finetune_steps'] = census['lam0_finetune_steps']
    man['train_manifest_source'] = os.path.relpath(CENSUS, ROOT)
    man['train_manifest_written_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    json.dump(man, open(TRAIN_MANIFEST, 'w'), indent=1, ensure_ascii=False)
    return man


def _gate_selfcheck(census):
    """★러너의 스탬프로 만든 **합성 행**을 동결 `gates_m7.gates()` 에 그대로 먹인다.

    ⛔게이트를 모형하지 않는다(그러면 「의도표」다) — 진짜 게이트 함수를 부르고,
    이 실행이 **결정할 수 있는 행만** 강제한다.  나머지는 아직 재료가 없어 FAIL 로 보이며
    그것은 정상이다(평가·분석 단계가 채운다)."""
    man = {}
    st = census.get('stream', {})
    if 'HARD' in st:
        man['hard_scenes_n'] = st['HARD'].get('n_scenes')
        man['hard_scenes_below_repr'] = st['HARD'].get('below_repr')
        man['hard_kHs_median'] = st['HARD'].get('kHs_median')
        man['anchors_excluded'] = st['HARD'].get('anchors_excluded')
    if 'BASE' in st:
        man['base_kHs_median'] = st['BASE'].get('kHs_median')
    man['train_scene_seeds_sample'] = census.get('train_scene_seeds_sample', [])
    # ⛔`eval_records` 를 **여기서 주입하지 않는다** — 그러면 `G-8` 이 항등적으로 통과한다.
    #   그 절반(*"평가 12 레코드를 제외하지 않았다"*)은 E1 행에서만 확인할 수 있고
    #   `analyze_m7.build_manifest` 가 행 되읽기로 만든다.  여기서 강제하는 것은
    #   앵커 절반뿐이고, 그것은 `gen_m7.hard_anchors()` 가 **스트림 생성 시점에 정지**로 지킨다.
    if 'lam0_finetune_steps' in census:
        man['lam0_finetune_steps'] = census['lam0_finetune_steps']
    if census.get('ckpt_sha256'):
        man['ckpt_sha256'] = census['ckpt_sha256']
    man['shipped_sha256'] = census.get('shipped_sha256')

    owned = {'★G-6', '★G-7', 'G-9', '★G-11', '★G-3'}      # ⛔G-8 은 행이 있어야 판정된다
    rows = G.gates(man)
    print("\n  ── 게이트 자가검사 (이 실행이 결정할 수 있는 행만 강제) ──", flush=True)
    if man.get('anchors_excluded') is not None:
        print(f"    [정지로 강제] 앵커 물리 스크린 제외 = {man['anchors_excluded']} "
              f"(★G-8 의 앵커 절반. 나머지 절반 '평가 미제외' 는 E1 행이 있어야 판정된다)",
              flush=True)
    fail = []
    for name, ok, detail in rows:
        key = name.split()[0]
        mine = key in owned
        # G-2/G-11 은 다섯 칸이 다 돌아야 결정된다 — 미완이면 '미결'로 표시하고 강제하지 않는다.
        if key == '★G-11' and 'lam0_finetune_steps' not in man:
            mine = False
        if key in ('★G-6', '★G-7') and not ('HARD' in st and 'BASE' in st):
            mine = key == '★G-6' and 'HARD' in st
        tag = ('강제' if mine else '미결')
        print(f"    [{tag}] {'PASS' if ok else 'fail'}  {name}  ({detail})", flush=True)
        if mine and not ok:
            fail.append(name)
    if fail:
        _stop(f"게이트 자가검사 실패: {fail}")
    return [dict(gate=n, ok=bool(o), detail=d) for n, o, d in rows]


# ── 공통: 장면 스트림 · 검증 ────────────────────────────────────────────────
@torch.no_grad()
def val_loss(model, val_scenes, stats, abar, bsz=32, seed=0):
    """`train_mixed.val_loss` 와 같은 규약 — 고정 잡음 ε-MSE, **보고 전용**
    (⛔체크포인트를 고르지 않는다: 이 배치는 학습 도중 어떤 선택도 하지 않는다)."""
    model.eval()
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    tot, nb = 0.0, 0
    for i in range(0, len(val_scenes), bsz):
        chunk = val_scenes[i:i + bsz]
        x0, Bmap = gm.batch_from_scenes(chunk, stats, DEVICE)
        b = x0.shape[0]
        t = torch.randint(0, cd.T, (b,), device=DEVICE, generator=g)
        ab = abar[t][:, None, None]
        eps = torch.randn(x0.shape, device=DEVICE, generator=g)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        tot += float(F.mse_loss(model(xt, t, Bmap), eps))
        nb += 1
    model.train()
    return tot / max(nb, 1)


def _row(**kw):
    return {k: v for k, v in kw.items() if v is not None}


# ── Phase A ─────────────────────────────────────────────────────────────────
def phase_a(a, stats, abar, bank):
    stream = g7.SceneStream(a.data, a.seed, bank, target_mode=a.target_mode)
    print(f"  장면 스트림: data={a.data} seed={a.seed} target_mode={stream.target_mode}",
          flush=True)
    # ★필터 항진 탐침 — 거절 없이 원시 장면을 뽑아 문턱 미달 비율을 잰다(분모가 살아 있다).
    probe = g7.probe_repr_filter(a.data, n=a.probe_n, seed=a.seed + 1, bank=bank,
                                 anchors=stream.anchors)
    print(f"  [항진 탐침] 거절 없이 {probe['n_scored']}장면 중 문턱 미달 {probe['below']} "
          f"({probe['frac_below_min']:.4f})  min {probe['min']:.4f}", flush=True)
    # ⛔새 문턱을 발명하지 않는다.  `sample_scene_ok` 가 20회 재시도하므로 탈출 확률은 p^20 이고,
    #   전체 런에서 기대되는 탈출 수 = steps·bsz·p^20 이다.  그 수가 1 이상이면 런 도중에
    #   반드시 멈추게 되므로 **한 시간을 쓰기 전에** 알려준다(파생량이지 고른 값이 아니다).
    pfb = probe['frac_below_min'] or 0.0
    exp_escape = (a.smoke or a.steps) * a.bsz * (pfb ** 20)
    print(f"           탈출 기대수 = {a.smoke or a.steps}×{a.bsz}×p^20 = {exp_escape:.3e}",
          flush=True)
    if pfb >= 1.0:
        _stop(f"{a.data} 스트림의 모든 원시 장면이 표현 문턱 미달이다 — 팔이 구조적으로 빈다")
    if exp_escape >= 1.0:
        print(f"  ⚠경고: 이 런에서 표현 필터 탈출이 {exp_escape:.2f}회 기대된다 — "
              f"탈출하면 G-6 위반으로 즉시 멈춘다.", flush=True)

    val_stream = g7.SceneStream(a.data, VAL_SEED, bank, verbose=False)
    val_scenes = val_stream.batch(VAL_N)
    print(f"  유보 검증: {len(val_scenes)} 장면 (seed {VAL_SEED}, 같은 분포)", flush=True)

    torch.manual_seed(a.seed)
    model = cd.CondSetDenoiser().to(DEVICE)          # from scratch
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    n_steps = a.smoke or a.steps
    curve, t0 = [], time.time()
    for s in range(n_steps):
        sc = stream.batch(a.bsz)
        x0, Bmap = gm.batch_from_scenes(sc, stats, DEVICE)
        t = torch.randint(0, cd.T, (a.bsz,), device=DEVICE)
        ab = abar[t][:, None, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        loss = F.mse_loss(model(xt, t, Bmap), eps)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if a.smoke:
            if s % 50 == 0:
                print(f"  smoke step {s:4d}  train {loss.item():.5f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            continue
        if s % a.val_every == 0 or s == n_steps - 1:
            v = val_loss(model, val_scenes, stats, abar)
            curve.append(dict(step=s, train=float(loss), val=float(v), sec=time.time() - t0))
            print(f"  step {s:6d}  train {loss.item():.5f}  val {v:.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            json.dump(dict(curve=curve, args=vars(a), stream=stream.stats(), probe=probe),
                      open(a.curve, 'w'), indent=1, ensure_ascii=False)
        elif s % 250 == 0:
            print(f"  step {s:6d}  train {loss.item():.5f} ({time.time()-t0:.0f}s)", flush=True)

    wall = time.time() - t0
    ms = 1000.0 * wall / n_steps
    if a.smoke:
        print(f"\nsmoke[phase A data={a.data}]: {ms:.1f} ms/step  ->  {G.PRETRAIN_STEPS} 스텝 = "
              f"{ms * G.PRETRAIN_STEPS / 3.6e6:.2f} h   (장면 {stream.n}개 소모)", flush=True)
        return dict(kind='smoke', ms_per_step=ms, n_steps=n_steps, wall_s=wall,
                    stream=stream.stats(), probe=probe)
    return dict(kind='phaseA', model=model, curve=curve, wall_s=wall, ms_per_step=ms,
                stream=stream.stats(), probe=probe, val_final=curve[-1]['val'] if curve else None)


# ── Phase B ─────────────────────────────────────────────────────────────────
def phase_b(a, stats, abar, bank):
    """`e2e_finetune.finetune` 의 이식.  손실 함수 자체는 그 파일에서 **임포트**한다."""
    stream = g7.SceneStream(a.data, a.seed, bank, target_mode=a.target_mode)
    gb = g7.bank_bounds(bank)
    print(f"  ★Phase B 분포 = 칸 정의(gates_m7.CELLS[{a.cell!r}]['data'])에서 파생 = {a.data} "
          f"— 요인 D 를 Phase A/B 양쪽에 적용한다(공시 §보고)", flush=True)
    print(f"  장면 스트림: data={a.data} seed={a.seed}  |  유보 격자 "
          f"{G.NGRID}x{G.NGRID} over x[{gb[0]:.1f},{gb[1]:.1f}] y[{gb[2]:.1f},{gb[3]:.1f}]",
          flush=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)                    # e2e_finetune.finetune 축자
    model = cd.CondSetDenoiser().to(DEVICE)
    model.load_state_dict(torch.load(a.init, map_location=DEVICE))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    Bsz = a.bsz
    nf = Bsz // 2
    lam = a.lambda_field
    t_lo = int(G.T_LO_FRAC * cd.T)
    n_steps = a.smoke or a.steps
    curve, t0 = [], time.time()
    grad_probe = None
    for s in range(n_steps):
        sc = stream.batch(Bsz)
        batch = g7.field_batch(sc, stats, DEVICE, gb)
        x0 = batch['x0']
        t_anchor = torch.randint(0, cd.T, (Bsz - nf,), device=DEVICE)
        t_field = torch.randint(0, t_lo, (nf,), device=DEVICE)
        t = torch.cat([t_anchor, t_field], 0)
        ab = abar[t][:, None, None]
        eps = torch.randn_like(x0)
        xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
        eps_pred = model(xt, t, batch['Bmap'])
        anchor = F.mse_loss(eps_pred, eps)
        xf = xt[Bsz - nf:]; epf = eps_pred[Bsz - nf:]; abf = ab[Bsz - nf:]
        x0hat = ((xf - (1 - abf).sqrt() * epf) / abf.sqrt()).clamp(-1, 1)
        fbatch = {k: (v[Bsz - nf:] if torch.is_tensor(v) and v.shape[0] == Bsz else v)
                  for k, v in batch.items()}
        fld = e2e.field_loss_from_x0hat(x0hat, fbatch, stats)      # ⛔동결 손실, 무수정
        loss = anchor + lam * fld

        if s == 0:
            # ★개입 계수기 — *"λ 항이 실제로 기울기를 만드는가"* 를 **런 안에서** 확인한다.
            #   (저장소 처방: 개입 기대값은 원자 카운터로 런 0회 확인)
            ga = torch.autograd.grad(anchor, list(model.parameters()),
                                     retain_graph=True, allow_unused=True)
            gf = torch.autograd.grad(lam * fld, list(model.parameters()),
                                     retain_graph=True, allow_unused=True)
            nrm = lambda gs: float(torch.sqrt(sum((g.detach() ** 2).sum()
                                                  for g in gs if g is not None)))
            grad_probe = dict(anchor_grad_norm=nrm(ga), field_grad_norm=nrm(gf),
                              lambda_field=lam)
            print(f"  [개입 계수기] |∇anchor| {grad_probe['anchor_grad_norm']:.4e}   "
                  f"|∇(λ·field)| {grad_probe['field_grad_norm']:.4e}  (λ={lam})", flush=True)
            if lam > 0 and grad_probe['field_grad_norm'] <= 0.0:
                _stop("λ_field > 0 인데 장 손실이 기울기를 만들지 않는다 — 개입 미발화")
            if lam == 0 and grad_probe['field_grad_norm'] != 0.0:
                _stop("λ_field = 0 인데 장 손실이 기울기를 만든다 — 대조 팔이 오염됐다")

        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), G.GRAD_CLIP)
        opt.step(); sched.step()
        if s % 100 == 0 or s == n_steps - 1:
            curve.append(dict(step=s, anchor=float(anchor), field=float(fld),
                              total=float(loss), sec=time.time() - t0))
            print(f"  step {s:5d}  anchor {anchor.item():.5f}  field {fld.item():.5f}  "
                  f"total {loss.item():.5f}  ({time.time()-t0:.0f}s)", flush=True)
            if not a.smoke:
                json.dump(dict(curve=curve, args=vars(a), stream=stream.stats(),
                               grad_probe=grad_probe, grid_bounds=list(gb)),
                          open(a.curve, 'w'), indent=1, ensure_ascii=False)

    wall = time.time() - t0
    ms = 1000.0 * wall / n_steps
    if a.smoke:
        print(f"\nsmoke[B cell={a.cell} data={a.data} λ={lam}]: {ms:.1f} ms/step  -> "
              f"{G.FINETUNE_STEPS} 스텝 = {ms * G.FINETUNE_STEPS / 6e4:.1f} min", flush=True)
        return dict(kind='smoke', ms_per_step=ms, n_steps=n_steps, wall_s=wall,
                    stream=stream.stats(), grad_probe=grad_probe)
    return dict(kind='phaseB', model=model, curve=curve, wall_s=wall, ms_per_step=ms,
                stream=stream.stats(), grad_probe=grad_probe, grid_bounds=list(gb))


# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['A', 'B'], required=True)
    ap.add_argument('--data', choices=list(g7.DATA_MODES), default=None,
                    help='Phase A 필수. Phase B 는 기본값 = gates_m7.CELLS[cell].data')
    ap.add_argument('--cell', choices=list(G.CELL_ORDER), default=None, help='Phase B 필수')
    ap.add_argument('--init', default=None, help='Phase B 시작 체크포인트')
    ap.add_argument('--lambda_field', type=float, default=None, help='Phase B: 기본값 = 칸 정의')
    ap.add_argument('--steps', type=int, default=None)
    ap.add_argument('--bsz', type=int, default=G.BSZ)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--curve', default=None)
    ap.add_argument('--val_every', type=int, default=1000)
    ap.add_argument('--probe_n', type=int, default=400)
    ap.add_argument('--target_mode', default='replicate',
                    choices=['replicate', 'graded', 'graded2'],
                    help="배포 v1 의 규약 = 'replicate'(train_curve.json 이 그 실행을 증언한다)")
    ap.add_argument('--smoke', type=int, default=0, help='n 스텝만 돌고 저장하지 않는다')
    ap.add_argument('--nshard', type=int, default=1)
    a = ap.parse_args()

    if a.nshard != 1:
        _stop(f"병렬 학습 금지: --nshard {a.nshard} (단일 GPU · 단일 순차 RNG)")

    # ── 팔 정의를 gates_m7 에서 되읽는다 (⛔빈 dict 팔 금지) ──
    if a.phase == 'A':
        if a.data is None:
            _stop('Phase A 는 --data 가 필요하다')
        a.cell = None
        a.steps = a.steps or G.PRETRAIN_STEPS
        a.lr = a.lr if a.lr is not None else G.PRETRAIN_LR
        cand = {k: v['seed'] for k, v in G.PRETRAINS.items() if v['data'] == a.data}
        if a.seed is None:
            if not a.smoke:
                _stop(f'Phase A 본실행은 --seed 를 명시해야 한다 (사전등록 후보: {cand})')
            # ⚠smoke 는 비용 측정 전용(체크포인트 저장 0)이라 사전등록의 대표 시드를 쓴다.
            a.seed = G.PRETRAINS[f'PRE-{a.data}-1']['seed']
            print(f"  (smoke: --seed 생략 ⟹ 사전등록 대표 시드 {a.seed} 사용, 저장 없음)",
                  flush=True)
        a.lambda_field = None
        tag = next((k for k, v in G.PRETRAINS.items()
                    if v['data'] == a.data and v['seed'] == a.seed), f'PRE-{a.data}-seed{a.seed}')
    else:
        if a.cell is None:
            _stop('Phase B 는 --cell 이 필요하다')
        spec = G.CELLS[a.cell]                       # ★칸 정의 = gates_m7 단일출처
        a.data = a.data or spec['data']
        a.lambda_field = (spec['lam_field'] if a.lambda_field is None else a.lambda_field)
        a.steps = a.steps or G.FINETUNE_STEPS
        a.lr = a.lr if a.lr is not None else G.FINETUNE_LR
        a.seed = FT_SEED if a.seed is None else a.seed
        if a.init is None:
            _stop('Phase B 는 --init(사전학습 체크포인트)이 필요하다')
        if a.data != spec['data'] or a.lambda_field != spec['lam_field']:
            _stop(f"칸 {a.cell} 의 정의(data={spec['data']} λ={spec['lam_field']})와 "
                  f"인자(data={a.data} λ={a.lambda_field})가 다르다")
        # ★`--init` 이 정말 이 칸의 사전학습인가 — **객체 되읽기로** 확인한다.
        #   (원장의 sha256 이 있으면 그것이 권위, 없으면 이름 규약)
        want = spec['pretrain']
        cen = ((json.load(open(CENSUS)) if os.path.exists(CENSUS) else {})
               if not a.smoke else {})
        want_sha = (cen.get('pretrain_sha256') or {}).get(want)
        if want_sha:
            got = sha256(a.init)
            if got != want_sha:
                _stop(f"칸 {a.cell} 은 사전학습 {want} 를 요구하는데 --init 의 sha256 가 다르다 "
                      f"({got[:16]}… != {want_sha[:16]}…)")
        elif not a.smoke and os.path.basename(a.init) != f'm7_{want}.pt':
            _stop(f"칸 {a.cell} 은 사전학습 {want} 를 요구한다: "
                  f"--init {os.path.basename(a.init)} != m7_{want}.pt")
        tag = a.cell
    a.curve = a.curve or os.path.join(HERE, f'curve_{tag}.json')

    if not a.smoke:
        if os.path.abspath(a.out) in (SHIPPED_CKPT, LEGACY_CKPT):
            _stop('배포/레거시 체크포인트를 덮으려 한다')
        if os.path.exists(a.out):
            _stop(f'이미 있는 체크포인트를 덮으려 한다: {a.out}')
        if os.path.basename(a.out) != f'm7_{tag}.pt':
            # ★칸/사전학습 이름과 파일 이름이 갈리면 `jobs_m7.cell_ckpt_rel` 이 다른 파일을
            #   집는다 — 팔이 조용히 뒤바뀌는 부류다.  이름 규약을 여기서 강제한다.
            _stop(f"--out 이름이 팔 이름과 다르다: {os.path.basename(a.out)} != m7_{tag}.pt")
        if os.path.abspath(a.curve).startswith(os.path.join(ROOT, 'tmp/wp_e2e_2026-07-25')):
            _stop('⛔wp_e2e 의 동결 곡선 파일을 덮으려 한다 (e2e_finetune __main__ 의 함정)')

    print(f"=== M7 train  phase {a.phase}  tag {tag}  data {a.data}  seed {a.seed}  "
          f"steps {a.smoke or a.steps}  λ_field {a.lambda_field}  device {DEVICE} ===", flush=True)
    print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  "
          f"num_threads={torch.get_num_threads()}  "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}", flush=True)
    consts = _constant_parity()
    shipped = _shipped_intact()
    print(f"  [G-3] 배포 체크포인트 무손상 {shipped[:16]}…", flush=True)

    if os.path.exists(CENSUS):
        stale = _census_is_stale(json.load(open(CENSUS)))
        if stale:
            _stop(f"원장 {os.path.relpath(CENSUS, ROOT)} 가 파일계와 어긋난다 {stale}\n"
                  f"           ⟹ 지난(시험/중단된) 실행의 행이다. 착수 전에 "
                  f"stream_census.json · train_manifest_m7.json · trainrow_*.json 을 지워라.")

    stats = torch.load(cd.PRIOR_CKPT, map_location=DEVICE)['stats']
    betas = cd.cosine_betas(cd.T).to(DEVICE)
    abar = torch.cumprod(1 - betas, 0)
    bank = g7.GeometryBank(verbose=False)
    print(f"  기하 은행: {len(bank.donors)} 공여 에피소드", flush=True)

    res = phase_a(a, stats, abar, bank) if a.phase == 'A' else phase_b(a, stats, abar, bank)

    if _shipped_intact() != shipped:
        _stop('학습 중 배포 체크포인트가 바뀌었다')

    if a.smoke:
        print("\n(smoke: 체크포인트를 저장하지 않는다)", flush=True)
        # ⚠비용 smoke 산출물은 배치 안이 아니라 **--out 옆**에 둔다(AMENDMENT §5.4 가
        #   비용 smoke 를 저장소 밖에서 돌린 규약).  배치 디렉터리를 결과로 오염시키지 않는다.
        json.dump(dict(smoke=res, args=vars(a), consts=consts),
                  open(a.out + '.smoke.json', 'w'), indent=1, ensure_ascii=False)
        print(f"  (비용 기록 -> {a.out}.smoke.json)", flush=True)
        return

    torch.save(res.pop('model').state_dict(), a.out)
    out_sha = sha256(a.out)
    print(f"\n저장 {a.out}  sha256 {out_sha[:16]}…  ({res['wall_s']:.0f}s, "
          f"{res['ms_per_step']:.1f} ms/step)", flush=True)

    stream = res['stream']
    if not stream['seeds_sample']:
        _stop('학습 장면 시드 표본이 비었다 — G-9 가 공허 PASS 가 된다 (F-1)')

    row = _row(
        tag=tag, phase=a.phase, cell=a.cell, data=a.data, seed=a.seed,
        lambda_field=a.lambda_field, steps=a.steps, lr=a.lr, bsz=a.bsz,
        target_mode=a.target_mode, out=os.path.relpath(a.out, ROOT), ckpt_sha256=out_sha,
        init=(os.path.relpath(a.init, ROOT) if a.init else None),
        init_sha256=(sha256(a.init) if a.init else None),
        shipped_sha256=shipped, wall_s=res['wall_s'], ms_per_step=res['ms_per_step'],
        val_final=res.get('val_final'), grad_probe=res.get('grad_probe'),
        grid_bounds=res.get('grid_bounds'), probe=res.get('probe'), stream=stream,
        phaseB_data_from_cell=(a.phase == 'B'),
        device=str(DEVICE), torch_version=torch.__version__,
        torch_num_threads=torch.get_num_threads(), git_head=_git_head(),
        md5_gen_m7=md5(os.path.join(HERE, 'gen_m7.py')),
        md5_train_m7=md5(os.path.join(HERE, 'train_m7.py')),
        md5_gates_m7=md5(os.path.join(HERE, 'gates_m7.py')),
        md5_gen_mixed=md5(os.path.join(ROOT, 'tmp/mixed_retrain_2026-07-26/gen_mixed.py')),
        md5_e2e_finetune=md5(os.path.join(ROOT, 'tmp/wp_e2e_2026-07-25/e2e_finetune.py')),
        finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    json.dump(dict(row=row, curve=res['curve'], consts=consts),
              open(os.path.join(HERE, f'trainrow_{tag}.json'), 'w'), indent=1, ensure_ascii=False)

    upd = {'runs': {tag: row}, 'shipped_sha256': shipped}
    if a.phase == 'A':
        upd['stream'] = {a.data: stream}
        upd['pretrain_sha256'] = {tag: out_sha}
    else:
        upd['ckpt_sha256'] = {tag: out_sha}
        if a.lambda_field == 0.0:
            upd['lam0_finetune_steps'] = a.steps
    census = _merge_census(upd)
    seeds = sorted(set(census.get('train_scene_seeds_sample', [])) | set(stream['seeds_sample']))
    census = _merge_census({'train_scene_seeds_sample': seeds[:256],
                            'n_pretrain': sum(1 for r in census['runs'].values()
                                              if r['phase'] == 'A'),
                            'n_finetune': sum(1 for r in census['runs'].values()
                                              if r['phase'] == 'B')})
    _project_manifest(census)
    _gate_selfcheck(census)
    print(f"\n원장 갱신: {os.path.relpath(CENSUS, ROOT)} + "
          f"{os.path.relpath(TRAIN_MANIFEST, ROOT)}  "
          f"(사전학습 {census['n_pretrain']} · 미세조정 {census['n_finetune']})", flush=True)


if __name__ == '__main__':
    main()
