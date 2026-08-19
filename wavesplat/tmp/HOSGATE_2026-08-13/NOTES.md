# HOSGATE 2026-08-13 — 진행 로그 (append-only)

배치: `#7 HOS 재현 게이트` — S13 의 혼잡 우위가 HOS M=3 비선형 GT 에서도 서는가.

## 2026-08-13T14:48 시작
- 디렉터리 생성. 이전 시도가 세션 한도로 죽어 파일 0개 남김 ⟹ 이번엔 즉시·점진 저장.
- 참고 러너 `tmp/S13b_seed_robust_2026-08-13/run_s13b.py` 읽음 (읽기전용).
  - `sys.path.insert(0, ROOT)` + `os.chdir(ROOT)` where ROOT=`.../wavesplat/wavesplat`
  - config: `from wavesplat.run import DEFAULT_CONFIG`
  - `effective_config` 는 `WR.make_updater` 를 monkeypatch 로 감싸 **객체에서 되읽기**(`readback()`)
  - READBACK_KEYS = seed, lambda_l1, lr, init_method, use_confidence, use_dispersion,
    adaptive_n, max_n, tracking_mode, prune_threshold, grow_residual_threshold,
    lambda_smooth, tracking_steps, tracking_lr, depth, device, member_seeds,
    ensemble_k, k_requested
  - 결과: jsonl append, `(episode, arm)` 키로 resume
  - 에피소드 파일명 규약: `episode_(BF\d)_(\d+)comp_seed(\d+)_(.+)\.h5`

## 2026-08-13T14:5x — PROBE 1 (probe_hos.log)
조사 사실 (재조사 불필요):
- `make_grid(x_range,y_range,nx,ny)` 는 **(ny,nx)** 배열을 준다. `X[i,j]=x[j]`, `Y[i,j]=y[i]`.
- `HOSSimulator.initialize_from_components` 는 t=0 에서 선형 GT 와 **1.07e-14 m** 일치 (P-a PASS).
  단 `pipeline.run_episode` 는 이걸 안 쓰고 **`initialize_jonswap(Hs,Tp,seed)`** 를 부른다
  (`pipeline.py:106`) ⟹ 그대로 두면 두 팔의 파랑이 **다른 바다**가 되어 짝 설계가 깨진다.
  ⟹ 이 배치는 `wavesim.pipeline.HOSSimulator` 를 **서브클래스로 몽키패치**해서
     `initialize_jonswap` 이 `initialize_from_components(스냅된 성분)` 로 위임하게 한다.
- `_to_physical` 이 항상 `np.real(...)` 을 취하므로 비대칭 스펙트럼 초기화도 물리가 옳다
  (Re[] 가 선형연산이라 미분·곱 전부 통과). 확인함.
- ★**원본 `evaluate_at` 은 배치를 불가능하게 한다**: 1 스텝 후 16384 모드 전부 켜짐 ⟹
  gt 격자(60x80) **4.37 s**, 메시(200x167) **30.9 s** = 프레임당 35 s x 301 = **2.9 h/에피소드**.
  ⟹ **분리가능(separable) 재작성**으로 대체: 0.9 ms / 2.5 ms (동일한 합의 인수분해).
  ⚠PROBE1 의 분리가능 구현은 축을 뒤집어 **불일치(0.98 m)** — make_grid 의 (ny,nx) 때문. 수정 필요.
- `step` 비용 M=3, 128x128: **55.7 ms/step** ⟹ 30 s 에피소드(dt=0.02, 1500 step) = **84 s**.
- Hs 안정: 1.2000 -> 1.2019 m (30 s), max|eta| 1.03->1.23 m. 발산 없음.
- ⚠**3차 구속고조파 해상 부족**: Lx=600/Nx=128 -> kx_nyq=0.6702. `3k>k_nyq` 인 성분이
  분산의 **33.0%** (BF4 15comp seed1581). dealiasing_factor=2 라 **앨리어싱이 아니라 절단**이지만
  고파수 비선형성이 과소표현된다 ⟹ 격자 재검토 필요(PROBE 2).

## 2026-08-13T15:02 — PREREG 동결
- `PREREG.md` md5 = `d841ae53c38eed4541d54bba59a50d80`, 동결 15:02:25 (+09:00), HEAD `5f0846c`.
- PRIMARY: HOS 팔 혼잡 16 쌍, 단측 α=0.025, **문턱 13/16** (P(X>=13)=0.010635; 12/16=0.038406 미달).
- 검정력: π=0.875(S13 실측) -> 0.870, π=0.80 -> 0.598.
- 소모 시드 **1581–1588** (혼잡 1581–1584, 희소 1585–1588). 충돌 0 (전 data* 스캔 완료).
- HOS 격자 결정: **M=3, Nx=192, Ny=128, Lx=600, Ly=400, dt=0.02** (출하 128 은 kx_nyq 0.670 이라
  3k 절단이 33% -> 192 로 올려 kx_nyq=ky_nyq=1.005 대칭).
- ⚠**BF4×2 는 병적**(2k 절단 99.8%): 지배 성분 λ≈8.9 m, 2차 고조파 λ≈4.5 m 는 메시(dx 1.21)도
  못 그린다. 희소 SECONDARY 에서 **BF4×2 제외 12쌍**을 사전지정해 병기.
- PROBE2 Q-a: 분리가능 evaluate_at == 원본, max|Δ| 3e-15 (축 수정 후 PASS).

## 다음: gen_hos.py -> run_hosgate.py -> analyze_hosgate.py -> 스모크 2쌍

## 2026-08-13T15:10 — 스크립트 3종 작성 완료
- `gen_hos.py` — 짝 코퍼스. 출력은 배치 디렉터리 안 `data_lin/`, `data_hos/` (에피소드당 11.6 MB,
  64개 ≈ 745 MB). 몽키패치 3종은 전부 이 파일 안 (`wavesim/**` 무수정).
- `run_hosgate.py` — 팔 SHIP / LETKF_PLUS / ZERO. 게이트 G-1..G-7. jsonl append+resume.
  각 행에 `code_md5`(임포트된 저장소 .py 전부 + configs/*.yaml 의 단일 digest) + `git_head` +
  `git_dirty_n`. `effective_config` 는 `make_updater` 를 감싸 **객체에서 되읽기**.
- `analyze_hosgate.py` — PREREG §4~§7 그대로. 판정표 5분기.
- 실측: LIN 에피소드 **169 s** (S9 의 ~160 s 와 일치).

## 2026-08-13T15:12 — ⚠**PREREG 정정 #1** (동결 후 발견, PREREG.md 는 수정 안 함)
`gen_manifest.json` 실측: **BF4×15 의 HOS 팔 `mesh_verdict` = `DEGRADED`**
(`var_under_resolved = 0.211`). PREREG §2 는 "혼잡(PRIMARY) 4셀은 전부 건전하다" 라고 썼는데
그건 **내가 계산 안 하고 추정한 값**이었다 (λ_min/3 = 2.85 m 를 dx=1.206 m 메시가 6점으로
못 그린다 -> `pts=2.37`). 같은 메시의 LIN 팔은 `ADEQUATE`(hos_order=None 이라 λ 를 안 나눔).
- 실질: HOS 바다의 **3차 구속고조파는 레이캐스트 메시에도 안 실린다**. 스펙트럼 절단(13.7%)과
  대체로 겹치는 제약이므로 새로운 결함이 아니라 **같은 결함의 두 번째 얼굴**이다.
- 메시를 506×421 로 올려야 고칠 수 있는데 그러면 레이캐스트 비용이 폭증하고 S9/S13 코퍼스와
  비교가 깨진다 ⟹ **메시 200×167 유지 + 공개**. 각 에피소드의 verdict 는 manifest 에 있다.
- ⟹ 결론 문장은 "**HOS M=3 의 2차 비선형성**에서도 서는가" 로 좁혀 읽어야 한다.
  3차는 이 코퍼스에서 부분적으로만 존재한다.

측정된 비선형 신호: `gt_eta_absmax` LIN 1.107 m vs HOS **1.218 m** (+10.0%) — 마루-골 비대칭.

## 2026-08-13T15:35 — 스모크 완주 (SMOKE.md 참조)
- 생성 4/4 ok · 게이트 G-1..G-7 **전부 PASS** · 추정 12/12 ok · 분석기 완주 · 2회 실행 비트동일.
- 실측: LIN **167 s**, HOS **290 s** (쌍 457 s) · SHIP 17–18 s(learned) / 87–89 s(v2_konly) ·
  LETKF_PLUS 41 s · ZERO 0.3 s.
- **전량 예상 ≈ 5.1–5.6 h** (생성 30쌍 3.8 h + 추정 64ep 1.2–1.7 h). ⚠병렬 금지(CPU 공유 병목).
- ⚠**정정 #2**: `code_md5` 가 팔 순서에 의존 -> `preload()` 추가로 digest 1개 확정.
  1차 출력은 `smoke_results_run1_before_preload_fix.jsonl` 로 증거 보존(값은 비트동일).
- ⚠**사전노출 2쌍**(스모크가 전량 코퍼스의 부분집합) -> 분석기에 **S-6 제외 민감도** 추가.
  **PRIMARY 논리 미변경**. 이 추가는 스모크 결과를 본 뒤라 SMOKE.md §4 에 명시.
- 실행: `cd /home/vexit/wavesplat/wavesplat && nohup bash tmp/HOSGATE_2026-08-13/chain.sh > tmp/HOSGATE_2026-08-13/chain_stdout.log 2>&1 &`

## 2026-08-13T16:09 — ⚠전량 체인이 **외부에서** 시작됐다 (내가 띄운 것 아님)
- `bash tmp/HOSGATE_2026-08-13/chain.sh` (PID 51920 -> python 51924) 가 **15:37:48** 에 시작.
  부모가 `/init` 로 재부모화된 detach 프로세스이고 출력이 `chain_stdout.log` 로 간다
  = SMOKE.md 에 적어둔 그 한 줄 명령. 임무 지시("전량은 내가 띄운다")와 일치하므로
  **건드리지 않는다**. 나는 전량을 실행하지 않았다.
- 건전성 확인(16:05 기준): 스모크 2쌍 **정상 skip**, 이후 LIN 165–168 s / HOS 283–288 s
  (쌍 ~7.5 min) 로 스모크 예측과 일치. 에러 0.
- ⚠러너가 인쇄하는 ETA 는 **skip 된 쌍을 평균에 넣어 과소평가**한다. 실제 잔여는
  27쌍 × 7.6 min ≈ **3.4 h** (생성 완료 ~19:30), 그 뒤 게이트+추정 1.2–1.7 h ⟹ ~21:00–21:15 완료 예상.
- ★`mesh_verdict` 는 **시드마다 다르다**(1582 ADEQUATE · 1583 DEGRADED · 1584 ADEQUATE).
  정정 #1 은 "혼잡 전체가 DEGRADED" 가 아니라 **에피소드별 공개 사항**이다 —
  값은 `gen_manifest.json` 에 전부 기록된다.

## 실행 출처 확정 (2026-08-13, 주 세션)
전량 체인은 **주 세션(내가) 15:39 에 직접 띄웠다** — `SMOKE.md` 의 한 줄 명령 그대로:
`nohup bash tmp/HOSGATE_2026-08-13/chain.sh > tmp/HOSGATE_2026-08-13/chain_stdout.log 2>&1 &`
준비 에이전트는 지시대로 전량을 돌리지 않았다(그래서 자기 이력에 없다). **미상 프로세스가 아니다.**
`chain_stdout.log` 첫 줄 `[15:39:39] [1/32]` 가 그 기동이다.

## 정정 — `mesh_verdict` 는 에피소드별이다
`SMOKE.md` 정정 #1 을 "혼잡 전체가 DEGRADED" 로 읽으면 안 된다. 실측: 1582 `ADEQUATE` ·
1583 `DEGRADED` · 1584 `ADEQUATE`. 전량 값은 `gen_manifest.json` 에 기록된다.
⟹ 결론 작성 시 "3차 비선형성이 부분적으로만 표현된다" 는 문장은 **DEGRADED 로 찍힌 에피소드에만**
적용하고, 층 전체로 일반화하지 마라.
