# Session Notes — 2026-06-25

## 환경/인프라 (완료)
- numpy 2.0 버그 수정: `np.math.factorial` → `math.factorial` (hos_simulator.py)
- PyTorch CUDA 12.6 (RTX 4080 Super 인식), 30개 에피소드 생성, 테스트 137개 통과
- **Discord 브릿지 설정**: Bun 1.3.14 설치, 토큰 → `~/.claude/channels/discord/.env`(600),
  MCP 의존성 설치, 페어링 완료(user `1519659211284414635`, dmPolicy allowlist).
  메모리 `discord_bridge.md`에 MCP 시작 크래시 해결법 기록됨.

## T-NSE 연구 — 핵심 결론: feedforward 방향추정 붕괴

### 진단 타임라인
| 단계 | 결과 | 해석 |
|------|------|------|
| reconstruction-only | val 0.275 > zero-baseline 0.25 | 파면 0 붕괴, 실패 |
| + 방향 지도학습 (Chamfer), 단일 ep | 0.21로 돌파 | 메모리제이션은 됨 |
| + 방향 지도학습, 24ep 일반화 | val 0.28 정체 | 일반화 실패 |
| **계측: kx_std(출력 다양성)** | **≈0.002 ≈ 0** | **입력 불변 상수 예측기로 붕괴** |
| avg-pool → cross-attention 디코더 | 동일 붕괴(kx_std≈0.003) | 구조 아님, **손실 문제** |

### 근본 원인
에너지가중 Chamfer(true→nearest pred)는 "k-범위를 적당히 덮는 고정 15방향 dictionary"
만으로도 모든 에피소드에서 낮게 나옴 → 모델이 입력별 방향을 식별할 유인이 없음.
reconstruction gradient(LS+분산전파 경유)는 약해서 입력 의존 해로 끌어내지 못함.

### 연구적 가치 (논문)
sparse aperture-limited LiDAR에서 **feedforward amortized 방향추정은 붕괴**한다는
negative result. WaveSplat가 에피소드별 beamforming+iterative 최적화를 쓰는 이유를
정당화함. (aperture problem)

## 코드 상태 (중요)
- `wavesplat/model/tnse.py` — **cross-attention(DETR식 query) 디코더로 변경됨**.
  → 구 체크포인트 `results/tnse_stage{1,2}_best.pt`(avg-pool)는 **로드 불가**.
- `scripts/train_tnse.py` — `direction_loss`(에너지가중 Chamfer, lambda_dir=5.0).
- `scripts/eval_tnse.py` — 상대시간 정규화 일치.
- 유효한 버그수정: aug 제거, 상대시간, full-window LS (모두 유지).

## 다음 할 일 (우선순위)
1. **손실 재설계 (핵심)**: 커버리지 Chamfer → **Hungarian 이분매칭**으로 query를 특정
   true 성분에 1:1 배정 + DETR식 "no-object"(15 query vs 5~30 true). 입력 의존성 강제.
   - 구현: `scipy.optimize.linear_sum_assignment`로 배치별 매칭, 매칭된 쌍에 L2,
     미매칭 query는 no-object(낮은 confidence) 페널티. confidence head 추가 필요.
2. 손실 교체 후 6ep 진단으로 kx_std↑ & recon<0.25 확인 → 통과 시 Stage 2 전체학습.
3. 그래도 붕괴하면 feedforward 포기, negative result로 문서화 → WaveSplat 논거 강화.
4. (병렬) GP 30ep/UQ/Forecast 재실행 — 이전 세션서 소실:
   `scripts/spacetime_gp_30ep.py`, `_uq.py`, `_forecast.py` (에피소드당 느림 주의: ~40-60분/ep,
   이전에 예상보다 10배 느렸음 → 코드 최적화 또는 N_TARGETS 축소 검토).

## 참조 수치 (30ep 프로토콜)
Oracle 0.126 · WaveSplat 0.223 · GP K_disp 0.223 · zero-pred 0.25 · ETKF 0.319
