# 내용 공부용 — 돌려볼 프로그램 & 대표 데이터

> `STUDY_GUIDE.md`와 함께 읽으며 "직접 돌려보고 figure를 눈으로 확인"하는 실습 가이드.
> 순서대로 따라 하면 논문 전체 흐름을 1시간 안에 체험할 수 있습니다.

---

## 대표 데이터 5개 (이것만 있으면 충분)

학습/발표용으로 **난이도를 점진적으로 올리는** 5개 에피소드를 선정했습니다.
나머지 95개는 통계 검증용이므로 공부 단계에서는 불필요합니다.

| # | 파일 | 이유 |
|---|---|---|
| 1 | `episode_BF2_1comp_seed0_uniform.h5` | **가장 단순**: 단일 성분, 약한 바다. 알고리즘 동작 직관 이해 |
| 2 | `episode_BF3_5comp_seed0_uniform.h5` | **중간 난이도**: 다중 성분. 컴포넌트 분리 능력 체감 |
| 3 | `episode_BF4_5comp_seed0_uniform.h5` | **논문 대표 케이스**: 실험 표/그림에서 가장 자주 인용 |
| 4 | `episode_BF4_15comp_seed0_uniform.h5` | **adaptive N 동작 관찰**: N이 자동 증가하는 과정 |
| 5 | `episode_BF5_30comp_seed0_slope.h5` | **최악 조건**: 거친 바다 + slope LiDAR 모델. 견고성 확인 |

필요시 디스크 절약을 위해 이 5개만 남기고 나머지 삭제 가능:
```bash
# 5개만 유지하는 예시 (주의: 되돌릴 수 없음)
cd /home/avikus/personal/wavesplat/wavesplat/data
mkdir ../data_keep
cp episode_BF2_1comp_seed0_uniform.h5 ../data_keep/
cp episode_BF3_5comp_seed0_uniform.h5 ../data_keep/
cp episode_BF4_5comp_seed0_uniform.h5 ../data_keep/
cp episode_BF4_15comp_seed0_uniform.h5 ../data_keep/
cp episode_BF5_30comp_seed0_slope.h5 ../data_keep/
```

---

## 실행 순서 — 3단계 커리큘럼

### Stage 1. "알고리즘이 어떻게 동작하는가" 체감 (15분)

**1.1. 가장 단순한 케이스부터 — 단일 성분**
```bash
cd /home/avikus/personal/wavesplat
python3 -m wavesplat.run \
    wavesplat/data/episode_BF2_1comp_seed0_uniform.h5 \
    --adaptive_n
```
- 출력: 콘솔에 프레임별 nRMSE, 컴포넌트 수 N, 런타임
- **확인 포인트**: N이 1로 수렴하는지, nRMSE가 0.1 이하로 떨어지는지

**1.2. 중간 난이도 — 5성분**
```bash
python3 -m wavesplat.run \
    wavesplat/data/episode_BF3_5comp_seed0_uniform.h5 \
    --adaptive_n
```
- **확인 포인트**: N이 5 근처로 자동 수렴, 각 컴포넌트의 (kx, ky) 분포

**1.3. 논문 대표 케이스**
```bash
python3 -m wavesplat.run \
    wavesplat/data/episode_BF4_5comp_seed0_uniform.h5 \
    --adaptive_n
```
- **확인 포인트**: 논문 Table 수치와 맞는지 비교

---

### Stage 2. "결과를 눈으로 보기" — Figure 생성 (10분)

**2.1. 정량적 비교 그래프 (논문 Figure 5~8)**
```bash
python3 wavesplat/scripts/generate_paper_figures.py
```
- 생성 위치: `wavesplat/paper/figures/`
- 주요 그림:
  - `nrmse_comparison.pdf`: WaveSplat vs ETKF vs GPR 막대 그래프
  - `runtime_comparison.pdf`: 런타임 비교
  - `adaptive_n_evolution.pdf`: N이 시간에 따라 변하는 곡선

**2.2. 정성적 파면 복원 그림 (heatmap)**
```bash
python3 wavesplat/scripts/generate_qualitative_figure.py
```
- 출력: ground truth vs WaveSplat 복원 파면 heatmap 비교
- **발표 슬라이드 #13용 그림**

**2.3. 추적 애니메이션 (시간 진화)**
```bash
python3 wavesplat/scripts/generate_tracking_animation.py
```
- 출력: 프레임별 컴포넌트 추적 과정 시각화 (gif 또는 mp4)
- **w_phi 모드의 효과를 직관적으로 보여줌**

---

### Stage 3. "하이드로포일 FF 제어" 시뮬레이션 (15분)

**3.1. Phase 1 — 기본 동역학 검증**
```bash
cd /home/avikus/personal/wavesplat
python3 -m wavesplat.hydrofoil.run_phase1
```
- 출력: heave/pitch 응답, static lift 검증

**3.2. Phase 2 — FF+FB 제어 비교**
```bash
python3 -m wavesplat.hydrofoil.run_phase2
```
- 출력: FB only vs FF+FB 가속도 RMS 비교
- **발표 슬라이드 #17용 그림**

**3.3. Phase 3 — End-to-End (WaveSplat + Hydrofoil)**
```bash
python3 -m wavesplat.hydrofoil.run_phase3_e2e
```
- **두 논문 통합 파이프라인 전체 동작**
- WaveSplat 추정 → Feedforward → 승차감 개선률 출력

**3.4. 필요시 — Preview time sensitivity**
```bash
python3 -m wavesplat.hydrofoil.run_sensitivity
```
- 출력: τ(preview time) vs 개선률 곡선
- **발표 슬라이드 #18용 그림**

---

## 추가 실습 (선택)

### 경쟁 기법과 직접 비교
```bash
# ETKF 대비 (논문 Table II)
python3 wavesplat/scripts/run_etkf_comparison.py

# GPR 3-way 비교 (논문 Table III)
python3 wavesplat/scripts/run_gpr_comparison.py

# 전체 12 에피소드 종합 (논문 main result)
python3 wavesplat/scripts/run_p1_comparison.py
```
- 결과는 `results/*.json`에 저장
- 전체 재현에 10~30분 소요

### 테스트로 핵심 모듈 이해
```bash
# 빠른 정합성 체크 (약 1분)
python3 -m pytest wavesim/tests/ wavesplat/tests/ -v
```
- 57개 테스트 모두 통과 확인
- `test_*.py` 파일들을 읽으면 각 모듈의 입출력 계약을 이해하기 좋음

---

## 데이터 파일 내부 구조 탐색 (HDF5 직접 열어보기)

HDF5 에피소드 파일을 한 번 열어보면 "LiDAR 포인트가 어떻게 생겼는지"
직관이 생깁니다. 간단한 탐색 스크립트:

```python
import h5py
import numpy as np

path = "wavesplat/data/episode_BF4_5comp_seed0_uniform.h5"
with h5py.File(path, "r") as f:
    print("Keys:", list(f.keys()))
    for k in f.keys():
        print(f"  {k}: shape={f[k].shape}, dtype={f[k].dtype}")
    # 첫 프레임 LiDAR 점 수
    if "lidar_points" in f:
        print("Frame 0 points:", f["lidar_points"][0].shape)
```

---

## 발표 준비 워크플로우 (권장 순서)

```
[1일차]
 ├─ STUDY_GUIDE.md Part A,B 정독 (1시간)
 ├─ Stage 1 실행하며 알고리즘 동작 확인 (15분)
 └─ Stage 2 figure 생성 후 논문 그림과 대조 (30분)

[2일차]
 ├─ STUDY_GUIDE.md Part C 정독 (45분)
 ├─ Stage 3 실행 (20분)
 └─ PPT 초안 작성 (슬라이드 1~10) (2시간)

[3일차]
 ├─ PPT 슬라이드 11~20 작성 (2시간)
 ├─ STUDY_GUIDE.md Part D 예상질문 리허설 (1시간)
 └─ 발표 연습 (타이머 돌리며 2회) (1시간)
```

---

## 문제 해결 팁

- **HDF5 에러**: `pip install h5py` 필요
- **GPU 사용 원함**: `--device cuda` 옵션 추가 (기본은 CPU)
- **런타임 측정 부정확**: 노트북이면 전원 연결 후 재측정 (전력 모드 영향)
- **figure 생성 실패**: `matplotlib` 백엔드 문제 → `export MPLBACKEND=Agg`

---

*모든 명령은 프로젝트 루트(`/home/avikus/personal/wavesplat`)에서 실행합니다.
Stage 1→2→3 순서를 지키면 "데이터 → 알고리즘 → 결과 시각화 → 제어 적용"
흐름이 머릿속에 남아 발표가 훨씬 자연스러워집니다.*
