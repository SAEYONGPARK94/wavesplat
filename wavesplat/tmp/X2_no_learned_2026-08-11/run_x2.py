"""X2 (탐색적) — **가우시안 경로에서 "학습된 초기화"만 빼면 어떻게 되나.**

## 가설 (X1 이 낳은 것)
X1: 시뮬레이터에서 가우시안(0.016/0.158)이 운용 한방풀이(0.158/0.214)를 압도한다.
그러나 가우시안 기본 설정은 **합성 바다로 학습된 초기화**를 쓴다 ⟹ 홈그라운드일 수 있다.
실측에서는 반대로 학습 팔이 30창 중 21창 열세였다.

⟹ **학습만 제거하고 나머지는 그대로** 두면 갈린다:
  - 여전히 한방풀이를 크게 이긴다  → **방법 자체가 우수** (실측 라우팅 재검토 대상)
  - 이득이 크게 줄어든다          → **이득의 정체는 학습**이고 그것은 실측에 안 간다

## 설계
셀 `V2K` = `init_method='v2_konly'` (학습 제안 제거) + `analytic_ensemble_k=3` (**앙상블은 유지**)
+ `lambda_l1=1e-4` (오늘의 출하 기본값).
⚠`X-BASE` 를 쓰지 않는 이유: 그것은 학습 **과 앙상블을 동시에** 제거해 두 요인이 교락된다.
비교 대상 = S9 의 `AUTO-L1`(학습 포함) · X1 의 bank. **같은 96 에피소드·같은 시드.**

⚠**탐색적**: 사전등록 없음. 판정 문구를 만들지 않는다.
"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'S9_lr_x_l1_2026-08-11'))
import run_s9 as M                                    # noqa: E402  에피소드·러너 재사용

M.R.CELLS['V2K'] = dict(cfg=dict(init_method='v2_konly', analytic_ensemble_k=3))
M.S5.S4.EXTRA['V2K'] = {'lambda_l1': M.L1_TREAT}      # 오늘의 출하 기본값 1e-4
M.CELLS = ['V2K']
M.OUT = os.path.join(HERE, 'results.jsonl')

if __name__ == '__main__':
    sys.argv = [sys.argv[0], 'V2K']
    M.main()
