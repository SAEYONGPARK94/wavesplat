"""
fix_track_notes_20min.py

docs/labmeeting_20260513_박세용_20min.pptx 의 slide 10 (Track + Adaptive N)
발표자 노트에 다음 4가지 nuance 추가:

1. w_phi 의 정확한 의미 (w=amplitude, phi=phase)
2. Phase prediction 의미 (kinematic preview before Adam)
3. Adaptive N growth 의 kx, ky 출처 (residual 에 beamforming 재적용)
4. Rollback RMS 의 정확한 정의 (residual RMS, 단위 m)
"""
import os
from pptx import Presentation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, 'docs', 'labmeeting_20260513_박세용_20min.pptx')


NEW_NOTES_TRACK = (
    "*Track 단계* 가 매 frame 13 ms 안에 끝나는 비결을 말씀드립니다. *Adaptive N* 도 이 "
    "단계에서 작동해요.\n\n"
    "*핵심 — 파수 k 는 Init 에서 결정한 값으로 고정* 합니다. WaveSplat 의 가장 중요한 "
    "설계 결정이에요. 한 순간 50 개 점만 보면 k 추정이 *극도로 불안정* 하거든요 — "
    "Fisher 정보 (k 변화에 데이터가 반응하는 정도) 가 (A, φ) 보다 약 1000 배 작습니다. "
    "k 자유 시 50 점에 over-fit 해서 비물리적으로 발산해요. 그래서 k 는 고정, *진폭 A 와 "
    "위상 φ 만 매 frame update*, ω 는 dispersion 으로 자동.\n\n"
    "*Adam optimizer* — 학습률 0.005, frame 당 10 step 정도 돌리면 약 13 ms / frame. "
    "Tracking window 3 frames 의 buffer 로 안정적인 gradient 계산.\n\n"
    "*Adaptive N* — 성분 수를 동적으로 조정합니다. (1) Pruning: cumulative energy 90% "
    "이내만 유지 (작은 성분 제거). (2) Growth: 잔차 RMS > 5 cm 이면 새 (k_x, k_y) 추가, "
    "최대 30 개. (3) Safety rollback: 변경 후 잔차 RMS 가 1.3× 초과 증가 시 즉시 복구. "
    "cascade failure 방지가 핵심입니다.\n\n"
    "─────────────────────────────────\n"
    "**[Q&A 대비 — 자주 받는 질문 4가지]**\n"
    "─────────────────────────────────\n\n"
    "**Q1. w_phi 의 w 와 phi 가 각각 뭔가요?**\n"
    "코드 변수명에서 따왔습니다. `self.w` 는 amplitude (진폭) — sine wave 의 weight 라는 "
    "관점에서 w 라고 명명. `self.phi` 는 phase (위상). 즉 *w_phi mode = amplitude 와 "
    "phase 만 update* 한다는 의미예요. 슬라이드에서는 헷갈리지 않게 (A, φ) 로 표기합니다.\n\n"
    "**Q2. Phase prediction 이 정확히 뭘 하는 건가요?**\n"
    "새 frame 도착 시 Adam optimization *시작 전* 에 모든 성분의 phase 를 *kinematic 으로 "
    "미리 진행* 시키는 step 입니다. 코드: `φ_n ← φ_n + ω_n · Δt`. 이전 frame 의 (k, ω, "
    "φ) 추정이 맞다면 그 사이 Δt 동안 phase 는 *물리적으로* ω·Δt 만큼 진행됐을 거니까 — "
    "Adam 의 *시작점을 정답에 더 가깝게* 옮기는 거예요. 비유: 시계 분침이 1 분 지나면 "
    "6° 진행한다는 prior knowledge 를 활용해 미리 돌려두는 것. 결과적으로 10 step 안에 "
    "수렴 가능해집니다.\n\n"
    "**Q3. Adaptive N 의 growth 에서 새 (k_x, k_y) 는 어디서 가져오나요?**\n"
    "*잔차 (residual) 에 beamforming 을 다시 적용* 해서 가져옵니다. 처리 흐름:\n"
    "  ① 현재 model 예측 − 실측 = residual (η_obs − η_pred)\n"
    "  ② residual 의 RMS > 5 cm 이면 growth 트리거\n"
    "  ③ residual 에 `_beamform_coarse` → coarse grid 위 peak 찾기\n"
    "  ④ `_beamform_refine` 으로 sub-grid 정밀화\n"
    "  ⑤ Linear projection 으로 해당 (kx, ky) 의 amplitude 계산\n"
    "  ⑥ 주변 zero-out → 다음 peak 반복 (n_add 개)\n"
    "즉 *Init 의 beamforming 을 잔차 위에서 다시 돌리는* 것입니다. 현재 model 이 *놓친 "
    "wavenumber 패턴* 이 residual 에 남아있을 거라는 발상이에요.\n\n"
    "**Q4. Rollback 의 RMS 는 어떤 RMS 인가요?**\n"
    "**잔차 (residual) 의 RMS** 입니다. 정의:\n"
    "  RMS = √( mean ( (η_pred − η_obs)² ) )\n"
    "단위는 m (파고 단위와 동일). 즉 *현재 model 이 측정 데이터를 얼마나 못 맞추는가* 의 "
    "평균 오차 크기예요. Rollback 임계값 1.3× 는 *변경 후 잔차가 30% 이상 증가하면 "
    "즉시 이전 상태로 복원* 한다는 뜻입니다. 잘못된 prune (중요 성분 제거) 이나 잘못된 "
    "growth (잡음 끼워맞춤) 의 cascade failure 를 방지하는 안전장치예요. 참고로 Adam 의 "
    "Loss 값과는 *다른* 지표 — Loss 는 L1·smoothness 정규화 항까지 포함하지만, 이 RMS 는 "
    "*순수 data fit* 만 봅니다."
)


def main():
    prs = Presentation(PATH)
    slide = prs.slides[9]  # slide 10 — Track + Adaptive N
    notes_slide = slide.notes_slide
    notes_slide.notes_text_frame.text = NEW_NOTES_TRACK
    prs.save(PATH)
    print(f'Updated: {PATH}')
    print(f'New notes length: {len(NEW_NOTES_TRACK)} chars')


if __name__ == '__main__':
    main()
