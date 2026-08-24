# 브리프 — 2026-08-24 페르소나 토론: **novelty 와 박사학위로서의 가치**

사용자 지시: *"논문 작성되면 novelty 와 박사학위로써 가치 페르소나 토론해줘"*.

## 규칙 (이 저장소의 측정 규율 — 어기면 그 발언은 무효다)
1. **런 0회.** 새 실험을 돌리지 마라. **읽고 논증하라.**
2. **열람 범위 = 주장 범위.** 실제로 연 파일만 근거로 삼고, 문장마다 `파일:줄` 을 붙여라.
3. ⛔**인용 금지 목록을 먼저 열어라**: `/home/vexit/.claude/projects/-home-vexit-wavesplat/memory/reference_citation_bans.md`.
   금지된 수치를 논거로 쓰면 그 문단 전체가 버려진다.
4. **측정보다 넓은 문장을 쓰지 마라.** 이 저장소가 반복해서 대가를 치른 결함이 정확히 그것이다.
5. **자기 진영에 불리한 증거를 먼저 적어라.** 유리한 것만 모으면 심사에서 그대로 뒤집힌다.
6. 한국어로 쓰고, 기술 용어는 영어를 그대로 둔다.

## 반드시 읽을 것 (최소)
- `/home/vexit/wavesplat/CLAUDE.md` (프로젝트 규율·배포 기본값)
- `/home/vexit/.claude/projects/-home-vexit-wavesplat/memory/MEMORY.md` (RESUME 두 개 + 08-22~24 항목)
- `/home/vexit/.claude/projects/-home-vexit-wavesplat/memory/reference_citation_bans.md`
- `wavesplat/tmp/STATUS_2026-08-23/STATUS.md` (현황판 — 맨 아래 08-24 절 포함)
- `wavesplat/thesis/thesis_main.tex` **제1장**(462–825행: 배경·한계·목적/기여·"주장하지 않는 것")
  과 **제2장 §최근접 선행 연구**(1184–1232행)
- 선행 토론: `wavesplat/tmp/DEBATE_NOVELTY_PHD_2026-08-21/` (B1_committee_chair.md · B2_external_examiner.md ·
  A3_control_reviewer.md · R3_*.md) 와 `wavesplat/tmp/DEBATE_PHD_2026-08-23/` (DEBATE.md · DEBATE_R2.md · DEBATE_R3.md)
  ⟹ ★**같은 지적을 되풀이하지 마라.** 그 뒤에 바뀐 것(08-22 배포 3건 · 08-23 실측 아크 · `B-11`/`B-12` · 08-24 그림/분석기)을
  반영해 **새 판정**을 내라.

## 살아 있는 측정 (2026-08-24 기준, 요약 — 원본을 직접 확인할 것)
- 합성 혼잡 우위: `B-12` `SHIP<SPEC25` **43/48 · −40.19% · p=6.8e-09**, 음성대조 `SPEC25<ZERO` 47/48
  (`tmp/B12_HOS_EXTEND_2026-08-23/RESULT.md`). HOS M=3 비선형 참값 · 신규 시드 32개 단독 29/32.
- 실측(12 레코드, 고정 관측소 스테레오 4 사이트): **제안 표현은 음성**(`B-7`/`B-8`),
  배포 실측 경로는 lead 1–3 s 양성 · 4 s 부터 소멸 · 2 s 재적합이 4–14 s 유지
  (`tmp/B3_REFIT_2026-08-23/FINAL_REAL_TIMEAXIS.md`, 재현 스크립트 `final6.py`).
- 외부 기준선: 실측에서 **두 종 다 무기술**(`B-5`, Wang 2025 재현) ⟹ 우열 주장 불가.
- 배포: `varpro_steps=400` · `learned_init_iters=0` · `auto_switch_thresh=0`(08-22),
  `bank_refit_every_s=None`(08-23 배포 후 당일 철회), `bank_refit_async` 구현·기본 False.
- 시험 **245 passed**(08-24 실측).

## 산출물
너에게 지정된 파일 하나만 쓴다(아래 프롬프트에 경로가 있다). 형식:
1. **판정 한 줄**(라벨 + 한 문장)
2. **근거** — 각 항목에 `파일:줄`
3. **내 진영에 불리한 것**
4. **구체적 처방** — *"무엇을 하면 판정이 바뀌는가"* (측정 가능한 형태로)
