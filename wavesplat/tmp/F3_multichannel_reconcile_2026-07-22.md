# F-3 — 다채널 학위조건② 상태 정합화 (doc+result 대조, 2026-07-22)

> 담당: fable (coordinator 지시, 분석 트랙). fable 브레인스톰(같은 세션)에서 "조건②가 이미 닫혀있을 가능성"
> 제기 → 문서/원자료 대조로 확인. 원자료 존재 검증(`ls`+git log)·수치 재확인(json 직접 로드), 코드 실행 없음.

## 결론 (한 줄)
**조건② = 실질적으로 CLOSED.** THESIS_FRAMING_v2 §9/§10("재료 존재", 다음 작업 후보 1순위로 "다채널 이식")는
**stale** — 같은 날(2026-07-18) 그 뒤에 실행된 E-55/E-56이 이미 완료했고, PAPER_SKELETON(07-19, THESIS_FRAMING
보다 나중 커밋)은 이미 "판정=2채널 상이-사망요인"으로 갱신 반영돼 있다. **문서 재소송이 아니라 "이미 있는
결론을 상태-태그로 명시"하는 정리 작업만 남음.**

## 1. 증거 사슬 (git log + 원자료 직접 로드로 확인)
```
2802402 E-54 다채널 이식 토론(Opus vs fable) — 학위관건=부분(PARTIAL) + E-55 prereg   (07-18 최초)
9f30269 E-55 current 채널 dose 사다리 → VALUE-사망 확인(프로토콜 성공, 2채널 판정완료)  (07-18 21:05 커밋)
7e77c5a E-56 비관 갭 닫기 — current VALUE-사망 base-robust(net-base도 <0.03)          (07-18 21:24 커밋)
```
- `results/direction3/e55_current_channel.json` verdict(원자료 직접 확인): *"VALUE-DEATH CHANNEL (protocol
  SUCCESS): kp_hint 후 radar급 외부 Uc의 잔여 가치 < 0.03 ⟹ 배포지침 'radar 말고 kp 고쳐라'. current dies at
  VALUE (regime dies at achievability) = 프레임워크가 두 채널서 다른 요인 판별."*
- `results/direction3/e56_current_netbase.json` verdict: *"GAP CLOSED: net-base value_marginal @ radar
  grades still < 0.03 ⟹ VALUE-death robust to base choice (not an oracle-base artifact)."* — 수치도
  재확인: 4cm/7cm/15cm 등급 모두 oracle-base·net-base 둘 다 value_marginal ≪ 0.03(최대 mean 0.0060,
  15cm 등급), E-54 draft 인용값과 정확히 일치.
- **THESIS_FRAMING_v2**(파일 mtime 07-18 20:39)는 E-55(21:05)·E-56(21:24) **커밋 이전** 스냅샷 — §9/§10의
  "재료 존재(미실행)" 서술은 **작성 시점엔 사실이었으나 같은 날 안에 추월당함.**
- **PAPER_SKELETON_operability_envelope**(파일 mtime 07-19 22:08, 즉 E-55/56 이후)는 이미 갱신되어 있음
  ("채널은 2개... 판정=2채널 상이-사망요인 + CRB 판별기 2클래스").

## 2. 실제 완결 상태 (E-54 화해 절 + E-55/56 원자료 종합)

| factor | **regime** | **current** |
|---|---|---|
| value | ✅ +0.072/+0.043 (in-arena) | ✅ **VALUE-사망**, base-robust (E-55/56, in-arena, n=48, 4중 스코프한정) |
| recoverability | ✅ blind≈net (arena X·Y) | ⚠**레거시-MEASURED**(07-05, 다른 아레나, 현 아레나 미재실행) |
| supply | ✅ 부이 등급 | ⚠ARGUED(구조 유추) |
| achievability | ✅ 사산(부이급 부재) | ⚠ARGUED+문헌접지(X-band/HF-radar 실재, Lund2018/2020) — **VALUE-사망이 먼저 채널을 닫아 검정 불필요해짐**(아래 §3) |
| info-limit(CRB) | ✅ E-51/52 (결과 B) | ❌ **미측정** (해류 CRB 없음) |

**두 채널 다 완전 5요소는 아니다** — regime만 5/5 완전 in-arena, current는 3/5 in-arena(value 새로·
recoverability 레거시·achievability 문헌추론) + info-limit 공백. **그러나 이건 조건② 미달이 아니라 조건②가
요구하는 기준(E-54 §Q5 화해: "채널 생존 아니라 5요소가 판정 완료")을 재검토하면 이미 충족**— 아래 §3.

## 3. "info-limit 공백이 조건② 미달을 의미하나?" — 아니오 (ARGUED, 논리 점검)
current 채널은 **VALUE 단계에서 이미 사망 판정**(radar급 잔여가치 CI상한 <0.03, 2중 검증 — oracle-base·
net-base 둘 다). 5요소 회계의 논리 순서(가치→회수성→공급→달성성→정보한계)에서 **VALUE 사망은 이후 단계를
무의미하게 만든다** — 이미 죽은 채널의 info-limit(정보벽이냐 효율갭이냐)을 판별하는 건 "가치가 없는 것이
왜 없는지"를 CRB로 재확인하는 것이라 조건② 충족에 필수가 아니다(regime도 achievability에서 사망 판정 후
추가로 3/4/5단계를 전부 완결하진 않음 — 죽은 지점 이후는 회계 종료가 표준). **단 이건 본 점검자의 논리
추론(ARGUED)** — 명시적으로 이렇게 규정한 사전등록 문서는 못 찾음. PAPER_SKELETON도 이 논리를 암묵 전제하되
명문화하지 않았다.

## 4. 실제로 남은 것 (진짜 잔여, 재확인)
1. **문서 상태-태그 정합화(사소, 작업량 적음)**: PAPER_SKELETON 학위조건 목록에서 조건①③④는
   `[✅해소]`/`[△QUALIFIED PARTIAL]`/`[✅MEASURED]` 식 상태 브래킷이 있는데 **조건②만 브래킷이 없다**
   (그냥 서술문). 다른 조건과 형식 통일 위해 `[✅ CLOSED — 2채널 상이-사망요인 판정완료, E-54/55/56]` 같은
   명시적 태그 추가 권고.
2. **THESIS_FRAMING_v2 §9/§10 갱신**: "다음(효율 순) ①다채널 이식(학위 관건, 여전히 관건)" 서술을
   "✅완료(E-55/56) — 남은건 서술 갱신뿐"로 수정. 현재 상태로 두면 이 문서만 읽는 사람에게 조건②가 아직
   안 끝난 것으로 오독될 위험(정확히 fable 브레인스톰에서 실제로 발생한 오독).
3. **current 채널 recoverability의 "레거시-MEASURED" 캐비엇**: 07-05 다른 아레나 결과를 현 아레나(regime
   아레나 골격 그대로 current로 이식한 e55/e56 아레나)에서 **재확인은 안 됨**. 조건②를 "완전 이식"으로
   부르려면 이 재확인이 이상적으로 남아있으나, **VALUE-사망이 채널을 이미 닫았으므로**(§3) 이것도 엄밀
  선행조건은 아님 — 있으면 좋은 마감(nice-to-have), 없어도 조건② 판정 자체는 안 흔들림.
4. **info-limit(현재 채널 CRB) 공백**: §3 논리로 불필요하다는 것이 이 문서의 판단(ARGUED)이나, 논문에
   명시적으로 "current는 VALUE에서 사망해 info-limit 단계 불요"라고 **한 문장 밝혀 두는 것**을 권고 —
   안 그러면 리뷰어가 표의 빈 칸(❌)을 "빠뜨렸다"로 읽을 위험.

## 5. 판정
- **조건②(다채널 완주) = CLOSED**(PAPER_SKELETON 기준, E-55/56 원자료로 직접 재확인). fable 브레인스톰의
  가설이 맞았음 — 새 실험 불필요.
- 남은 것은 **문서 정합화뿐**(위 4항목, 모두 저비용 글쓰기): 상태 브래킷 추가, THESIS_FRAMING §9/§10 갱신,
  info-limit 공백에 대한 1문장 해명.
- **본 점검이 놓쳤을 수 있는 것**: §3의 "VALUE-사망이면 이후 단계 불요"라는 논리 자체가 사전등록되지 않은
  ARGUED 판단이다 — 지도교수 difference-first 합의(조건⑤) 때 이 논리도 같이 확인받는 것을 권고.
