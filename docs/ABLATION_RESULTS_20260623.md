# Ablation Pipeline 결과 (M6 → M5 → M7)

> 실행: 2026-06-23 18:00~20:47 (minum 서버, `monapi-train` 브랜치)
> `scripts/run_ablation.sh` (2026-06-02 작성, 이번에 처음 실행) — 중간에 브랜치 전환 실수로
> M5 eval 단계가 한 번 끊겼다가(`scripts/eval_closedloop.py`가 `monapi-driving`에는 없는 파일이라
> working tree에서 사라짐) `monapi-train`에 고정한 채 재실행해 완료.

## M6 — ODE steps 3/5/10 (재학습 없음, eval만)

베이스라인(`checkpoints/best`)으로 ODE 적분 스텝만 바꿔 품질/속도 트레이드오프 확인.

| ODE steps | SR (regular, n=38) | FPE |
|---|---|---|
| 3 | 100.0% | 0.0707m |
| 5 (현재 기본값) | 100.0% | 0.0697m |
| 10 | 100.0% | 0.0703m |

**결론**: 3/5/10 전부 거의 동일 — **ODE steps를 3으로 줄여도 품질 손해 없음**. 추론 속도를 줄일 수 있는 여유가 있다는 뜻 (현재 기본 5 대비 추가 단축 가능, 실측 latency 비교는 별도 필요).

## M5 — 증강 없이 학습 (244ep, mobile_vla_dataset_merged) vs baseline(증강 있음)

| 구성 | eval 대상 | SR | FPE |
|---|---|---|---|
| baseline (증강 있음, `checkpoints/best`) | full (n=53) | 73.6% | 0.3906m |
| M5 (증강 없음) | full (n=53) | 73.6% | 0.3868m |
| M5 (증강 없음) | regular (n=38) | 100.0% | 0.0694m |

baseline의 `full` eval도 추가로 돌려서 직접 비교함 — **증강 유/무가 SR·FPE에 거의 차이를 만들지 않음(73.6%/73.6%, FPE 차이 0.004m로 노이즈 수준)**. `full`에만 포함되는 나머지 15개 에피소드(center_left/center_right/straight 등 약한 카테고리로 추정)에서 둘 다 동일하게 크게 떨어짐.

✅ **결론**: "증강이 약한 카테고리를 보완해줄 것"이라는 기대는 **이 실험에서 확인되지 않음** — 증강 여부와 무관하게 동일한 카테고리에서 동일하게 실패. 원인은 모델 구조나 augmentation 종류가 아니라 **해당 카테고리의 실제 학습 데이터 절대량 부족일 가능성이 높다**(`docs/MONAVLA_CROSSCHECK_20260623.md`의 center/straight 약점 가설과 일치).

## M7 — 116ep 원본 v5 데이터만 (mobile_vla_dataset_v5, 증강 없음)

| eval 대상 | SR | FPE |
|---|---|---|
| regular (n=24) | 100.0% | 0.0485m |
| full (n=24) | 100.0% | 0.0485m |

⚠️ **해석 주의**: M7의 `regular`와 `full`이 완전히 동일(둘 다 n=24)하다는 건, **116ep 원본 v5 데이터셋 자체에 "hard" 카테고리 평가 에피소드가 거의/전혀 없다는 뜻** — 즉 "원본 데이터만으로도 100%"가 "작은 데이터셋이 더 낫다"를 의미하지 않음. 단지 그 데이터셋의 eval 셋이 쉬운 케이스로만 구성돼 있어서 어려운 케이스를 아예 평가하지 않은 것. M5(244ep, full=53)와 M7(116ep, full=24)을 직접 비교하는 건 **apples-to-apples가 아님** — 서로 다른 eval 모집단.

## 종합

- ✅ **ODE steps 축소(3)는 안전** — 바로 적용 가능한 최적화.
- ✅ **증강은 약한 카테고리(center/straight 계열)에 측정 가능한 효과가 없음** — baseline과 M5(no-aug)가 73.6%/73.6%로 동일. 약점의 원인은 augmentation이 아니라 데이터 절대량 부족 쪽으로 무게가 실림.
- ❓ **"116ep로도 충분하다"는 결론은 못 냄** — M7의 eval 모집단(n=24)이 애초에 쉬운 케이스로만 구성돼 있어 M5(n=53)와 직접 비교 불가(apples-to-apples 아님). 116ep 데이터셋에도 244ep 수준의 hard-category 평가셋을 적용해야 진짜 비교 가능 — 후속 작업으로 남김.
