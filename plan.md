# MoNa-pi 장기 로드맵 (2026-06-24)

> 이전 plan.md(AdaLN-Zero 교체)는 완료되어 `docs/archive_plan_adaln_zero.md`로 옮김.
> 이 문서는 M9(train/val 누출 수정) 이후 다음 단계 장기 계획. **승인 전 구현 금지.**

## 0. 현재까지 확정된 사실 (재구성 없이 그대로 인용)

| 발견 | 결론 |
|---|---|
| M5 vs M7 (244ep vs 116ep, apples-to-apples) | 데이터 2배 늘려도 전체 SR 거의 무변화 (71.7% vs 73.6%, 노이즈 수준) |
| M8 (이미지 합성 OOD 증강) | 라벨 불변 증강은 새 image→action 대응을 가르치지 못함 — null result |
| M9 (train/val 누출 수정) | 정형 9종 100%, `free_*` 0% 패턴은 누출 수정 후에도 동일 — 구조적 한계, 버그가 아니었음 |
| Gemma Expert 아키텍처 | BF16/INT8 모두 0 missing/unexpected key로 로드 확인, 양쪽 브랜치 프로덕션 기본값으로 전환 완료 |
| `free_*` val 표본 | stratify_free로 0개 사고는 막았지만 현재 n=2 — 통계적으로 결론 내리기엔 너무 작음 |
| MoNaVLA 횡단 비교 | end-to-end VLM(Exp11) PM 58.6%인데 closed-loop 0% / decomposition(bbox+image MLP) PM 75.9%인데 closed-loop 66.7% — grounding 분리가 직접 효과 큼 |

**결론**: 지금 병목은 모델 구조도, 데이터 "양"도 아니라 **`free_*` 같은 OOD 상황을 표현하는 실데이터의 부재**다. 합성 증강과 데이터 볼륨 증가라는 두 가지 "공짜" 레버는 이미 시도해서 둘 다 죽었다 (M7, M8). 남은 레버는 실데이터 수집과 grounding 신호 주입뿐이다.

---

## 1. 장기 목표

closed-loop 평가에서 `free_*` 카테고리 SR을 0%에서 의미 있게(목표: 정형 경로의 절반 수준, 즉 ~40~50%) 끌어올린다. 이게 안 되면 MoNa-pi는 "정형 경로 한정 데모"로 남고, 교수님 프로토콜의 Step 3(33/33/33 완전 동작) 기준을 못 넘김.

---

## 2. Phase 1 — 측정 인프라 정비 (선행 작업, 코드 변경 적음)

현재 `free_*` val 표본이 n=2라 어떤 후속 실험도 신뢰 가능한 비교가 안 됨. 데이터/아키텍처를 더 손대기 전에 측정을 먼저 고정한다.

- **고정 free_* eval set 분리**: `mobile_vla_dataset_merged/` 내 `free_*` 전체(현재 21개 + 신규 수집분)를 train/val 분할에서 완전히 제외하고 별도 `eval_free_holdout/` 디렉토리(또는 파일 리스트)로 고정. 매 실험마다 동일 모집단으로 비교 가능하게.
- **`eval_closedloop.py`에 `--free-only` 플래그 추가**: 지금 `--regular-only`의 반대 버전. 정형/`free_*` 두 숫자를 항상 같이 리포트하도록 출력 포맷 변경.
- 변경 파일: `data/dataset.py`(분할 로직에 holdout 제외 옵션), `scripts/eval_closedloop.py`.
- 트레이드오프: holdout으로 빼면 학습에 쓸 `free_*` 표본이 더 줄어듦 → Phase 2에서 신규 수집량이 더 중요해짐.

## 3. Phase 2 — 실데이터 수집 캠페인 (가장 ROI 높은 레버, 비코드)

M7/M8이 보여준 것: "더 많은 같은 데이터"와 "가짜 변형"은 안 통한다. 실제 `free_*` 변형 자체를 더 찍어야 한다.

- 현재 `free_*` 카테고리(분포): `free_center/left/right_*` × `basket_left_extreme, basket_right_extreme, robot_close, robot_far, diagonal_left, diagonal_right, lighting_diff` — 카테고리당 1~3개뿐.
- 목표: 카테고리당 최소 10~15개로 확충 (현재 21개 → 70~100개 수준).
- 수집 후 `scripts/sync_dataset_from_monavla.py` 패턴 재사용 가능(다만 이건 MoNaVLA 쪽 소스가 따로 있을 때고, 신규 `free_*`는 MoNa-pi 로봇으로 직접 찍어야 함 — 사용자 확인 필요).
- 이 Phase는 코드 작업이 아니라 로봇 운용 스케줄링 항목. Claude 쪽에서는 수집 후 데이터 정합성 검증(포맷, instruction 태깅, episode 길이) 스크립트만 준비.

## 4. Phase 3 — Grounding 신호 주입 실험 (MoNaVLA 교훈 이식)

MoNaVLA의 핵심 교훈: end-to-end VLM은 PM이 높아도 closed-loop에서 방향 오류가 누적되지만, bbox 같은 명시적 grounding을 분리해서 주면 closed-loop이 극적으로 좋아짐. MoNa-pi는 현재 순수 end-to-end(이미지 패치 → PaliGemma → action expert)라 같은 함정에 빠질 위험이 있다.

- 가설: `free_*` 실패가 "바구니/목표를 못 찾는" grounding 문제라면, bbox 또는 목표 중심 좌표를 보조 컨디셔닝으로 주입하면 개선될 것.
- 실험 설계(스케치, 구현 전 검토 필요):
  - SigLIP/PaliGemma의 attention map 또는 별도 경량 detector로 목표 바구니의 (cx, cy, area) 추출.
  - `Pi0VLA.forward_backbone()` 출력에 이 3~4차원 벡터를 concat하거나, action expert의 cross-attn cond에 추가 토큰으로 주입.
  - MoNaVLA에서 이미 검증된 ablation 패턴(`scripts/ablate_bbox_image_features.py`) 참고해 bbox_only / image_only / bbox+image 세 조건 비교.
- 이 Phase는 Phase 1(측정 인프라)이 끝나야 의미 있는 비교가 가능. Phase 2(데이터)와는 병행 가능.
- 리스크: MoNaVLA 쪽도 "BBox는 보조, image가 핵심"이라는 결론이었음(bbox_only 67.4% vs image_only 75.6%) — grounding 신호가 만능 해법은 아닐 수 있음. 작은 규모 파일럿으로 먼저 검증.

## 5. Phase 4 — 통합 재학습 + 재평가

Phase 2(데이터 확충) + Phase 1(고정 eval) 완료 후, 필요시 Phase 3(grounding 주입) 포함해서 Gemma Expert 기준으로 재학습.

- 학습: `python training/train.py --config configs/train.yaml` (use_gemma_expert: true 기본 확인됨)
- 평가: `scripts/eval_closedloop.py --free-only`와 `--regular-only` 둘 다 리포트, M9 이전/이후/Phase 4 세 시점 비교표로 문서화.
- 성공 기준: free_* SR이 0%에서 통계적으로 의미 있게(고정 holdout 기준 최소 절반 이상 표본에서) 상승.

## 6. 우선순위 및 순서

```
Phase 1 (측정 인프라) → 즉시 시작 가능, 코드량 적음
Phase 2 (실데이터 수집) → 가장 오래 걸림, 동시에 시작 권장 (로봇 스케줄 필요)
Phase 3 (grounding 주입 파일럿) → Phase 1 완료 후, Phase 2와 병행
Phase 4 (통합 재학습/재평가) → Phase 2 데이터 일정량 확보 후
```

## 7. 다음 결정이 필요한 지점

- [ ] Phase 1을 지금 바로 구현 승인할지 (`data/dataset.py` holdout 분리 + `eval_closedloop.py --free-only`)
- [ ] Phase 2 실데이터 수집 일정 (로봇/현장 작업이라 Claude가 직접 못함 — 사용자 스케줄)
- [ ] Phase 3 grounding 주입 방식(별도 detector vs PaliGemma 내부 attention 재사용) 중 어느 쪽으로 파일럿할지

---

## [ ] 체크리스트 (승인 후 갱신)
- [x] Phase 1 구현 승인
- [x] Phase 1 구현 완료 (`data/dataset.py` exclude_free_holdout + `build_free_holdout()`, `eval_closedloop.py --free-only`. 상세: `docs/ABLATION_RESULTS_20260623.md` M10)
- [ ] Phase 2 수집 시작
- [ ] Phase 3 파일럿 설계 승인
- [ ] Phase 4 재학습/재평가
