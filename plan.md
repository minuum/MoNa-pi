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

가설: `free_*` 실패가 "바구니/목표를 못 찾는" grounding 문제라면, bbox 또는 목표 중심 좌표를 보조 컨디셔닝으로 주입하면 개선될 것.

**리서치로 확인된 출발선**: MoNaVLA와 달리 MoNa-pi는 bbox 라벨이 전혀 없다(`mobile_vla_dataset_merged/`의 H5엔 `actions`/`language_instruction`/`observations/images`뿐). `language_instruction`은 거의 전부 "Navigate ... gray basket ..." 형태로 **타겟 객체가 "gray basket"으로 고정**돼 있음을 확인 — 고정 프롬프트로 detect 가능.

**Detector 방식 결정(2026-06-24, 사용자 승인)**: PaliGemma 자체 `generate()` detect 기능 사용. 이미 로드해둔 `paligemma-3b-pt-224`에 `"detect gray basket"` 프롬프트로 `generate()` 호출 → `<locXXXX>` 토큰 파싱 → (cx, cy, area) 정규화 좌표. 추가 모델/의존성 불필요. 오프라인 1회성 캐시 생성이라 inference 핫루프에는 안 들어감(MoNaVLA가 경고한 "실시간 generate() 금지"는 다른 backbone·다른 상황이라 여기 적용 안 됨). **단, 박스 정확도가 별도 fine-tune 없이 보장되지 않으므로 — Step A(추출 품질 확인)를 게이트로 두고, 품질이 나쁘면 여기서 멈추고 Phase 2로 우선순위를 옮긴다.**

### Step A — 오프라인 bbox 추출 + 품질 확인 (게이트)
- `scripts/extract_bbox_cache.py` 신규: `PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-pt-224")` 로드, 각 H5 에피소드의 각 프레임에 `"detect gray basket"` 프롬프트로 `generate()`, `<locy1><locx1><locy2><locx2>` 토큰 파싱(1024 그리드 정규화) → `(cx, cy, area)` ∈ [0,1].
- 소규모 샘플(예: 3~5 에피소드, 각 에피소드 첫/중간/마지막 프레임)에 대해 박스를 이미지에 오버레이해서 PNG로 저장 → 눈으로 품질 확인. 박스가 바구니를 못 잡으면(예: 배경/로봇 자신을 잡거나 박스가 전체화면) 여기서 멈추고 보고.
- 품질 통과 시 전체 245 에피소드(또는 holdout 제외 train/val 224개)에 대해 전체 캐시 생성, `bbox_cache.json`(또는 episode별 `.npy`)으로 저장.

### Step B — 데이터셋 통합
- `ActionChunkDataset.__getitem__`에 `use_bbox_cond` 옵션 추가: 캐시에서 윈도우(8프레임)에 대응하는 bbox 시퀀스 로드, `(window_size, 3)` 텐서 + 탐지 실패 프레임용 valid mask 반환.
- HFlip 증강 시 `cx → 1-cx`로 같이 반전(MoNaVLA의 동일 버그 패턴을 미리 피함).

### Step C — 모델 통합
- `Pi0VLA`에 `bbox_proj: Linear(4, cond_dim)`(cx,cy,area,valid) 추가, bbox 토큰을 VLM cond 시퀀스에 concat해서 action expert에 전달. cross-attn은 가변 길이 시퀀스를 그대로 받으므로 구조 변경 없음 — 새 투영 레이어만 학습 필요.
- 세 조건(image_only=현재 baseline, bbox_only=vision_tower 출력을 0으로 마스킹, bbox+image=둘 다)을 같은 config로 플래그 전환만으로 만들 수 있게 설계 — MoNaVLA `ablate_bbox_image_features.py`와 동일 철학.

### Step D — 파일럿 학습 + 평가
- 정형 9종(train/val, Phase 1 split) 기준 짧은 파일럿 학습(전체 재학습 아님, 빠른 검증용 epoch 수로) 후 `eval_closedloop.py --free-only`/`--regular-only`로 세 조건 비교.
- 성공 기준: bbox+image 조건이 image_only(현재 baseline) 대비 `free_*` SR을 의미 있게 올리면 Phase 4(전체 재학습)로 승격, 아니면 MoNaVLA와 동일하게 "grounding 신호 분리만으론 부족"으로 결론 내리고 Phase 2(실데이터)에 집중.

리스크: MoNaVLA 쪽도 "BBox는 보조, image가 핵심"이라는 결론이었음(bbox_only 67.4% vs image_only 75.6%) — grounding 신호가 만능 해법은 아닐 수 있음. Step A 게이트와 Step D 작은 파일럿으로 투자를 최소화하면서 검증.

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

- [x] Phase 1 구현 승인
- [ ] Phase 2 실데이터 수집 일정 (로봇/현장 작업이라 Claude가 직접 못함 — 사용자 스케줄)
- [x] Phase 3 grounding 주입 방식 결정 — **PaliGemma 자체 generate() detect** (사용자 승인, 2026-06-24)

---

## [ ] 체크리스트 (승인 후 갱신)
- [x] Phase 1 구현 승인
- [x] Phase 1 구현 완료 (`data/dataset.py` exclude_free_holdout + `build_free_holdout()`, `eval_closedloop.py --free-only`. 상세: `docs/ABLATION_RESULTS_20260623.md` M10)
- [ ] Phase 2 수집 시작
- [x] Phase 3 Detector 방식 승인 (PaliGemma generate() detect)
- [x] Phase 3 Step A — bbox 추출 + 품질 확인 (게이트 통과, 44% 탐지율. `scripts/extract_bbox_cache.py`, `logs/bbox_cache.json` 244 에피소드)
- [x] Phase 3 Step B — 데이터셋 통합 (`data/dataset.py`의 `bbox_cache_path`/`use_bbox_cond`, HFlip cx 반전 포함, unit test 통과)
- [x] Phase 3 Step C — 모델 통합 (`Pi0VLA`의 `use_bbox_cond`/`bbox_only`, 3-way 조건 forward/loss 단위 테스트 통과)
- [x] Phase 3 Step D — 파일럿 학습/평가 완료 (image_only 15% / bbox_only 20% / bbox+image 0%-깨짐, free n=20. 상세: ABLATION_RESULTS M11)
- [ ] Phase 4 재학습/재평가 — bbox+image concat 버그 수정 또는 bbox_only 채택 여부 결정 필요
