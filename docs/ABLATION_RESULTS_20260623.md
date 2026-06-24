# Ablation Pipeline 결과 (M6 → M5 → M7)

> ⚠️ **2026-06-24 중요 정정**: 이 문서의 M5/M6/M7/M8 숫자는 모두 **train/val 누출이 있던
> 구버전 split**으로 측정됐다(아래 "M9 — train/val 누출 버그 수정" 참고). 윈도우 단위
> random_split 때문에 "val 에피소드"의 92.5%가 train에도 자기 자신의 다른 윈도우를 갖고
> 있었음 — 즉 이 문서의 SR 숫자들은 진짜 held-out 성능이 아니라 다소 낙관적인 값일 수 있다.
> 다만 M5/M6/M7/M8/baseline이 전부 **동일한 leaky split**을 공유했으므로 그들 사이의
> **상대 비교**(증강 효과 없음, ODE steps 무관, 116ep≈244ep)는 여전히 유효하다고 판단함 —
> 절대 SR 수치만 곧이곧대로 읽지 말 것.

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

baseline의 `full` eval도 추가로 돌려서 직접 비교함 — **증강 유/무가 SR·FPE에 거의 차이를 만들지 않음(73.6%/73.6%, FPE 차이 0.004m로 노이즈 수준)**.

### 14건 실패의 정체 — `episode_idx`를 실제 파일명으로 역추적 (정정)

처음엔 "`full`에만 포함되는 나머지 15개 에피소드가 center_left/center_right/straight 등 약한 카테고리일 것"으로 추정했으나, `episode_idx`를 실제 h5 파일명으로 역매핑해 확인한 결과 **완전히 다른 그림**이었다.

| 카테고리 | 구성 | 실패 |
|---|---|---|
| 정형 9종(center/left/right × straight/left/right) | 38개 | **0개 (0%)** |
| `free_center_*` (basket 극단오프셋/대각선/로봇거리/조명) | 5개 | 4개 (1개만 성공: robot_far) |
| `free_left_*` | 5개 | **5개 (100%)** |
| `free_right_*` | 5개 | **5개 (100%)** |

baseline과 M5(no-aug) 둘 다 동일하게 **free fail/total = 14/15, regular fail = 0/38**.

✅ **정정된 결론**: 약점은 "center/straight 카테고리"가 아니라 **`free_*` 의도적 OOD 프로브 에피소드 전체**(basket 극단 오프셋, 대각선 시작, 로봇 원거리, 조명 변화 — 거의 모든 변형에서 실패). 정형 9종 경로는 (증강 유무와 무관하게) 100% 성공하므로, CH7의 add_free 244ep 통합(center_straight 21ep 추가 등)이 실제로 "정형 경로" 문제는 해결한 것으로 보인다. 남은 문제는 augmentation으로 풀리는 종류가 아니라 **학습 분포 밖(OOD) 일반화 자체의 한계** — `free_*` 부류 자체를 학습 데이터에 더 포함시키거나, OOD에 강한 정규화/증강(예: 극단 오프셋·대각선 시작을 흉내내는 합성 증강)이 필요할 것으로 보임.

## M8 — OOD 흉내 증강 (robot_close/far zoom, basket_extreme shift) — null 결과

`data/ood_augment.py` 신규 구현(윈도우 8프레임 전체에 동일 zoom/shift, p=0.35) 후 244ep 동일 조건으로 재학습.

| 구성 | full(53) SR | FPE |
|---|---|---|
| baseline | 73.6% | 0.3906m |
| M5(no-aug) | 73.6% | 0.3868m |
| **M8(ood-aug)** | **73.6%** | 0.3944m |

❌ **완전한 null 결과 — 실패한 14개 에피소드가 baseline/M5와 1:1로 정확히 동일**(같은 episode_idx, 성공한 것도 free_center_robot_far 단 1개로 동일). 증강을 추가했는데 단 1건도 바뀌지 않음.

**원인 추정**: 이 증강은 이미지를 zoom/shift로 변형하지만 **액션 레이블은 원본 그대로 둠** — 실제 `basket_left_extreme` 에피소드는 "바구니가 화면 왼쪽 끝에 있을 때 그에 맞는 회피/접근 궤적"이 액션에 반영돼 있는데, 합성 증강은 "다른 정상 궤적의 이미지만 좌우로 밀어서" 만든 것이라 이미지-액션 대응 관계 자체가 실제 OOD 시나리오와 다르다. 모델이 증강된 이미지 변형을 보고도 원래 정답 액션을 그대로 맞추도록 학습되면서 **시각적 invariance만 약하게 학습되고, 새로운 image→action 매핑은 배우지 못했을 가능성**.

✅ **결론**: 이미지 레벨 합성 증강으로는 이 OOD 문제를 풀 수 없음(적어도 이번 설계로는) — 실제 `free_*` 부류 데이터를 더 수집하는 것이 유일하게 검증된 해법으로 좁혀짐. (`diagonal_left/right`는 처음부터 2D 증강 대상에서 제외했었음.)

## M7 — 116ep 원본 v5 데이터만 (mobile_vla_dataset_v5, 증강 없음)

| eval 대상 | SR | FPE |
|---|---|---|
| regular (n=24) | 100.0% | 0.0485m |
| full (n=24) | 100.0% | 0.0485m |

⚠️ **해석 주의**: M7의 `regular`와 `full`이 완전히 동일(둘 다 n=24)하다는 건, **116ep 원본 v5 데이터셋 자체에 "hard" 카테고리 평가 에피소드가 거의/전혀 없다는 뜻** — 즉 "원본 데이터만으로도 100%"가 "작은 데이터셋이 더 낫다"를 의미하지 않음. 단지 그 데이터셋의 eval 셋이 쉬운 케이스로만 구성돼 있어서 어려운 케이스를 아예 평가하지 않은 것. M5(244ep, full=53)와 M7(116ep, full=24)을 직접 비교하는 건 **apples-to-apples가 아님** — 서로 다른 eval 모집단.

### M7 진짜 apples-to-apples 비교 (2026-06-24 추가)

위 비교의 한계를 해소하기 위해 M7의 체크포인트(`checkpoints/ep116/best`)를 **M5/M8과 동일한 eval 모집단**(`configs/train.yaml`의 244ep val split, full n=53)에 직접 통과시켰다(`--config configs/train.yaml --ckpt checkpoints/ep116/best`). 모델 아키텍처 설정(`use_gemma_expert`, `use_paligemma` 등)이 두 config에서 동일함을 먼저 확인 후 진행.

| 구성 | eval 대상(동일 모집단, n=53) | SR | FPE |
|---|---|---|---|
| baseline (244ep, 증강 있음) | full | 73.6% | 0.3906m |
| M5 (244ep, 증강 없음) | full | 73.6% | 0.3868m |
| M8 (244ep, OOD 증강) | full | 73.6% | 0.3944m |
| **M7 (116ep, 증강 없음)** | full | **71.7%** | **0.3790m** |

✅ **결론**: 116ep와 244ep는 동일 eval 모집단에서 **71.7% vs 73.6% — 차이 1개 에피소드(53개 중), 노이즈 수준**. `add_free`로 데이터를 116→244ep(2배)로 늘렸지만 전체 성능 지표는 거의 변하지 않았다. 위의 "14건 실패=free_* OOD 전체" 진단과 합쳐 보면, add_free 통합이 실제로 고친 것은 정형 9종 경로(0% 실패)이고, 데이터 양 자체를 늘리는 것만으로는 `free_*` OOD 일반화가 개선되지 않는다는 그림과 일치한다. **"데이터를 더 모으면 풀린다"는 가정도 무조건은 아님** — 단순 양적 증가가 아니라 `free_*`류(극단 오프셋/대각선/원거리/조명)를 직접 타겟한 데이터 수집이 필요해 보인다(M8 null 결과와 함께 고려할 것).

## 종합

- ✅ **ODE steps 축소(3)는 안전** — 바로 적용 가능한 최적화.
- ✅ **약점은 `free_*` OOD 프로브 전체(정형 9종 경로는 100%)** — baseline과 M5(no-aug)가 73.6%/73.6%로 동일하고, 실패 14건 전부 `free_*`(14/15), 정형 경로 실패 0건(0/38)으로 정확히 일치.
- ❌ **이미지 레벨 합성 OOD 증강(M8)은 효과 없음** — baseline/M5/M8 전부 동일한 14개 에피소드가 실패. 액션 레이블이 안 바뀌는 합성 증강의 구조적 한계로 추정. **실제 free_* 데이터 추가 수집이 유일한 다음 단계.**
- ✅ **116ep vs 244ep, 진짜 apples-to-apples 비교 완료(6/24)**: 동일 eval 모집단(n=53)에서 71.7% vs 73.6% — 노이즈 수준 차이. `add_free`로 데이터를 2배로 늘려도 전체 SR은 거의 그대로. 정형 9종 경로 보강에는 효과가 있었겠지만(M5에서 0% 실패 확인), `free_*` OOD 일반화는 데이터 양만으로 개선되지 않음 — M8의 합성 증강 null 결과와 같은 결론으로 수렴.

## M9 — train/val 누출 버그 발견 + 수정 (2026-06-24)

`알고리즘적 미스 audit` 중 `data/dataset.py`의 `build_train_val_split()`을 들여다보다 발견.

### 버그

기존 구현은 **윈도우(샘플) 생성 후** `torch.utils.data.random_split()`으로 train/val을 나눴다. 윈도우는 `window_size=8`, `k=10`(horizon) 기준으로 한 에피소드 안에서 `t`를 1씩 밀며 슬라이딩하므로, 인접한 두 윈도우는 입력 이미지 8프레임 중 7프레임이 겹친다. random_split이 윈도우 단위로 작동하면 **같은 에피소드의 인접 윈도우가 train과 val에 양쪽으로 갈리는 일이 빈번**하게 일어난다.

실측(`configs/train.yaml`, 244ep, seed=42): val로 분류된 53개 에피소드 중 **49개(92.5%)가 train 쪽에도 자기 자신의 다른 윈도우를 갖고 있었다.** 즉 "held-out" 평가가 실제로는 거의 다 봤던 에피소드를 다른 프레임 오프셋으로 다시 보는 것에 가까웠다 — `eval_closedloop.py`가 episode 단위로 전체 궤적을 재생하긴 했지만, 그 episode의 다른 윈도우는 이미 gradient에 반영된 상태였던 것.

MoNaVLA의 `robovlm_nav/datasets/nav_h5_dataset_impl.py`(`stratified_split` 옵션 포함)는 처음부터 **에피소드(파일) 단위**로 train/val을 나눈 뒤 윈도우를 생성한다 — 이 프로젝트가 참고해야 했던 "좋은 센스"였는데 MoNa-pi 쪽엔 포팅되지 않았던 부분.

### 수정

`build_train_val_split()`을 파일 단위 분할로 재작성, `ActionChunkDataset`에 `file_list` 파라미터 추가(주어지면 디렉토리 전체 글롭 대신 그 목록만 사용). 추가로 `stratify_free=True`(기본값) 옵션 — `free_*`/정형 9종을 따로 묶어 각각 `val_split` 비율로 나눠 합침. 이유: 단순 통째 shuffle로는 `free_*`가 전체의 ~8%뿐이라 val에 0개 뽑히는 경우가 실제로 발생함(seed=42, stratify 끄면 확인됨) — `free_*` 실패율 추적이 이 프로젝트 평가의 핵심이라 stratify 없이는 eval이 무의미해질 위험이 있었음.

검증:
- 수정 후 train 파일/val 파일 교집합 = **0개** (245개 중 train 221 / val 24).
- `train.py`의 DataLoader 빌드 sanity test 정상 동작 확인(배치 shape 정상).
- `scripts/eval_closedloop.py`도 `val_ds`가 더 이상 `Subset`이 아니므로(이제 파일 단위로 분리된 독립 `ActionChunkDataset`) `--regular-only` 필터의 파일명 매핑을 `val_ds.h5_files` 기준으로 수정(기존엔 `train_path` 전체 글롭 기준이라 인덱스가 안 맞을 뻔했음).

### 누출 수정 후 첫 baseline (checkpoints/best, stratified split, n=17)

| 카테고리 | 구성 | 결과 |
|---|---|---|
| 정형 9종 | 15개 | **15/15 = 100%** (FPE 0.026~0.183m) |
| `free_*` | 2개(diagonal_right ×2) | **0/2 = 0%** (FPE 1.11m, 1.35m) |
| 전체 | 17개 | 88.2% |

✅ **결론**: 누출을 제거한 진짜 held-out 평가에서도 **정형 9종 100% / free_* 0%** 패턴이 그대로 재현됨 — CH9-2/9-4의 "약점은 카테고리가 아니라 free_* OOD 전체"라는 결론이 누출과 무관하게 견고하다는 뜻. 다만 stratify된 val의 free_* 표본이 단 2개뿐이라 통계적으로는 약함 — `free_*` 21개 전체를 별도 고정 eval set으로 떼어 쓰는 방식이 더 안정적일 수 있음(후속 검토).

⚠️ **이 수정으로 인해 위 M5~M8 섹션의 절대 SR 수치는 더 이상 재현되지 않는다** — config는 그대로지만 내부 split 로직이 바뀌어 val 모집단 자체가 달라졌기 때문. M5~M8을 다시 정확히 재현하려면 동일 buggy split이 필요한데, 그건 의미가 없으므로 재실행하지 않음. 위 M5~M8 절들은 "그 당시 비교들 사이의 상대적 결론"으로만 읽을 것.

## M10 — free_* 고정 holdout 분리 (Phase 1, 2026-06-24)

M9에서 남긴 후속 검토(`free_*` stratified val 표본 n=2는 통계적으로 너무 약함)를 바로 처리. 장기 로드맵(`plan.md`) Phase 1.

### 변경

- `data/dataset.py`: `build_train_val_split()`의 `stratify_free` 파라미터를 `exclude_free_holdout`(기본 `True`)으로 교체. `free_*` 에피소드는 더 이상 train/val 분할 대상에 들어가지 않고 완전히 빠진다.
- `build_free_holdout()` / `get_free_holdout_files()` 신규 — `free_*` 21개 전체를 매 실험마다 동일하게 평가할 수 있는 고정 데이터셋으로 반환.
- `scripts/eval_closedloop.py`: `--free-only` 플래그 추가(`build_free_holdout()` 사용). `--regular-only`는 이제 val이 항상 정형뿐이라 no-op이지만 하위호환 위해 유지. 출력/JSON summary에 `population`("regular_val" | "free_holdout") 필드 추가.

### 근거

M7(데이터 2배)·M8(합성 OOD 증강) 둘 다 `free_*` SR을 못 바꿨다 — `free_*`를 학습에 일부 끼워 넣는 것 자체가 효과가 없다는 게 이미 확인된 상태. 반면 stratify로 val에 어떤 free_* 1~2개가 뽑히는지가 seed/실험마다 흔들리면 향후 어떤 개선(Phase 2 실데이터 수집, Phase 3 grounding 주입)을 시도해도 "효과가 있었는지"를 판단할 기준 모집단이 안정적이지 않다. 학습 신호로서는 버려도 되는 손실인 반면, 고정 모집단으로 얻는 측정 안정성은 이후 모든 비교의 전제조건이라 우선순위를 높게 잡음.

### 검증

```
free holdout files: 21개 (get_free_holdout_files)
build_train_val_split → train 202 / val 22, 둘 다 free_* 0개 (exclude 확인)
build_free_holdout    → 21개 에피소드, 368 샘플, train+val과 파일 교집합 0개
train.py DataLoader sanity: images (4,8,3,224,224), actions (4,10,3) — 정상
```

기존에 알려지지 않았던 손상 파일 1개 발견(`episode_260506_194517_target_left_straight_path__core__fixed_center.h5`, `Unable to synchronously open file (bad object header version number)`) — `_read_episode`에서 예외로 스킵되어 로딩엔 영향 없음. 이번 변경과 무관, 별도 데이터 정합성 이슈로 기록.

### 다음

Phase 2(실데이터 `free_*` 수집)·Phase 3(grounding 신호 파일럿) 진행 시 `scripts/eval_closedloop.py --free-only` 결과를 baseline으로 사용. 자세한 로드맵은 `plan.md` 참고.
