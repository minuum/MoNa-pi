# v5 학습 검토 리포트 (2026-04-30)

> v5 데이터셋 전환 이후의 학습 기록을 전수 분석. 다음 학습(train v2, D3 예정)에 반영할 시사점 정리.

## 1. 학습 횟수 — 단 1회

git commit 기준 v5 도입 이후의 학습 관련 변경:

| commit | 날짜 | 내용 |
|--------|------|------|
| `f05a2d8` | 2026-04-29 | Update dataset + training for v5 data and real PaliGemma backbone |
| `d1d6cc8` | 2026-04-29 | Fix PaliGemmaBackbone for transformers 5.x API |
| `37c863d` | 2026-04-29 | Switch training to BF16 to fix FP16 gradient overflow |

`logs/train.log`는 단일 run. **v5 데이터셋으로 학습된 모델은 `checkpoints/{mona_pi_best, mona_pi_epoch_05, mona_pi_epoch_10}` 한 세트뿐**이고 epoch 10이 best (val=0.0646).

## 2. Loss 곡선 — 매끄러운 수렴, LR cosine 끝까지 소진

| epoch | train | val | LR | 비고 |
|-------|-------|-----|----|------|
| 01 | 0.5502 | 0.6113 | 2.00e-05 | best |
| 02 | 0.3512 | 0.3786 | 1.92e-05 | best |
| 03 | 0.2818 | 0.2699 | 1.74e-05 | best |
| 04 | 0.1814 | 0.2547 | 1.47e-05 | best |
| 05 | 0.1689 | 0.1576 | 1.15e-05 | best, ckpt 저장 |
| 06 | 0.1370 | 0.0958 | 8.04e-06 | best |
| 07 | 0.1238 | **0.1790** | 4.86e-06 | val 튐 (regression) |
| 08 | 0.1099 | 0.1121 | 2.28e-06 | |
| 09 | 0.1149 | 0.1011 | 5.91e-07 | |
| 10 | 0.1129 | **0.0646** | **4.08e-11** | best, 최종 ckpt |

- val loss 9.5× 개선(0.6113 → 0.0646)
- **LR이 cosine schedule 끝(4e-11 ≈ 0)까지 도달** → 다음 학습은 LR 재시작 필요
- epoch 7 val 튐(0.0958 → 0.179)은 val set이 26 sample로 매우 작아 노이즈가 큼 → **val 신뢰성 낮음**

## 3. **🚨 데이터 누락 이슈 — episode 22.7%가 학습에 0 sample 기여**

### 데이터셋 통계
- 총 150 episode, 2,626 frame
- frame 분포: min=14 / median=18 / max=19 / mean=17.5

| episode 길이 | 개수 | 1 episode당 sample |
|--------------|------|---------------------|
| 14 frame | **20** | **0** ❌ |
| 16 frame | **14** | **0** ❌ |
| 17 frame | 1 | 1 |
| 18 frame | 80 | 2 |
| 19 frame | 35 | 3 |

### 원인
`data/dataset.py:72`의 인덱싱:
```python
for t in range(window_size - 1, n - k + 1):  # range(7, n-9)
```
- `window_size=8, k=10` → `n - 16` 만큼 sample 생성
- **n ≤ 16 인 episode는 빈 range** → 통째로 누락
- 누락 episode: 20 + 14 = **34 / 150 = 22.7%**

### 영향
- 학습 sample 266개(train=240, val=26)는 **150 episode가 아니라 116 episode에서만** 나옴
- 데이터셋이 이미 작은데(150 ep) 23% 더 줄어드는 셈

### 🚨🚨 카테고리별 누락 — 두 카테고리는 사실상 학습 안 됨

| 카테고리 | 총 episode | usable | 누락 | 비고 |
|---------|-----------|--------|------|------|
| target_center_straight | 20 | **0** | **20** | ❌ **전 카테고리 누락**, 모델이 한 번도 못 본 시나리오 |
| target_right_right | 15 | 1 | **14** | ❌ 단 1 sample로만 학습 |
| target_left_straight | 20 | 20 | 0 | ✓ |
| target_right_straight | 20 | 20 | 0 | ✓ |
| target_center_left | 15 | 15 | 0 | ✓ |
| target_center_right | 15 | 15 | 0 | ✓ |
| target_left_left | 15 | 15 | 0 | ✓ |
| target_left_right | 15 | 15 | 0 | ✓ |
| target_right_left | 15 | 15 | 0 | ✓ |

**시연/실로봇 영향**: D10 시연 시나리오 1번(Straight-1m, 자연어 "go forward")은 **`target_center_straight`에 해당하는데 모델이 단 한 번도 본 적 없음**. 시연 즉시 실패 가능성이 매우 높음 → D2 데이터 보강에서 최우선 카테고리.

### 권고 (업데이트)
- **즉시 (D2)**: `center_straight` 20개 + `right_right` 14개 = **34개 episode를 17+ frame으로 재수집** — 최대 3 카테고리 신규 수집 한도 안에서 우선
- **OR `data/dataset.py:72` 인덱싱 수정**: `t = max(window_size-1, n-k)`로 clamp해서 짧은 episode도 1 sample 확보 (코드 한 줄, D3 train v2 직전)
- 두 가지 병행 추천: 코드 수정으로 즉시 막고, 데이터 보강으로 카테고리 다양성 확보

### 권고
- **Option A** (간단): `data/dataset.py`에서 누락 episode를 마지막 valid window로 채워 1 sample이라도 확보 (`t = max(window_size-1, n-k)` clamp)
- **Option B** (정석): 데이터 수집 시 episode 최소 길이 17 frame 강제. 짧은 14-frame episode는 보강 수집 시 우선순위
- **Option C** (제안 D2 결합): augmentation의 시간축 jitter ±2가 짧은 episode를 깨트리지 않게 가드 추가

D2의 데이터 보강 시 **누락된 34 episode에 해당하는 두 카테고리(center_straight, right_right)부터 우선 수집**해야 train v2에서 같은 손실이 반복되지 않음.

## 4. **🚨 Sample 단위 random_split으로 episode 누수 가능성**

`training/train.py:88`:
```python
train_ds, val_ds = random_split(full_ds, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))
```

### 문제
- 같은 episode의 다른 window(window 시작 step만 다른 sample들)가 train과 val에 동시에 들어갈 수 있음
- 18-frame episode당 2 sample, 19-frame당 3 sample이므로 누수 가능 episode 비율이 매우 높음
- val=26 sample 중 상당수가 train과 같은 episode에서 왔을 가능성 → **val loss 0.0646이 실제 일반화 성능을 과대평가**할 수 있음

### 권고
- `splits.json`을 episode 단위로 고정 (D1-B 작업)
- train v2 (D3) 부터는 episode-level split 사용
- v1 vs v2 비교 시 v1의 val loss는 sample-split 누수 분이 포함됐다고 명시

## 5. 다음 학습(train v2, D3)에 반영할 사항

| 사항 | v1 | v2 권고 |
|------|-----|---------|
| 정밀도 | BF16 | BF16 유지 (FP16 gradient overflow는 `37c863d`에서 확인) |
| Epoch | 10 | **20** (v1이 cosine 끝까지 갔는데 train loss 추가 감소 여지 있음) |
| LR | 2e-5 cosine | 2e-5 cosine 재시작 (warmup 100 step 동일) |
| Split | sample-level random | **episode-level (`splits.json`)** |
| Dataset 누락 | 34 episode (22.7%) | dataset.py 인덱싱 수정 OR 짧은 episode 재수집 |
| Augmentation | ColorJitter만 | + RandomErasing, 시간축 jitter, Gaussian noise (D2 추가) |
| Val 신뢰성 | 26 sample, 노이즈 큼 | episode-level split 후 ≥30 sample 권장 |

## 6. 산출물 / 파일

- 학습 로그: `logs/train.log` (133KB, full)
- 학습 ckpt: `checkpoints/mona_pi_{best, epoch_05, epoch_10}/`
- 데이터셋: `/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5/` (150 ep, 2,626 frame)
- 본 리포트: `reports/v5_training_audit.md`
- 다음 단계: `splits.json` 생성 (D1-B), `scripts/eval_offline.py` 작성 (D1-C)
