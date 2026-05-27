# MoNa-pi 개발 상태 문서

**최종 업데이트**: 2026-05-28  
**작성 목적**: 다른 서버·다른 세션에서 현재 상태를 즉시 파악하고 이어서 작업하기 위한 핸드오버 문서

---

## 1. 브랜치 구조

```
origin/
├── master            ← GX10 학습 서버 기준선. 평가 결과·데이터 모듈 정비
├── monapi-train      ← AdaLN-Zero 아키텍처 브랜치 (다른 서버에서 작업)
├── monapi-driving    ← Jetson AGX 추론 서버 전용 (monapi-train과 공유 커밋)
└── feat/inference-server  ← (로컬 GX10) inference/ 3파일만 분리한 임시 브랜치
```

### 브랜치별 역할

| 브랜치 | 서버 | 용도 |
|---|---|---|
| `master` | GX10 (ASCENT GB10) | 학습·평가·실험 결과 관리 |
| `monapi-train` | GPU 학습 서버 | π0 정통 아키텍처(AdaLN-Zero) 개발 |
| `monapi-driving` | Jetson AGX | 실기기 주행·추론 서버 |

---

## 2. 각 브랜치 핵심 변경 내용

### master (GX10 기준)

최근 커밋 순서:
```
9e66f08  Add meeting reports (Apr-May) and update paper draft
a1949a3  Add experiment results: ablation JSONs, CL eval, server latency, attention maps
31e5894  Add eval and utility scripts: offline FPE, closed-loop sim, grad flow, splits builder
26c8a5a  Training v2: BF16, episode split, instruction map, num_workers=4, epochs=20
8c7a5c1  Dataset v5: episode-level split, instruction pool, augmentations, thread-local h5 cache
37c863d  Switch training to BF16 to fix FP16 gradient overflow   ← monapi-train 분기점
```

**master에만 있는 것:**
- `data/augmentations.py` — ColorJitter + temporal jitter
- `data/instruction_map.json` / `instruction_pool.json` — 9카테고리 × 15 paraphrase
- `splits.json` / `splits.v2.json` — 에피소드 레벨 분할 (train=92, val=24)
- `scripts/eval_offline.py` / `eval_closed_loop.py` — FPE·Success@T 평가
- `reports/E*.json` / `H*.json` / `CL*.json` — 전체 실험 결과
- 평가 결과 요약 → §4 참고

### monapi-train (다른 서버에서 작업됨)

```
044c077  docs: add README_BRANCHES.md manual
144d3b1  feat(driving): implement AdaLN-Zero, MoNaActionExpert normalization
e450225  fix: replace flow_head alias with action_expert throughout
4ef45ad  feat(inference): wire PaliGemma params into engine & server
5b77ad0  feat(arch): replace backbone with PaliGemma + Action Expert (real π0)
```

**monapi-train에만 있는 것:**
- `models/heads/flow_head.py` 전체 재작성 — **AdaLN-Zero** 방식
- `models/heads/mona_action_expert.py` 신규 — 정규화 래퍼 (`normalize`/`unnormalize`)
- `data/preprocessing.py` 신규 — `EpisodePreprocessor`, `ActionNormalizer`, v5/v3 포맷 자동 감지
- `data/dataset.py` 대폭 수정 — `data/augmentations.py` 대신 `preprocessing.py` 사용
- `scripts/verify_mona_expert.py` — 정규화 수치 정합성 검증
- `research.md` / `plan.md` — 아키텍처 분석 및 구현 계획
- `configs/train.yaml` 보강
- `README_BRANCHES.md` — 브랜치 운영 가이드

---

## 3. 아키텍처 변화 요약 (master → monapi-train)

### Flow Head: 단순 덧셈 → AdaLN-Zero

**master (구버전):**
```python
t_emb = time_mlp(t)           # (B, 1, 512)
h = action_proj(x_t) + t_emb  # 시간 임베딩을 그냥 더함
# self-attn: query만 norm, key/value는 raw
```

**monapi-train (π0 정통):**
```python
cond_emb = TimestepEmbedder(t)          # (B, 512)
# 레이어마다 AdaLN modulation
(α1, β1, γ1, α2, β2, γ2) = AdaLNModulation(cond_emb)
h = h + γ1 * self_attn(norm1(h) * (1+α1) + β1)   # scale+shift+gate
h = h + cross_attn(norm2(h), vlm_cond)              # VLM cross-attn (timestep 미적용)
h = h + γ2 * mlp(norm3(h) * (1+α2) + β2)          # scale+shift+gate
```

Zero-init → 학습 초기 모든 modulation이 identity → 안정적 수렴 시작

### 정규화 위치 이동

| | master | monapi-train |
|---|---|---|
| 정규화 위치 | `dataset.py` (ActionNormalizer) | `MoNaActionExpert.get_loss()` 내부 |
| dataset 출력 | 정규화된 값 | RAW 물리값 (±1.15) |
| configs | `data.normalize: true` | `data.normalize: false` |

---

## 4. 평가 결과 (GX10/master 기준)

### Offline 평가 (53 samples)

| 체크포인트 | FPE↓ | MSE↓ |
|---|---|---|
| random_init | 13.49 | 1.485 |
| mona_pi_best_fp16 | 2.09 | 0.074 |
| **mona_pi_best BF16** | **0.673** | **0.056** |
| 텍스트 없음 (ablation) | 1.085 | 0.060 |

### Closed-Loop 평가 (24 episodes, threshold=0.5)

| 모델 | FPE 평균 | Success@0.5 |
|---|---|---|
| BF16 best | 0.823 | **45.8%** |
| FP16 | 2.286 | 0% |

### 카테고리별 약점 (CL 기준)

- 잘 됨: `left_left` 100%, `right_right` 100%, `right_left` 67%, `left_right` 67%
- 약함: `center_left` **0%**, `center_right` 33%, `left/right_straight` 25%

### Success@T Sweep (mona_pi_best, 24 episodes)

| T=0.5 | T=1.0 | T=1.5 | T=2.0 |
|---|---|---|---|
| 41.7% | 79.2% | 83.3% | 91.7% |

### 서버 레이턴시 (GX10, D9)

- 첫 요청 (JIT cold): 885ms
- 웜 요청 평균: **~240ms** (4Hz 예산 250ms 충족)

---

## 5. 체크포인트

```
checkpoints/
├── v3a/mona_pi_best   ← 현재 최적 (epoch 14, val loss 0.0714)
├── v3/mona_pi_best
├── mona_pi_best       ← 메인 베스트 (BF16)
├── mona_pi_best_fp16
├── mona_pi_epoch_05/10/15/20
└── random_init
```

---

## 6. 서버별 빠른 시작

### GX10 학습 서버 (master)

```bash
git checkout master
git pull origin master

# 평가 실행
python scripts/eval_offline.py --ckpt checkpoints/mona_pi_best
python scripts/eval_closed_loop.py --ckpt checkpoints/mona_pi_best

# 학습 재개
python training/train.py
```

### monapi-train 브랜치로 전환 (AdaLN-Zero 아키텍처)

```bash
git fetch origin
git checkout monapi-train

# 아키텍처 검증
python scripts/verify_mona_expert.py
# 결과: Loss: ~1.09, Sampled actions shape: (2, 10, 3), ✅ Logic Verification Success!

# 학습
python training/train.py --config configs/train.yaml
# configs/train.yaml 에서 data.normalize: false 확인
```

### monapi-driving (Jetson AGX)

```bash
git checkout monapi-driving
python inference/server.py --ckpt checkpoints/best --port 8080
# ROS2 환경:
source /opt/ros/humble/setup.bash
python robot/ros2_controller.py --server-url http://localhost:8080
```

---

## 7. master ↔ monapi-train 동기화 주의사항

두 브랜치가 **같은 파일을 다르게 수정**해서 직접 merge 시 충돌 발생:

| 파일 | 충돌 여부 | 비고 |
|---|---|---|
| `data/dataset.py` | **충돌** | master=augmentations 기반, monapi-train=preprocessing 기반 |
| `data/augmentations.py` | **충돌** | master에만 존재 (monapi-train에서 삭제) |
| `training/train.py` | 충돌 가능 | 양쪽 수정됨 |
| `models/heads/flow_head.py` | **충돌** | monapi-train에서 전체 재작성 |

**권장 전략**: monapi-train 아키텍처를 기준으로 잡고, master의 실험 결과 JSON과 scripts만 cherry-pick

```bash
# monapi-train을 베이스로, master의 eval scripts·결과만 가져오기
git checkout monapi-train
git cherry-pick 31e5894  # eval scripts
git cherry-pick a1949a3  # experiment results JSONs
```

---

## 8. 다음 작업 목록

- [ ] monapi-train에서 BF16 학습 재실행 (AdaLN-Zero 적용 후 첫 학습)
- [ ] v3a best vs AdaLN-Zero 모델 성능 비교
- [ ] center_left/right 에피소드 추가 수집 → 재학습
- [ ] Jetson AGX 실기기 배포 (monapi-driving)
- [ ] async 데이터 수집 파이프라인 (Camera 10Hz + Teleop 50Hz 분리)
- [ ] 논문 실험 결과 섹션 완성
