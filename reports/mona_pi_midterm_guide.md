# MoNa-pi 중간발표 편집 가이드
> 2026-05-14 / 총 12슬라이드 / 발표 10분

---

## 전체 구조 한눈에

| # | 슬라이드 제목 | 핵심 메시지 | 사진 필요 | 수정 여부 |
|---|---|---|---|---|
| 1 | 제목 | MoNa-pi 프로젝트 소개 | — | ✅ 완성 |
| 2 | 연구 배경 & 문제 정의 | 어떤 로봇, 왜 어려운가 | 🔴 로봇 정면 + 주행 환경 | ⚠️ 사진 삽입 필요 |
| 3 | 핵심 설계 포인트 & 기대 효과 | 왜 이렇게 설계했는가 + 무엇을 기대하는가 | — | ✅ 완성 |
| 4 | 제안 방법 3가지 전환 | Flow Matching / Chunking / Instruction | — | ✅ 완성 |
| 5 | MoNa-pi 아키텍처 | 전체 파이프라인 | 🟡 다이어그램 교체 권장 | ⚠️ ASCII → 정식 figure |
| 6 | 데이터셋 & 학습 설정 | 150 에피소드, 9 카테고리 | 🔴 카테고리 예시 이미지 3장 | ⚠️ 사진 삽입 필요 |
| 7 | Ablation Study | text-off, FP16 결과 | — | ✅ 완성 |
| 8 | 주요 결과 | v3-A success@1.0=86.7% | — | ✅ 완성 |
| 9 | 카테고리별 분석 | center_left 0% 실패 원인 | — | ✅ 완성 |
| 10 | 추론 서버 검증 (GX10) | 243ms warm, 4Hz 예산 내 | — | ✅ 완성 |
| 11 | 한계 & 토의 | center_left, val≠CL, 실로봇 미배포 | — | ✅ 완성 |
| 12 | 결론 & 향후 계획 | 기여 4가지 + 단기/중기/장기 | — | ✅ 완성 |

---

## 슬라이드별 상세 가이드

---

### Slide 1 — 제목

**핵심 메시지:** 프로젝트 이름 + 부제

**현재 내용:**
- 대형 타이틀: `MoNa-pi`
- 부제: `Flow Matching 기반 고주파 모바일 내비게이션 VLA`
- 날짜: 2026. 05. 14 중간발표
- 팀명: Mobile Navigation π0 Team

**수정 포인트:** 없음

---

### Slide 2 — 연구 배경 및 문제 정의

**핵심 메시지:** "어떤 로봇을 제어하는가 + 왜 어려운가"

**현재 내용 (왼쪽):**
- 3자유도 옴니휠 로봇 소개 (linear_x / linear_y / angular_z)
- 자연어 명령 + Fish-eye 카메라 8프레임 입력
- 목표: 4Hz 재계획 + 50Hz 로컬 제어
- `(사진) 로봇 정면` 플레이스홀더
- `(사진) 주행 환경` 플레이스홀더

**현재 내용 (오른쪽):**
- ① 연속 고주파 제어 (이산 분류 한계)
- ② 행동 일관성 (단발 예측 → jitter)
- ③ 데이터 효율성 (150 에피소드)
- ④ 실시간 배포 요건 (GX10 ↔ Jetson)

**📷 사진 삽입 방법:**
1. 로봇 정면 사진: 플레이스홀더 위에 이미지 삽입 → 크기 맞춤
   - 권장 비율: 세로형 (1:1.1~1:1.3)
   - 권장 크기: 약 **2.55 × 2.85 인치**
2. 주행 환경 사진: 복도/실험실 주행 환경 전경
   - 권장 크기: 약 **2.75 × 2.85 인치**

---

### Slide 3 — 핵심 설계 포인트 & 기대 효과

**핵심 메시지:** "왜 이렇게 설계했고, 무엇을 기대하는가"

**현재 내용 (왼쪽) — 신경 쓴 설계 결정:**

| 항목 | 결정 내용 |
|---|---|
| Flow Matching | 이산 class 대신 연속 경로 생성. CFM으로 ODE 적분 |
| Action Chunking | horizon=10 동시 예측 → 50Hz 로컬 루프 실행 |
| Instruction Pool | 카테고리당 15개 paraphrase, 매 step 무작위 선택 |
| Episode split | train/val 에피소드 엄격 분리 → 데이터 누수 방지 |
| BF16 | Gemma-2B gradient 안정성 + 배포 호환성 확보 |

**현재 내용 (오른쪽) — 예상 기대 효과:**

| 설계 결정 | 기대 효과 |
|---|---|
| Flow Matching | 부드럽고 연속적인 주행 궤적 |
| Action Chunking | 행동 jitter 감소 + 고주파 실로봇 제어 가능 |
| Instruction Pool | 다양한 자연어 표현에 강건한 일반화 |
| Episode split | 신뢰성 있는 평가, 실 배포 시나리오 모사 |
| BF16 | GX10·Jetson 네이티브 지원, 메모리 절반 |

**발표 포인트:**
> "각 설계 결정마다 이유가 있습니다. 연속 경로, 미래 예측, 언어 다양성 —
>  이 세 가지가 합쳐져야 실로봇에서 안정적으로 동작할 것으로 기대합니다."

---

### Slide 4 — 제안 방법: 세 가지 핵심 전환

**핵심 메시지:** "무엇을 어떻게 바꿨는가"

**현재 내용 (카드 3장):**

| 카드 | 전환 | 내용 |
|---|---|---|
| ① | 이산 분류 → Flow Matching | CFM loss, ODE 적분, 연속 경로 생성 |
| ② | 단발 예측 → Action Chunking | horizon=10, 50Hz 로컬 루프 실행 |
| ③ | 고정 문장 → Instruction 다양화 | instruction_pool (카테고리당 15개) |

---

### Slide 5 — MoNa-pi 아키텍처

**핵심 메시지:** "어떻게 만들었는가"

**현재 내용 (왼쪽) — 파이프라인 다이어그램:**
```
[이미지 8프레임] + [자연어 명령]
     │                   │
SigLIP SO400M        Tokenizer
     └──────────┬────────┘
          Gemma-2B LM  →  (B, 64, 2048)
               │
    Action Expert (Cross-Attn Transformer)
    4-layer, 8-head, dim=256
               │
    velocity field  v = x₁ - x₀
               │
    ODE 적분 (n_steps=5)
               │
    [action chunk (B, 10, 3)]
```

**현재 내용 (오른쪽) — 상세:**
- 백본: google/paligemma-3b-pt-224
- Flow Matching: L = ||vθ - (x₁-x₀)||²
- 추론: FastAPI 서버 → Action Chunk Buffer → Jetson ROS2

**🟡 2차 수정 권장:**
- 왼쪽 ASCII 다이어그램을 draw.io / Figma로 만든 정식 figure로 교체
- 권장 크기: **6.5 × 6.1 인치**
- 박스+화살표 형태, 같은 다크 배경 컬러 유지

---

### Slide 6 — 데이터셋 및 수집 파이프라인

**핵심 메시지:** "어떤 데이터로, 어떻게 수집하고, 어떻게 학습했는가"

**현재 내용 (왼쪽 위) — 데이터셋 구성:**
- HDF5, 150 에피소드, 9 카테고리
- Episode-level split: Train 120 / Val 30
- Instruction Pool: 카테고리당 15개, 무작위 선택
- 증강: ColorJitter / RandomErasing / TemporalJitter

**현재 내용 (왼쪽 중간) — 비동기 수집 파이프라인 (Jetson 16GB 기준):**
```
[Camera Thread]  10 Hz  →  frame queue (timestamp)
[Teleop Thread]  50 Hz  →  action queue (timestamp)
       ↓  timestamp 정렬 (±50ms nearest)
Episode Manager  →  HDF5 일괄 저장
```

| 항목 | 값 | 근거 |
|---|---|---|
| 카메라 | 10 Hz | window 0.8s 커버 |
| 액션 수집 | 50 Hz | 제어 루프 동기화 |
| 정렬 방식 | nearest timestamp | 프레임 드랍 안전 |
| 메모리/에피소드 | ~15 MB | JPEG 50KB × 10Hz × 30s |

**현재 내용 (오른쪽 위) — 학습 설정 표:**

| 항목 | 값 |
|---|---|
| window_size | 8 프레임 |
| horizon | 10 액션 |
| batch_size | 4 |
| learning rate | 1e-4 (Cosine+warmup) |
| dtype | BF16 |
| optimizer | AdamW |

**현재 내용 (오른쪽 아래) — 카테고리 예시 사진:**
- `(사진) center` 플레이스홀더
- `(사진) left_*` 플레이스홀더
- `(사진) right_*` 플레이스홀더

**📷 사진 삽입 방법:**
- 각 크기: **1.75 × 1.75 인치**
- HDF5에서 프레임 추출 스니펫 (아래 참조)

**비동기 수집 구현 노트 (Jetson 기준):**
- ROS2 사용 시: `image_transport (10Hz)` + `cmd_vel (50Hz)` → rosbag2 녹화 → 오프라인 HDF5 변환
- 순수 Python 사용 시: `threading.Thread` × 2 + `collections.deque` + `time.monotonic_ns()` 타임스탬프
- 50Hz 액션을 10Hz로 다운샘플 시: 카메라 프레임 timestamp 기준 nearest-neighbor 매칭
- Jetson 16GB에서 1 에피소드(30s) ≈ 15MB → 버퍼 메모리 여유 충분

---

### Slide 7 — Ablation Study

**핵심 메시지:** "Instruction이 유효하고, FP16은 배포 불가"

**현재 내용:**

| 실험 | 설명 | FPE | 결론 |
|---|---|---|---|
| E3 (v2 best) | baseline | 0.673 (green) | 기준 |
| E4 (text-off) | instruction=0 벡터 | 1.085 (red) | text +61% 기여 |
| E6 (FP16) | BF16→FP16 다운캐스트 | 2.093 (red) | 배포 불가 |

- 자동 생성된 바 차트 포함
- 핵심 해석 박스 (오른쪽)

---

### Slide 8 — 주요 결과: 폐루프 시뮬레이션

**핵심 메시지:** "v3-A가 최고 성능 — success@1.0 86.7%"

**현재 내용 (표):**

| 모델 | val loss | FPE ↓ | s@1.0 ↑ | s@1.5 |
|---|---|---|---|---|
| Random | — | 13.49 | — | — |
| v2 best | 0.0619 | 0.857 | ~79% | — |
| v3 (inst. map) | 0.0682 | 0.740 | 76.7% | 93.3% |
| **v3-A ★** | 0.0714 | **0.731** | **86.7%** | **93.3%** |

- FPE 비교 바 차트 + Success@1.0 바 차트 자동 생성
- ⚠️ val loss와 실제 성능 역전 현상 강조 필요

**발표 포인트:**
> "val loss는 v2가 가장 낮지만, 실제 폐루프 성공률은 v3-A가 최고입니다.
>  단순 val loss가 아닌 episode-level 분할과 instruction 다양화가
>  실질 일반화를 개선한다는 것을 보여줍니다."

---

### Slide 9 — 카테고리별 성능 분석

**핵심 메시지:** "center_left는 모든 버전에서 0% — 데이터 문제"

**현재 내용 (표):**

| 카테고리 | n | FPE | s@1.0 |
|---|---|---|---|
| center_straight | 4 | 0.582 | 25% |
| **center_left** | 3 | **1.055** | **0%** ← 모든 버전 공통 |
| center_right | 3 | 1.328 | 33% |
| left_straight | 4 | 0.782 | 25% |
| left_left | 3 | 0.451 | 67% |
| left_right | 3 | 0.747 | 0% |
| right_straight | 4 | 0.650 | 50% |
| right_left | 3 | 0.572 | 33% |
| right_right | 3 | 0.471 | 67% |

- 이중축 카테고리 차트 (FPE cyan / Success amber) 자동 생성

---

### Slide 10 — 추론 서버 검증 (GX10)

**핵심 메시지:** "실제 하드웨어에서 4Hz 예산 내 구동 확인"

**현재 내용:**
- 서버: ASUS ASCENT GX10 (NVIDIA GB10 Superchip)
- GPU Memory: 5.97 GB → Jetson AGX Orin 호환

| 시나리오 | Latency | 상태 |
|---|---|---|
| cold start | 885ms | ❌ |
| warm (평균) | **~243ms** | ✅ 4Hz 예산(250ms) 내 |

- 배포 아키텍처: GX10 서버 → Action Chunk Buffer → Jetson ROS2

---

### Slide 11 — 한계 및 토의

**핵심 메시지:** "정직한 한계 인정 + 원인 분석 + 해결책 제시"

**현재 내용 (카드 3장):**

| 카드 | 제목 | 핵심 |
|---|---|---|
| 🔴 | center_left 0% 실패 | 시각 모호성 + n=3 통계 불확실 → H6 에피소드 수집 |
| 🟡 | val loss ≠ 실제 성능 | episode-split + 다양화가 진짜 일반화 지표 |
| ⬜ | 실로봇 미배포 | GX10 서버·코드 준비됨, Jetson 배포 남음 |

---

### Slide 12 — 결론 및 향후 계획

**핵심 메시지:** "뭘 했고, 다음에 뭘 할 것인가"

**현재 내용 (왼쪽) — 주요 기여:**
1. VLA 파이프라인 구현 (PaliGemma + Action Expert + Flow Matching)
2. 데이터 전략 개선 → success@1.0 +7~10%p
3. 배포 검증 (GX10 243ms, 5.97GB)
4. 실용적 진단 (FP16 불가, threshold 기준, center_left 병목)

**현재 내용 (오른쪽) — 향후 계획:**
- 단기: Jetson 실로봇 배포 + 실환경 테스트 (D10-D11)
- 중기: center_left 에피소드 수집 → v4 재학습 (H6)
- 장기: 실로봇 결과 업데이트, MoNaVLA 비교, 시연 영상

---

## 사진 체크리스트

| 위치 | 슬라이드 | 권장 크기 | 내용 | 파일 |
|---|---|---|---|---|
| 로봇 정면 | 2 (왼쪽 아래 좌) | 2.55 × 2.85 in | 옴니휠 로봇 전면 사진 | — |
| 주행 환경 | 2 (왼쪽 아래 우) | 2.75 × 2.85 in | 복도/실험 환경 전경 | — |
| center 계열 | 6 (왼쪽 아래 좌) | 1.75 × 1.6 in | center_straight 등 1프레임 | HDF5에서 추출 |
| left_* 계열 | 6 (왼쪽 아래 중) | 1.75 × 1.6 in | left_straight 등 1프레임 | HDF5에서 추출 |
| right_* 계열 | 6 (왼쪽 아래 우) | 1.75 × 1.6 in | right_straight 등 1프레임 | HDF5에서 추출 |

### HDF5에서 이미지 추출하는 방법

```python
import h5py, cv2, numpy as np

hdf5_path = "path/to/dataset.hdf5"
with h5py.File(hdf5_path, 'r') as f:
    # 에피소드 목록 확인
    print(list(f.keys()))
    # 첫 번째 에피소드의 첫 프레임
    ep = list(f.keys())[0]
    frame = f[ep]['images'][0]  # (H, W, 3)
    cv2.imwrite('sample_frame.jpg', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
```

---

## 아키텍처 다이어그램 교체 가이드 (Slide 5)

현재 ASCII 텍스트 → 정식 figure로 교체 권장

**draw.io (diagrams.net) 추천 설정:**
- 배경 색: `#0D1B2A`
- 박스 배경: `#1A3456`
- 화살표 색: `#22D3EE` (cyan)
- 텍스트: 흰색
- 폰트: Malgun Gothic 또는 Noto Sans KR

**블록 구성:**
```
[이미지 8프레임] ──┐
                   ├→ [SigLIP SO400M] → [Linear Projector] ──┐
[자연어 명령]  ──→ [Tokenizer] ────────────────────────────→ [Gemma-2B LM]
                                                                    │
                                                         [Action Expert]
                                                     (Cross-Attn Transformer)
                                                                    │
                                                         [Flow Matching ODE]
                                                                    │
                                                    [action chunk (B, 10, 3)]
```

완성된 이미지를 Slide 5 왼쪽 영역 (6.5 × 6.1 인치)에 삽입

---

## 발표 타임 배분 (10분)

| # | 슬라이드 | 시간 | 발표 포인트 |
|---|---|---|---|
| 1 | 제목 | 30s | 팀명, 발표 제목 소개 |
| 2 | 연구 배경 | 1m | 로봇 보여주면서 "이걸 자연어로 제어합니다" |
| 3 | 설계 포인트 & 기대 효과 | 1m | 각 설계 결정의 이유 + 기대하는 것 설명 |
| 4 | 제안 방법 | 1m | 3가지 전환 카드 설명 |
| 5 | 아키텍처 | 1.5m | 데이터 흐름 따라가며 설명 |
| 6 | 데이터셋 | 1m | 카테고리 사진 보여주면서 설명 |
| 7 | Ablation | 1m | 차트 보면서 "instruction 있어야 잘 된다" |
| 8 | 주요 결과 | 1.5m | v3-A 86.7% 강조, val loss 역전 현상 |
| 9~10 | 카테고리 + 서버 | 1m | center_left 문제 + 4Hz 동작 확인 |
| 11~12 | 한계 + 결론 | 30s | 정직한 한계 인정 + 앞으로 할 일 |
