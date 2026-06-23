# MoNaVLA 연구 결과 → MoNa-pi 적용 가능성 분석

> 작성: 2026-06-23 (minum 서버, MoNaVLA 세션에서 작성)
> 목적: MoNaVLA에서 최근 발견한 파이프라인 버그/교훈을 MoNa-pi에도 동일한 클래스의 문제가 있는지 점검.
> 방법: 가정으로 옮기지 않고, 실제 코드(`monapi-train`/`monapi-driving` 양 브랜치)를 읽어 검증된 것만 적용.

---

## 1. 적용함 — train/inference 리사이즈 보간 불일치

**MoNaVLA에서 발견된 클래스의 버그**: 학습 시 이미지 리사이즈와 추론 시 리사이즈가 암묵적으로 다른 보간법을 쓸 수 있음(`Image.resize()`에 resample을 명시 안 하면 Pillow 버전별 기본값에 의존).

**MoNa-pi에서 확인된 사실**:

| | 파일 | 코드 |
|---|---|---|
| 학습 | `data/dataset.py` (monapi-train) | `PILImage.fromarray(img).resize((self.image_size, self.image_size), PILImage.BILINEAR)` — **BILINEAR 명시** |
| 추론 | `inference/engine.py` (monapi-driving) | `PILImage.fromarray(image).resize((self.IMG_SIZE, self.IMG_SIZE))` — **보간법 미지정 → Pillow 기본값(최신 버전 BICUBIC)** |

→ 학습은 BILINEAR, 추론은 (버전에 따라) BICUBIC일 수 있어 train/inference mismatch 가능성 확인됨.

**조치**: `inference/engine.py:181`을 학습과 동일하게 `PILImage.BILINEAR` 명시로 수정함 (이미 적용, 커밋 전).

```diff
- pil = PILImage.fromarray(image).resize((self.IMG_SIZE, self.IMG_SIZE))
+ pil = PILImage.fromarray(image).resize((self.IMG_SIZE, self.IMG_SIZE), PILImage.BILINEAR)
```

색상 정규화(`/255.0`, CLIP mean/std 미적용)는 양쪽 동일하게 raw 0~1 스케일이라 문제 없음.

---

## 2. 점검했으나 문제 없음 — BGR/RGB 색공간

MoNaVLA에서 데이터 수집기(BGR8 디코드)와 추론 노드(BGR→RGB 변환) 간 불일치 가능성을 의심했다가 픽셀 통계로 기각한 사례(CH45-1)와 같은 클래스 점검.

**MoNa-pi 확인**:
- `robot/camera_node.py:115` — `cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)` 후 `rgb8`로 publish
- `collect/ros2_recorder.py:79` — `imgmsg_to_cv2(msg, desired_encoding="rgb8")`
- `robot/ros2_controller.py:256` — 동일하게 `rgb8` 요청

→ 수집·추론 전체 경로가 일관되게 RGB로 통일돼 있음. **문제 없음, 조치 불필요.**

---

## 3. 검증 못함 — 적용 보류 (각자 점검 필요)

| MoNaVLA 발견 | MoNa-pi에 적용 가능한가 | 비고 |
|---|---|---|
| **text attention 거의 0%** (Google-robot post-train Kosmos-2에서) | ❓ 미검증 | MoNa-pi는 base PaliGemma(`google/paligemma-3b-pt-224`) + 별도 flow-matching action expert로 cross-attn 조건화(`pi0_core.py forward_backbone`) — 아키텍처가 완전히 다르고, MoNaVLA의 원인은 특정 체크포인트(Google-robot)의 post-training 손상이었음. **그대로 전이 가정 불가** — 직접 attention 측정 스크립트로 따로 검증해야 함 (MoNaVLA의 `scripts/measure_attention.py` 패턴 참고 가능) |
| **stale annotation/feature 캐시가 최신 모델과 불일치** (bbox_dataset_full.json이 5월 구 모델로 생성된 채 6월 신모델 실험에 계속 쓰인 사례) | ⚠️ 일반 원칙으로 점검 권장 | MoNa-pi도 `checkpoints/` 여러 버전(v3/v3a/메인)과 `splits.json`/`instruction_pool.json` 등 사전 계산 파일을 쓰는 구조라, **새 체크포인트/아키텍처(monapi-train AdaLN-Zero)로 바뀔 때마다 캐시 파일들이 최신 가정과 맞는지** 재확인하는 습관 필요. 자동 검증 스크립트는 만들지 않음(우리가 임의로 만들면 두 프로젝트 결합도가 너무 높아짐) — MoNa-pi 쪽에서 필요시 요청 |
| **단일 run 헤드라인 수치가 noise-high였던 사례** (CH43, 96.85%→실제 95.39%±0.20%p) | ⚠️ 권장 사항 | DEVSTATE.md의 CL Success@0.5=45.8%, Success@T sweep 등이 전부 단일 run으로 보임. 헤드라인으로 쓰기 전 핵심 수치 1~2개만 다른 seed로 재현해보는 걸 권장 |
| **center/straight 경로군 약점** | 🔍 흥미로운 평행 현상, 원인 불명 | MoNa-pi: `center_left 0%`, `center_right 33%`, `*_straight 25%` / MoNaVLA(MONAPI_HANDOFF 문서): `center_straight 경로 0% 병목`. 두 프로젝트가 **다른 아키텍처(이산분류 vs flow matching)인데도 같은 경로 유형에서 약함** — 데이터 부족(중앙 시작 에피소드가 절대적으로 적음, MoNaVLA 쪽도 73개 추가분 중 center_straight는 1개뿐이었음) 같은 **데이터 분포 문제일 가능성**. 모델 구조 문제가 아니라 수집 우선순위 문제일 수 있다는 가설만 제시 — 검증/조치는 각 프로젝트가 따로 진행 |

---

## 4. 조치 요약

- ✅ **적용**: `inference/engine.py` 리사이즈 보간 명시 (1줄, 커밋 예정)
- ✅ **점검 완료, 문제 없음**: BGR/RGB
- ❓ **보류**: text attention, stale-cache 일반원칙, 5-seed 재현 습관, center/straight 약점 — 근거가 MoNa-pi 자체 코드로 직접 검증되지 않았거나 두 프로젝트 결합도를 과하게 높일 수 있어 **분석만 남기고 자동 조치하지 않음**
