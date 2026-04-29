# 주간 미팅 보고 — 2026-04-28

## 지난 미팅 이후 진행 사항

- `full_seq_cond` 플래그 추가: Gemma 전체 시퀀스(시각+언어 토큰)를 Action Expert에 전달하는 π0 방식 완성
- `inference/server.py`에 `--mock` 플래그 추가 (모델 없이 서버 동작 테스트 가능)
- `flow_head` alias 제거 및 `action_expert` 명칭으로 통일 (아키텍처 명확화)

---

## 추론 서버 환경 구축 — ASUS ASCENT GX10

로컬 추론 전용 서버로 ASUS ASCENT GX10 세팅 완료.

| 항목 | 사양 |
|------|------|
| **칩셋** | NVIDIA GB10 Superchip (Grace Blackwell) |
| **AI 성능** | 1 PFLOP (FP4) |
| **메모리** | 128 GB LPDDR5x 통합 메모리 (CPU+GPU 공유) |
| **CPU** | NVIDIA Grace (ARM v9.2-A, 20코어) |
| **인터커넥트** | NVLink-C2C (PCIe 5.0 대비 5배 대역폭) |
| **스토리지** | 2 TB NVMe PCIe 4.0 |
| **OS** | NVIDIA DGX OS (Ubuntu 기반) |
| **소비전력** | 240 W |

**환경 세팅 절차**

1. NVIDIA DGX OS 기본 설치 (CUDA, PyTorch 사전 탑재)
2. 프로젝트 클론 및 의존성 설치 (`pip install -r requirements.txt`)
3. HuggingFace 캐시 경로 설정 — SigLIP SO400M, Gemma-2B 가중치 로컬 캐싱
4. `inference/server.py` 서비스 등록 및 포트 개방 (`8000`)
5. `tests/test_model_loss.py` 실행으로 GPU 연산 정상 동작 확인

**GB10 통합 메모리의 이점**: CPU·GPU가 128GB 메모리 풀을 공유하므로 SigLIP+Gemma-2B 전체를 VRAM 제한 없이 단일 프로세스에서 로드 가능. 기존 이산 GPU 환경에서 발생하던 호스트↔디바이스 복사 병목이 없음.

---

## 로컬 모델 로드 및 이미지 추론 테스트

교수님 요청에 따라 ASCENT GX10 위에서 모델 로드 및 실제 이미지 입력 테스트 수행.

**테스트 구성**
- 백본: SigLIP SO400M (384×384) + Gemma-2B, 로컬 캐시 사용 (`load_pretrained_paligemma=False`)
- 입력: 이미지 window 8프레임 (384×384), 자연어 명령어
- 추론: Heun ODE 솔버, 5스텝

**결과**
- 모델 FP16 로드 성공 (SigLIP + Gemma-2B + Action Expert 전체)
- 이미지 입력 → VLM 피처 추출 → Flow Matching 액션 샘플링 전 파이프라인 동작 확인
- 출력 shape `(B, 10, 3)` — horizon 10스텝, 3-DOF (linear_x, linear_y, angular_z) 정상 반환
- Flow Matching Loss 계산 및 역전파 1-iter 확인 (loss ≈ 2.07)
- 비전 피처 캐싱(TTL 0.5s) 동작 확인 → 동일 이미지 재요청 시 캐시 히트

**현재 한계**
- 사전학습 가중치 미적용 (랜덤 초기화) → 액션 출력값 자체는 아직 의미 없음
- 실 데이터셋 학습 미실시

---

## 다음 목표

- `mobile_vla_dataset_v3` 실 데이터셋으로 학습 루프 실행
- 사전학습 가중치(PaliGemma 3B) 로드 후 fine-tuning 시작
- 로봇 실기동 테스트를 위한 Jetson ↔ ASCENT GX10 서버 통신 구성
