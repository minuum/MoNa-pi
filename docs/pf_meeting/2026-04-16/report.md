# 주간 미팅 보고 — 2026-04-16

## 진행 사항

### 설계 확정 및 문서화 (4/13)
- 프로젝트 제안서 및 설계 제안서 v1 작성 (`docs/MoNa-pi_Design_Proposal_v1.md`)
- 로컬 모델 로드 테스트 스크립트 초안 작성 (`scripts/test_loading.py`)

### 엔드투엔드 파이프라인 구현 (4/15)

**백본 교체 — 실제 π0 아키텍처 적용**
- 기존 Placeholder → **SigLIP (비전) + Gemma-2B (언어)** 조합으로 교체
- `models/backbones/paligemma_backbone.py`: SigLIP → Linear Projector(1152→2048) → Gemma 인코더
- `load_pretrained_paligemma` 플래그로 로컬 캐시 vs HuggingFace 다운로드 전환 가능
- Action Expert(`models/heads/action_expert.py`): VLM 전체 토큰 시퀀스를 컨디션으로 사용 (π0 방식)

**데이터 파이프라인 강화**
- `data/preprocessing.py`: Savitzky-Golay 액션 스무딩, 다국어(KO/EN) 인텐트 프리픽스, Counterfactual 주입
- 데이터 증강: HFlip (좌우 반전 시 `angular_z` 부호 + 명령어 방향 동시 반전), ColorJitter
- HDF5 v5/v6 자동 감지 및 하드코딩 경로 제거

**학습/추론 인프라**
- `configs/train.yaml`: 전체 하이퍼파라미터 통합 설정
- `training/train.py`: 검증 루프, Cosine+Warmup LR 스케줄러, W&B/TensorBoard 로깅, Top-k 체크포인트 관리
- `inference/ode_solver.py`: Euler / Heun / DPM 솔버 구현
- `inference/engine.py`: 비전 피처 캐싱(TTL 0.5s), FP16 지원
- `inference/server.py`: FastAPI REST 서버 (`POST /predict`, `GET /health`, `/metrics`)

**검증**
- dtype/API 수정 및 1-iter 학습 루프 실행 확인 (loss=2.07)
- `tests/test_integration.py`: ActionChunkBuffer + mock 서버 통합 테스트 추가

## 다음 목표

- 로컬에서 실제 모델 로드 및 이미지 입력 테스트
- 실 데이터셋으로 학습 루프 실행
