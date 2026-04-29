# 주간 미팅 보고 — 2026-04-02

## 진행 사항

- **프로젝트 기반 구축 완료** (커밋: `a0bb336`, 2026-03-17)
  - π0(Physical Intelligence) 논문 스터디 및 아키텍처 분석 문서 작성 (`docs/pi0_study.md`, `mona_pi_impl_design.md`)
  - Flow Matching Action Head 스켈레톤 구현 (`models/heads/flow_head.py`)
    - Sinusoidal Time Embedding, Cross-Attention 기반 Transformer 구조
    - Conditional Flow Matching Loss (`x_t = (1-t)x_0 + tx_1`, target: `v = x_1 - x_0`)
  - HDF5 데이터셋 로더 초안 (`data/dataset.py`)
    - Action Chunking: window 8프레임 입력 → horizon 10 액션 예측
    - 3-DOF 옴니휠 액션 정규화 (`[-1.15, 1.15] → [-1, 1]`)
  - 학습 루프 초안 (`training/train.py`, HuggingFace Accelerate 기반)

## 다음 목표

- 모델 통합 설계 확정 (백본 선택: Kosmos-2 vs SigLIP+Gemma)
- 엔드투엔드 학습 파이프라인 완성
- 로컬 실행 가능한 테스트 스크립트 작성
