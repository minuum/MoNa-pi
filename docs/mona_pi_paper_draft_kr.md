# MoNa-pi: Flow Matching 기반 고주파 모바일 내비게이션 VLA

> 본 문서는 `MoNa-pi`와 인접 레포(`../MoNaVLA`)의 **로컬 문서/코드만**을 근거로 작성된 논문 초안이다. 외부 참고문헌은 포함하지 않으며, 외부 인용이 필요한 부분은 `[TODO: citation]`으로 표기한다.

---

## 초록

본 연구는 모바일 로봇의 자연어 기반 장애물 회피 주행을 위한 VLA(vision-language-action) 시스템을, 기존 MoNaVLA의 분류/회귀 기반 액션 헤드에서 **Flow Matching 기반 연속 액션 생성**으로 확장하는 방법을 제안한다. MoNaVLA는 Kosmos-2 백본에 LoRA를 적용하고 LSTM 정책 헤드를 통해 이산 9‑class 또는 2D 연속 속도 출력을 수행하며, 오프라인 성능은 높으나 타이밍 암기 문제로 인한 인과적 시각 이해 부족이 확인되었다. 본 연구는 이러한 한계를 해결하기 위해 (1) 연속 액션 경로 생성, (2) 고주파 제어를 위한 action chunking, (3) 데이터 다양화 기반의 인과 혼란 완화 전략을 통합한 Pi0‑VLA 설계를 제시한다. 본 초안은 모델 구조, 학습 목표, 추론/서빙 설계를 MoNaVLA 대비 체계적으로 정리하고, 검증 실험 계획과 평가 지표를 제안한다.

---

## 1. 서론

모바일 로봇 내비게이션에서 VLA는 **자연어 지시를 시각 입력과 결합해 즉시 행동으로 변환**하는 것이 핵심 목표다. MoNaVLA는 Kosmos‑2 VLM과 LoRA, LSTM 정책 헤드를 통해 장애물 회피 주행을 구현했으며, 오프라인 정답 일치율(Perfect Match, Direction Match) 100%를 달성한 사례가 보고되었다. 그러나 이는 실제 시각적 이해가 아니라 **타이밍 암기**에 의해 형성될 수 있음을 분석 문서에서 확인했다. 따라서 실제 환경에서의 일반화 성능을 확보하려면 **행동 표현과 학습 목표** 자체를 재설계해야 한다.

본 연구는 MoNaVLA의 설계를 기반으로, **Flow Matching 기반 연속 액션 생성**과 **고주파 제어용 action chunking**을 통합한 Pi0‑VLA 설계를 제시한다. 목표는 (1) 연속 제어의 표현력 확보, (2) 고주파 제어 안정성, (3) 인과적 시각‑행동 연결 강화이다.

---

## 2. 배경 및 베이스라인: MoNaVLA

### 2.1 입력/출력 정의

- 입력: 720×1280 RGB(Fisheye) 이미지 + 자연어 명령
- 출력: README 기준 9‑class 이산 행동, 코드에는 2D 연속 속도(LSTM 회귀) 또는 분류 헤드가 공존

근거: `../MoNaVLA/README.md`, `../MoNaVLA/robovlm_nav/models/policy_head/nav_policy_impl.py`

### 2.2 모델 구조

MoNaVLA는 Kosmos‑2 백본을 사용하고 LoRA로 시각‑언어 그라운딩을 강화하며, `[LRN]` 액션 토큰의 hidden state를 추출해 정책 헤드로 전달한다. 정책 헤드는 LSTM 기반이며, window size 및 fwd_pred_next_n(미래 N‑step 예측)을 사용한다.

근거: `../MoNaVLA/README.md`, `../MoNaVLA/docs/PROJECT_STRUCTURE_AND_ARCH_20251120.md`, `../MoNaVLA/README_INFERENCE.md`

### 2.3 정책 헤드 구현(코드 기준)

- `MobileVLALSTMDecoder`: 2D 속도(Linear X/Y) 예측, Huber loss
- `MobileVLAClassificationDecoder`: 클래스 분류(logits) + class‑weighted CE loss
- `HybridActionHead`: 방향 분류 + 크기 회귀

근거: `../MoNaVLA/robovlm_nav/models/policy_head/nav_policy_impl.py`, `../MoNaVLA/robovlm_nav/models/policy_head/hybrid_action_head.py`

### 2.4 관찰된 한계

- 오프라인 지표 100% 수렴이 **타이밍 암기**에 의해 발생할 수 있음
- Dataset v3에서는 동일 time step에 상이한 시각 맥락을 부여하는 변형(Close/Far/Offset/No‑obstacle)으로 인과 혼란을 완화하려는 계획이 제시됨

근거: `../MoNaVLA/README.md`

---

## 3. 제안: Pi0‑VLA의 목표와 차별점

Pi0‑VLA는 기존 분류/회귀 중심의 액션 헤드를 **Flow Matching(또는 Diffusion) 기반 연속 액션 생성**으로 전환하고, 멀티스텝 액션 청킹을 통해 **고주파(50Hz+) 제어**를 목표로 한다. 필요 시 미래 시각 토큰까지 예측하는 **Unified World Model**을 옵션으로 포함한다.

근거: `README.md`

### 3.1 핵심 전환점

1. 행동 표현: 이산 분류 → 연속 경로 생성
2. 정책 헤드: LSTM 예측 → Flow Matching 생성
3. 제어 주기: 저주파 → 고주파 action chunking
4. 데이터 전략: 타이밍 암기 방지 데이터 설계 내재화

---

## 4. 방법

### 4.1 전체 아키텍처 개요

**(제안)** VLM 백본(Kosmos‑2 계열 유지 가능) 위에 Flow Matching action head를 탑재하고, 멀티스텝 chunk를 생성하도록 설계한다. Perceiver Resampler/Visual Tokens 등 기존 설계 철학은 유지하되, 정책 헤드와 손실 함수를 교체한다.

근거(목표): `README.md`

### 4.1.1 입력 및 조건부 구조

**(제안)** 입력은 기존 MoNaVLA와 동일하게 이미지와 자연어 명령을 사용한다. 액션 생성기는 VLM의 출력 임베딩(또는 `[LRN]` 토큰 hidden state)에 조건부로 연결되며, chunk 길이와 예측 간격을 명시적 하이퍼파라미터로 둔다. 이는 기존 LSTM 정책 헤드의 시퀀스 처리 방식을 **연속 생성 문제**로 재정의하는 과정이다.

근거(기존 조건부 구조): `../MoNaVLA/docs/PROJECT_STRUCTURE_AND_ARCH_20251120.md`

### 4.2 Flow Matching 기반 연속 액션 생성

**(제안)** Action sequence를 연속 확률 경로로 모델링하고, Flow Matching 목적 함수로 학습한다. 이는 분류/회귀 기반 LSTM 헤드보다 **연속 제어의 표현력**을 높이며, 고주파 제어 시 발생하는 불연속성을 완화할 수 있다.

근거(목표): `README.md`

### 4.2.1 액션 표현 및 스케일링

**(제안)** MoNaVLA의 2D 속도 표현(Linear X/Y)을 유지하되, 연속 경로 생성에 맞게 스케일링과 정규화를 통일한다. 기존 Huber/CE 기반 손실에서 Flow Matching 기반 손실로 전환할 때, 액션 분포의 범위를 명확히 정의해야 한다.

근거(기존 액션 표현): `../MoNaVLA/robovlm_nav/models/policy_head/nav_policy_impl.py`

### 4.2.2 손실 함수와 학습 목표

**(제안)** Flow Matching 목적 함수로 연속 경로를 학습하되, MoNaVLA가 사용하던 분류/회귀 손실과 직접 비교 가능한 보조 지표(예: chunk 단위 MSE, 방향 일치율)를 함께 기록한다. 이렇게 하면 새로운 손실의 학습 안정성을 기존 체계와 연결해 검증할 수 있다.

근거(기존 손실 체계): `../MoNaVLA/robovlm_nav/models/policy_head/nav_policy_impl.py`

### 4.3 Action Chunking (Multi‑step Prediction)

**(제안)** MoNaVLA의 fwd_pred_next_n 개념을 확장해 16~50 step 이상의 액션 청킹을 예측한다. 이를 통해 컨트롤 주기를 50Hz+로 끌어올리는 것을 목표로 한다.

근거(목표): `README.md`

### 4.3.1 Chunk 스케줄링

**(제안)** 고주파 제어를 위해 chunk 길이 N과 재계획 주기 M을 분리한다. 즉, N step을 한 번에 생성하되, M step마다 새 chunk를 재생성해 폐루프 안정성을 확보한다. MoNaVLA의 LSTM 히스토리 메모리 구조는 이러한 재계획 설계의 비교 기준으로 활용할 수 있다.

근거(기존 히스토리 구조): `../MoNaVLA/robovlm_nav/models/policy_head/nav_policy_impl.py`

### 4.4 데이터 설계 및 인과 혼란 완화

**(제안)** MoNaVLA에서 보고된 타이밍 암기 문제를 해결하기 위해, 동일 step에 서로 다른 시각 맥락을 제공하는 데이터 구성(Close/Far/Offset/No‑obstacle)을 Pi0‑VLA 학습에도 기본 전제로 포함한다.

근거: `../MoNaVLA/README.md`

### 4.4.1 데이터 분할 및 검증 규약

**(제안)** 동일 에피소드 기반 편향을 완화하기 위해, split 기준을 에피소드 단위로 유지하고, 동일 시간 index를 공유하는 변형 샘플을 서로 다른 split에 동시에 넣지 않도록 규약을 명시한다. 이를 통해 타이밍 암기의 부정적 영향을 줄인다.

근거(문제 정의): `../MoNaVLA/README.md`

### 4.5 추론/서빙 설계

**(제안)** 기존 API 기반 추론 구조는 유지 가능하나, 고주파 제어를 위해 **action chunk buffer** 및 **재계획 주기**를 재설계해야 한다. 기존 INT8 추론 구조를 활용할 수 있으나 목표 주기(50Hz+)를 만족하는지 검증이 필요하다.

근거(기존 서빙 구조): `../MoNaVLA/README.md`, `../MoNaVLA/README_INFERENCE.md`

### 4.5.1 고주파 제어 경로

**(제안)** 서버‑클라이언트 구조에서 네트워크 왕복이 병목이 될 수 있으므로, 제어 루프를 분리한다. 즉, 서버는 chunk 생성만 담당하고, Jetson 측에서 chunk를 버퍼링하며 로컬 제어 주기를 유지한다. 이는 MoNaVLA의 2Hz 추론 구조와 명확히 구분되는 설계 포인트다.

근거(기존 2Hz 구조): `../MoNaVLA/README_INFERENCE.md`

---

## 5. 실험 설계(제안)

### 5.1 비교 대상

- Baseline: MoNaVLA V3‑EXP08 (분류/LoRA/LSTM)
- Proposed: Pi0‑VLA (Flow Matching + action chunking)

근거: `../MoNaVLA/README.md`, `README.md`

### 5.2 비교 축

- 행동 표현: 이산 분류 vs 연속 생성
- 예측 길이: 짧은 chunk vs 긴 chunk (고주파)
- 데이터: 기존 v2 데이터 vs v3 다양화 데이터

근거: `../MoNaVLA/README.md`, `README.md`

### 5.3 평가 지표(제안)

- 기존 PM/DM 지표 유지
- 실제 주행 성공률(충돌률, 목표 도달률)
- 제어 안정성(속도 변화율, 커맨드 지터)
- 인과성 평가: 동일 step에서 시각 변화에 대한 행동 민감도

### 5.4 실험 프로토콜(제안)

**(제안)** 아래 프로토콜로 비교 실험을 구성한다.

1. 오프라인: 동일 데이터셋에서 MoNaVLA(V3‑EXP08)와 Pi0‑VLA를 동일 평가 지표로 비교
2. 시각 다양성: Dataset v3 변형에서 타이밍 암기 민감도 측정
3. 실로봇: 동일 경로/장애물 구성에서 성공률 및 충돌률 측정
4. 주기성 평가: 50Hz+ 제어 유지 여부와 커맨드 지터 측정
근거(데이터 변형 문제): `../MoNaVLA/README.md`

---

## 6. 논의 및 한계

- MoNaVLA는 오프라인 지표가 높더라도 타이밍 암기 문제가 존재했으며, Pi0‑VLA에서도 데이터 설계 없이는 동일 문제가 재현될 가능성이 높다.
- 고주파 제어(50Hz+)는 모델 구조뿐 아니라 **서빙/추론 파이프라인**의 재설계를 요구한다.
- Flow Matching 기반 헤드가 실제 모바일 로봇 환경에서 안정적으로 작동하는지 실증이 필요하다.

근거: `../MoNaVLA/README.md`, `README.md`

---

## 7. 결론

본 초안은 MoNaVLA 기반 모바일 내비게이션 VLA를 Flow Matching 기반 연속 액션 생성 모델로 확장하는 Pi0‑VLA 설계를 제안했다. 핵심은 행동 표현의 연속화, 고주파 action chunking, 인과 혼란 완화 데이터 설계이며, 이는 기존 MoNaVLA의 타이밍 암기 한계를 직접적으로 겨냥한다. 이후 단계는 구현 검증, 고주파 서빙 최적화, 실제 로봇 환경에서의 성능 평가이다.

---

## 8. 구현 계획 (실행 지침)

이 섹션은 논문 방법을 실제 코드로 옮기기 위한 실행 계획이다. 각 단계는 `MoNa-pi` 구조를 기준으로 적었다.

### 8.0 실행 체크리스트 (TODO)

- [ ] `models/heads/`에 Flow Matching action head 스켈레톤 추가
- [ ] 기존 분류/회귀 헤드와 동일한 I/O 텐서 규격 정의 문서화
- [ ] `configs/`에 chunk 길이 및 제어 주기 하이퍼파라미터 추가
- [ ] `data/` 로더에 연속 액션 스케일링 규칙 고정
- [ ] Dataset v3 변형(Close/Far/Offset/No‑obstacle) 반영
- [ ] 에피소드 단위 split 규칙 문서화 및 검증 스크립트 추가
- [ ] `training/`에 Flow Matching loss 및 스케줄러 통합
- [ ] PM/DM 비교 가능 보조 지표 로깅 추가
- [ ] action chunk 길이 vs 성능 실험 행렬 정리
- [ ] `inference/`에 chunk buffer 및 재계획 주기 로직 추가
- [ ] 서버‑클라이언트 분리 재설계(고주파 제어 경로)
- [ ] 50Hz+ 제어 주기 실측 로깅 추가
- [ ] 오프라인 비교 리포트 자동 생성
- [ ] 실로봇 로그에서 성공률/충돌률/지터 측정
- [ ] `docs/`에 실험 리포트 축적 체계화

### 8.1 모델/헤드 구현

1. `models/heads/`에 Flow Matching action head 추가
2. 기존 분류/회귀 헤드와 동일한 I/O 텐서 규격 정의
3. chunk 길이와 제어 주기 하이퍼파라미터를 `configs/`에 명시
### 8.2 데이터 파이프라인

1. `data/` 로더에서 연속 액션 스케일링 규칙 고정
2. Dataset v3 변형(Close/Far/Offset/No‑obstacle)을 학습 split에 반영
3. 에피소드 단위 split 규칙을 문서화하고 검증 스크립트 추가
### 8.3 학습 루프

1. `training/`에 Flow Matching loss 및 스케줄러 통합
2. 기존 MoNaVLA 지표(PM/DM)와 비교 가능한 보조 지표 로그 추가
3. action chunk 길이와 성능의 trade‑off를 실험 행렬로 관리
### 8.4 추론/서빙

1. `inference/`에 chunk buffer 및 재계획 주기 로직 추가
2. 서버‑클라이언트 분리를 고주파 제어에 맞게 조정
3. 목표 50Hz+를 만족하는지 실측 로깅 추가
### 8.5 검증 및 리포트

1. 오프라인 평가 스크립트에서 MoNaVLA 대비 비교 리포트 생성
2. 실로봇 주행 로그에서 성공률/충돌률/지터 측정
3. 결과를 `docs/`에 실험 리포트로 축적
---

## 부록 A. 구현 대응표(근거 기반)

- MoNaVLA 정책 헤드 코드: `../MoNaVLA/robovlm_nav/models/policy_head/nav_policy_impl.py`
- MoNaVLA 하이브리드 헤드: `../MoNaVLA/robovlm_nav/models/policy_head/hybrid_action_head.py`
- MoNaVLA 개요/성과/한계: `../MoNaVLA/README.md`
- MoNaVLA 구조 설명: `../MoNaVLA/docs/PROJECT_STRUCTURE_AND_ARCH_20251120.md`
- MoNaVLA 추론 구조/INT8: `../MoNaVLA/README_INFERENCE.md`
- Pi0‑VLA 목표: `README.md`

---

## 부록 B. TODO (논문 완성용)

- [TODO: citation] Flow Matching/Diffusion 관련 선행연구 정리
- [TODO: citation] VLA 및 Mobile Navigation 관련 선행연구 정리
- [TODO] Pi0‑VLA 구현 결과 및 수치 실험 추가
- [TODO] 고주파 추론 파이프라인 실측 결과
- [TODO] 인과성 평가 실험 설계 고도화
