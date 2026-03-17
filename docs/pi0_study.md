# Pi0 스터디 (arXiv + openpi 기반, 상세)

이 문서는 **arXiv 논문(π0, 2410.24164)**과 **Physical‑Intelligence/openpi GitHub README**만을 근거로 정리했다. 다른 출처는 사용하지 않았다.

---

## 1. 논문 핵심 요약 (arXiv:2410.24164)

### 1.1 문제 정의

- 로봇 학습의 범용성 한계(데이터 부족, 일반화, 강건성)를 **로봇 파운데이션 모델** 접근으로 해결하려는 프레임을 제시한다.
- 이를 위해 **사전학습된 VLM + Flow Matching 기반 액션 생성**을 결합한 구조를 제안한다.

### 1.2 아키텍처 핵심

- **VLM 백본 + Action Expert(Flow Matching)** 구조.
- 연속 액션을 Flow Matching으로 모델링하고, **action chunking**으로 고주파 제어를 가능하게 한다.
- 논문은 **최대 50Hz 제어**를 언급하며, 이는 고주파/정교 작업을 가능하게 하는 핵심 설계로 설명된다.

### 1.3 데이터 및 스케일

- **7개 로봇 구성, 68개 태스크**의 데이터로 사전학습함을 명시한다.
- **약 10,000시간 규모**의 로봇 시연 데이터와 공개 OXE 데이터가 포함된다고 서술한다.

### 1.4 학습 레시피 (Pre‑training / Post‑training)

- 논문은 **pre‑training과 post‑training 분리**가 필수적이라고 설명한다.
- pre‑training은 폭넓은 일반화와 기본 능력을 확보하고, post‑training은 고품질 데이터로 지시 수행/복잡 과제를 정밀하게 맞춘다.
- post‑training만 수행하면 복구/일반화가 약하고, zero‑shot만으로는 복잡한 장기 과제가 부족하다는 논지를 제시한다.

### 1.5 액션 청킹과 시간 해상도

- 논문은 **action chunk** 단위로 연속 액션을 생성하며, 특정 설정에서 **H=50** 길이의 chunk를 사용한다고 설명한다.
- 또한 Flow Matching 기반 설계가 **최대 50Hz 제어**를 가능하게 한다고 명시한다.

---

## 2. openpi GitHub 요약 (실행 관점)

### 2.1 모델 계열

- `π0`: Flow 기반 VLA
- `π0‑FAST`: FAST action tokenizer 기반 autoregressive VLA
- `π0.5`: open‑world generalization 강화 버전
- openpi README는 **π0.5에 대해 flow matching head만 지원**한다고 명시한다.

### 2.2 체크포인트 및 스케일

- base 모델 체크포인트는 **10k+ 시간 로봇 데이터**로 사전학습된 것으로 설명된다.
- base 모델과 robot‑specific fine‑tuned 체크포인트를 제공한다.

### 2.3 실행 요구사항

- 단일 GPU 기준 메모리 요구량(README):
  - Inference > 8GB
  - LoRA fine‑tuning > 22.5GB
  - Full fine‑tuning > 70GB

---

## 3. Pi0 → Pi0‑VLA(우리 프로젝트) 정렬 포인트

### 3.1 핵심 정렬

- **Flow Matching 기반 연속 액션 생성**: Pi0 논문의 핵심과 Pi0‑VLA 목표가 일치
- **Action chunking + 50Hz 제어**: Pi0의 고주파 제어 목표와 일치
- **Pre‑training/Post‑training 분리**: 학습 파이프라인 설계에 필수 반영

### 3.2 설계 충돌 지점(검토 필요)

- Pi0는 대규모 cross‑embodiment 데이터 전제를 가지므로, 우리의 데이터 규모/다양성과의 격차를 전략적으로 보완해야 한다.
- openpi는 flow head와 autoregressive head가 공존하므로, **Pi0‑VLA에서 Flow Matching만 유지할지** 결정이 필요하다.

---

## 4. 스터디 체크리스트

- [ ] Flow Matching action expert 구조를 논문 기반으로 도식화
- [ ] action chunking에서 chunk 길이/재계획 주기 정의
- [ ] openpi 추론 서버 구조를 `inference/` 설계에 매핑
- [ ] Pi0 학습 레시피를 `training/` 설계 문서로 전개

---

## 5. Sources

- arXiv PDF: https://arxiv.org/pdf/2410.24164
- openpi README: https://github.com/Physical-Intelligence/openpi
