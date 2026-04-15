# Deep Dive Analysis: 2025-2026 Mobile VLA State-of-the-Art 논문 3편

이 문서는 2025~2026년에 발표된 가장 진보적인 Mobile VLA(Vision-Language-Action) 논문 3편을 심층적으로 분석하고, MoNa-pi(MoNaVLA) 프로젝트에서 어떻게 데이터를 수집하고 최적화해야 할지에 대한 구체적 청사진을 제공합니다. 

각 논문의 핵심 아이디어, 데이터 수집 파이프라인(Data Collection), 액션 표현(Action Space) 방식 및 우리 연구에의 시사점을 다룹니다.

---

## Ⅰ. TIC-VLA: A Think-in-Control Vision-Language-Action Model (arXiv:2602.02459)

### 1. 핵심 철학: "인지 지연(Latency)의 명시적 수용"
VLM 기반 모델의 가장 큰 단점은 추론 속도가 느리다는 것입니다. 로봇이 1 프레임을 이해하는 동안 상황이 변하기 때문에 발생하는 "Temporal Mismatch(시간 불일치)"를 TIC-VLA는 정면으로 돌파합니다. VLM 추론 지연을 숨기려 하지 않고 학습 데이터와 아키텍처에 명시적으로 반영했습니다.

### 2. 구조 및 액션 스페이스 (Dual-System Architecture)
*   **분리된 추론(Decoupled Inference)**: 거대한 VLM(Slow Thinker)과 가벼운 Action Expert(Fast Controller)를 분리했습니다.
    *   **VLM Backbone**: 2~10Hz로 작동하여 전반적인 상황 판단과 거시적 경로를 구상합니다. 
    *   **Action Expert**: 50Hz 이상의 고주파수(High-frequency)로 작동하며, 현재의 빠른 감각(Proprioception/State)과 **과거 거시적 판단의 KV Cache**를 조합해 즉각적으로 반응합니다.
*   **Action Space**: $(v_x, v_y, \omega_z)$의 연속 속도 제어를 수행하며, 여기서 생성하는 액션은 점진적으로 업데이트되는 Action Queries의 Cross-attention 결과물입니다. 즉, "미래 여러 스텝의 궤적"을 한 단위로 방출(Action Chunking)합니다.

### 3. 데이터 수집 및 학습(Training) 관점
*   **Latency-consistent Training**: 오프라인 텔레오퍼레이션으로 수집한 데이터에 인위적으로 지연(Delay)을 삽입하여 학습시켰습니다. 즉, t시점의 이미지를 보고 t+$\delta$ 시점의 액션을 예측하도록 학습시켜, 실제 배포 시의 지연과 학습 환경을 일치시켰습니다.

### 💡 MoNa-pi 적용 방안 (Action Chunking & Async Handling)
현재 MoNa-pi v5 데이터가 비동기적(2Hz 수집, 제어는 독립적)으로 쌓이고 있는 점을 역이용할 수 있습니다. 
1.  **지연 학습(Latency Injection)**: 수집된 v5 이미지와 정확히 동시점의 액션을 매칭하려 애쓰지 말고, "이미지가 들어온 시점 (t) -> 실제 모델이 해석을 마치는 시점 (t+0.5s) -> 이때부터 실행될 10스텝 연속 액션($A_{t+0.5 \sim t+1.5}$)" 을 쌍(Pair)으로 엮어 Flow Matching을 학습시킵니다.
2.  이 철학을 완벽히 흡수하는 것이 바로 현재 코드에 작성된 **`n_steps=10` 형태의 Action Chunking**입니다. 

---

## Ⅱ. OmniVLA: Omni-Modal Goal Conditioning (ICRA 2026 Accept)

### 1. 핵심 철학: "데이터의 형식에 구애받지 않는 범용성 (Omni-Modal)"
로봇 데이터셋은 수집 주체가 다르고, 환경이 다르며, 입력 모달리티(이미지/자연어/2D Pose 등)가 뒤섞여 있습니다. OmniVLA는 이를 통일된 토큰 공간(Unified Token Space)으로 밀어넣고, 일부 목표 정보가 누락되어도 강인하게 주행하는 것에 초점을 맞췄습니다.

### 2. 데이터셋 구성 및 수집 (Data Collection)
*   **매머드급 데이터 통합**: 총 10종의 로봇 하드웨어에 걸쳐 **9,500시간**의 거대한 주행 데이터를 모았습니다. 
*   **수집 방식 (중요)**: 주행은 철저하게 인간 조종자(Human expert)의 원격 조작(Teleoperation)으로 매끄러운(Smooth) 궤적이 수집됩니다. 조이스틱 등을 이용해 연속적인 궤적을 그리며, 특히 이미지는 1Hz(초당 1장) 수준의 상대적으로 낮은 프레임으로 저장됩니다.
*   **Modality Dropout**: 학습 중 랜덤하게 일부 모달리티(예: 자연어 지시 생략, 목표 이미지 블러 처리 등)를 삭제(Masking)하여, 로봇이 단일 모달리티에 과적합(Overfitting)되지 않게 만들었습니다.

### 3. 액션 스페이스 (Action Space)
*   VLA 백본 상단에 Linear Action Head를 달아 $(v_{linear}, \omega_{angular})$ 형태의 2D 내비게이션 연속 속도를 $N$ 스텝 동안 나열하여 출력합니다. 

### 💡 MoNa-pi 적용 방안 (수집 퀄리티 & 강인함)
1.  **부드러운 조종(Teleop)의 중요성**: 이 논문 역시 1Hz라는 매우 낮은 주사율로 비전을 확보했지만 압도적 성능을 보였습니다. 즉, "초당 몇 장의 이미지가 수집되느냐"보다 **"인간이 কতটা 부드럽게(Continuous) 가속/감속하며 운전한 궤적이냐"**가 핵심입니다. 키보드로 수집하더라도 반드시 **가속/감속 램프(Acceleration Ramp)** 필터를 통과한 데이터여야 합니다.
2.  **Instruction Dropout 적용**: 향후 학습 코드에 "텍스트 지시어"를 10% 확률로 빈 문자열(`""`)로 바꾸는 Dropout 로직을 추가하면, 텍스트에 집착하지 않고 시각 정보(이미지 경로)만으로도 주행 능력을 유지할 수 있게 됩니다.

---

## Ⅲ. MobileVLA-R1: Reinforcing Vision-Language-Action (arXiv:2511.17889)

### 1. 핵심 철학: "강화학습과 생각의 사슬(Chain-of-Thought)로 간극 좁히기"
단순히 "이미지 -> 모터 윙~"이라는 End-to-End 방식은 한계(Generalization failure)가 명확합니다. MobileVLA-R1은 거대 백본이 직접 모터를 돌리는 대신, 중간에 **논리적인 생각(CoT)** 단계를 배치하고 이후 강화학습(RL)을 통해 연속 제어를 가다듬습니다.

### 2. 구조 및 액션 스페이스 (Hierarchical Action Space)
*   **1단계: 이산 사고(Discrete Logic)**: 모델은 먼저 $\alpha \in \{FORWARD, TURN\_LEFT, AVOID\}$와 같은 고수준의 '이도(Intent)'를 텍스트 또는 이산 토큰으로 예측합니다.
*   **2단계: 연속 제어(Continuous Control)**: 위의 $\alpha$ 조건과 시각 정보를 결합하여 실제 모바일베이스의 연속 속도 3차원 벡터 $(v_x, v_y, \omega_{yaw})$를 정밀하게 토해내는 계층(Hierarchical) 형태입니다.

### 3. 학습 방법론 (Two-Stage Training with GRPO)
*   **Stage 1 (SFT)**: 사람의 궤적(CoT 포함)을 활용해 정답을 맞추는 모방학습(Behavior Cloning) 진행.
*   **Stage 2 (GRPO - 강화학습)**: 훈련 후 시뮬레이션 환경에 던져놓고, '장애물 회피', '도착 시간'에 보상(Reward)을 주어 Group Relative Policy Optimization으로 미세조정(Fine-Tuning)합니다. 이를 통해 단순히 정답 궤적을 흉내 내는 것을 넘어, 실제로 안전하고 강인한 제어 곡선을 스스로 찾아냅니다.

### 💡 MoNa-pi 적용 방안 (하이브리드 액션과 RL 도입)
이 논문은 현재 우리의 고민(이산형 9-class와 연속형 제어 사이클)에 명쾌한 해답을 제공합니다.
1.  **계층적 프롬프팅 (Hierarchical Control)**:
    MoNa-pi VLM(Gemma)에 단순 지시문만 주지 맙시다. 이전 버전의 MoNaVLA에서 사용하던 9-class(FL, FR, F 등) 분류기를 버리지 말고, **VLM의 중간 생성 지시문(Prefix)**으로 활용하세요.
    *   *Prompt Input*: `<image> 목표 위치로 이동하세요.`
    *   *CoT Output(모델 생성)*: `현재 장애물이 우측에 있습니다. 따라서 <TURN_LEFT_FORWARD> 모드가 필요합니다. 이에 따른 연속 제어 속도는 다음과 같습니다:` 
    *   *Flow Matching 입력 조건*: 이 CoT 텍스트 임베딩을 Flow Head에 Conditioning.
2.  이렇게 구성하면 VLM은 '개념적 판단'을 하고 Flow Head는 '물리적 제어'를 하는 계층구조가 완성되어 성능과 설명력(Explainability)이 비약적으로 상승합니다.

---

## ▣ 최종 요약: MoNa-pi 파이프라인 업데이트 지침서

세 논문의 인사이트를 종합하여 우리의 데이터 수집/가공 프로세스를 다음과 같이 확정해야 합니다.

1.  **가속 램프(Analog Simulation) 적용 수집**:
    현재의 1.15 / 0 식의 이산 수치를 폐기해야 합니다. 키보드를 떼면 1.15 -> 0.8 -> 0.4 -> 0 식으로 0.2초에 걸쳐 부드럽게 감속되도록 ROS 조작 노드를 즉시 수정하십시오 (OmniVLA).
2.  **Delay-Aware 10-Step Chunking**:
    현재 `test_loading.py`에서 확인한 shape `(1, 10, 3)`을 적극 이용하여 미래 10단계 궤적을 학습시킵니다. 이미지가 비동기적으로 천천히 수집되어도 그 사이에 채워질 10 프레임 분량의 고주파 연속 액션을 하나의 타겟으로 묶으십시오 (TIC-VLA).
3.  **9-Class Label 재활용 (CoT Prompting)**:
    v5 데이터를 수집할 때마다 현재의 연속 속도만 저장하지 말고, 해당 구간을 대표하는 9-class 상태(직전의 상태)를 함께 Labeling 해두십시오. 학습 시 "나는 지금 TURN_LEFT 상황이다"라는 명시적 임베딩을 Flow Head에 밀어 넣어 모델이 헷갈리지 않게 유도해야 합니다 (MobileVLA-R1).
4.  **Z-score Normalization**:
    Raw Velocity는 절대로 그대로 학습에 쓰면 안 됩니다. 반드시 `(action - mean) / std`로 정규화하여 Flow Matching이 편안하게 안정적으로 노이즈를 예측할 수 있게 처리하십시오.
