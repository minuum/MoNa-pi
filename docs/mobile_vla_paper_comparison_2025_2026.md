# Mobile VLA 논문 비교 분석: Action Space & Data Collection

> **분석 목적**: 2025~2026년 최신 Mobile VLA 논문 3편과 MoNa-pi의 데이터 수집 방식 및 Action Space를 비교하여, 최선의 전략을 도출한다.

---

## 1. 비교 대상 논문 개요

| 구분 | TIC-VLA (2026) | OmniVLA (2025/ICRA 2026) | MobileVLA-R1 (2025) |
| :--- | :--- | :--- | :--- |
| **arXiv** | 2602.02459 | (ICRA 2026 accept) | AIGeeksGroup/MobileVLA-R1 |
| **플랫폼** | 4륜 이동 로봇 (Dynamic env) | 다종 휠 로봇 (FrodoBots, VizBot 등) | 4족 보행 로봇 (Unitree Go2) |
| **핵심 기여** | VLM 추론 latency를 명시적으로 해결 | 멀티 모달 목표 조건부 내비게이션 | CoT 기반 Chain-of-Thought 행동 계획 |
| **Action Space** | $v_x, v_y, \omega_z$ (연속 속도) | $v_{linear}, \omega_{angular}$ (연속 속도) | $v_x, v_y, \omega_{yaw}$ (연속 속도) |

---

## 2. Action Space 상세 비교

### 2.1 TIC-VLA (arXiv 2602.02459, Zhiyu Huang et al., 2026)

**핵심 아이디어**: VLM의 느린 추론 속도와 실시간 제어 간의 "시간 불일치(Temporal Mismatch)"를 해결  
**아키텍처**: Slow VLM + Fast Action Expert (Dual-system)

```
[입력 흐름]
RGB + Language → VLM (Low frequency, ~2-10Hz)
               ↓ (지연된 KV Cache Feature 전달)
RGB + State  → Action Expert (High frequency, ~50Hz)
               ↓
          vx, vy, ωz (Continuous)
```

| 속성 | 내용 |
| :--- | :--- |
| **Action 차원** | 3D 연속 벡터: $(v_x, v_y, \omega_{yaw})$ |
| **제어 주파수** | 50Hz (Action Expert가 독립적으로 가동) |
| **수집 방식** | 원격 조종(Teleoperation) - 상세 미공개이나 연속값 기록 |
| **핵심 문제 인식** | VLM이 2Hz로 생각하더라도, 로봇은 50Hz로 움직여야 한다 |

**MoNa-pi에의 시사점**: 바로 우리 연구의 핵심 문제를 다루고 있음. "VLM은 느린데 로봇은 빨라야 한다"는 모순을 해결하기 위해 **두 주기를 분리(Decouple)** 함. MoNa-pi의 Action Chunking 전략과 맥락을 같이 함.

---

### 2.2 OmniVLA (ICRA 2026)

**핵심 아이디어**: 언어, 이미지, 2D Pose를 모두 "목표 조건(Goal Condition)"으로 수용하는 범용 내비게이션 VLA  
**데이터**: 9,500시간 이상, 10개 로봇 플랫폼, 13개 공개 데이터셋(GNM, LeLaN, Frodobots-2K 등)

```
[Action Space]
- 출력: Linear velocity, Angular velocity (2D Navigation)
- 액션 시퀀스 길이: N개 연속 예측 (Action Head 적용)
- 학습 데이터 수집: 사람이 텔레오퍼(원격 조작)로 시연 → 고정 주파수로 기록
```

| 속성 | 내용 |
| :--- | :--- |
| **Action 차원** | 2D: $(v_{linear}, \omega_{angular})$ |
| **수집 주파수** | **1Hz** (Goal Image용), 내부 제어는 더 높음 |
| **수집 방식** | 텔레오퍼레이션 (Joystick/Gamepad) → 연속 궤적 |
| **특이점** | 데이터 불균형 해소를 위해 **Modality Dropout** 사용 |

**MoNa-pi에의 시사점**: 1Hz 이미지로도 학습 가능했지만, 수집은 부드러운 텔레오퍼레이션으로 했음. 1Hz 수집이 아니라 수집 시 Smooth한 Velocity가 핵심.

---

### 2.3 MobileVLA-R1 (Nov 2025, AIGeeksGroup)

**핵심 아이디어**: VLM이 단순히 액션을 예측하는 것이 아니라, **Chain-of-Thought(CoT)로 먼저 추론**하고 그 결과를 연속 제어로 변환

```
[Action Space]
vx (전진/후진) + vy (측면 이동) + ω_yaw (회전)
↓
+ 고차원 이산 행동 α ∈ {FORWARD, TURN_LEFT, TURN_RIGHT, STOP}를 병행

[학습 방식]
Stage 1: CoT-annotated SFT (지도 학습)
Stage 2: GRPO (강화 학습으로 정교화)
```

| 속성 | 내용 |
| :--- | :--- |
| **Action 차원** | 연속 3D $(v_x, v_y, \omega_{yaw})$ + 이산 고수준 액션 α |
| **수집 방식** | 대규모 CoT 데이터셋 + 시뮬레이션 궤적 |
| **특이점** | 이산과 연속을 "계층적(Hierarchical)"으로 결합 |

**MoNa-pi에의 시사점**: 우리의 기존 이산 분류(9-class) 방식을 완전히 버릴 필요 없이, 이를 "고수준 의도(High-level Intent)"로 활용하고 연속 제어와 결합하는 절충안(Hybrid)이 가능함을 시사.

---

## 3. 핵심 비교 표: 3 논문 vs MoNa-pi (v5 현재)

| 비교 항목 | TIC-VLA | OmniVLA | MobileVLA-R1 | **MoNa-pi v5 (현재)** |
| :--- | :---: | :---: | :---: | :---: |
| **Action 유형** | 연속 속도 | 연속 속도 | 연속 + 이산 혼합 | **이산형 Bang-bang** |
| **Action 차원** | (vx, vy, ωz) | (v, ω) | (vx, vy, ωyaw) + α | (vx, vy, ωz) |
| **제어 주파수** | 50Hz | >1Hz | >1Hz | **2Hz (비동기)** |
| **수집 방식** | 텔레오퍼 (Analogue) | 텔레오퍼 (Joystick) | CoT+시뮬레이션 | **키보드 (Binary)** |
| **VLM 처리 주기** | 2~10Hz | 사전학습 기반 | SFT+GRPO | 1프레임 당 1회 |
| **Action Chunking** | ✅ 분리 (Dual-system) | ✅ N-step 예측 | ✅ CoT 계획 | ❌ 미적용 |
| **Action 정규화** | [-1, 1] 정규화 추정 | 정규화 추정 | 정규화 적용 | **❌ 미적용 (1.15 raw)** |

---

## 4. Findings: MoNa-pi에 적용 가능한 전략

### Priority 1: Action Chunking 도입 (TIC-VLA 영향)
현재 MoNa-pi는 이미지 1장 → 1개 액션을 예측하는 구조. → **1장 이미지 → 10개 미래 액션 청크**로 전환하면 비동기 타이밍 문제가 자동으로 완화됨. Flow Matching이 이 구조에 최적화되어 있음.

### Priority 2: Action 정규화 파이프라인 (3논문 공통)
모든 논문이 raw 속도값을 정규화하여 학습함. `1.150` → `1.0` 스케일링 이상으로, 데이터셋 전체의 **Z-score 표준화** 또는 **Min-Max [-1, 1] 정규화**가 필수적.

### Priority 3: 수집 방식 개선 (OmniVLA 영향)
수집 빈도(1Hz~2Hz) 자체보다 **수집 시 속도 프로파일의 연속성**이 중요함. 키보드 방식도 유지하되, **가속/감속 Ramp 로직**을 소프트웨어적으로 삽입하면 연속 속도 데이터를 만들 수 있음.

### Priority 4: 고수준 의도 + 연속 제어 혼합 (MobileVLA-R1 영향)
기존 MoNaVLA의 9-class 이산 분류(FORWARD, TURN_LEFT, STOP 등)를 **Policy의 "고수준 조건(Prefix)"**으로 활용하고, Flow Matching이 세밀한 속도값을 생성하도록 구조화하면 학습 안정성이 크게 향상될 수 있음.

---

## 5. Conclusion: 최선의 전략 로드맵

```
[지금 바로 적용 가능]
1. Action Normalization: 학습 데이터 로더 레벨에서 Z-score 정규화 적용
2. Action Chunking: 1프레임 분석 → 미래 10스텝 예측으로 Flow Head 재설계

[단기 (다음 데이터 수집 전)]  
3. 키보드에 Ramp 가속 로직 삽입 (소프트웨어 수정만으로 가능)
4. 수집 주파수 2Hz → 5Hz로 상향 조정

[중기 (가능하다면)]
5. Xbox 패드나 저가형 조이스틱으로 Analog 수집 전환
6. 첫 수집 후 Action Smoothing 필터로 후처리
```

---

*분석 출처: TIC-VLA (arXiv 2602.02459), OmniVLA (ICRA 2026), MobileVLA-R1 (AIGeeksGroup, Nov 2025)*
