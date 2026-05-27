# MoNa-pi 브랜치 구조 및 서버별 운영 가이드

MoNa-pi 프로젝트의 정통 π0 아키텍처(AdaLN-Zero) 적용 및 수치 정규화 변경 사항에 따른 브랜치 분할과 서버별 사용법 안내 가이드입니다.

---

## 📌 1. 브랜치 개요

코드베이스는 역할에 따라 두 개의 브랜치로 관리됩니다:

1. **`monapi-driving` (로봇 서버 - 주행/데모/추론 전용)**
   - **대상 기기**: NVIDIA Jetson AGX (로봇 온디바이스)
   - **주요 기능**: FastAPI 추론 서버 기동 (`inference/server.py`), ROS2 제어 노드 연동 (`robot/ros2_controller.py`), 실시간 주행 테스트.
2. **`monapi-train` (학습 서버 - 모델 트레이닝 전용)**
   - **대상 기기**: GPU 학습 서버 (예: Minum/A5000 서버 등)
   - **주요 기능**: 모델 학습 (`training/train.py`), 데이터 전처리 및 수렴성 검증.

---

## 🛠️ 2. 핵심 변경 사항 (AdaLN-Zero & 정규화 통합)

기존 베이스라인 대비 정통 π0 구조를 반영하며 아래 내용이 수정되었습니다.

- **AdaLN-Zero 시간 컨디셔닝 (`models/heads/flow_head.py`)**
  - 기존의 단순 시간 임베딩 합산 방식을 제거하고, Transformer 각 블록에 Adaptive Layer Normalization (AdaLN-Zero) 기법을 적용.
  - 가중치를 `0`으로 초기화(`Zero-init`)하여 학습 초기 모델 안정성을 극대화.
- **모델 중심의 정규화 파이프라인 (`models/heads/mona_action_expert.py`)**
  - MoNa-pi v5 데이터셋 물리 범위(최대 약 1.15)에 맞게 `normalize` (물리값 $\rightarrow$ $[-1, 1]$ 범위) 및 `unnormalize`를 수행하는 `MoNaActionExpert` 도입.
  - 이로 인해 **데이터셋은 정규화하지 않은 RAW 물리값을 출력**하고, **모델 내부(`get_loss` / `sample_actions`)에서 정규화 및 역정규화를 일괄 전담**하는 정통 π0 구조로 정렬 완료.

---

## 💻 3. 서버별 적용 및 실행 가이드

### A. 학습 서버 (Training Server) 운영 방법
학습 서버에서는 모델 내부에서 정규화가 처리될 수 있도록 데이터셋 정규화 옵션을 끈 상태로 학습을 수행해야 합니다.

1. **브랜치 전환**
   ```bash
   git checkout monapi-train
   ```
2. **설정 확인 (`configs/train.yaml`)**
   `data.normalize` 옵션이 `false`로 되어 있는지 확인합니다. (기본값 `false`)
   ```yaml
   data:
     train_path: /path/to/v5_dataset/
     normalize: false  # 중요: 데이터셋 단의 정규화를 끄고 모델 내 정규화 활성화
   ```
3. **학습 실행**
   ```bash
   python training/train.py --config configs/train.yaml
   ```

### B. 로봇 서버 (Robot Server - Jetson AGX) 운영 방법
로봇 서버에서는 학습 완료된 체크포인트를 활용하여 추론 서버를 띄우고 주행 노드를 가동합니다.

1. **브랜치 전환**
   ```bash
   git checkout monapi-driving
   ```
2. **FastAPI 추론 서버 기동**
   ```bash
   python inference/server.py --ckpt checkpoints/best --port 8080
   ```
3. **ROS2 제어 컨트롤러 노드 실행**
   ```bash
   # ROS2 환경 로드 후 실행
   source /opt/ros/humble/setup.bash
   python robot/ros2_controller.py --server-url http://localhost:8080 --instruction "Navigate to the goal"
   ```

---

## 🔍 4. 동작 검증 스크립트
설정 및 아키텍처가 정상 동작하는지 테스트하려면 아래 검증 스크립트를 활용합니다.

- **수치 정합성 검증 (정규화/역정규화 오차 확인)**:
  ```bash
  python scripts/verify_mona_expert.py
  # 결과: Max Reconstruction Error: 0.000001 (Success)
  ```
- **Fast 통합 테스트 (액션 버퍼 및 Mock 서버 통신 검증)**:
  ```bash
  pytest tests/test_integration.py -v
  ```
