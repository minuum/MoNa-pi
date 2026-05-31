# MoNa-pi 로봇 배포 가이드 (monapi-driving)

> **대상**: Serbot2 온보드 또는 GX10 → Serbot2 분리 실행  
> **모델**: monapi-train 브랜치 best checkpoint (val_loss=0.0414, CL SR 100%)  
> **아키텍처**: PaliGemma 3B + AdaLN-Zero Flow Matching, BF16/INT8

---

## 1. 사전 조건

### 스왑 설정 (Serbot2 온보드 INT8 필수)
```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
# 영구 적용
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### 패키지 설치
```bash
pip install bitsandbytes          # INT8 양자화
pip install fastapi uvicorn requests pillow safetensors
source /opt/ros/humble/setup.bash # ROS2 Humble
```

### 체크포인트 위치 확인
```
checkpoints/best/model.safetensors   ← 배포용 (val_loss=0.0414)
```

---

## 2. 빠른 시작

### 전체 스택 (Serbot2 온보드 INT8)
```bash
./start.sh           # hybrid 모드: 키 우선 → 없으면 VLA
```

### 모드별 실행
```bash
./start.sh server    # 추론 서버만 (포트 8080)
./start.sh manual    # 키보드 수동만
./start.sh vla       # VLA 자동 (서버는 별도)
```

### GX10 서버 + Serbot2 분리 실행
```bash
# [GX10에서]
./start.sh server
# → http://<GX10_IP>:8080

# [Serbot2에서]
SERVER_URL=http://<GX10_IP>:8080 ./start.sh vla
```

---

## 3. 설정 파일

### `configs/serbot2.yaml` — 배포 전용
```yaml
model:
  use_paligemma: true
  load_pretrained_paligemma: true
  use_int8: true          # Serbot2 온보드: BF16 ~6GB → INT8 ~3GB
  paligemma_id: "google/paligemma-3b-pt-224"

data:
  image_size: 224

robot:
  image_topic: /cam_high/image_raw
  cmd_vel_topic: /cmd_vel
  control_hz: 10
  replan_ratio: 0.5
  n_ode_steps: 5
```

INT8 끄려면 (GX10 등 메모리 여유 있는 경우):
```yaml
use_int8: false
```

---

## 4. ROS2 파이프라인

```
[카메라]                [추론 서버]            [로봇]
camera_node.py  →  /cam_high/image_raw  →  ros2_controller.py  →  /cmd_vel  →  POP driver
                        ↕ HTTP POST
                   inference/server.py
                   (checkpoints/best, INT8)
```

### 노드별 실행 (수동)
```bash
# 1. 추론 서버
python inference/server.py --config configs/serbot2.yaml --ckpt checkpoints/best --port 8080

# 2. 카메라 노드
python robot/camera_node.py --backend gstreamer --fps 30
# USB 카메라: --backend usb --device 0
# 테스트: --backend dummy

# 3. 컨트롤러
python robot/ros2_controller.py \
    --server-url http://localhost:8080 \
    --instruction "Navigate to the goal"

# 4. 키보드 (모드 전환)
python robot/keyboard_controller.py --mode hybrid
```

---

## 5. 키보드 제어

| 키 | 동작 |
|---|---|
| **M** | Manual 모드 (WASD 직접 제어) |
| **V** | VLA 모드 (모델 자동) |
| **H** | Hybrid 모드 (키 우선, 없으면 VLA) |
| **X** | 비상정지 |
| W/S | 전진/후진 |
| A/D | 좌/우 이동 (옴니) |
| Q/E | 좌/우 회전 |
| SPACE | 정지 |

> **실기동 순서**: M으로 수동 확인 → H로 VLA 믹스 테스트 → V로 완전 자동

---

## 6. 성능 수치 (참고)

| 지표 | 수치 |
|---|---|
| Val Loss | 0.0414 |
| CL Success Rate | 100% (24/24 val ep) |
| Mean FPE | 0.048 m |
| Mean TLD | 0.957 |
| GX10 warm latency | ~240 ms (4Hz) |
| 모델 메모리 (BF16) | ~6 GB |
| 모델 메모리 (INT8) | ~3 GB |

---

## 7. 트러블슈팅

### 서버가 느리게 뜨는 경우 (첫 실행 ~30초)
PaliGemma 3B 로딩 시간. `GET /health` 로 준비 확인:
```bash
curl http://localhost:8080/health
# {"status":"ok","engine_ready":true,...}
```

### NaN 액션 출력
`configs/serbot2.yaml`에서 `use_int8: false`로 바꾸고 BF16으로 실행.

### `/cam_high/image_raw` 토픽 없음
```bash
# 카메라 노드 확인
ros2 topic list | grep cam_high
# 없으면 camera_node.py 실행
python robot/camera_node.py --backend gstreamer
```

### POP 드라이버 없음 (시뮬레이션 모드)
`from pop.driving import Driving` 실패 시 ROS `/cmd_vel`만 발행.  
실로봇 필요 시 pop 패키지 설치 확인.

---

## 8. 브랜치 구조

| 브랜치 | 용도 |
|---|---|
| `master` | MoNaVLA 베이스라인, 평가 스크립트 |
| `monapi-train` | π0 AdaLN-Zero 학습 코드, 체크포인트 |
| `monapi-driving` | **이 브랜치** — 로봇 배포 전용 |

`monapi-driving` = `monapi-train`에서 학습 코드 제외, 배포/실행 파일만.

---

## 9. 체크포인트 전송 (다른 기기로)

```bash
# GX10 → Serbot2
scp -r checkpoints/best/ user@serbot2:/path/to/MoNa-pi/checkpoints/best/

# 크기 확인 (~5.9GB)
du -sh checkpoints/best/
```
