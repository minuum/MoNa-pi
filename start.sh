#!/bin/bash
# MoNa-pi 빠른 시작 스크립트 (5/29 Serbot2 테스트용)
#
# 사용법:
#   ./start.sh                        # 전체 스택 (hybrid 모드, INT8 온보드)
#   ./start.sh server                 # 추론 서버만 (Serbot2 온보드)
#   ./start.sh manual                 # 수동 키보드만
#   ./start.sh vla                    # VLA 자동 (서버는 별도 실행)
#
# Serbot2 온보드 INT8 추론 (스왑 필요):
#   configs/serbot2.yaml 의 use_int8: true 로 BF16 ~6GB → INT8 ~3GB
#   sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile
#   sudo mkswap /swapfile && sudo swapon /swapfile

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-configs/serbot2.yaml}"
CKPT="${CKPT:-checkpoints/best}"
SERVER_URL="${SERVER_URL:-http://localhost:8080}"
INSTRUCTION="${INSTRUCTION:-[FORWARD] 목표물을 향해 이동해}"
MODE="${1:-hybrid}"

# ROS2 환경
ROS_SETUP="/opt/ros/humble/setup.bash"
if [ -f "$ROS_SETUP" ]; then
    source "$ROS_SETUP"
fi

# 가상환경
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

echo "=== MoNa-pi 시작 (mode=$MODE) ==="
echo "  config : $CONFIG"
echo "  ckpt   : $CKPT"
echo "  server : $SERVER_URL"
echo ""

case "$MODE" in

  server)
    echo "[1/1] 추론 서버 시작..."
    python inference/server.py \
        --config "$CONFIG" \
        --ckpt   "$CKPT" \
        --host   0.0.0.0 \
        --port   8080
    ;;

  manual)
    echo "[1/2] 카메라 노드 시작..."
    python robot/camera_node.py --backend gstreamer --fps 30 &
    CAM_PID=$!

    echo "[2/2] 키보드 컨트롤러 시작 (manual 모드)..."
    python robot/keyboard_controller.py --mode manual

    kill $CAM_PID 2>/dev/null || true
    ;;

  vla)
    echo "[1/3] 카메라 노드 시작..."
    python robot/camera_node.py --backend gstreamer --fps 30 &
    CAM_PID=$!

    echo "[2/3] 컨트롤러 노드 시작..."
    python robot/ros2_controller.py \
        --server-url  "$SERVER_URL" \
        --instruction "$INSTRUCTION" \
        --control-hz  10.0 &
    CTRL_PID=$!

    echo "[3/3] 키보드 (VLA 모드, X=비상정지)..."
    python robot/keyboard_controller.py --mode vla

    kill $CAM_PID $CTRL_PID 2>/dev/null || true
    ;;

  hybrid | *)
    echo "[1/4] 추론 서버 시작..."
    python inference/server.py \
        --config "$CONFIG" \
        --ckpt   "$CKPT" \
        --host   0.0.0.0 --port 8080 &
    SRV_PID=$!

    echo "  추론 서버 워밍업 대기 (30s)..."
    sleep 30

    echo "[2/4] 카메라 노드 시작..."
    python robot/camera_node.py --backend gstreamer --fps 30 &
    CAM_PID=$!

    echo "[3/4] 컨트롤러 노드 시작..."
    python robot/ros2_controller.py \
        --server-url  "$SERVER_URL" \
        --instruction "$INSTRUCTION" \
        --control-hz  10.0 &
    CTRL_PID=$!

    echo "[4/4] 키보드 (hybrid — 키 우선, 없으면 VLA, X=비상정지)..."
    python robot/keyboard_controller.py --mode hybrid

    echo "종료 중..."
    kill $SRV_PID $CAM_PID $CTRL_PID 2>/dev/null || true
    ;;

esac
