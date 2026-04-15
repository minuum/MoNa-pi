"""
MoNa-pi ROS2 토픽 레코더

역할:
    ROS2 토픽에서 이미지/속도 명령을 실시간으로 구독하여
    HDF5 v6 포맷으로 저장.

구독 토픽:
    /cam_high/image_raw   (sensor_msgs/Image)
    /cmd_vel              (geometry_msgs/Twist)

실행:
    python collect/ros2_recorder.py \
        --output  /data/v6/ep001.h5 \
        --task    "장애물 회피" \
        --duration 30

의존성:
    source /opt/ros/humble/setup.bash
    pip install rclpy cv_bridge h5py numpy
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[ROS2Recorder] ROS2 없음. source /opt/ros/humble/setup.bash")


class ROS2Recorder(Node if ROS2_AVAILABLE else object):
    """
    ROS2 토픽 구독 + HDF5 저장 노드.
    이미지와 cmd_vel을 각각 독립적으로 수집한 뒤,
    에피소드 종료 시 nearest-neighbor 동기화 후 저장.
    """

    def __init__(self, output_path: str, task: str, duration: float = 0.0):
        if ROS2_AVAILABLE:
            super().__init__("mona_pi_recorder")

        self.output_path = Path(output_path)
        self.task = task
        self.duration = duration  # 0이면 Ctrl+C까지 수집

        self._images: list[tuple[float, np.ndarray]] = []
        self._actions: list[tuple[float, np.ndarray]] = []
        self._lock = threading.Lock()
        self._recording = True
        self._bridge = CvBridge() if ROS2_AVAILABLE else None
        self._start_time = time.monotonic()

        if ROS2_AVAILABLE:
            self.create_subscription(Image, "/cam_high/image_raw", self._image_cb, 10)
            self.create_subscription(Twist, "/cmd_vel", self._twist_cb, 10)
            self.get_logger().info(
                f"레코더 시작: output={output_path}, task='{task}', duration={duration}s"
            )

    # ── 토픽 콜백 ──────────────────────────────────────────────────

    def _image_cb(self, msg: "Image"):
        if not self._recording:
            return
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except Exception:
            return

        ts = time.monotonic()
        with self._lock:
            self._images.append((ts, cv_img.copy()))

        # duration 초과 시 자동 종료
        if self.duration > 0 and (ts - self._start_time) >= self.duration:
            self._recording = False
            self.get_logger().info("지정 시간 도달, 레코딩 완료")

    def _twist_cb(self, msg: "Twist"):
        if not self._recording:
            return
        action = np.array(
            [msg.linear.x, msg.linear.y, msg.angular.z],
            dtype=np.float32,
        )
        ts = time.monotonic()
        with self._lock:
            self._actions.append((ts, action))

    # ── 저장 ───────────────────────────────────────────────────────

    def save(self):
        with self._lock:
            images = list(self._images)
            actions = list(self._actions)

        if not images:
            print("[Recorder] 저장할 이미지 없음")
            return
        if not actions:
            print("[Recorder] 저장할 액션 없음")
            return

        img_ts  = np.array([t for t, _ in images])
        act_ts  = np.array([t for t, _ in actions])
        act_arr = np.stack([a for _, a in actions])

        # nearest-neighbor 동기화
        synced = []
        for ts in img_ts:
            idx = int(np.argmin(np.abs(act_ts - ts)))
            synced.append(act_arr[idx])
        synced = np.stack(synced)

        img_arr = np.stack([img for _, img in images])

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output_path, "w") as f:
            obs = f.create_group("observations")
            obs.create_dataset("images", data=img_arr, compression="gzip", compression_opts=4)
            f.create_dataset("actions", data=synced)
            f.attrs["language_instruction"] = self.task
            f.attrs["num_frames"] = len(img_arr)
            f.attrs["collector"] = "ros2_recorder"

        print(f"[Recorder] 저장 완료: {self.output_path} ({len(img_arr)} 프레임)")

    def is_done(self) -> bool:
        return not self._recording


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoNa-pi ROS2 레코더")
    parser.add_argument("--output",   default="data/collected/episode.h5")
    parser.add_argument("--task",     default="Navigate to the goal")
    parser.add_argument("--duration", type=float, default=0.0,
                        help="수집 시간(초). 0이면 Ctrl+C까지")
    args, ros_args = parser.parse_known_args()

    if not ROS2_AVAILABLE:
        print("ROS2를 설치하고 source /opt/ros/humble/setup.bash 를 실행하세요.")
        sys.exit(1)

    rclpy.init(args=ros_args)
    recorder = ROS2Recorder(args.output, args.task, args.duration)

    try:
        while rclpy.ok():
            rclpy.spin_once(recorder, timeout_sec=0.1)
            if recorder.is_done():
                break
    except KeyboardInterrupt:
        print("\n[Recorder] 키보드 인터럽트 수신, 저장 중...")
    finally:
        recorder.save()
        recorder.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
