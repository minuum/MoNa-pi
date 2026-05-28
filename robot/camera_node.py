"""
MoNa-pi 카메라 퍼블리셔 노드

Serbot2 카메라 → /cam_high/image_raw 발행
MoNaVLA camera_publisher_node.py 참고하여 MoNa-pi용으로 작성.

지원 백엔드:
    1. GStreamer (Jetson CSI 카메라) — 기본
    2. OpenCV USB 카메라 (--device /dev/video0)
    3. ROS2 republisher (기존 토픽 → /cam_high/image_raw 리매핑)

실행:
    # Jetson CSI 카메라
    python robot/camera_node.py

    # USB 카메라
    python robot/camera_node.py --backend usb --device 0

    # 테스트 (더미 이미지)
    python robot/camera_node.py --backend dummy
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CompressedImage
    from cv_bridge import CvBridge
    from rcl_interfaces.msg import ParameterDescriptor
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[CameraNode] ROS2 없음. source /opt/ros/humble/setup.bash 후 재시도")


def _gstreamer_pipeline(width=1280, height=720, fps=30, flip=0) -> str:
    """Jetson Nano/Orin CSI 카메라 GStreamer 파이프라인 (MoNaVLA 동일)"""
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={fps}/1 ! "
        f"nvvidconv flip-method={flip} ! "
        f"video/x-raw, width={width}, height={height}, format=BGRx ! "
        f"videoconvert ! video/x-raw, format=BGR ! appsink"
    )


def _open_camera(backend: str, device: int, width: int, height: int, fps: int):
    """카메라 열기 (백엔드별 분기)"""
    if backend == "gstreamer":
        cap = cv2.VideoCapture(_gstreamer_pipeline(width, height, fps), cv2.CAP_GSTREAMER)
    elif backend == "usb":
        cap = cv2.VideoCapture(device)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
    else:  # dummy
        return None
    if not cap.isOpened():
        raise RuntimeError(f"카메라 열기 실패 (backend={backend}, device={device})")
    return cap


if ROS2_AVAILABLE:
    class CameraPublisherNode(Node):
        """
        카메라 이미지를 /cam_high/image_raw 로 발행하는 ROS2 노드.
        MonaPiController가 이 토픽을 구독해서 추론 트리거.
        """

        def __init__(self, backend: str, device: int, width: int, height: int,
                     fps: int, publish_compressed: bool):
            super().__init__("mona_pi_camera")
            self.bridge = CvBridge()
            self.backend = backend
            self.publish_compressed = publish_compressed

            # 발행자
            self.img_pub = self.create_publisher(Image, "/cam_high/image_raw", 5)
            if publish_compressed:
                self.cmp_pub = self.create_publisher(
                    CompressedImage, "/cam_high/image_raw/compressed", 5)

            # 카메라 열기
            if backend == "dummy":
                self.cap = None
                self.get_logger().warn("DUMMY 모드 — 랜덤 이미지 발행")
            else:
                self.cap = _open_camera(backend, device, width, height, fps)
                self.get_logger().info(
                    f"카메라 열기 완료 (backend={backend}, {width}x{height}@{fps}fps)")

            # 퍼블리시 타이머
            period = 1.0 / fps
            self.create_timer(period, self._publish_frame)
            self._frame_count = 0

        def _publish_frame(self):
            if self.cap is not None:
                ret, frame_bgr = self.cap.read()
                if not ret:
                    self.get_logger().warn("프레임 읽기 실패")
                    return
            else:
                # Dummy: 랜덤 컬러 이미지
                frame_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            now = self.get_clock().now().to_msg()

            # raw Image
            msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
            msg.header.stamp = now
            msg.header.frame_id = "cam_high"
            self.img_pub.publish(msg)

            # CompressedImage (선택)
            if self.publish_compressed:
                _, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                cmp = CompressedImage()
                cmp.header.stamp = now
                cmp.header.frame_id = "cam_high"
                cmp.format = "jpeg"
                cmp.data = buf.tobytes()
                self.cmp_pub.publish(cmp)

            self._frame_count += 1
            if self._frame_count % 30 == 0:
                self.get_logger().debug(f"[Camera] {self._frame_count} frames published")

        def destroy_node(self):
            if self.cap:
                self.cap.release()
            super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="MoNa-pi 카메라 퍼블리셔")
    parser.add_argument("--backend",    default="gstreamer",
                        choices=["gstreamer", "usb", "dummy"])
    parser.add_argument("--device",     type=int, default=0, help="USB 카메라 인덱스")
    parser.add_argument("--width",      type=int, default=1280)
    parser.add_argument("--height",     type=int, default=720)
    parser.add_argument("--fps",        type=int, default=30)
    parser.add_argument("--compressed", action="store_true",
                        help="CompressedImage도 함께 발행")
    args, ros_args = parser.parse_known_args()

    if not ROS2_AVAILABLE:
        print("ROS2가 필요합니다.")
        sys.exit(1)

    rclpy.init(args=ros_args)
    node = CameraPublisherNode(
        backend=args.backend,
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        publish_compressed=args.compressed,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
