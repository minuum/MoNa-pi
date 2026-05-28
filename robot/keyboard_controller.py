"""
MoNa-pi 키보드 제어 노드

MoNaVLA robot_control_node.py 참고하여 3-DOF(vx, vy, wz)에 맞게 재작성.

모드:
    M — Manual   : WASD 키보드 직접 제어
    V — VLA      : 추론 서버 자동 제어 (MonaPiController 활성화)
    H — Hybrid   : 키 입력 시 수동 우선, 없을 때 VLA

키 매핑 (Manual / Hybrid):
    W/S  — 전진/후진 (linear_x)
    A/D  — 좌/우 이동 (linear_y, 옴니휠)
    Q/E  — 좌/우 회전 (angular_z)
    SPACE — 정지

실행 (ROS2 환경):
    python robot/keyboard_controller.py --mode vla
    python robot/keyboard_controller.py --mode manual
"""

import argparse
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np

try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from std_msgs.msg import Bool, String
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[KeyboardController] ROS2 없음")

try:
    from pop.driving import Driving
    POP_AVAILABLE = True
except ImportError:
    POP_AVAILABLE = False


# ─── 키 → 액션 매핑 ───────────────────────────────────────────────────────────
KEY_MAP = {
    "w": ( 1.15,  0.00,  0.00),   # 전진
    "s": (-1.15,  0.00,  0.00),   # 후진
    "a": ( 0.00,  1.15,  0.00),   # 좌이동 (옴니)
    "d": ( 0.00, -1.15,  0.00),   # 우이동 (옴니)
    "q": ( 0.00,  0.00,  1.15),   # 좌회전
    "e": ( 0.00,  0.00, -1.15),   # 우회전
    # 대각선
    "1": ( 1.15,  1.15,  0.00),   # 좌전방
    "2": ( 1.15, -1.15,  0.00),   # 우전방
    "3": (-1.15,  1.15,  0.00),   # 좌후방
    "4": (-1.15, -1.15,  0.00),   # 우후방
    " ": ( 0.00,  0.00,  0.00),   # 정지
}

HELP_TEXT = """
┌─────────────────────────────────────────┐
│     MoNa-pi 키보드 제어                  │
├─────────────────────────────────────────┤
│  모드 전환:                              │
│    M — Manual  (키보드 직접 제어)         │
│    V — VLA     (모델 자동 제어)           │
│    H — Hybrid  (키 우선, 없으면 VLA)      │
│    X — 비상정지                          │
├─────────────────────────────────────────┤
│  이동 (Manual / Hybrid):                │
│    W/S  — 전진/후진                      │
│    A/D  — 좌/우 이동 (옴니)              │
│    Q/E  — 좌/우 회전                    │
│    SPACE — 정지                          │
│    Ctrl+C — 종료                         │
└─────────────────────────────────────────┘
"""


def _read_key():
    """터미널에서 단일 키 읽기 (non-blocking)"""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


if ROS2_AVAILABLE:
    class KeyboardControllerNode(Node):
        """
        모드 전환 + 키보드 제어 ROS2 노드.
        /cmd_vel 직접 발행 + /emergency_stop 발행.
        """

        def __init__(self, initial_mode: str, throttle: int):
            super().__init__("mona_pi_keyboard")
            self.mode = initial_mode.lower()   # "manual" | "vla" | "hybrid"
            self.throttle = throttle

            # POP 드라이버
            if POP_AVAILABLE:
                self.driver = Driving()
                self.get_logger().info("POP 드라이버 연결됨")
            else:
                self.driver = None
                self.get_logger().warn("POP 드라이버 없음 — ROS 토픽만 발행")

            # 발행자
            self.cmd_pub   = self.create_publisher(Twist,  "/cmd_vel",         10)
            self.estop_pub = self.create_publisher(Bool,   "/emergency_stop",  10)
            self.mode_pub  = self.create_publisher(String, "/mona_pi/mode",    10)

            # 구독자 (VLA가 보내는 cmd_vel — hybrid 모드에서 pass-through)
            self._vla_action: tuple = (0.0, 0.0, 0.0)
            self.create_subscription(Twist, "/mona_pi/vla_cmd", self._vla_cb, 10)

            self._manual_active = False   # 키 누름 중
            self._last_key_time = 0.0
            self._lock = threading.Lock()

            # 키보드 입력 스레드
            self._running = True
            self._key_thread = threading.Thread(target=self._key_loop, daemon=True)
            self._key_thread.start()

            # 제어 루프 (10Hz) — hybrid 모드에서 VLA 폴백
            self.create_timer(0.1, self._control_loop)

            print(HELP_TEXT)
            self._announce_mode()

        # ── 키보드 루프 ──────────────────────────────────────────────
        def _key_loop(self):
            while self._running:
                try:
                    key = _read_key().lower()
                except Exception:
                    break

                if key == "\x03":  # Ctrl+C
                    self._running = False
                    break

                with self._lock:
                    if key == "m":
                        self.mode = "manual"
                        self._announce_mode()
                    elif key == "v":
                        self.mode = "vla"
                        self._announce_mode()
                    elif key == "h":
                        self.mode = "hybrid"
                        self._announce_mode()
                    elif key == "x":
                        self._emergency_stop()
                    elif key in KEY_MAP and self.mode in ("manual", "hybrid"):
                        lx, ly, az = KEY_MAP[key]
                        self._publish(lx, ly, az)
                        self._manual_active = True
                        self._last_key_time = time.monotonic()

        def _vla_cb(self, msg: Twist):
            with self._lock:
                self._vla_action = (msg.linear.x, msg.linear.y, msg.angular.z)

        def _control_loop(self):
            """Hybrid: 키 입력 없을 때 VLA 액션 pass-through"""
            if self.mode != "hybrid":
                return
            with self._lock:
                # 0.3s 이상 키 입력 없으면 VLA로 폴백
                if time.monotonic() - self._last_key_time > 0.3:
                    lx, ly, az = self._vla_action
                    self._publish(lx, ly, az)

        # ── 발행 헬퍼 ────────────────────────────────────────────────
        def _publish(self, lx: float, ly: float, az: float):
            msg = Twist()
            msg.linear.x  = float(lx)
            msg.linear.y  = float(ly)
            msg.angular.z = float(az)
            self.cmd_pub.publish(msg)

            if self.driver and not (lx == 0 and ly == 0 and az == 0):
                try:
                    if abs(az) > 0.1:
                        self.driver.spin(int(np.sign(az) * self.throttle))
                    else:
                        angle = int(np.degrees(np.arctan2(ly, lx))) % 360
                        self.driver.move(angle, self.throttle)
                except Exception as e:
                    self.get_logger().warn(f"POP 오류: {e}")
            elif self.driver:
                try:
                    self.driver.stop()
                except Exception:
                    pass

        def _emergency_stop(self):
            msg = Bool()
            msg.data = True
            self.estop_pub.publish(msg)
            self._publish(0.0, 0.0, 0.0)
            self.get_logger().warn("비상정지 발행!")

        def _announce_mode(self):
            msg = String()
            msg.data = self.mode
            self.mode_pub.publish(msg)
            self.get_logger().info(f"모드 전환 → {self.mode.upper()}")

        def destroy_node(self):
            self._running = False
            self._publish(0.0, 0.0, 0.0)
            if self.driver:
                try:
                    self.driver.stop()
                except Exception:
                    pass
            super().destroy_node()


def main():
    parser = argparse.ArgumentParser(description="MoNa-pi 키보드 컨트롤러")
    parser.add_argument("--mode",     default="manual",
                        choices=["manual", "vla", "hybrid"])
    parser.add_argument("--throttle", type=int, default=50,
                        help="POP 드라이버 속도 (0-100)")
    args, ros_args = parser.parse_known_args()

    if not ROS2_AVAILABLE:
        print("ROS2가 필요합니다.")
        sys.exit(1)

    rclpy.init(args=ros_args)
    node = KeyboardControllerNode(args.mode, args.throttle)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
