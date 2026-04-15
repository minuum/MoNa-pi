"""
MoNa-pi ROS2 컨트롤러 노드

역할:
    - /cam_high/image_raw 구독 → 추론 서버에 HTTP POST
    - ActionChunkBuffer에서 10Hz 로 액션 꺼내기 → /cmd_vel 발행
    - /emergency_stop 구독 → 즉시 정지

실행:
    # 추론 서버가 먼저 실행 중이어야 함
    python robot/ros2_controller.py \
        --instruction "장애물을 피해 직진" \
        --server-url  http://localhost:8080

ROS2 의존성:
    source /opt/ros/humble/setup.bash
    pip install rclpy cv_bridge requests
"""

import argparse
import base64
import io
import sys
import threading
import time
from pathlib import Path

import numpy as np
import requests

# ROS2 임포트 (ROS2 환경에서만 유효)
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image
    from std_msgs.msg import Bool, String
    from cv_bridge import CvBridge
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    print("[Controller] ROS2를 찾을 수 없음. 시뮬레이션 모드로 실행됩니다.")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from robot.action_buffer import ActionChunkBuffer


# ─────────────────────────────────────────────
# VLAControlManager — MoNaVLA 이관
# 정밀 타이밍 제어: 0.4s 이동 + 자동 정지 + 다중 stop 패킷
# ─────────────────────────────────────────────

class VLAControlManager:
    """
    MoNaVLA `vla_control_utils.VLAControlManager` 직접 이관.

    - publish_and_move(): ROS2 Twist 발행 + (선택적) 하드웨어 직접 제어
    - robust_stop():      연속 N회 정지 패킷으로 버퍼 클리어
    - move_and_stop_timed(): 0.4s 이동 후 자동 정지 (MoNaVLA 동일 패턴)
    """

    def __init__(self, node, move_duration: float = 0.4):
        self.node = node
        self.move_duration = move_duration
        self._movement_lock = threading.Lock()
        self._movement_timer: threading.Timer | None = None
        self._cmd_counter = 0

        # pop 드라이버 (하드웨어 있을 때만)
        try:
            from pop.driving import Driving
            self._driver = Driving()
        except Exception:
            self._driver = None

    def publish_and_move(self, lx: float, ly: float, az: float, source: str = "vla"):
        self._cmd_counter += 1
        is_stop = abs(lx) < 0.01 and abs(ly) < 0.01 and abs(az) < 0.01

        if hasattr(self.node, "cmd_pub"):
            msg = Twist()
            msg.linear.x  = float(lx)
            msg.linear.y  = float(ly)
            msg.angular.z = float(az)
            self.node.cmd_pub.publish(msg)

        if self._driver is not None:
            try:
                import numpy as _np
                if not is_stop:
                    if abs(az) > 0.1:
                        self._driver.spin(int(_np.sign(az) * 50))
                    else:
                        angle = int(_np.degrees(_np.arctan2(ly, lx))) % 360
                        self._driver.move(angle, 50)
                else:
                    self._driver.stop()
            except Exception as e:
                if hasattr(self.node, "get_logger"):
                    self.node.get_logger().warn(f"[VLAControl] HW error: {e}")

    def robust_stop(self, count: int = 5, source: str = "robust_stop"):
        """연속 N회 정지 패킷 — MoNaVLA 동일"""
        for _ in range(count):
            self.publish_and_move(0.0, 0.0, 0.0, source=source)
            time.sleep(0.05)

    def move_and_stop_timed(self, lx: float, ly: float, az: float, source: str = "timed"):
        """0.4s 이동 후 자동 정지 — MoNaVLA 동일 패턴"""
        with self._movement_lock:
            if self._movement_timer:
                self._movement_timer.cancel()
            self.publish_and_move(lx, ly, az, source=source)
            self._movement_timer = threading.Timer(
                self.move_duration,
                lambda: self.robust_stop(source=f"{source}_autostop"),
            )
            self._movement_timer.daemon = True
            self._movement_timer.start()


# ─────────────────────────────────────────────
# 추론 서버 HTTP 클라이언트
# ─────────────────────────────────────────────

class InferenceCaller:
    """
    별도 스레드에서 추론 서버에 비동기적으로 요청을 보내는 클라이언트.
    블로킹 HTTP 호출이 제어 루프를 막지 않도록 스레드 분리.
    """

    def __init__(self, server_url: str, timeout_ms: float = 500.0):
        self.server_url = server_url.rstrip("/") + "/predict"
        self.timeout_s = timeout_ms / 1000.0
        self._session = requests.Session()

    def call(
        self,
        image_np: np.ndarray,
        instruction: str,
        callback,  # callback(actions: np.ndarray)
    ):
        """비차단 요청 — 새 스레드에서 HTTP 호출 후 callback 실행"""
        threading.Thread(
            target=self._call_sync,
            args=(image_np, instruction, callback),
            daemon=True,
        ).start()

    def _call_sync(self, image_np: np.ndarray, instruction: str, callback):
        try:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(image_np)
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()

            resp = self._session.post(
                self.server_url,
                json={"image_b64": b64, "instruction": instruction},
                timeout=self.timeout_s,
            )
            resp.raise_for_status()
            data = resp.json()
            actions = np.array(data["actions"], dtype=np.float32)  # (10, 3)
            callback(actions)
        except Exception as e:
            print(f"[InferenceCaller] 요청 실패: {e}")


# ─────────────────────────────────────────────
# ROS2 노드
# ─────────────────────────────────────────────

if ROS2_AVAILABLE:
    class MonaPiController(Node):
        """
        MoNa-pi 메인 ROS2 제어 노드.

        구독:
            /cam_high/image_raw    (sensor_msgs/Image)
            /emergency_stop        (std_msgs/Bool)

        발행:
            /cmd_vel               (geometry_msgs/Twist) @ control_hz
            /mona_pi/status        (std_msgs/String)

        파라미터:
            inference_server_url   (str,   default "http://localhost:8080")
            instruction            (str,   default "Navigate to the goal")
            control_hz             (float, default 10.0)
            max_linear_vel         (float, default 1.15)
            max_angular_vel        (float, default 1.15)
        """

        def __init__(self):
            super().__init__("mona_pi_controller")

            # 파라미터 선언
            self.declare_parameter("inference_server_url", "http://localhost:8080")
            self.declare_parameter("instruction", "Navigate to the goal")
            self.declare_parameter("control_hz", 10.0)
            self.declare_parameter("max_linear_vel", 1.15)
            self.declare_parameter("max_angular_vel", 1.15)

            server_url  = self.get_parameter("inference_server_url").value
            self.instruction = self.get_parameter("instruction").value
            control_hz  = self.get_parameter("control_hz").value
            self.max_lv = self.get_parameter("max_linear_vel").value
            self.max_av = self.get_parameter("max_angular_vel").value

            # 내부 컴포넌트
            self.buffer = ActionChunkBuffer(horizon=10, replan_ratio=0.5)
            self.caller = InferenceCaller(server_url, timeout_ms=2000.0)
            self.bridge = CvBridge()
            self.ctrl = VLAControlManager(self)  # MoNaVLA 이관
            self._requesting = False  # 중복 요청 방지 플래그

            # 구독자
            self.create_subscription(Image, "/cam_high/image_raw", self._image_cb, 10)
            self.create_subscription(Bool, "/emergency_stop",      self._estop_cb, 10)

            # 발행자
            self.cmd_pub    = self.create_publisher(Twist,  "/cmd_vel",       10)
            self.status_pub = self.create_publisher(String, "/mona_pi/status", 10)

            # 제어 루프 타이머
            period = 1.0 / control_hz
            self.create_timer(period, self._control_loop)

            self.get_logger().info(
                f"MonaPiController 시작 "
                f"[server={server_url}, hz={control_hz}, instruction='{self.instruction}']"
            )

        # ── 이미지 콜백 ────────────────────────────────────────────────
        def _image_cb(self, msg: Image):
            """새 이미지 도착 시 재계획 필요 여부를 확인하고 추론 요청"""
            if self.buffer.is_emergency():
                return

            if not self.buffer.should_replan() and not self.buffer.is_empty():
                return  # 재계획 불필요

            if self._requesting:
                return  # 이미 요청 중

            try:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            except Exception as e:
                self.get_logger().warn(f"이미지 변환 실패: {e}")
                return

            self._requesting = True
            self.caller.call(cv_image, self.instruction, self._on_actions_received)

        def _on_actions_received(self, actions: np.ndarray):
            """추론 서버 응답 콜백 — 새 청크를 버퍼에 주입"""
            self.buffer.push(actions)
            self._requesting = False
            self.get_logger().debug(f"새 청크 수신 (remaining={self.buffer.remaining()})")

        # ── 제어 루프 (10Hz) ──────────────────────────────────────────
        def _control_loop(self):
            """버퍼에서 액션을 꺼내 /cmd_vel 발행"""
            if self.buffer.is_emergency():
                self._publish_zero()
                self._publish_status("EMERGENCY_STOP")
                return

            action = self.buffer.pop()  # (3,) [vx, vy, wz]
            self._publish_twist(action)
            self._publish_status(
                f"ok | remaining={self.buffer.remaining()} | "
                f"vx={action[0]:.2f} vy={action[1]:.2f} wz={action[2]:.2f}"
            )

        # ── 비상 정지 콜백 ────────────────────────────────────────────
        def _estop_cb(self, msg: Bool):
            if msg.data:
                self.buffer.emergency_stop()
                self._publish_zero()
                self.get_logger().warn("비상 정지 수신!")
            else:
                self.buffer.resume()
                self.get_logger().info("비상 정지 해제")

        # ── 발행 헬퍼 ─────────────────────────────────────────────────
        def _publish_twist(self, action: np.ndarray):
            lx = float(np.clip(action[0], -self.max_lv, self.max_lv))
            ly = float(np.clip(action[1], -self.max_lv, self.max_lv))
            az = float(np.clip(action[2], -self.max_av, self.max_av))
            # VLAControlManager 경유 — 하드웨어 + ROS 동시 제어 (MoNaVLA 계승)
            self.ctrl.publish_and_move(lx, ly, az, source="action_chunk")

        def _publish_zero(self):
            self.ctrl.robust_stop(count=3, source="zero_cmd")

        def _publish_status(self, text: str):
            msg = String()
            msg.data = text
            self.status_pub.publish(msg)


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoNa-pi ROS2 컨트롤러")
    parser.add_argument("--instruction", default="Navigate to the goal")
    parser.add_argument("--server-url",  default="http://localhost:8080")
    parser.add_argument("--control-hz",  type=float, default=10.0)
    args, ros_args = parser.parse_known_args()

    if not ROS2_AVAILABLE:
        print("[Controller] ROS2 없이는 실행할 수 없습니다.")
        print("  source /opt/ros/humble/setup.bash 후 재시도하세요.")
        sys.exit(1)

    rclpy.init(args=ros_args)

    node = MonaPiController()
    # 파라미터를 CLI args로 덮어쓰기
    node.instruction = args.instruction
    node.caller = InferenceCaller(args.server_url, timeout_ms=2000.0)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
