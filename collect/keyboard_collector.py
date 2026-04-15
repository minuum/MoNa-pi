"""
MoNa-pi 키보드 수집기 — 가감속(Ramp) 적용

문제:
    기존 키보드 수집은 키 누름 즉시 max 속도(1.15)로 점프 → Bang-bang 데이터
    Flow Matching 모델은 이런 스텝 함수를 학습하기 어려움

해결:
    RampController: 키 누름 지속 시간에 따라 속도를 이차 함수로 가속/감속
    - 0.0s 누름 → 0.0
    - 0.3s 누름 → ~0.5
    - 0.5s+ 누름 → 1.15 (최대 속도 도달)
    - 키 뗌 → 0.15s 동안 감속

실행:
    python -m collect.keyboard_collector \
        --output /data/v6/ep001.h5 \
        --task   "장애물을 피해 직진" \
        --hz     20

    python -m collect.keyboard_collector --dry-run  # 속도 플롯만 출력

키 맵핑:
    W / ↑  : 전진 (linear_x +)
    S / ↓  : 후진 (linear_x -)
    A / ←  : 좌측 이동 (linear_y +)
    D / →  : 우측 이동 (linear_y -)
    Q      : 좌회전 (angular_z +)
    E      : 우회전 (angular_z -)
    SPACE  : 정지 (모든 속도 0)
    ESC    : 수집 종료 + 저장

의존성:
    pip install pynput h5py numpy Pillow

카메라:
    OpenCV 웹캠 (--cam-id) 또는 ROS2 토픽 이미지 (--ros-image-topic)
"""

import argparse
import queue
import threading
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

try:
    from pynput import keyboard as pk
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[Collector] pynput 없음. pip install pynput")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ─────────────────────────────────────────────
# 가감속 Ramp 컨트롤러
# ─────────────────────────────────────────────

class RampController:
    """
    키 누름/뗌 이벤트를 받아 부드러운 속도 명령을 생성.

    Args:
        max_vel:      최대 속도 절댓값 (기본 1.15)
        ramp_up_s:    최대 속도까지 도달하는 시간 (초)
        ramp_down_s:  키 뗌 후 0까지 감속하는 시간 (초)
        curve:        "quadratic" | "linear"
    """

    # 키 → (action_dim_idx, sign)
    KEY_MAP = {
        "w": (0, +1.0), "up":    (0, +1.0),
        "s": (0, -1.0), "down":  (0, -1.0),
        "a": (1, +1.0), "left":  (1, +1.0),
        "d": (1, -1.0), "right": (1, -1.0),
        "q": (2, +1.0),
        "e": (2, -1.0),
    }

    def __init__(
        self,
        max_vel: float = 1.15,
        ramp_up_s: float = 0.5,
        ramp_down_s: float = 0.15,
        curve: str = "quadratic",
    ):
        self.max_vel = max_vel
        self.ramp_up_s = ramp_up_s
        self.ramp_down_s = ramp_down_s
        self.curve = curve
        self.action_dim = 3

        # 각 차원별 상태
        self._press_time: dict[int, float] = {}     # dim → 누름 시작 시각
        self._release_time: dict[int, float] = {}   # dim → 뗌 시각
        self._direction: dict[int, float] = {}      # dim → +1 or -1
        self._lock = threading.Lock()

    def key_down(self, key_name: str):
        if key_name not in self.KEY_MAP:
            return
        dim, sign = self.KEY_MAP[key_name]
        with self._lock:
            if dim not in self._press_time:
                self._press_time[dim] = time.monotonic()
                self._direction[dim] = sign
                # 방향 반전 시 release 정보 리셋
                self._release_time.pop(dim, None)

    def key_up(self, key_name: str):
        if key_name not in self.KEY_MAP:
            return
        dim, _ = self.KEY_MAP[key_name]
        with self._lock:
            if dim in self._press_time:
                del self._press_time[dim]
                self._release_time[dim] = time.monotonic()

    def get_action(self) -> np.ndarray:
        """현재 시각 기준 속도 벡터 반환 (action_dim,)"""
        now = time.monotonic()
        action = np.zeros(self.action_dim, dtype=np.float32)

        with self._lock:
            for dim in range(self.action_dim):
                if dim in self._press_time:
                    # 가속 구간
                    elapsed = now - self._press_time[dim]
                    ratio = min(elapsed / self.ramp_up_s, 1.0)
                    if self.curve == "quadratic":
                        speed = self.max_vel * ratio ** 2
                    else:
                        speed = self.max_vel * ratio
                    action[dim] = self._direction.get(dim, 1.0) * speed

                elif dim in self._release_time:
                    # 감속 구간
                    elapsed = now - self._release_time[dim]
                    if elapsed >= self.ramp_down_s:
                        del self._release_time[dim]
                    else:
                        ratio = 1.0 - elapsed / self.ramp_down_s
                        if self.curve == "quadratic":
                            speed = self.max_vel * ratio ** 2
                        else:
                            speed = self.max_vel * ratio
                        action[dim] = self._direction.get(dim, 1.0) * speed

        return action


# ─────────────────────────────────────────────
# 동기화 레코더
# ─────────────────────────────────────────────

class SyncRecorder:
    """
    카메라 이미지(저주파)와 모터 명령(고주파)을 별도로 수집 후
    타임스탬프 기반 nearest-neighbor 동기화하여 HDF5에 저장.
    """

    def __init__(self):
        self._images: list[tuple[float, np.ndarray]] = []   # (ts, HWC)
        self._actions: list[tuple[float, np.ndarray]] = []  # (ts, 3)
        self._lock = threading.Lock()

    def record_image(self, image: np.ndarray):
        with self._lock:
            self._images.append((time.monotonic(), image.copy()))

    def record_action(self, action: np.ndarray):
        with self._lock:
            self._actions.append((time.monotonic(), action.copy()))

    def save(self, output_path: str, task: str):
        """수집 데이터를 HDF5 v6 포맷으로 저장"""
        with self._lock:
            images = list(self._images)
            actions = list(self._actions)

        if not images or not actions:
            print("[Recorder] 저장할 데이터 없음")
            return

        # 이미지 타임스탬프에 맞게 액션 보간 (nearest-neighbor)
        img_ts  = np.array([t for t, _ in images])
        act_ts  = np.array([t for t, _ in actions])
        act_arr = np.stack([a for _, a in actions])   # (M, 3)

        synced_actions = []
        for ts in img_ts:
            idx = int(np.argmin(np.abs(act_ts - ts)))
            synced_actions.append(act_arr[idx])
        synced_actions = np.stack(synced_actions)  # (N, 3)

        img_arr = np.stack([img for _, img in images])  # (N, H, W, 3)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f:
            # v5/v6 포맷
            obs_grp = f.create_group("observations")
            obs_grp.create_dataset("images", data=img_arr, compression="gzip", compression_opts=4)
            f.create_dataset("actions", data=synced_actions)
            f.attrs["language_instruction"] = task
            f.attrs["num_frames"] = len(img_arr)
            f.attrs["collector"] = "keyboard_ramp"

        print(f"[Recorder] 저장 완료: {output_path} ({len(img_arr)} 프레임)")

    def clear(self):
        with self._lock:
            self._images.clear()
            self._actions.clear()


# ─────────────────────────────────────────────
# 메인 수집 루프
# ─────────────────────────────────────────────

class KeyboardCollector:
    def __init__(self, args):
        self.args = args
        self.ramp = RampController(max_vel=1.15, ramp_up_s=0.5, ramp_down_s=0.15)
        self.recorder = SyncRecorder()
        self._stop = False
        self._paused = False

    def _on_key_press(self, key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name.lower()
        except AttributeError:
            return

        if name == "space":
            self._paused = not self._paused
            print(f"[일시정지: {self._paused}]")
        elif name == "escape":
            self._stop = True
            return False  # pynput: 리스너 중단
        else:
            self.ramp.key_down(name)

    def _on_key_release(self, key):
        try:
            name = key.char.lower() if hasattr(key, "char") else key.name.lower()
        except AttributeError:
            return
        self.ramp.key_up(name)

    def _capture_loop(self, cam_id: int, target_hz: float):
        """카메라 이미지 캡처 스레드"""
        if not CV2_AVAILABLE:
            print("[Collector] OpenCV 없음. 더미 이미지를 사용합니다.")
            interval = 1.0 / target_hz
            while not self._stop:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                if not self._paused:
                    self.recorder.record_image(dummy)
                time.sleep(interval)
            return

        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"[Collector] 카메라 {cam_id} 열기 실패. 더미 이미지 사용.")
            self._capture_loop_dummy(target_hz)
            return

        interval = 1.0 / target_hz
        while not self._stop:
            t0 = time.monotonic()
            ret, frame = cap.read()
            if ret and not self._paused:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.recorder.record_image(rgb)
            elapsed = time.monotonic() - t0
            time.sleep(max(0, interval - elapsed))

        cap.release()

    def _capture_loop_dummy(self, target_hz: float):
        interval = 1.0 / target_hz
        while not self._stop:
            if not self._paused:
                dummy = np.zeros((480, 640, 3), dtype=np.uint8)
                self.recorder.record_image(dummy)
            time.sleep(interval)

    def run(self):
        if self.args.dry_run:
            self._dry_run()
            return

        if not PYNPUT_AVAILABLE:
            print("[Collector] pynput을 설치해야 합니다: pip install pynput")
            return

        print(f"[Collector] 수집 시작 | 태스크: '{self.args.task}'")
        print("  W/S/A/D/Q/E: 이동 | SPACE: 일시정지 | ESC: 종료+저장")

        # 카메라 스레드 시작
        cam_thread = threading.Thread(
            target=self._capture_loop,
            args=(self.args.cam_id, self.args.cam_hz),
            daemon=True,
        )
        cam_thread.start()

        # 액션 레코딩 스레드
        interval = 1.0 / self.args.hz

        def action_loop():
            while not self._stop:
                t0 = time.monotonic()
                action = self.ramp.get_action()
                if not self._paused:
                    self.recorder.record_action(action)
                elapsed = time.monotonic() - t0
                time.sleep(max(0, interval - elapsed))

        action_thread = threading.Thread(target=action_loop, daemon=True)
        action_thread.start()

        # 키보드 리스너 (블로킹)
        with pk.Listener(on_press=self._on_key_press, on_release=self._on_key_release) as listener:
            listener.join()

        self._stop = True
        cam_thread.join(timeout=2.0)
        action_thread.join(timeout=2.0)

        self.recorder.save(self.args.output, self.args.task)

    def _dry_run(self):
        """가감속 속도 프로필을 콘솔에 출력 (matplotlib 없이)"""
        print("[DRY RUN] W 키 0.6초 누름 → 뗌 시 속도 시뮬레이션")
        ramp = RampController()
        ramp.key_down("w")
        timeline = []
        for i in range(90):
            t = i * 0.01
            if t > 0.6:
                ramp.key_up("w")
            v = ramp.get_action()[0]
            timeline.append((t, v))
            bar = "#" * int(v * 30)
            print(f"  t={t:.2f}s  vx={v:.3f}  {bar}")
        print("[DRY RUN] 완료")


# ─────────────────────────────────────────────
# 진입점
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoNa-pi 키보드 수집기 (Ramp 가감속)")
    parser.add_argument("--output",  default="data/collected/episode.h5", help="저장 경로")
    parser.add_argument("--task",    default="Navigate to the goal", help="언어 지시어")
    parser.add_argument("--hz",      type=float, default=20.0, help="액션 레코딩 Hz")
    parser.add_argument("--cam-id",  type=int,   default=0,    help="OpenCV 카메라 인덱스")
    parser.add_argument("--cam-hz",  type=float, default=2.0,  help="이미지 캡처 Hz")
    parser.add_argument("--dry-run", action="store_true", help="속도 프로필 테스트만 실행")
    args = parser.parse_args()

    collector = KeyboardCollector(args)
    collector.run()


if __name__ == "__main__":
    main()
