"""
종단간 통합 테스트

실제 모델/ROS2 없이 다음 연결을 검증:
  ActionChunkBuffer  ←  mock 액션 주입
  inference server   ←  --mock 모드로 기동 후 /predict POST
"""

import base64
import io
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────────────────────────────────────────
# ActionChunkBuffer 통합 테스트
# ─────────────────────────────────────────────────────────────

class TestActionChunkBuffer:
    def setup_method(self):
        from robot.action_buffer import ActionChunkBuffer
        self.buf = ActionChunkBuffer(horizon=10)

    def test_push_and_pop(self):
        chunk = np.random.randn(10, 3).astype(np.float32)
        self.buf.push(chunk)
        for _ in range(10):
            action = self.buf.pop()
            assert action.shape == (3,), f"pop shape: {action.shape}"

    def test_replan_at_half(self):
        chunk = np.random.randn(10, 3).astype(np.float32)
        self.buf.push(chunk)
        triggered = []
        for _ in range(10):
            self.buf.pop()
            triggered.append(self.buf.should_replan())
        # 50% 소진(5번 pop) 이후 최소 1번 True
        assert any(triggered), "should_replan이 한 번도 True가 아님"

    def test_emergency_stop_clears_buffer(self):
        chunk = np.random.randn(10, 3).astype(np.float32)
        self.buf.push(chunk)
        self.buf.emergency_stop()
        action = self.buf.pop()
        assert np.allclose(action, np.zeros(3)), f"비상정지 후 0이 아님: {action}"

    def test_empty_buffer_returns_zeros(self):
        action = self.buf.pop()
        assert action.shape == (3,)
        assert np.allclose(action, np.zeros(3))


# ─────────────────────────────────────────────────────────────
# inference/server.py mock 모드 통합 테스트
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_server():
    """--mock 모드로 서버를 기동하고 종료 시 정리"""
    proc = subprocess.Popen(
        [sys.executable, "inference/server.py", "--mock", "--port", "18080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    # 서버 준비 대기 (최대 5초)
    for _ in range(10):
        time.sleep(0.5)
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:18080/health", timeout=1)
            break
        except Exception:
            pass
    yield "http://localhost:18080"
    proc.terminate()
    proc.wait()


@pytest.mark.skipif(
    subprocess.run(
        [sys.executable, "-c", "import uvicorn, fastapi"],
        capture_output=True
    ).returncode != 0,
    reason="fastapi/uvicorn 미설치",
)
class TestMockServer:
    def _dummy_image_b64(self) -> str:
        from PIL import Image
        img = Image.new("RGB", (64, 64), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def test_health(self, mock_server):
        import urllib.request
        resp = urllib.request.urlopen(f"{mock_server}/health")
        data = json.loads(resp.read())
        assert data["status"] == "ok"

    def test_predict_returns_10x3(self, mock_server):
        import urllib.request
        payload = json.dumps({
            "image_b64": self._dummy_image_b64(),
            "instruction": "[FORWARD] 직진",
        }).encode()
        req = urllib.request.Request(
            f"{mock_server}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        actions = data["actions"]
        assert len(actions) == 10, f"horizon: {len(actions)}"
        assert len(actions[0]) == 3, f"action_dim: {len(actions[0])}"
        assert data["latency_ms"] >= 0

    def test_metrics_after_predict(self, mock_server):
        import urllib.request
        resp = urllib.request.urlopen(f"{mock_server}/metrics")
        data = json.loads(resp.read())
        assert "request_count" in data
        assert data["request_count"] >= 1
