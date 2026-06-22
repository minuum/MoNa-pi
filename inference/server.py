"""
MoNa-pi 통합 추론 서버 (포트 8082)

MoNa-pi 네이티브 + MoNaVLA proxy_inference_server API를 한 프로세스에서 제공.
엔진은 하나만 띄워 메모리를 공유한다.

MoNa-pi 네이티브:
    POST /predict       image_b64 필드 → actions (10×3 청크)
    GET  /health
    GET  /metrics

MoNaVLA 호환 (proxy_inference_server 동일 프로토콜):
    POST /predict       image 필드 → action / action_3d / predicted_label
    POST /reset         no-op (stateless)
    POST /config        mode(single|chunk) / n_ode_steps 변경
    GET  /model/info
    POST /model/load    런타임 ckpt 교체
    GET  /              서비스 정보

실행:
    python inference/server.py --config configs/serbot2.yaml --ckpt checkpoints/best
    python inference/server.py --mock   # 테스트용 (모델 로드 없음)
"""

import argparse
import asyncio
import base64
import io
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, model_validator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.engine import MoNaPiEngine
from inference.ode_solver import build_solver


# ─────────────────────────────────────────────
# 액션 클래스 분류
# ─────────────────────────────────────────────

def classify_action(vx: float, vy: float, wz: float) -> tuple[int, str]:
    if abs(vx) < 0.1 and abs(vy) < 0.1 and abs(wz) < 0.1:
        return 0, "STOP"
    if abs(wz) > 0.15 and abs(vx) < 0.2 and abs(vy) < 0.2:
        return (6, "ROT_L") if wz > 0 else (7, "ROT_R")
    if vx > 0.3 and vy > 0.3:  return 4, "FWD+L"
    if vx > 0.3 and vy < -0.3: return 5, "FWD+R"
    if vx > 0.3:                return 1, "FORWARD"
    if vy > 0.3:                return 2, "LEFT"
    if vy < -0.3:               return 3, "RIGHT"
    return 1, "FORWARD"


# ─────────────────────────────────────────────
# 인증 (선택적)
# ─────────────────────────────────────────────

_api_key: str | None = os.getenv("MONAPI_API_KEY") or os.getenv("VLA_API_KEY")


async def _verify_key(x_api_key: Optional[str] = Header(default=None)):
    if _api_key and x_api_key != _api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")


# ─────────────────────────────────────────────
# 스키마
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    """image_b64(MoNa-pi) 또는 image(MoNaVLA) 둘 다 허용"""
    image_b64: Optional[str] = None
    image: Optional[str] = None
    instruction: str = "Navigate to the goal"
    # MoNaVLA 호환 필드 (무시)
    vlm_model: Optional[str] = None
    snap_to_grid: Optional[bool] = None
    snap_threshold: Optional[float] = None

    @model_validator(mode="after")
    def coerce_image(self) -> "PredictRequest":
        if self.image_b64 is None and self.image is not None:
            self.image_b64 = self.image
        if self.image_b64 is None:
            raise ValueError("image_b64 또는 image 중 하나는 필수")
        return self


class PredictResponse(BaseModel):
    """MoNa-pi 네이티브 + MoNaVLA 호환 필드를 모두 포함"""
    # MoNa-pi 네이티브
    actions: list[list[float]]       # (horizon, 3) 전체 청크
    latency_ms: float
    # MoNaVLA 호환
    action: list[float]              # actions[0][:2]
    action_3d: list[float]           # actions[0]
    model_name: str = "monapi"
    strategy: str = "single_action"
    source: str = "monapi"
    chunk_size: int = 10
    buffer_status: dict[str, Any] = {}
    predicted_class: int = 1
    predicted_label: str = "FORWARD"
    goal_near_proxy: bool = False
    grounding_latency_ms: float = 0.0
    bbox: Optional[dict[str, Any]] = None
    grounding_caption: Optional[str] = None
    instruction_used: bool = True
    # chunk 모드 전용
    chunk_index: int = 0
    # 추가 호환성 필드
    chunk: Optional[list[list[float]]] = None
    speed_scale: Optional[float] = 1.0
    grounding_cached: Optional[bool] = False
    matched_path_type: Optional[str] = None


class ConfigRequest(BaseModel):
    # MoNaVLA 호환 (무시되는 것)
    speed_scaling: Optional[bool] = None
    grounding_skip_n: Optional[int] = None
    smooth_enabled: Optional[bool] = None
    smooth_alpha_xy: Optional[float] = None
    smooth_alpha_az: Optional[float] = None
    model: Optional[str] = None
    # MoNa-pi 전용
    mode: Optional[str] = None         # "single" | "chunk"
    n_ode_steps: Optional[int] = None


class ModelLoadRequest(BaseModel):
    checkpoint_path: str
    config_path: Optional[str] = None
    precision: Optional[str] = "fp16"
    refresh: Optional[bool] = False


class SwitchRequest(BaseModel):
    model_name: str


# ─────────────────────────────────────────────
# 전역 상태
# ─────────────────────────────────────────────

app = FastAPI(title="MoNa-pi Inference Server", version="1.0")

_engine: MoNaPiEngine | None = None
_mock_mode: bool = False
_active_mode: str = "single"   # "single" | "chunk"
_request_count: int = 0
_latency_history: deque[float] = deque(maxlen=100)
_inference_lock = asyncio.Lock()
_chunk_idx: int = 0


def _gpu_memory() -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            return {
                "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
                "reserved_gb":  round(torch.cuda.memory_reserved()  / 1e9, 3),
                "device_name":  torch.cuda.get_device_name(0),
            }
    except Exception:
        pass
    return {}


# ─────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {"service": "MoNa-pi Inference Server", "version": "1.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy" if (_engine is not None or _mock_mode) else "not_ready",
        # MoNa-pi 네이티브
        "engine_ready": _engine is not None or _mock_mode,
        "request_count": _request_count,
        # MoNaVLA 호환
        "active_model": "monapi",
        "model_loaded": _engine is not None or _mock_mode,
        "active_mode": _active_mode,
        "gpu_memory": _gpu_memory(),
    }


@app.get("/metrics")
async def metrics():
    avg_lat = float(np.mean(_latency_history)) if _latency_history else 0.0
    p95_lat = float(np.percentile(list(_latency_history), 95)) if len(_latency_history) >= 5 else 0.0
    return {
        "request_count": _request_count,
        "avg_latency_ms": round(avg_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
    }


@app.get("/model/info")
async def model_info():
    is_ready = _engine is not None or _mock_mode
    ckpt_path = str(_engine.model_path) if _engine else "mock"
    return {
        "model_name":   "monapi",
        "checkpoint":   ckpt_path,
        "action_dim":   _engine.action_dim if _engine else 3,
        "horizon":      _engine.horizon    if _engine else 10,
        "n_ode_steps":  _engine.n_ode_steps if _engine else 5,
        "device":       str(_engine.device) if _engine else "cpu",
        "engine_ready": is_ready,
        "active_mode":  _active_mode,
        # MoNaVLA 프로토콜 호환용 필드 추가
        "model_loaded":    is_ready,
        "checkpoint_path": ckpt_path,
        "precision":       "fp32",
        "config_path":     "N/A",
    }


@app.post("/reset")
async def reset(x_api_key: Optional[str] = Header(default=None)):
    await _verify_key(x_api_key)
    global _chunk_idx
    _chunk_idx = 0
    return {"status": "ok"}


@app.post("/config")
async def set_config(req: ConfigRequest, x_api_key: Optional[str] = Header(default=None)):
    global _active_mode
    await _verify_key(x_api_key)
    changed: dict[str, Any] = {}
    if req.mode is not None:
        if req.mode not in ("single", "chunk"):
            raise HTTPException(status_code=400, detail="mode는 'single' 또는 'chunk'")
        _active_mode = req.mode
        changed["mode"] = _active_mode
    if req.n_ode_steps is not None and _engine is not None:
        _engine.n_ode_steps = req.n_ode_steps
        _engine.solver = build_solver("heun", req.n_ode_steps)
        changed["n_ode_steps"] = req.n_ode_steps
    return {"status": "ok", "active_model": "monapi", "active_mode": _active_mode, "changed": changed}


@app.post("/model/switch")
async def model_switch(req: SwitchRequest, x_api_key: Optional[str] = Header(default=None)):
    await _verify_key(x_api_key)
    return {"status": "ok", "model": req.model_name}


@app.post("/model/load")
async def model_load(req: ModelLoadRequest, x_api_key: Optional[str] = Header(default=None)):
    await _verify_key(x_api_key)
    global _engine
    if _mock_mode:
        return {"status": "ok", "checkpoint": req.checkpoint_path, "note": "mock mode"}
    if _engine is None:
        raise HTTPException(status_code=503, detail="엔진 미초기화")
    _engine.model_path = Path(req.checkpoint_path)
    _engine.warmup()
    return {"status": "ok", "checkpoint": req.checkpoint_path}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, x_api_key: Optional[str] = Header(default=None)):
    global _request_count, _chunk_idx
    await _verify_key(x_api_key)

    if _mock_mode:
        _request_count += 1
        _latency_history.append(1.0)
        return _build_response(np.zeros((10, 3), dtype=np.float32), latency_ms=1.0)

    if _engine is None:
        raise HTTPException(status_code=503, detail="엔진이 초기화되지 않았습니다")

    try:
        img_bytes = base64.b64decode(req.image_b64)  # type: ignore[arg-type]
        from PIL import Image as PILImage
        pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {e}")

    async with _inference_lock:
        try:
            actions_raw, latency_ms = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: _engine.predict_latency(image_np, req.instruction),  # type: ignore[union-attr]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"추론 실패: {e}")

    _request_count += 1
    _latency_history.append(latency_ms)
    _chunk_idx = 0
    return _build_response(actions_raw, latency_ms=latency_ms)


def _build_response(actions: np.ndarray, latency_ms: float) -> PredictResponse:
    global _chunk_idx
    step = actions[_chunk_idx] if _active_mode == "chunk" else actions[0]
    vx, vy, wz = float(step[0]), float(step[1]), float(step[2])
    cls_idx, label = classify_action(vx, vy, wz)
    if _active_mode == "chunk":
        _chunk_idx = min(_chunk_idx + 1, len(actions) - 1)
    horizon = len(actions)
    remaining = horizon - _chunk_idx if _active_mode == "chunk" else horizon
    return PredictResponse(
        actions=actions.tolist(),
        latency_ms=round(latency_ms, 2),
        action=[vx, vy],
        action_3d=[vx, vy, wz],
        strategy="chunk_playback" if _active_mode == "chunk" else "single_action",
        chunk_size=horizon,
        buffer_status={"mode": _active_mode, "horizon": horizon,
                       "remaining": remaining, "chunk_idx": _chunk_idx},
        predicted_class=cls_idx,
        predicted_label=label,
        goal_near_proxy=(label == "STOP"),
        chunk_index=_chunk_idx,
        chunk=actions.tolist(),
        speed_scale=1.0,
        grounding_cached=False,
        matched_path_type=None,
    )


# ─────────────────────────────────────────────
# 서버 시작
# ─────────────────────────────────────────────

def build_engine(args, cfg: dict) -> MoNaPiEngine:
    m = cfg.get("model", {})
    d = cfg.get("data", {})
    return MoNaPiEngine(
        model_path=args.ckpt,
        device=args.device,
        solver=args.solver,
        n_ode_steps=args.steps,
        action_dim=m.get("action_dim", 3),
        horizon=m.get("horizon", 10),
        hidden_dim=m.get("hidden_dim", 512),
        use_paligemma=m.get("use_paligemma", True),
        load_pretrained_paligemma=m.get("load_pretrained_paligemma", False),
        use_int8=m.get("use_int8", False),
        paligemma_id=m.get("paligemma_id", "google/paligemma-3b-pt-224"),
        vision_model_id=m.get("vision_model_id", "google/siglip-so400m-patch14-384"),
        lang_model_id=m.get("lang_model_id", "google/gemma-2b"),
        image_size=d.get("image_size", 224),
    )


def main():
    parser = argparse.ArgumentParser(description="MoNa-pi 통합 추론 서버")
    parser.add_argument("--config", default="configs/serbot2.yaml")
    parser.add_argument("--ckpt",   default="checkpoints/best")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8082)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solver", default="heun", choices=["euler", "heun", "dpm"])
    parser.add_argument("--steps",  type=int, default=5)
    parser.add_argument("--mock",   action="store_true")
    args = parser.parse_args()

    global _engine, _mock_mode
    _mock_mode = args.mock

    if _mock_mode:
        print("[Server] MOCK 모드 — 모델 로드 없이 실행")
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        _engine = build_engine(args, cfg)
        _engine.warmup()

    print(f"[Server] 시작: http://{args.host}:{args.port}")
    print(f"  MoNa-pi 네이티브: image_b64 → actions (청크)")
    print(f"  MoNaVLA 호환:     image     → action_3d / predicted_label")
    print(f"  Dashboard 연결:   VLA_API_SERVER=http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
