"""
MoNa-pi → MoNaVLA 호환 서버 (포트 8082)

proxy_inference_server.py (8001) 와 동일한 API 프로토콜을 사용하므로
VLA_API_SERVER=http://localhost:8082 한 줄로 기존 Dashboard(7865) / Hub(7860) 에서
MoNa-pi(π0 Flow Matching)를 "모드"로 선택 가능.

실행 모드:
    single (기본) — 매 프레임 predict, actions[0] 즉시 실행 (MoNaVLA 동일 패턴)
    chunk         — 10스텝 청크 반환, 청크 인식 클라이언트가 소비

실행:
    python inference/monavla_server.py --mock --port 8082
    python inference/monavla_server.py --ckpt checkpoints/best --port 8082
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
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.engine import ActionNormalizer, MoNaPiEngine
from inference.ode_solver import build_solver


# ─────────────────────────────────────────────
# 액션 클래스 분류 (π0 연속 출력 → 8-class)
# ─────────────────────────────────────────────

ACTION_CLASSES: dict[int, str] = {
    0: "STOP",   1: "FORWARD", 2: "LEFT",  3: "RIGHT",
    4: "FWD+L",  5: "FWD+R",  6: "ROT_L", 7: "ROT_R",
}


def classify_action(vx: float, vy: float, wz: float) -> tuple[int, str]:
    if abs(vx) < 0.1 and abs(vy) < 0.1 and abs(wz) < 0.1:
        return 0, "STOP"
    if abs(wz) > 0.15 and abs(vx) < 0.2 and abs(vy) < 0.2:
        return (6, "ROT_L") if wz > 0 else (7, "ROT_R")
    if vx > 0.3 and vy > 0.3:   return 4, "FWD+L"
    if vx > 0.3 and vy < -0.3:  return 5, "FWD+R"
    if vx > 0.3:                 return 1, "FORWARD"
    if vy > 0.3:                 return 2, "LEFT"
    if vy < -0.3:                return 3, "RIGHT"
    return 1, "FORWARD"


# ─────────────────────────────────────────────
# 스키마 (proxy_inference_server 호환)
# ─────────────────────────────────────────────

class InferenceRequest(BaseModel):
    image: str
    instruction: str = "Navigate to the goal"
    vlm_model: Optional[str] = "monapi"


class InferenceResponse(BaseModel):
    action: list[float]
    action_3d: list[float]
    latency_ms: float
    model_name: str = "monapi"
    strategy: str
    source: str = "monapi"
    buffer_status: dict[str, Any]
    predicted_class: int
    predicted_label: str
    goal_near_proxy: bool = False
    grounding_latency_ms: float = 0.0
    bbox: Optional[dict[str, Any]] = None
    grounding_caption: Optional[str] = None
    instruction_used: bool = True
    speed_scale: Optional[float] = None
    grounding_cached: Optional[bool] = None
    matched_path_type: Optional[str] = None
    chunk: Optional[list[list[float]]] = None
    chunk_index: int = 0


class ConfigRequest(BaseModel):
    speed_scaling: Optional[bool] = None
    grounding_skip_n: Optional[int] = None
    smooth_enabled: Optional[bool] = None
    smooth_alpha_xy: Optional[float] = None
    smooth_alpha_az: Optional[float] = None
    model: Optional[str] = None
    mode: Optional[str] = None
    n_ode_steps: Optional[int] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    checkpoint: str
    action_dim: int
    horizon: int
    n_ode_steps: int
    device: str
    engine_ready: bool
    active_mode: str


class ModelLoadRequest(BaseModel):
    checkpoint_path: str
    config_path: Optional[str] = None
    precision: Optional[str] = "fp16"
    refresh: Optional[bool] = False


# ─────────────────────────────────────────────
# 인증
# ─────────────────────────────────────────────

_api_key: str | None = os.getenv("MONAPI_API_KEY")


async def _verify_key(x_api_key: Optional[str] = Header(default=None)):
    if _api_key and x_api_key != _api_key:
        raise HTTPException(status_code=403, detail="Invalid API Key")


# ─────────────────────────────────────────────
# 전역 상태
# ─────────────────────────────────────────────

app = FastAPI(title="MoNa-pi MoNaVLA-Compat Server", version="1.0")

_engine: MoNaPiEngine | None = None
_mock_mode: bool = False
_active_mode: str = "single"
_request_count: int = 0
_latency_history: deque[float] = deque(maxlen=100)
_inference_lock = asyncio.Lock()

_current_chunk: np.ndarray | None = None
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

@app.get("/health")
async def health():
    return {
        "status": "healthy" if (_engine is not None or _mock_mode) else "not_ready",
        "active_model": "monapi",
        "model_loaded": _engine is not None or _mock_mode,
        "active_mode": _active_mode,
        "gpu_memory": _gpu_memory(),
    }


@app.get("/model/info", response_model=ModelInfoResponse)
async def model_info():
    if _engine is None and not _mock_mode:
        raise HTTPException(status_code=503, detail="엔진 미초기화")
    return ModelInfoResponse(
        model_name="monapi",
        checkpoint=str(_engine.model_path) if _engine else "mock",
        action_dim=_engine.action_dim if _engine else 3,
        horizon=_engine.horizon if _engine else 10,
        n_ode_steps=_engine.n_ode_steps if _engine else 5,
        device=str(_engine.device) if _engine else "cpu",
        engine_ready=_engine is not None or _mock_mode,
        active_mode=_active_mode,
    )


@app.post("/reset")
async def reset(x_api_key: Optional[str] = Header(default=None)):
    await _verify_key(x_api_key)
    global _current_chunk, _chunk_idx
    _current_chunk = None
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


@app.post("/predict", response_model=InferenceResponse)
async def predict(req: InferenceRequest, x_api_key: Optional[str] = Header(default=None)):
    global _request_count, _current_chunk, _chunk_idx
    await _verify_key(x_api_key)

    if _mock_mode:
        _request_count += 1
        return _build_response(np.zeros((10, 3), dtype=np.float32), latency_ms=1.0)

    if _engine is None:
        raise HTTPException(status_code=503, detail="엔진 미초기화")

    try:
        img_bytes = base64.b64decode(req.image)
        from PIL import Image as PILImage
        pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {e}")

    async with _inference_lock:
        try:
            actions, latency_ms = await asyncio.get_event_loop().run_in_executor(
                None, lambda: _engine.predict_latency(image_np, req.instruction),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"추론 실패: {e}")

    _request_count += 1
    _latency_history.append(latency_ms)
    _current_chunk = actions
    _chunk_idx = 0
    return _build_response(actions, latency_ms=latency_ms)


def _build_response(actions: np.ndarray, latency_ms: float) -> InferenceResponse:
    global _chunk_idx
    step = actions[_chunk_idx] if _active_mode == "chunk" else actions[0]
    vx, vy, wz = float(step[0]), float(step[1]), float(step[2])
    cls_idx, label = classify_action(vx, vy, wz)
    if _active_mode == "chunk":
        _chunk_idx = min(_chunk_idx + 1, len(actions) - 1)
    horizon = len(actions)
    remaining = horizon - _chunk_idx if _active_mode == "chunk" else horizon
    return InferenceResponse(
        action=[vx, vy],
        action_3d=[vx, vy, wz],
        latency_ms=round(latency_ms, 2),
        strategy="chunk_playback" if _active_mode == "chunk" else "single_action",
        buffer_status={"mode": _active_mode, "horizon": horizon,
                       "remaining": remaining, "chunk_idx": _chunk_idx},
        predicted_class=cls_idx,
        predicted_label=label,
        chunk=actions.tolist() if _active_mode == "chunk" else None,
        chunk_index=_chunk_idx,
    )


# ─────────────────────────────────────────────
# 서버 시작
# ─────────────────────────────────────────────

def _build_engine(args, cfg: dict) -> MoNaPiEngine:
    m = cfg.get("model", {})
    d = cfg.get("data", {})
    engine = MoNaPiEngine(
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
    engine.warmup()
    return engine


def main():
    parser = argparse.ArgumentParser(description="MoNa-pi MoNaVLA-호환 서버 (포트 8082)")
    parser.add_argument("--config", default="configs/serbot2.yaml")
    parser.add_argument("--ckpt",   default="checkpoints/best")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8082)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solver", default="heun", choices=["euler", "heun", "dpm"])
    parser.add_argument("--steps",  type=int, default=5)
    parser.add_argument("--mode",   default="single", choices=["single", "chunk"])
    parser.add_argument("--mock",   action="store_true")
    args = parser.parse_args()

    global _engine, _mock_mode, _active_mode
    _active_mode = args.mode
    _mock_mode = args.mock

    if _mock_mode:
        print(f"[MonaVLA-Server] MOCK 모드 (mode={_active_mode})")
    else:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        _engine = _build_engine(args, cfg)

    print(f"[MonaVLA-Server] 시작: http://{args.host}:{args.port}  mode={_active_mode}")
    print(f"  → VLA_API_SERVER=http://localhost:{args.port} 으로 Dashboard 연결 가능")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
