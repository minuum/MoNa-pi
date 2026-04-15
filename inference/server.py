"""
MoNa-pi FastAPI 추론 서버

엔드포인트:
    POST /predict   — 이미지 + instruction → 액션 청크 (10x3)
    GET  /health    — 서버 상태 확인
    GET  /metrics   — 요청 수 / 평균 지연시간

실행:
    python inference/server.py \
        --ckpt  checkpoints/best \
        --host  0.0.0.0 \
        --port  8080 \
        --steps 5

의존성:
    pip install fastapi uvicorn pydantic numpy pillow
"""

import argparse
import asyncio
import base64
import io
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference.engine import MoNaPiEngine


# ─────────────────────────────────────────────
# 요청 / 응답 스키마
# ─────────────────────────────────────────────

class PredictRequest(BaseModel):
    image_b64: str          # base64 인코딩된 PNG/JPEG 이미지
    instruction: str = "Navigate to the goal"

class PredictResponse(BaseModel):
    actions: list[list[float]]  # (horizon, 3) — [[vx, vy, wz], ...]
    latency_ms: float


# ─────────────────────────────────────────────
# FastAPI 앱
# ─────────────────────────────────────────────

app = FastAPI(title="MoNa-pi Inference Server", version="1.0")

# 전역 상태
_engine: MoNaPiEngine | None = None
_request_count: int = 0
_latency_history: deque[float] = deque(maxlen=100)
_inference_lock = asyncio.Lock()  # 동시 추론 방지 (GPU 단일 스트림)


_mock_mode: bool = False  # --mock 플래그 시 모델 로드 없이 0 액션 반환


def get_engine() -> MoNaPiEngine:
    if not _mock_mode and _engine is None:
        raise HTTPException(status_code=503, detail="엔진이 초기화되지 않았습니다")
    return _engine


# ─────────────────────────────────────────────
# 엔드포인트
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine_ready": _engine is not None,
        "request_count": _request_count,
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


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    global _request_count

    # ── Mock 모드: 모델 없이 즉시 0 액션 반환 ──────────────────
    if _mock_mode:
        _request_count += 1
        _latency_history.append(1.0)
        return PredictResponse(
            actions=np.zeros((10, 3)).tolist(),
            latency_ms=1.0,
        )

    engine = get_engine()

    # base64 → numpy 이미지 디코딩
    try:
        img_bytes = base64.b64decode(req.image_b64)
        from PIL import Image as PILImage
        pil_img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        image_np = np.array(pil_img, dtype=np.uint8)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 디코딩 실패: {e}")

    # 추론 (GPU 락으로 직렬화)
    async with _inference_lock:
        try:
            actions_raw, latency_ms = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: engine.predict_latency(image_np, req.instruction),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"추론 실패: {e}")

    _request_count += 1
    _latency_history.append(latency_ms)

    return PredictResponse(
        actions=actions_raw.tolist(),
        latency_ms=round(latency_ms, 2),
    )


# ─────────────────────────────────────────────
# 서버 시작
# ─────────────────────────────────────────────

def build_engine(args) -> MoNaPiEngine:
    engine = MoNaPiEngine(
        model_path=args.ckpt,
        device=args.device,
        solver=args.solver,
        n_ode_steps=args.steps,
        use_paligemma=not args.legacy_backbone,
        load_pretrained_paligemma=args.pretrained_paligemma,
    )
    engine.warmup()
    return engine


def main():
    parser = argparse.ArgumentParser(description="MoNa-pi 추론 서버")
    parser.add_argument("--ckpt",   default="checkpoints/best", help="체크포인트 경로")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--solver", default="heun", choices=["euler", "heun", "dpm"])
    parser.add_argument("--steps",  type=int, default=5, help="ODE 스텝 수")
    parser.add_argument("--legacy-backbone",     action="store_true",
                        help="이전 Pi0Backbone 사용 (use_paligemma=False)")
    parser.add_argument("--pretrained-paligemma", action="store_true",
                        help="google/paligemma-3b-pt-224 다운로드 사용")
    parser.add_argument("--mock",   action="store_true",
                        help="모델 로드 없이 0 액션 반환 (통합 테스트용)")
    args = parser.parse_args()

    global _engine, _mock_mode
    _mock_mode = args.mock
    if _mock_mode:
        print("[Server] MOCK 모드 — 모델 로드 없이 실행")
    else:
        _engine = build_engine(args)

    print(f"[Server] 서버 시작: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
