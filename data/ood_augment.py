"""
OOD(분포 밖) 흉내 증강 — CH9-2(6/23) 진단 대응.

free_* 평가 14/15 실패가 정형 9종과 무관하게 전부 OOD 프로브(basket 극단
오프셋, 로봇 원거리/근접, 조명 변화)에 집중됨을 확인 — 기존 ColorJitter
(0.4)/RandomResizedCrop(0.8~1.0)로는 그 정도의 변형을 못 흉내냄.

다루는 것:
    - robot_close / robot_far  → random_zoom_pad (확대=크롭+업스케일, 축소=다운스케일+패딩)
    - basket_left/right_extreme → random_horizontal_shift (크롭 없이 평행이동+패딩)
    - lighting_diff             → 기존 ColorJitter 범위를 별도로 더 강하게 호출

다루지 못하는 것:
    - diagonal_left/right(로봇 시작 방향) — 3D 시점 변화라 2D 이미지 증강으로
      흉내내기 부적절(perspective warp는 왜곡이 너무 인위적). 실데이터 수집 필요.
"""
import random

import numpy as np
from PIL import Image as PILImage


def _mean_border_color(arr: np.ndarray) -> tuple:
    """가장자리 픽셀 평균색 — 패딩에 사용(검은 테두리보다 자연스러움)."""
    edge = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return tuple(int(v) for v in edge.mean(axis=0))


def random_zoom_pad(img: PILImage.Image, min_scale: float = 0.55, max_scale: float = 1.6) -> PILImage.Image:
    """scale>=1: 중앙 크롭 후 업스케일(robot_close 흉내) / scale<1: 다운스케일+패딩(robot_far 흉내)."""
    w, h = img.size
    scale = random.uniform(min_scale, max_scale)
    if scale >= 1.0:
        crop_w, crop_h = max(1, int(w / scale)), max(1, int(h / scale))
        x0, y0 = (w - crop_w) // 2, (h - crop_h) // 2
        return img.crop((x0, y0, x0 + crop_w, y0 + crop_h)).resize((w, h), PILImage.BILINEAR)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    small = img.resize((new_w, new_h), PILImage.BILINEAR)
    pad_color = _mean_border_color(np.array(small))
    canvas = PILImage.new("RGB", (w, h), pad_color)
    canvas.paste(small, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas


def random_horizontal_shift(img: PILImage.Image, max_shift_frac: float = 0.35) -> PILImage.Image:
    """좌우 평행이동 + 패딩 — basket이 화면 끝에 몰린 상황(basket_left/right_extreme) 흉내."""
    w, h = img.size
    shift = int(w * random.uniform(-max_shift_frac, max_shift_frac))
    pad_color = _mean_border_color(np.array(img))
    canvas = PILImage.new("RGB", (w, h), pad_color)
    canvas.paste(img, (shift, 0))
    return canvas


def sample_ood_params(p: float = 0.35) -> dict:
    """샘플(에피소드 윈도우) 단위로 한 번만 호출 — 윈도우 내 모든 프레임에 동일 변형 적용.

    프레임마다 독립적으로 다시 뽑으면 robot_close/far·basket_extreme 같은
    "그 순간의 정적인 상황"이 아니라 매 프레임 흔들리는 비현실적 지터가 됨.
    """
    return {
        "zoom": random.uniform(0.55, 1.6) if random.random() < p else None,
        "shift": random.uniform(-0.35, 0.35) if random.random() < p else None,
    }


def apply_ood_aug(img: PILImage.Image, params: dict) -> PILImage.Image:
    """sample_ood_params()로 윈도우당 한 번 뽑은 params를 모든 프레임에 동일 적용."""
    if params.get("zoom") is not None:
        img = random_zoom_pad(img, min_scale=params["zoom"], max_scale=params["zoom"])
    if params.get("shift") is not None:
        w, h = img.size
        pad_color = _mean_border_color(np.array(img))
        canvas = PILImage.new("RGB", (w, h), pad_color)
        canvas.paste(img, (int(w * params["shift"]), 0))
        img = canvas
    return img
