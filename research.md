# Action Expert 코드리서치

## 현재 구현 스택

```
ActionExpert (action_expert.py)
  └── FlowMatchingHead (flow_head.py)   ← 실제 구현
          ├── action_proj : Linear(3, 512)
          ├── time_mlp    : SinusoidalPosEmb → Linear(512,2048) → GELU → Linear(2048,512)
          ├── cond_proj   : Linear(2048, 512)
          ├── 4 × ModuleDict (self_attn, cross_attn, mlp, norm1~3)
          └── output_head : Linear(512, 3)
```

### 현재 시간 컨디셔닝 방식
```python
t_emb = time_mlp(t)           # (B, 1, 512)
h = action_proj(x_t) + t_emb  # 단순 덧셈
```

### 현재 Transformer Block 패턴
```python
# Pre-norm이지만 불완전
attn_out, _ = self_attn(norm1(h), h, h)   # query만 norm, key/value는 raw
h = h + attn_out
cross_out, _ = cross_attn(norm2(h), c, c)
h = h + cross_out
h = h + mlp(norm3(h))
```

---

## π0 논문의 Action Expert (정확한 구조)

### 출처
- π0 논문: "A Vision-Language-Action Flow Model" (Black et al., 2024)
- lerobot 레퍼런스 구현
- DiT (Diffusion Transformer, Peebles & Xie 2023) — 동일한 AdaLN-Zero 기법 사용

### 논문 Action Expert 핵심: AdaLN-Zero 시간 컨디셔닝

```
t → SinusoidalEmb → MLP → cond_emb (B, hidden_dim)

각 레이어마다:
  cond_emb → Linear(hidden_dim, 6*hidden_dim) → split:
    (α1, β1, γ1, α2, β2, γ2)

self-attn block:
  h = h + γ1 * self_attn( α1 * norm1(h) + β1 )
                ↑ scale + shift norm, gate output

cross-attn block:
  h = h + cross_attn( norm2(h), cond_vlm )    ← timestep 미적용 (VLM cross-attn은 그대로)

mlp block:
  h = h + γ2 * mlp( α2 * norm3(h) + β2 )
```

### 단순 덧셈 vs AdaLN 차이

| 항목 | 현재 (단순 덧셈) | π0 논문 (AdaLN) |
|------|-----------------|-----------------|
| 시간 적용 위치 | 입력에 한 번 더함 | 각 레이어 norm마다 scale/shift |
| Gate | 없음 | γ × attn_out (레이어별 게이팅) |
| 표현력 | 낮음 | 높음 (t에 따라 각 레이어 동작 변화) |
| 초기화 | 무관 | Zero-init → 학습 초기 identity |

---

## 수정 대상 파일

- `models/heads/flow_head.py` : FlowMatchingHead 전체 재작성
- `models/heads/action_expert.py` : 변경 없음 (wrapper만)

## 유지 대상 (변경 불필요)
- `models/pi0_core.py` : ActionExpert 인터페이스 동일
- `models/backbones/paligemma_backbone.py` : 변경 없음
- `training/train.py`, `inference/engine.py` : 변경 없음
