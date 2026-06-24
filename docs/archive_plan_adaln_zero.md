# Plan: Action Expert → π0 논문 AdaLN-Zero 방식으로 교체 (완료, 보관용)

> 커밋 144d3b1로 구현 완료. 신규 장기 계획은 `/plan.md` 참조.

## 변경 파일
- `models/heads/flow_head.py` **전체 재작성**
- `models/heads/action_expert.py` 변경 없음

---

## 핵심 변경: AdaLN-Zero 모듈 추가

### 1. `TimestepEmbedder` (기존 `time_mlp` 교체)
```python
class TimestepEmbedder(nn.Module):
    # Sinusoidal → Linear(dim, 4*dim) → SiLU → Linear(4*dim, dim)
    # 출력: (B, hidden_dim)  ← unsqueeze(1) 제거
```

### 2. `AdaLNModulation` (핵심 신규)
```python
class AdaLNModulation(nn.Module):
    # Linear(hidden_dim, 6 * hidden_dim) + SiLU
    # Zero-init weights/bias → 학습 초기 identity 보장
    # 출력: (α1, β1, γ1, α2, β2, γ2)  각 (B, 1, hidden_dim)
```

### 3. `FlowMatchingHead` Transformer Block 재작성

**현재:**
```python
attn_out = self_attn(norm1(h), h, h)        # query만 norm
h = h + attn_out                             # gate 없음
h = h + cross_attn(norm2(h), c, c)
h = h + mlp(norm3(h))
```

**변경 후:**
```python
# per-layer modulation
(α1, β1, γ1, α2, β2, γ2) = adaLN_mod(cond_emb)

# self-attn with AdaLN
normed = norm1(h) * (1 + α1) + β1
attn_out = self_attn(normed, normed, normed)   # q=k=v 모두 modulated norm
h = h + γ1 * attn_out                         # gate

# cross-attn (timestep 미적용, VLM cond는 그대로)
h = h + cross_attn(norm2(h), c, c)

# mlp with AdaLN
normed = norm3(h) * (1 + α2) + β2
h = h + γ2 * mlp(normed)                      # gate
```

---

## 전체 forward 흐름

```
x_t   : (B, H, 3)
t     : (B,)
cond  : (B, T_vlm, 2048)

1. action_proj(x_t)              → (B, H, 512)       h 초기화
2. timestep_embedder(t)          → (B, 512)           cond_emb
3. cond_proj(cond)               → (B, T_vlm, 512)    c
4. for each layer:
     (α1,β1,γ1,α2,β2,γ2) = adaLN_mods[i](cond_emb)
     h = h + γ1 * self_attn( norm1(h)*(1+α1)+β1 )
     h = h + cross_attn( norm2(h), c )
     h = h + γ2 * mlp( norm3(h)*(1+α2)+β2 )
5. output_head(h)                → (B, H, 3)          v_pred
```

---

## get_loss 변경 없음
CFM linear interpolation 유지:
```
x_t = (1-t)*x_0 + t*x_1
v_target = x_1 - x_0
loss = MSE(v_pred, v_target)
```

---

## 인터페이스 변경 없음
- `forward(x_t, t, cond)` → 동일
- `get_loss(x_1, cond)` → 동일
- `pi0_core.py` 수정 불필요

---

## [x] 구현 체크리스트 (커밋 144d3b1 완료)
- [x] `TimestepEmbedder` 클래스
- [x] `AdaLNModulation` 클래스 (Zero-init)
- [x] `FlowMatchingHead.__init__`: `adaLN_mods` ModuleList 추가
- [x] `FlowMatchingHead.forward`: AdaLN 적용 블록으로 교체
- [x] `MoNaActionExpert` 정규화 래퍼 (`mona_action_expert.py`)
- [x] `verify_mona_expert.py` 수치 정합성 검증 통과
- [x] `configs/train.yaml` 데이터 경로 수정 (GX10)

## [x] 완료된 다음 작업
- [x] AdaLN-Zero 학습 실행 → 중간발표 loss curve 확보
- [x] v3a best vs AdaLN-Zero 성능 비교
- [x] 중간발표 슬라이드 업데이트
