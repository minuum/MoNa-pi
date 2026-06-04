#!/bin/bash
# MoNa-pi Ablation 순차 실행
# M6 (ODE eval) → M5 (no-aug 학습+eval) → M7 (116ep 학습+eval)
set -e
cd "$(dirname "$0")/.."
PY=".venv/bin/python3"

echo "====== Ablation Pipeline 시작: $(date) ======"

# ── M6: ODE steps 3/5/10 (이전 실행에서 이미 완료되었으므로 건너뜀) ───────────
echo "[M6] ODE steps eval... (Skipped: Completed in previous run)"

# ── M5: 증강 없는 버전 복구 및 이어서 학습 ─────────────────────────────────────
echo "[M5] no-aug 학습 재개..."
$PY training/train.py --config configs/train_no_aug_resume.yaml \
    > logs/train_no_aug.log 2>&1
echo "[M5] eval 시작..."
$PY scripts/eval_closedloop.py \
    --config configs/train_no_aug_resume.yaml \
    --ckpt checkpoints/no_aug/best \
    --fpe-thresh 0.3 --regular-only \
    --out logs/cl_eval_no_aug_regular.json \
    > logs/cl_eval_no_aug_regular.log 2>&1
$PY scripts/eval_closedloop.py \
    --config configs/train_no_aug_resume.yaml \
    --ckpt checkpoints/no_aug/best \
    --fpe-thresh 0.3 \
    --out logs/cl_eval_no_aug_full.json \
    > logs/cl_eval_no_aug_full.log 2>&1
echo "[M5] 완료: $(date)"

# ── M7: 116 에피소드 ───────────────────────────────────────────────────────
echo "[M7] 116ep 학습 시작..."
$PY training/train.py --config configs/train_116ep.yaml \
    > logs/train_116ep.log 2>&1
echo "[M7] eval 시작..."
$PY scripts/eval_closedloop.py \
    --config configs/train_116ep.yaml \
    --ckpt checkpoints/ep116/best \
    --fpe-thresh 0.3 --regular-only \
    --out logs/cl_eval_116ep_regular.json \
    > logs/cl_eval_116ep_regular.log 2>&1
$PY scripts/eval_closedloop.py \
    --config configs/train_116ep.yaml \
    --ckpt checkpoints/ep116/best \
    --fpe-thresh 0.3 \
    --out logs/cl_eval_116ep_full.json \
    > logs/cl_eval_116ep_full.log 2>&1
echo "[M7] 완료: $(date)"

# ── 최종 요약 ───────────────────────────────────────────────────────────────
echo ""
echo "====== 최종 Ablation 결과 ======"
$PY - << 'PYEOF'
import json

print(f"{'모델':<28} {'정형SR':>7} {'전체SR':>7}")
print('-'*45)
rows = [
    ('M1 base PG (244ep+aug)',   'logs/cl_eval_v5_regular.json',           'logs/cl_eval_v5_thresh03_20260601_2113.json'),
    ('M2 lerobot backbone',       'logs/cl_eval_lerobot_regular.json',      'logs/cl_eval_lerobot_backbone_trained.json'),
    ('M3 LoRA',                   'logs/cl_eval_lora_regular.json',         'logs/cl_eval_lora_full.json'),
    ('M4 GemmaExpert',            'logs/cl_eval_gemma_regular.json',        'logs/cl_eval_gemma_expert_thresh03_20260601_2113.json'),
    ('M5 no-aug (244ep)',         'logs/cl_eval_no_aug_regular.json',       'logs/cl_eval_no_aug_full.json'),
    ('M7 116ep+aug',              'logs/cl_eval_116ep_regular.json',        'logs/cl_eval_116ep_full.json'),
]
for name, reg_f, full_f in rows:
    try:
        r = json.load(open(reg_f)); f = json.load(open(full_f))
        print(f"{name:<28} {r['success_rate']:>7.1%} {f['success_rate']:>7.1%}")
    except:
        print(f"{name:<28} {'—':>7} {'—':>7}")

print()
print(f"{'ODE steps':<10} {'정형SR':>7} {'FPE':>8}")
for s in [3, 5, 10]:
    try:
        d = json.load(open(f'logs/cl_eval_ode{s}_regular.json'))
        print(f"{s:<10} {d['success_rate']:>7.1%} {d['mean_fpe']:>8.4f}m")
    except:
        print(f"{s:<10} {'—':>7}")
PYEOF

echo "====== Pipeline 완료: $(date) ======"
