#!/usr/bin/env bash
# run_all_eval_then_lora.sh
# 순서: v5 CL eval → LeRobot CL eval → Gemma Expert CL eval → LoRA 학습
# FPE threshold: 0.3m (2m×2m 맵 기준)
#
# 실행: bash scripts/run_all_eval_then_lora.sh 2>&1 | tee logs/run_all_$(date +%Y%m%d_%H%M).log

set -e
PYTHON="/home/minum/26CS/MoNa-pi/.venv/bin/python3"
THRESH=0.3
CONFIG_MAIN="configs/train.yaml"
CONFIG_GEMMA="configs/train_gemma_expert.yaml"
CONFIG_LORA="configs/train_lora.yaml"
TIMESTAMP=$(date +%Y%m%d_%H%M)

echo "=============================================="
echo "  MoNa-pi 전체 평가 + LoRA 학습 파이프라인"
echo "  FPE threshold: ${THRESH}m"
echo "  시작: $(date)"
echo "=============================================="

# ── 1. v5 (base PaliGemma) CL eval @ 0.3m ──────────────────────────────────
echo ""
echo "[1/4] v5 (base PaliGemma) CL eval — fpe_thresh=${THRESH}m"
${PYTHON} scripts/eval_closedloop.py \
    --config ${CONFIG_MAIN} \
    --ckpt checkpoints/epoch_021_loss_0.1499 \
    --fpe-thresh ${THRESH} \
    --out logs/cl_eval_v5_thresh03_${TIMESTAMP}.json \
    2>&1 | tee logs/cl_eval_v5_thresh03_${TIMESTAMP}.log
echo "[1/4] 완료 ✓"

# ── 2. LeRobot backbone CL eval @ 0.3m ──────────────────────────────────────
echo ""
echo "[2/4] LeRobot pi0 backbone CL eval — fpe_thresh=${THRESH}m"
${PYTHON} scripts/eval_closedloop.py \
    --config ${CONFIG_MAIN} \
    --ckpt checkpoints/lerobot_pi0_backbone \
    --fpe-thresh ${THRESH} \
    --out logs/cl_eval_lerobot_thresh03_${TIMESTAMP}.json \
    2>&1 | tee logs/cl_eval_lerobot_thresh03_${TIMESTAMP}.log
echo "[2/4] 완료 ✓"

# ── 3. Gemma Expert CL eval @ 0.3m ──────────────────────────────────────────
echo ""
echo "[3/4] Gemma Expert CL eval — fpe_thresh=${THRESH}m"
${PYTHON} scripts/eval_closedloop.py \
    --config ${CONFIG_GEMMA} \
    --ckpt checkpoints/gemma_expert/best \
    --fpe-thresh ${THRESH} \
    --out logs/cl_eval_gemma_expert_thresh03_${TIMESTAMP}.json \
    2>&1 | tee logs/cl_eval_gemma_expert_thresh03_${TIMESTAMP}.log
echo "[3/4] 완료 ✓"

# ── 4. LoRA 학습 ─────────────────────────────────────────────────────────────
echo ""
echo "[4/4] LoRA 학습 시작 (rank=16, epochs=20)"
${PYTHON} training/train.py --config ${CONFIG_LORA} \
    2>&1 | tee logs/train_lora_${TIMESTAMP}.log
echo "[4/4] LoRA 학습 완료 ✓"

echo ""
echo "=============================================="
echo "  모든 작업 완료! $(date)"
echo "  결과 파일:"
echo "    logs/cl_eval_v5_thresh03_${TIMESTAMP}.json"
echo "    logs/cl_eval_lerobot_thresh03_${TIMESTAMP}.json"
echo "    logs/cl_eval_gemma_expert_thresh03_${TIMESTAMP}.json"
echo "    logs/train_lora_${TIMESTAMP}.log"
echo "=============================================="
