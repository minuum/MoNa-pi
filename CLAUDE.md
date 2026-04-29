# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run the model logic test (no dataset required):**
```bash
python tests/test_model_loss.py
```

**Run training (requires HDF5 dataset):**
```bash
python training/train.py
```

**Analyze action distribution of HDF5 dataset:**
```bash
python scripts/analyze_actions.py
```

There is no test runner configured; run individual test files directly with `python`.

## Architecture

MoNa-pi is a Vision-Language-Action (VLA) model for mobile robot control, inspired by Physical Intelligence's π0. The core idea: replace discrete action classification with **Flow Matching** over a continuous action space, predicting a chunk of future actions at once (Action Chunking) for high-frequency (50Hz+) control.

**Action space:** 3-DOF omniwheel robot — `(linear_x, linear_y, angular_z)`, normalized to `[-1, 1]` via `ActionNormalizer` (stats hardcoded from `mobile_vla_dataset_v3`: ±1.15 range).

**Data flow:**
1. `ActionChunkDataset` (`data/dataset.py`) reads HDF5 files, returns a sliding window of `window_size=8` images and a future action chunk of `horizon=10`.
2. `Pi0VLA.forward_backbone()` (`models/pi0_core.py`) — currently a placeholder returning random tensors of shape `(B, 64, backbone_out_dim)`. Real backbone integration (SigLIP/Kosmos-2) is pending.
3. `FlowMatchingHead` (`models/heads/flow_head.py`) takes the conditioning features + noisy actions + timestep, and predicts the velocity field `v = x_1 - x_0` via a Transformer with cross-attention.
4. At inference, `Pi0VLA.sample_actions()` integrates the ODE using Euler or Heun's method over `n_steps` (default 5).

**Key constants (validated from dataset analysis):**
- `window_size = 8` (image history frames)
- `horizon = 10` (action chunk size)
- `backbone_out_dim = 1024`

**Two head files exist** — `flow_head.py` (the real implementation) and `flow_matching_head.py` (a wrapper skeleton around it). `Pi0VLA` imports directly from `flow_head.py`. `FlowMatchingActionHead` in `flow_matching_head.py` is an alternative interface; its `sample()` method is not yet implemented.

**Training** uses HuggingFace `Accelerate` for distributed/multi-GPU support. Checkpoints are saved under `checkpoints/mona_pi_epoch_{N}/`.

**Inference deployment** is designed as a client-server split: a policy server generates action chunks, and a Jetson-side local control loop buffers and executes them. The `inference/` directory is not yet populated.
