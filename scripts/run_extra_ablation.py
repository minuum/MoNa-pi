import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 기존 eval_closedloop 모듈의 핵심 함수들 임포트
import scripts.eval_closedloop as ev
from data.dataset import build_train_val_split

def run_extra_ablation():
    parser = argparse.ArgumentParser(description="MoNa-pi 추가 Ablation 실험 (M8, M9, M10)")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--ckpt", default="checkpoints/best")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["train"].get("seed", 42))

    # 데이터셋 로드 (val split)
    _, val_ds = build_train_val_split(
        directory=cfg["data"]["train_path"],
        val_split=cfg["data"].get("val_split", 0.1),
        k=cfg["model"]["horizon"],
        window_size=cfg["data"]["window_size"],
        image_size=cfg["data"].get("image_size", 224),
        preprocess=cfg["data"].get("preprocess", True),
        normalize=cfg["data"].get("normalize", False),
        seed=cfg["train"].get("seed", 42),
    )
    base_ds = val_ds.dataset if hasattr(val_ds, "dataset") else val_ds
    val_indices = val_ds.indices if hasattr(val_ds, "indices") else list(range(len(val_ds)))
    val_f_idxs = sorted({base_ds.samples[i][0] for i in val_indices})

    # 정형 9종 에피소드만 필터링 (비교 직관성을 위해)
    REGULAR_PATHS = [
        'center_straight', 'center_left', 'center_right',
        'left_straight',   'left_left',   'left_right',
        'right_straight',  'right_left',  'right_right',
    ]
    h5_files = sorted(Path(cfg["data"]["train_path"]).glob("*.h5"))
    idx_to_stem = {i: f.stem for i, f in enumerate(h5_files)}
    val_f_idxs = [i for i in val_f_idxs if any(kw in idx_to_stem.get(i, '') for kw in REGULAR_PATHS)]

    print(f"[Extra Eval] 정형 에피소드 수: {len(val_f_idxs)}")
    
    # 모델 로드
    model = ev.load_model(cfg, Path(args.ckpt), device)

    # ─────── [M8] ODE Steps = 1, 2 테스트 ───────
    for steps in [1, 2]:
        print(f"\n>>> [M8] ODE Steps = {steps} 평가 중...")
        ev._ODE_STEPS = steps
        results = []
        for f_idx in tqdm(val_f_idxs, desc=f"ODE={steps}"):
            ep = base_ds._ep_cache[f_idx]
            metrics = ev.eval_episode(model, ep, cfg, device)
            if not metrics.get("skipped"):
                metrics["episode_idx"] = f_idx
                results.append(metrics)
        
        n_ok = sum(1 for r in results if r["success"])
        sr = n_ok / len(results) if results else 0
        mean_fpe = np.mean([r["fpe"] for r in results]) if results else 0
        print(f"  Result -> SR: {sr*100:.1f}%, Mean FPE: {mean_fpe:.4f}m")
        
        out_path = f"logs/cl_eval_ode{steps}_regular.json"
        with open(out_path, "w") as f:
            json.dump({"success_rate": sr, "mean_fpe": mean_fpe, "n_episodes": len(results), "episodes": results}, f, indent=2)

    # ─────── [M9] Instruction Robustness (지시어 변조) 테스트 ───────
    ev._ODE_STEPS = 5 # 기본 솔버 스텝 복구
    
    # 9-1. 오타가 섞인 지시어 (Typo)
    # 9-2. 엉뚱한 지시어 (OOD)
    scenarios = {
        "typo": "Nvigat untl th gry bskt is cntrd and fils lower hlf.",
        "ood": "Go pick up a red block on the table."
    }

    for sc_name, sc_instr in scenarios.items():
        print(f"\n>>> [M9] 지시어 변조 [{sc_name.upper()}] 평가 중...")
        results = []
        for f_idx in tqdm(val_f_idxs, desc=f"Instr={sc_name}"):
            ep = base_ds._ep_cache[f_idx]
            
            # 원래 에피소드의 Instruction을 변조 지시어로 대체
            ep_modified = ep.copy()
            ep_modified["instruction"] = sc_instr
            
            metrics = ev.eval_episode(model, ep_modified, cfg, device)
            if not metrics.get("skipped"):
                metrics["episode_idx"] = f_idx
                results.append(metrics)
        
        n_ok = sum(1 for r in results if r["success"])
        sr = n_ok / len(results) if results else 0
        mean_fpe = np.mean([r["fpe"] for r in results]) if results else 0
        print(f"  Result -> SR: {sr*100:.1f}%, Mean FPE: {mean_fpe:.4f}m")
        
        out_path = f"logs/cl_eval_instr_{sc_name}_regular.json"
        with open(out_path, "w") as f:
            json.dump({"success_rate": sr, "mean_fpe": mean_fpe, "n_episodes": len(results), "episodes": results}, f, indent=2)

    # ─────── [M10] Initial Pose Perturbation (초기 Pose 편차 수렴력) 테스트 ───────
    # X, Y 위치에 인위적으로 +10cm 초기 에러(Perturbation)를 주입
    perturbations = {
        "x_plus_10cm": (0.1, 0.0, 0.0),
        "y_plus_10cm": (0.0, 0.1, 0.0),
    }

    for p_name, (px, py, pt) in perturbations.items():
        print(f"\n>>> [M10] 초기 위치 편차 주입 [{p_name.upper()}] 평가 중...")
        results = []
        for f_idx in tqdm(val_f_idxs, desc=f"Perturb={p_name}"):
            ep = base_ds._ep_cache[f_idx]
            
            # 에피소드 평가를 수동 전개하며 초기 포즈에 편차 주입
            images_np = ep["images"]
            gt_actions = ep["actions"]
            raw_instr = ep["instruction"]
            
            img_size = cfg["data"].get("image_size", 224)
            window = cfg["data"].get("window_size", 8)
            horizon = cfg["model"]["horizon"]
            
            from PIL import Image as PILImage
            pred_actions = []
            
            for t in range(window - 1, len(images_np) - horizon + 1):
                frames = []
                for i in range(t - window + 1, t + 1):
                    pil = PILImage.fromarray(images_np[i]).resize((img_size, img_size))
                    arr = np.array(pil).transpose(2, 0, 1).astype(np.float32) / 255.0
                    frames.append(torch.from_numpy(arr))
                img_tensor = torch.stack(frames).unsqueeze(0).to(device)
                
                t_end = min(t + horizon, len(gt_actions))
                chunk_gt = gt_actions[t: t_end]
                tagged_instr = ev._INJECTOR.inject(raw_instr, chunk_gt, is_training=False)
                
                chunk = model.sample_actions(img_tensor, [tagged_instr], n_steps=5)
                pred_actions.append(chunk[0, 0].float().cpu().numpy())

            if not pred_actions:
                continue
                
            pred_actions = np.array(pred_actions)
            gt_slice = gt_actions[window - 1: window - 1 + len(pred_actions)]
            
            expert_poses = ev.build_trajectory(gt_slice, 0.1)
            
            # [M10 핵심] 초기 편차 반영 궤적 구성
            # poses[0] = Pose(0,0,0) 이지만, 주행 궤적의 시작점에 px, py 오차를 더해 누적 경로 전개
            pred_poses = [ev.Pose(x=px, y=py, theta=pt)]
            for lx, ly, az in pred_actions:
                pred_poses.append(ev.pose_step(pred_poses[-1], float(lx), float(ly), float(az), 0.1))
            
            metrics = ev.compute_metrics(expert_poses, pred_poses, fpe_thresh=0.3)
            metrics["episode_idx"] = f_idx
            results.append(metrics)
            
        n_ok = sum(1 for r in results if r["success"])
        sr = n_ok / len(results) if results else 0
        mean_fpe = np.mean([r["fpe"] for r in results]) if results else 0
        print(f"  Result -> SR: {sr*100:.1f}%, Mean FPE: {mean_fpe:.4f}m")
        
        out_path = f"logs/cl_eval_perturb_{p_name}_regular.json"
        with open(out_path, "w") as f:
            json.dump({"success_rate": sr, "mean_fpe": mean_fpe, "n_episodes": len(results), "episodes": results}, f, indent=2)

    print("\n====== 모든 추가 Ablation 실험 완료! ======")

if __name__ == "__main__":
    run_extra_ablation()
