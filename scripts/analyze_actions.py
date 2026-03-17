import h5py
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

def analyze_dataset(directory):
    files = glob.glob(os.path.join(directory, "*.h5"))
    all_actions = []
    
    for f_path in files:
        try:
            with h5py.File(f_path, 'r') as f:
                if 'actions' in f:
                    all_actions.append(f['actions'][:])
        except Exception as e:
            print(f"Error reading {f_path}: {e}")
            
    if not all_actions:
        print("No actions found.")
        return
        
    all_actions = np.concatenate(all_actions, axis=0)
    
    print(f"Total samples: {len(all_actions)}")
    print(f"Action shape: {all_actions.shape}")
    
    for i in range(all_actions.shape[1]):
        col = all_actions[:, i]
        print(f"\nColumn {i} Stats:")
        print(f"  Min:  {np.min(col):.4f}")
        print(f"  Max:  {np.max(col):.4f}")
        print(f"  Mean: {np.mean(col):.4f}")
        print(f"  Std:  {np.std(col):.4f}")
        
    # Plot histograms
    plt.figure(figsize=(15, 5))
    for i in range(all_actions.shape[1]):
        plt.subplot(1, 3, i+1)
        plt.hist(all_actions[:, i], bins=50)
        plt.title(f"Column {i} Distribution")
    plt.savefig("/home/soda/MoNa-pi/scripts/action_distribution.png")
    print("\nHistogram saved to scripts/action_distribution.png")

if __name__ == "__main__":
    analyze_dataset("/home/soda/vla/ROS_action/mobile_vla_dataset_v3/")
