import json
import matplotlib.pyplot as plt
import numpy as np

# 데이터 파싱
rows = [
    ('M1: Baseline\n(244ep+aug)', 'logs/cl_eval_v5_thresh03_20260601_2113.json'),
    ('M2: LeRobot\nbackbone', 'logs/cl_eval_lerobot_backbone_trained.json'),
    ('M3: LoRA\n(Rank=16)', 'logs/cl_eval_lora_full.json'),
    ('M4: Gemma Expert\n(π0 faithful)', 'logs/cl_eval_gemma_expert_thresh03_20260601_2113.json'),
    ('M5: No-Aug\n(244ep)', 'logs/cl_eval_no_aug_full.json'),
    ('M7: Half-Ep\n(116ep+aug)', 'logs/cl_eval_116ep_full.json'),
]

models = []
success_rates = []
mean_fpes = []

for name, filepath in rows:
    try:
        with open(filepath) as f:
            data = json.load(f)
            models.append(name)
            success_rates.append(data.get("success_rate", 0) * 100)
            mean_fpes.append(data.get("mean_fpe", 0))
    except Exception as e:
        print(f"Skipping {name}: {e}")

# 그래프 그리기 (Dual Y-axis 한 프레임 구조)
fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
plt.title("MoNa-pi Closed-Loop Ablation Study (Threshold = 0.3m)", fontsize=14, fontweight='bold', pad=20)

# X축 위치 설정
x = np.arange(len(models))
width = 0.4

# Left Y-axis: Success Rate
color1 = '#3498db'
ax1.set_xlabel('Ablation Models', fontsize=11, labelpad=12)
ax1.set_ylabel('Success Rate (%)', color='#2980b9', fontsize=11, fontweight='bold')
bars = ax1.bar(x, success_rates, width, color=color1, alpha=0.85, edgecolor='#2980b9', linewidth=1.2, label='Success Rate (%)')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=9)
ax1.tick_params(axis='y', labelcolor='#2980b9')
ax1.set_ylim(0, 110)

# Right Y-axis: Mean FPE
ax2 = ax1.twinx()
color2 = '#e74c3c'
ax2.set_ylabel('Mean FPE (meters)', color='#c0392b', fontsize=11, fontweight='bold')
line = ax2.plot(x, mean_fpes, color=color2, marker='o', linewidth=2.5, markersize=8, label='Mean FPE (m)')
ax2.tick_params(axis='y', labelcolor='#c0392b')
ax2.set_ylim(0, max(mean_fpes) * 1.3)

# 막대 그래프 상단에 수치 텍스트 표시
for bar in bars:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 2, f'{height:.1f}%',
             ha='center', va='bottom', fontsize=9, color='#2c3e50', fontweight='bold')

# 선 그래프 마커 상단에 수치 텍스트 표시
for i, fpe in enumerate(mean_fpes):
    ax2.text(i, fpe + 0.02, f'{fpe:.4f}m', ha='center', va='bottom', fontsize=9, color='#7f8c8d')

# 격자 추가 및 레이아웃 정리
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)
fig.tight_layout()

# 이미지 저장
save_path = 'reports/ablation_fpe_chart.png'
plt.savefig(save_path, bbox_inches='tight')
print(f"Chart successfully saved to {save_path}")
