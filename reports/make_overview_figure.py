#!/usr/bin/env python3
"""π0-style system overview figure — MoNa-π adaptation"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

BASE       = Path(__file__).parent
SAMPLE_DIR = BASE / "sample_imgs"
ROBOT_TOP  = BASE / "robot_top.jpg"

# ── Color palette (v5 theme) ──────────────────────────────────────────────────
NAVY    = '#1D4ED8'; NAVY_T    = '#EBF3FF'
GREEN   = '#16A34A'; GREEN_T   = '#DCFCE7'
ORANGE  = '#EA580C'; ORANGE_T  = '#FFF7ED'
RED     = '#DC2626'; RED_T     = '#FEE2E2'
PURPLE  = '#7127CE'; PURPLE_T  = '#F3E8FF'
TEAL    = '#0D9488'; TEAL_T    = '#CCFBF1'
TEXT    = '#111827'
MUTED   = '#9CA3AF'
BG      = '#F8FAFC'
WHITE   = '#FFFFFF'

FW, FH = 18.0, 9.0   # figure data coords (coordinate space stays fixed)
FIGW, FIGH = 22.0, 11.0  # physical figure size in inches
FS = 1.55            # global font scale — change this one number to resize all text

# ── Helpers ───────────────────────────────────────────────────────────────────

def rbox(ax, x, y, w, h, fc=WHITE, ec='#CBD5E1', lw=1.4, radius=0.18, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder)
    ax.add_patch(p)

def txt(ax, s, x, y, sz=9, c=TEXT, ha='center', va='center', bold=False, mono=False, zorder=5):
    ax.text(x, y, s, ha=ha, va=va, color=c, fontsize=sz*FS, zorder=zorder,
            fontweight='bold' if bold else 'normal',
            fontfamily='monospace' if mono else 'sans-serif')

def arr(ax, x1, y1, x2, y2, c=NAVY, lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=c, lw=lw,
                                connectionstyle='arc3,rad=0.0'), zorder=4)

def img_at(fig, img_path, data_x, data_y, data_w, data_h):
    """Place an image using data coordinates (figure spans FW × FH)."""
    if not Path(img_path).exists():
        return
    im = mpimg.imread(str(img_path))
    ax2 = fig.add_axes([data_x/FW, data_y/FH, data_w/FW, data_h/FH])
    ax2.imshow(im, aspect='auto')
    ax2.axis('off')


# ── Build figure ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(FIGW, FIGH), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')

# ── GLOBAL TITLE ─────────────────────────────────────────────────────────────
txt(ax, "MoNa-π  ·  Flow Matching VLA for Omni-wheel Mobile Navigation",
    FW/2, FH-0.28, sz=13, bold=True, c=TEXT)
ax.plot([0.5, FW-0.5], [FH-0.55, FH-0.55], color='#CBD5E1', lw=1.0)

# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL: Dataset
# ══════════════════════════════════════════════════════════════════════════════
LX, LW = 0.25, 3.85
rbox(ax, LX, 0.35, LW, FH-0.80, fc='#EFF6FF', ec='#BFDBFE', lw=1.8)
txt(ax, "MoNa-pi Dataset", LX+LW/2, FH-0.85, sz=11, bold=True, c=NAVY)

# Robot hardware photo
img_at(fig, ROBOT_TOP, LX+0.10, 5.60, 1.65, 2.50)
txt(ax, "Serbot2", LX+0.92, 5.40, sz=8, c=NAVY, bold=True)
txt(ax, "Jetson Orin AGX 16GB", LX+0.92, 5.18, sz=7.5, c=MUTED)

# 9-category sample image grid (3 columns × 3 rows)
grid_cats = [
    ("c_str",  "center_straight_mid.jpg", GREEN),
    ("c_lft",  "center_left_mid.jpg",     RED),
    ("c_rgt",  "center_right_mid.jpg",    TEAL),
    ("l_str",  "left_straight_mid.jpg",   GREEN),
    ("l_lft",  "left_left_mid.jpg",       ORANGE),
    ("l_rgt",  "left_right_mid.jpg",      RED),
    ("r_str",  "right_straight_mid.jpg",  GREEN),
    ("r_lft",  "right_left_mid.jpg",      ORANGE),
    ("r_rgt",  "right_right_mid.jpg",     ORANGE),
]
cell_w, cell_h = 1.12, 0.88
x0_grid, y0_grid = LX+0.15, 1.42

for i, (label, fname, ec) in enumerate(grid_cats):
    col, row = i % 3, i // 3
    cx = x0_grid + col * cell_w
    cy = y0_grid + (2 - row) * cell_h
    img_at(fig, SAMPLE_DIR / fname, cx, cy, cell_w-0.06, cell_h-0.14)
    # small label below image
    txt(ax, label, cx+(cell_w-0.06)/2, cy-0.12, sz=7, c=ec, bold=True)

# Dataset stats footer
rbox(ax, LX+0.15, 0.50, LW-0.30, 0.72, fc=NAVY_T, ec=NAVY, lw=1.0)
txt(ax, "9 categories  ·  HDF5  ·  3-DOF", LX+LW/2, 0.96, sz=8.5, c=NAVY, bold=True)
txt(ax, "linear_x · linear_y · angular_z", LX+LW/2, 0.68, sz=8, c=MUTED)


# ══════════════════════════════════════════════════════════════════════════════
# CENTER PANEL: Architecture
# ══════════════════════════════════════════════════════════════════════════════
CX, CW = 4.40, 9.20
rbox(ax, CX, 0.35, CW, FH-0.80, fc='#F0FDF4', ec='#BBF7D0', lw=1.8)
txt(ax, "MoNa-pi VLA Model", CX+CW/2, FH-0.85, sz=11, bold=True, c=TEXT)

MID = CX + CW/2   # center x of panel

# ── Inputs ───────────────────────────────────────────────────────────────────
# Camera input (left)
rbox(ax, CX+0.30, 6.80, 3.80, 0.90, fc=GREEN_T, ec=GREEN, lw=1.3)
txt(ax, "Fish-eye Camera  (8-frame)", CX+2.20, 7.38, sz=9.5, c=GREEN, bold=True)
txt(ax, "(B, 8, 224, 224, 3)   ·   window = 0.8 s",
    CX+2.20, 7.06, sz=8.5, c=MUTED, mono=True)

# Language input (right)
rbox(ax, CX+4.40, 6.80, 4.60, 0.90, fc=ORANGE_T, ec=ORANGE, lw=1.3)
txt(ax, '"Move to the left"', CX+6.70, 7.38, sz=10, c=ORANGE, bold=True)
txt(ax, "Instruction Pool  —  15 paraphrases / category",
    CX+6.70, 7.06, sz=8.5, c=MUTED)

# Arrows → VLM
arr(ax, CX+2.20, 6.80, MID-0.80, 6.18, c=GREEN)
arr(ax, CX+6.70, 6.80, MID+0.80, 6.18, c=ORANGE)

# ── PaliGemma 3B ─────────────────────────────────────────────────────────────
rbox(ax, CX+0.40, 5.10, CW-0.80, 1.00, fc=NAVY_T, ec=NAVY, lw=1.8)
txt(ax, "PaliGemma 3B  (BF16 full fine-tuning)",
    MID, 5.74, sz=11, bold=True, c=NAVY)
txt(ax, "SigLIP SO400M  (image patches)   +   Gemma-2B  (language model)   →   context features (B, 64, 2048)",
    MID, 5.34, sz=8.5, c=MUTED)

arr(ax, MID, 5.10, MID, 4.55, c=NAVY)

# ── Action Expert ─────────────────────────────────────────────────────────────
rbox(ax, CX+0.40, 3.52, CW-0.80, 0.96, fc=ORANGE_T, ec=ORANGE, lw=1.8)
txt(ax, "Action Expert  —  4-layer Cross-Attention Transformer",
    MID, 4.12, sz=11, bold=True, c=ORANGE)
txt(ax, "8-head  ·  hidden_dim=256  ·  AdaLN-Zero time conditioning",
    MID, 3.76, sz=8.5, c=MUTED)

# Noise input
rbox(ax, CX+7.30, 3.52, 1.90, 0.96, fc=PURPLE_T, ec=PURPLE, lw=1.2)
txt(ax, "x₀ ~ N(0,I)", CX+8.25, 4.14, sz=9, c=PURPLE, bold=True)
txt(ax, "t ~ U(0,1)", CX+8.25, 3.80, sz=8.5, c=MUTED)
arr(ax, CX+7.30, 4.00, CX+7.00, 4.00, c=PURPLE)

arr(ax, MID, 3.52, MID, 2.97, c=ORANGE)

# ── Flow Matching ─────────────────────────────────────────────────────────────
rbox(ax, CX+0.40, 1.95, CW-0.80, 0.95, fc=GREEN_T, ec=GREEN, lw=1.8)
txt(ax, "Flow Matching ODE  (× 5 Euler steps)",
    MID, 2.56, sz=11, bold=True, c=GREEN)
txt(ax, "L = ||v_θ − (x₁−x₀)||²   ·   x₀ ~ N(0,I)   ·   x₁ = GT action chunk",
    MID, 2.20, sz=8.5, c=MUTED, mono=True)

arr(ax, MID, 1.95, MID, 1.40, c=GREEN)

# ── Output ───────────────────────────────────────────────────────────────────
rbox(ax, CX+1.20, 0.55, CW-2.40, 0.80, fc=NAVY, ec=NAVY, lw=0)
txt(ax, "(B, 10, 3) Action Chunk   →   4Hz replanning  ·  50Hz local control",
    MID, 0.95, sz=10, bold=True, c=WHITE)

# ── Arrow: Dataset → Model ───────────────────────────────────────────────────
arr(ax, LX+LW+0.05, FH/2, CX-0.05, FH/2, c=NAVY, lw=2.2)


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL: Results
# ══════════════════════════════════════════════════════════════════════════════
RX = CX + CW + 0.30
RW = FW - RX - 0.25
rbox(ax, RX, 0.35, RW, FH-0.80, fc='#FFFBEB', ec='#FDE68A', lw=1.8)
txt(ax, "Evaluation Results  (v3-A)", RX+RW/2, FH-0.85, sz=11, bold=True, c=ORANGE)

results = [
    ("[OK]  Straight navigation",
     "c_str · l_str · r_str",
     "Success@1.0 = 25-50%",
     GREEN, GREEN_T,
     "center_straight_mid.jpg"),
    ("[--]  Turning tasks",
     "l_lft · r_rgt · r_lft",
     "Success@1.0 = 33-67%",
     ORANGE, ORANGE_T,
     "left_left_mid.jpg"),
    ("[X]   Ambiguous path",
     "c_lft · l_rgt",
     "Success@1.0 = 0%",
     RED, RED_T,
     "center_left_mid.jpg"),
]

box_h = (FH - 1.60) / 3
for i, (title, cats, perf, ec, bg, img_f) in enumerate(results):
    by = 0.55 + (2 - i) * box_h
    rbox(ax, RX+0.15, by, RW-0.30, box_h-0.18, fc=bg, ec=ec, lw=1.4)

    # Category image
    img_at(fig,
           SAMPLE_DIR / img_f,
           RX+0.22, by+0.14,
           1.48, box_h-0.48)

    # Text right of image
    tx = RX + 1.82
    txt(ax, title,  tx, by+box_h-0.46, sz=10, bold=True, c=ec, ha='left')
    txt(ax, cats,   tx, by+box_h-0.82, sz=8.5, c=MUTED, ha='left')
    txt(ax, perf,   tx, by+0.36, sz=9.5, bold=True, c=ec, ha='left')

# Arrow: Model → Results
arr(ax, RX-0.05, FH/2, RX+0.15, FH/2, c=ORANGE, lw=2.2)


# ── Save ──────────────────────────────────────────────────────────────────────
out = BASE / "mona_pi_overview.png"
fig.savefig(str(out), dpi=180, bbox_inches='tight', facecolor=BG)
print(f"Saved: {out}")
