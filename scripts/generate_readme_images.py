"""Generate clean visualizations for the project README."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

OUT = os.path.join(os.path.dirname(__file__), '..', 'images')
os.makedirs(OUT, exist_ok=True)

# ── 1. Leaderboard progression ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

milestones = [
    ("Baseline\nGlobal Ridge", 0.00830),
    ("Per-Player\nModels", 0.00720),
    ("Frame\nOptimization", 0.00680),
    ("Kinetic Chain\n+ Hand Physics", 0.00647),
    ("Velocity\nCNN", 0.00631),
    ("Position\nCNN", 0.00627),
    ("Multi-Source\nBlend", 0.00624),
    ("Per-Player\nCNN Weights", 0.00615),
    ("Final\nSubmission", 0.00614),
]

labels = [m[0] for m in milestones]
scores = [m[1] for m in milestones]

colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(scores)))

bars = ax.bar(range(len(scores)), scores, color=colors, edgecolor='white', linewidth=1.5, width=0.7)

for i, (bar, score) in enumerate(zip(bars, scores)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005,
            f'{score:.5f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, fontsize=8.5, ha='center')
ax.set_ylabel('Leaderboard MSE (lower is better)', fontsize=11)
ax.set_title('Leaderboard Score Progression', fontsize=14, fontweight='bold', pad=15)
ax.set_ylim(0.0055, 0.009)
ax.axhline(y=0.00614, color='green', linestyle='--', alpha=0.4, linewidth=1)
ax.text(len(scores)-0.5, 0.00560, 'Final: 0.00614', color='green', fontsize=9, ha='right', fontstyle='italic')

# improvement arrow
ax.annotate('', xy=(len(scores)-1, scores[-1]), xytext=(0, scores[0]),
            arrowprops=dict(arrowstyle='->', color='#333333', lw=2, connectionstyle='arc3,rad=-0.15'))
pct = (scores[0] - scores[-1]) / scores[0] * 100
ax.text(len(scores)/2, 0.00570, f'{pct:.0f}% total reduction', ha='center', fontsize=10, color='#333333', fontweight='bold')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'leaderboard_progression.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── 2. Pipeline architecture ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))
ax.set_xlim(0, 14)
ax.set_ylim(0, 7)
ax.axis('off')

def draw_box(ax, x, y, w, h, text, color, fontsize=9, bold=False):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor='#333333', linewidth=1.5)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color='#555555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# Title
ax.text(7, 6.7, 'Final Prediction Pipeline', ha='center', fontsize=16, fontweight='bold')

# Input
draw_box(ax, 5.0, 5.8, 4.0, 0.7, '69 Keypoints x 240 Frames\n(Motion Capture @ 60fps)', '#E8E8E8', fontsize=10, bold=True)

# Feature extraction layer
draw_box(ax, 0.3, 4.2, 2.8, 1.2, 'Hoop-Relative\nPositions\n(198 features)', '#AED6F1')
draw_box(ax, 3.5, 4.2, 2.5, 1.2, 'Kinetic Chain\n+ Hand Physics\n(PLS compressed)', '#AED6F1')
draw_box(ax, 6.4, 4.2, 2.5, 1.2, 'Joint Angles\n+ Multi-Frame\nAveraging', '#AED6F1')
draw_box(ax, 9.3, 4.2, 2.0, 1.2, 'MiniRocket\nTemporal\n(5K features)', '#AED6F1')
draw_box(ax, 11.7, 4.2, 2.0, 1.2, 'Velocity &\nPosition\nSequences', '#AED6F1')

# Arrows from input
for x in [1.7, 4.75, 7.65, 10.3, 12.7]:
    draw_arrow(ax, 7.0, 5.8, x, 5.4)

# Models layer
draw_box(ax, 0.8, 2.4, 7.5, 1.2,
         'Per-Player Locally Weighted Ridge Regression\n'
         'Gaussian kernel  |  Player-specific bandwidth & frame  |  Per-target PLS',
         '#F9E79F', fontsize=10, bold=True)

draw_box(ax, 9.0, 2.4, 4.5, 1.2,
         '1D CNNs (Velocity + Position)\n'
         'Per-player weights by MSE quality\n'
         '(2-10% blend weight)',
         '#ABEBC6', fontsize=10, bold=True)

# Arrows from features to models
for x in [1.7, 4.75, 7.65, 10.3]:
    target_x = 4.55 if x < 9 else 11.25
    draw_arrow(ax, x, 4.2, target_x, 3.6)
draw_arrow(ax, 12.7, 4.2, 11.25, 3.6)

# Ensemble
draw_box(ax, 4.0, 0.8, 6.0, 1.1,
         'Weighted Ensemble\n(per-player, per-target blend weights)',
         '#D7BDE2', fontsize=11, bold=True)

draw_arrow(ax, 4.55, 2.4, 6.0, 1.9)
draw_arrow(ax, 11.25, 2.4, 8.5, 1.9)

# Output
draw_box(ax, 4.5, 0.0, 5.0, 0.6,
         'Angle  |  Depth  |  Left/Right', '#F5B7B1', fontsize=10, bold=True)
draw_arrow(ax, 7.0, 0.8, 7.0, 0.6)

plt.savefig(os.path.join(OUT, 'pipeline_architecture.png'), dpi=150, bbox_inches='tight',
            facecolor='white')
plt.close()

# ── 3. Temporal commitment ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 4))

frames = np.arange(0, 241)
time_sec = frames / 60.0

# Background regions
ax.axvspan(0, 2.0, alpha=0.08, color='blue', label='_')
ax.axvspan(2.0, 3.0, alpha=0.08, color='orange', label='_')
ax.axvspan(3.0, 4.0, alpha=0.08, color='red', label='_')

# Phase labels
ax.text(1.0, 0.92, 'Setup / Crouch', ha='center', fontsize=10, color='#333', transform=ax.get_xaxis_transform())
ax.text(2.5, 0.92, 'Launch', ha='center', fontsize=10, color='#333', transform=ax.get_xaxis_transform())
ax.text(3.5, 0.92, 'Follow-through', ha='center', fontsize=10, color='#333', transform=ax.get_xaxis_transform())

# Commitment lines
commit_depth = 150 / 60.0  # frame 150
commit_angle = 153 / 60.0  # frame 153
commit_lr = 170 / 60.0     # frame 170

ax.axvline(commit_depth, color='#2980B9', linewidth=3, linestyle='-', alpha=0.9)
ax.axvline(commit_angle, color='#E67E22', linewidth=3, linestyle='-', alpha=0.9)
ax.axvline(commit_lr, color='#27AE60', linewidth=3, linestyle='-', alpha=0.9)

# Labels with offset
ax.text(commit_depth - 0.05, 0.70, 'Depth\ncommits\n(f150)', ha='right', fontsize=9,
        color='#2980B9', fontweight='bold', transform=ax.get_xaxis_transform())
ax.text(commit_angle + 0.05, 0.50, 'Angle\ncommits\n(f153)', ha='left', fontsize=9,
        color='#E67E22', fontweight='bold', transform=ax.get_xaxis_transform())
ax.text(commit_lr + 0.05, 0.25, 'Left/Right\ncommits\n(f170)', ha='left', fontsize=9,
        color='#27AE60', fontweight='bold', transform=ax.get_xaxis_transform())

# Approximate velocity envelope
np.random.seed(42)
x = np.linspace(0, 4.0, 500)
# Simulated velocity envelope peaking around release
vel = np.exp(-0.5 * ((x - 2.7)/0.5)**2) * 0.8
vel += 0.1
ax.plot(x, vel, color='#333', linewidth=2, alpha=0.5)
ax.fill_between(x, 0, vel, alpha=0.05, color='gray')
ax.text(2.7, 0.82, 'velocity', ha='center', fontsize=8, color='#666', fontstyle='italic')

ax.set_xlim(0, 4.0)
ax.set_ylim(0, 1.0)
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Relative Magnitude', fontsize=11)
ax.set_title('When Each Target Is Decided: Temporal Commitment Points', fontsize=13, fontweight='bold', pad=10)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'temporal_commitment.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── 4. Per-player analysis (clean version) ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

players = ['P1', 'P2', 'P3', 'P4', 'P5']
samples = [70, 66, 68, 67, 74]
colors_p = ['#3498DB', '#2ECC71', '#E67E22', '#E74C3C', '#9B59B6']

# Samples
axes[0].bar(players, samples, color=colors_p, edgecolor='white', linewidth=1.5)
for i, (p, s) in enumerate(zip(players, samples)):
    axes[0].text(i, s + 1, str(s), ha='center', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Training Shots', fontsize=11)
axes[0].set_title('Training Data per Player', fontsize=12, fontweight='bold')
axes[0].set_ylim(0, 85)
axes[0].spines['top'].set_visible(False)
axes[0].spines['right'].set_visible(False)

# Error contribution
angle_err = [0.102, 0.139, 0.121, 0.199, 0.159]  # relative MSE contributions
depth_err = [0.127, 0.170, 0.161, 0.197, 0.161]
lr_err = [0.145, 0.186, 0.168, 0.193, 0.166]

x = np.arange(len(players))
w = 0.25
axes[1].bar(x - w, angle_err, w, label='Angle', color='#F39C12', edgecolor='white')
axes[1].bar(x, depth_err, w, label='Depth', color='#3498DB', edgecolor='white')
axes[1].bar(x + w, lr_err, w, label='Left/Right', color='#2ECC71', edgecolor='white')
axes[1].set_xticks(x)
axes[1].set_xticklabels(players)
axes[1].set_ylabel('Error Contribution (fraction)', fontsize=11)
axes[1].set_title('Error Budget by Player and Target', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].spines['top'].set_visible(False)
axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(os.path.join(OUT, 'player_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

print("Generated 4 images in images/")
