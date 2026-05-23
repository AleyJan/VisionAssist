import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Create graphs folder inside VisionAssist
os.makedirs('/home/raspberrypi/VisionAssist/graphs', exist_ok=True)

PURPLE = '#534AB7'
TEAL   = '#1D9E75'
RED    = '#E24B4A'
AMBER  = '#EF9F27'
BLUE   = '#378ADD'
GRAY   = '#888780'
BG     = '#F8F8F8'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.facecolor'] = BG
plt.rcParams['figure.facecolor'] = 'white'

print("Generating VisionAssist performance graphs...")

# ── Chart 1: Model FPS Comparison ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Model Performance Comparison — VisionAssist Pi 5',
             fontsize=14, fontweight='bold', y=1.01)

models = ['yolo11n\n(primary)', 'yolov8n', 'yolo11s\n(secondary)', 'yolov8m']
fps    = [21.8, 22.7, 10.7, 4.1]
inf_ms = [45.9, 44.1, 93.8, 244.3]
colors = [TEAL, BLUE, AMBER, RED]

ax = axes[0]
bars = ax.bar(models, fps, color=colors, edgecolor='white', linewidth=0.5)
ax.axhline(10, color=RED, ls='--', lw=1.5, label='Min safe (10 FPS)')
ax.axhline(15, color=AMBER, ls='--', lw=1.5, label='Good (15 FPS)')
for bar, val in zip(bars, fps):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.set_ylabel('FPS')
ax.set_title('Frames Per Second', fontsize=11)
ax.legend(fontsize=8)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 28)

ax2 = axes[1]
bars2 = ax2.bar(models, inf_ms, color=colors, edgecolor='white', linewidth=0.5)
ax2.axhline(100, color=RED, ls='--', lw=1.5, label='100ms limit')
for bar, val in zip(bars2, inf_ms):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2,
             f'{val:.0f}ms', ha='center', fontsize=10, fontweight='bold')
ax2.set_ylabel('Inference Time (ms)')
ax2.set_title('Inference Time per Frame', fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(axis='y', alpha=0.3)

fig.tight_layout()
fig.savefig('/home/raspberrypi/VisionAssist/graphs/01_model_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("[1/6] Model comparison saved")

# ── Chart 2: FPS Configurations ───────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle('FPS Across Different Configurations — VisionAssist',
             fontsize=14, fontweight='bold')

configs = [
    'YOLO11n\nalone',
    'Dual model\n(n+s)',
    'Threaded\ncamera+dual',
    'Full app\n(window ON)',
    'Full app\n(window OFF)\n[target]'
]
fps_vals   = [21.8, 16.6, 15.6, 8.5, 15.0]
bar_colors = [TEAL, TEAL, BLUE, RED, TEAL]

bars = ax.bar(configs, fps_vals, color=bar_colors,
              edgecolor='white', linewidth=0.5)
for bar, val in zip(bars, fps_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.2,
            f'{val:.1f}', ha='center', fontsize=11, fontweight='bold')

ax.axhline(10, color=RED, ls='--', lw=1.5, label='Min safe (10 FPS)')
ax.axhline(15, color=AMBER, ls='--', lw=1.5, label='Good target (15 FPS)')
ax.set_ylabel('FPS')
ax.set_title('How Each Optimization Changed FPS', fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 26)

fig.tight_layout()
fig.savefig('/home/raspberrypi/VisionAssist/graphs/02_fps_configurations.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("[2/6] FPS configurations saved")

# ── Chart 3: Feature Status ────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 8))
fig.suptitle('VisionAssist — Feature Implementation Status',
             fontsize=14, fontweight='bold')

features = [
    ('YOLO11n Object Detection',       'Phase 1', True),
    ('ONNX Runtime Optimization',      'Phase 1', True),
    ('Dual YOLO Pipeline (n+s)',       'Phase 1', True),
    ('Threaded Camera Capture',        'Phase 1', True),
    ('CLAHE Image Preprocessing',      'Phase 1', True),
    ('Direction Detection (L/C/R)',    'Phase 1', True),
    ('Urgency System',                 'Phase 1', True),
    ('Frame Stability (3 frames)',     'Phase 1', True),
    ('Cooldown System (6s)',           'Phase 1', True),
    ('Piper TTS Human Voice',         'Phase 2', True),
    ('Vosk Offline Voice Commands',    'Phase 2', True),
    ('EasyOCR Text Reading',          'Phase 2', True),
    ('Mode Manager (Nav/Read/Wait)',   'Phase 2', True),
    ('AirPods Bluetooth Audio',        'Phase 2', True),
    ('USB Webcam Support',             'Phase 2', True),
    ('Class Filtering (20 classes)',   'Phase 1', False),
    ('ByteTrack Object Tracking',      'Phase 3', False),
    ('Smart Scene Summary',            'Phase 3', False),
    ('Ultrasonic Sensor Distance',     'Phase 4', False),
    ('Auto Start on Boot',             'Phase 4', False),
    ('Moondream2 Scene Description',   'Phase 5', False),
    ('Fine-tune on Custom Data',       'Phase 5', False),
]

phase_colors = {
    'Phase 1': PURPLE,
    'Phase 2': BLUE,
    'Phase 3': AMBER,
    'Phase 4': RED,
    'Phase 5': GRAY
}

for i, (feat, phase, done) in enumerate(features):
    color = TEAL if done else '#D3D1C7'
    ax.barh(i, 1, color=color, edgecolor='white', height=0.7)
    status = '[DONE]' if done else '[TODO]'
    text_color = '#085041' if done else '#5F5E5A'
    ax.text(0.02, i, f'{status}  {feat}', va='center', fontsize=9,
            fontweight='bold' if done else 'normal', color=text_color)
    phase_col = phase_colors.get(phase, GRAY)
    ax.text(1.02, i, phase, va='center', fontsize=8,
            color=phase_col, fontweight='bold')

ax.set_xlim(0, 1.25)
ax.set_ylim(-0.5, len(features)-0.5)
ax.set_yticks([])
ax.set_xticks([])
done_count = sum(1 for _, _, d in features if d)
ax.set_title(f'Completed: {done_count}/{len(features)} features', fontsize=11)

done_p = mpatches.Patch(color=TEAL, label='Completed')
todo_p = mpatches.Patch(color='#D3D1C7', label='Pending')
ax.legend(handles=[done_p, todo_p], fontsize=9, loc='lower right')

fig.tight_layout()
fig.savefig('/home/raspberrypi/VisionAssist/graphs/03_feature_status.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("[3/6] Feature status saved")

# ── Chart 4: Issues and Fixes ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Current Issues and Planned Fixes — VisionAssist',
             fontsize=14, fontweight='bold')

issues = ['False detections\n(no class filter)', 'Object flickering\n(no tracking)',
          'FPS drop\n(window ON)', 'Voice mishearing', 'No real distance\nmeasurement']
severity = [9, 8, 7, 6, 8]

ax = axes[0]
bars = ax.barh(issues, severity, color=RED, edgecolor='white', height=0.5)
for bar, val in zip(bars, severity):
    ax.text(val+0.1, bar.get_y()+bar.get_height()/2,
            str(val), va='center', fontsize=11, fontweight='bold')
ax.set_xlabel('Severity (1-10)')
ax.set_title('Current Issues', fontsize=11)
ax.set_xlim(0, 11)
ax.grid(axis='x', alpha=0.3)

fixes = ['Class filtering\n(20 priority classes)', 'ByteTrack\nobject tracking',
         'SHOW_WINDOW=False\nproduction mode', 'Larger Vosk\nmodel',
         'HC-SR04P ultrasonic\nsensor']
effectiveness = [9, 8, 8, 7, 10]

ax2 = axes[1]
bars2 = ax2.barh(fixes, effectiveness, color=TEAL, edgecolor='white', height=0.5)
for bar, val in zip(bars2, effectiveness):
    ax2.text(val+0.1, bar.get_y()+bar.get_height()/2,
             str(val), va='center', fontsize=11, fontweight='bold')
ax2.set_xlabel('Effectiveness (1-10)')
ax2.set_title('Planned Fixes', fontsize=11)
ax2.set_xlim(0, 11)
ax2.grid(axis='x', alpha=0.3)

fig.tight_layout()
fig.savefig('/home/raspberrypi/VisionAssist/graphs/04_issues_and_fixes.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("[4/6] Issues and fixes saved")

# ── Chart 5: FPS Impact per Component ─────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
fig.suptitle('FPS Impact of Each Component — VisionAssist Full App',
             fontsize=14, fontweight='bold')

components = ['YOLO11n\nbaseline', '+ YOLO11s\n(x10 frames)',
              '+ Threaded\ncamera', '+ Vosk +\nSpeech', '+ cv2.imshow\nwindow']
fps_impact  = [21.8, 16.6, 15.6, 14.0, 8.5]
loss_labels = ['baseline', '-5.2 FPS', '-1.0 FPS', '-1.6 FPS', '-5.5 FPS']

ax.plot(range(len(components)), fps_impact, 'o-',
        color=PURPLE, lw=2.5, markersize=9)
for i, (y, loss_l) in enumerate(zip(fps_impact, loss_labels)):
    color = RED if i in [1, 4] else GRAY
    ax.annotate(f'{y:.1f} FPS\n({loss_l})',
                xy=(i, y), xytext=(0, 14),
                textcoords='offset points', ha='center',
                fontsize=8, fontweight='bold', color=color)

ax.axhline(10, color=RED, ls='--', lw=1.5, label='Min safe (10 FPS)')
ax.axhline(15, color=AMBER, ls='--', lw=1.5, label='Good (15 FPS)')

for i in range(len(fps_impact)-1):
    col = TEAL if fps_impact[i] >= 10 else RED
    ax.fill_between([i, i+1],
                    [fps_impact[i], fps_impact[i+1]], 10,
                    alpha=0.08, color=col)

ax.set_ylabel('FPS')
ax.set_xticks(range(len(components)))
ax.set_xticklabels(components)
ax.set_title('Cumulative FPS Impact as Components Are Added', fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_ylim(0, 26)

fig.tight_layout()
fig.savefig('/home/raspberrypi/VisionAssist/graphs/05_fps_impact_components.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("[5/6] FPS impact saved")

# ── Chart 6: Model Size vs Speed ──────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle('Model Size vs Speed Trade-off — VisionAssist',
             fontsize=14, fontweight='bold')

model_names = ['yolo11n', 'yolov8n', 'yolo11s', 'yolov8m']
sizes_mb    = [10.1, 12.1, 36.2, 98.9]
fps_vals2   = [21.8, 22.7, 10.7, 4.1]
dot_colors  = [TEAL, BLUE, AMBER, RED]
roles       = ['PRIMARY\n(current)', 'Similar speed\nto yolo11n',
               'SECONDARY\n(every 10fr)', 'Too slow\nfor Pi 5']

scatter = ax.scatter(sizes_mb, fps_vals2, c=dot_colors,
                     s=300, zorder=5, edgecolors='white', linewidth=2)

for name, x, y, role, col in zip(model_names, sizes_mb, fps_vals2, roles, dot_colors):
    ax.annotate(f'{name}\n{role}',
                xy=(x, y), xytext=(10, 5),
                textcoords='offset points',
                fontsize=8, color=col, fontweight='bold')

ax.axhline(10, color=RED, ls='--', lw=1.5, label='Min safe (10 FPS)')
ax.axhline(15, color=AMBER, ls='--', lw=1.5, label='Good (15 FPS)')
ax.fill_between([0, 110], 10, 25, alpha=0.05, color=TEAL, label='Safe zone')
ax.fill_between([0, 110], 0, 10, alpha=0.05, color=RED, label='Danger zone')

ax.set_xlabel('Model Size (MB)')
ax.set_ylabel('FPS on Pi 5')
ax.set_title('Bigger model = slower on Pi 5 ARM CPU', fontsize=11)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xlim(0, 110)
ax.set_ylim(0, 26)

fig.tight_layout()
fig.savefig('/home/raspberrypi/VisionAssist/graphs/06_model_size_vs_speed.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("[6/6] Model size vs speed saved")

print("\nAll graphs saved to ~/VisionAssist/graphs/")
print("Run: cd ~/VisionAssist && git add graphs/ && git commit -m 'Add performance graphs' && git push")
