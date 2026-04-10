import matplotlib.pyplot as plt
import matplotlib
 
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows 한글 폰트 (Mac: AppleGothic)
matplotlib.rcParams['axes.unicode_minus'] = False
 
labels = [
    'm1\n(float32)\nArduino TFLite',
    'm1\n(int8)\nArduino TFLite(CMSIS)',
    'm1\n(float32)\nTVM',
    'm1\n(int8)\nTVM',
    'm2\n(int8)\nArduino TFLite(CMSIS)',
    'm2\n(int8)\nTVM + CMSIS',
]
 
times_us = [427546, 133655, 278810, 340966, 12042, 6582]
times_ms = [t / 1000 for t in times_us]
 
colors = ['#3266ad', '#3266ad', '#3266ad', '#3266ad', '#1D9E75', '#1D9E75']
 
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, times_ms, color=colors, width=0.6, edgecolor='white')
 
for bar, ms in zip(bars, times_ms):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
            f'{ms:.1f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
 
ax.set_ylabel('(ms)', fontsize=12)
ax.set_title('모델 및 추론 엔진별 추론 시간 비교', fontsize=14, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.3)
 
plt.tight_layout()
plt.savefig('_etc/inference_time_chart.png', dpi=150)
plt.show()

labels = [
    'm1\n(float32)',
    'm1\n(int8)',
    'm2\n(float32)',
    'm2\n(int8)',
]
 
acc = [92.98, 92.96, 88.12, 88.13]
 
colors = ['#3266ad', '#3266ad', '#1D9E75', '#1D9E75']
 
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, acc, color=colors, width=0.6, edgecolor='white')
 
for bar, ac in zip(bars, acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f'{ac:.2f} %', ha='center', va='bottom', fontsize=10, fontweight='bold')
 
ax.set_ylabel('%', fontsize=12)
ax.set_title('모델 정확도 비교', fontsize=14, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.grid(axis='y', alpha=0.3)
 
plt.tight_layout()
plt.savefig('_etc/acc_chart.png', dpi=150)
plt.show()