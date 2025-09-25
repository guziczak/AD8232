import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

# Read the EKG data
df = pd.read_csv('ekg_5min_20250626_161335.csv')

# Convert timestamp from microseconds to seconds
df['time_seconds'] = (df['timestamp_us'] - df['timestamp_us'].iloc[0]) / 1_000_000

# Extract voltage data
voltage = df['voltage'].values
time = df['time_seconds'].values

# Calculate sampling rate
sampling_rate = 250  # Based on previous analysis

# Create a detailed view of different segments
fig, axes = plt.subplots(4, 1, figsize=(14, 12))

# Segment 1: First spike region (around 4 seconds)
start1 = 1070
end1 = 1150
time1 = time[start1:end1]
voltage1 = voltage[start1:end1]

axes[0].plot(time1, voltage1, 'b-', linewidth=2, marker='o', markersize=3)
axes[0].set_title('Spike Pattern 1 (Detailed View)', fontsize=12)
axes[0].set_ylabel('Voltage (V)')
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([1.5, 2.4])

# Annotate key features
peak_idx = np.argmax(voltage1)
axes[0].annotate(f'Peak: {voltage1[peak_idx]:.3f}V', 
                xy=(time1[peak_idx], voltage1[peak_idx]),
                xytext=(time1[peak_idx]+0.05, voltage1[peak_idx]+0.1),
                arrowprops=dict(arrowstyle='->'))

# Segment 2: Second spike region (around 6 seconds)
start2 = 1290
end2 = 1370
time2 = time[start2:end2]
voltage2 = voltage[start2:end2]

axes[1].plot(time2, voltage2, 'g-', linewidth=2, marker='o', markersize=3)
axes[1].set_title('Spike Pattern 2 (Detailed View)', fontsize=12)
axes[1].set_ylabel('Voltage (V)')
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim([1.5, 2.4])

# Segment 3: Baseline region
start3 = 1180
end3 = 1260
time3 = time[start3:end3]
voltage3 = voltage[start3:end3]

axes[2].plot(time3, voltage3, 'r-', linewidth=2, marker='o', markersize=3)
axes[2].set_title('Baseline Region (Between Spikes)', fontsize=12)
axes[2].set_ylabel('Voltage (V)')
axes[2].grid(True, alpha=0.3)
axes[2].set_ylim([1.5, 1.7])

# Segment 4: Comparison with textbook EKG
axes[3].text(0.05, 0.95, 'EKG Signal Analysis Summary:', 
             transform=axes[3].transAxes, fontsize=14, fontweight='bold',
             verticalalignment='top')

analysis_text = """
OBSERVED CHARACTERISTICS:
• Sampling rate: 250 Hz (adequate for EKG)
• Voltage range: ~0.85V (852 mV) - FAR TOO HIGH for physiological EKG
• Baseline voltage: ~1.65V
• Spike amplitude: ~0.6-0.7V above baseline
• Spike duration: ~120-160ms
• Spike frequency: ~1 Hz (60 bpm if these were heartbeats)

TEXTBOOK EKG CHARACTERISTICS:
• P wave: 80-120ms duration, <0.25mV amplitude
• QRS complex: 60-100ms duration, 0.5-2.0mV amplitude
• T wave: ~160ms duration, <0.5mV amplitude
• Total amplitude range: typically 1-2 mV peak-to-peak

CONCLUSION:
This signal does NOT represent a proper textbook EKG waveform:
1. Voltage amplitudes are ~400-800x too large (852mV vs typical 1-2mV)
2. No clear P waves, QRS complexes, or T waves visible
3. Spikes are too wide and simple (lacking QRS morphology)
4. Signal appears to be either:
   - Severely amplified/miscalibrated
   - Not from proper EKG electrodes
   - Contaminated with significant artifacts
   
For a proper EKG, you would need:
- Proper electrode placement and skin preparation
- Correct amplifier gain settings (typical EKG gain: 1000x)
- Appropriate filtering (0.5-150 Hz bandpass)
- Proper grounding to reduce noise
"""

axes[3].text(0.05, 0.85, analysis_text, transform=axes[3].transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('detailed_ekg_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis: Check if spikes show any QRS-like features
print("Detailed Spike Analysis:")
print("-" * 50)

# Analyze the first spike in detail
spike_region = voltage[1070:1120]
spike_time = time[1070:1120]

# Calculate rise and fall times
max_idx = np.argmax(spike_region)
baseline = np.mean(spike_region[:10])
peak = spike_region[max_idx]
amplitude = peak - baseline

# Find 10% and 90% points for rise time
thresh_10 = baseline + 0.1 * amplitude
thresh_90 = baseline + 0.9 * amplitude

rise_region = spike_region[:max_idx]
rise_start = np.where(rise_region > thresh_10)[0][0] if np.any(rise_region > thresh_10) else 0
rise_end = np.where(rise_region > thresh_90)[0][0] if np.any(rise_region > thresh_90) else max_idx

rise_time = (rise_end - rise_start) * 4  # 4ms per sample
print(f"Spike rise time (10-90%): {rise_time} ms")
print(f"Spike amplitude: {amplitude*1000:.1f} mV")
print(f"Spike width at half maximum: ~{np.sum(spike_region > baseline + 0.5*amplitude) * 4} ms")

print("\nFor comparison, typical QRS complex:")
print("- Duration: 60-100 ms")
print("- Amplitude: 0.5-2.0 mV")
print("- Sharp, narrow morphology")

# Create a comparison plot with a synthetic textbook EKG
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Plot observed signal
ax1.plot(spike_time, spike_region, 'b-', linewidth=2)
ax1.set_title('Observed Signal Spike', fontsize=14)
ax1.set_ylabel('Voltage (V)')
ax1.set_ylim([1.5, 2.4])
ax1.grid(True, alpha=0.3)

# Create and plot a synthetic textbook EKG complex
t_synth = np.linspace(0, 0.8, 200)
# P wave
p_wave = 0.00015 * np.exp(-((t_synth - 0.1)**2) / (2 * 0.01**2))
# QRS complex
qrs = -0.0003 * np.exp(-((t_synth - 0.22)**2) / (2 * 0.003**2))  # Q
qrs += 0.0015 * np.exp(-((t_synth - 0.24)**2) / (2 * 0.005**2))  # R
qrs += -0.0002 * np.exp(-((t_synth - 0.26)**2) / (2 * 0.003**2))  # S
# T wave
t_wave = 0.0003 * np.exp(-((t_synth - 0.4)**2) / (2 * 0.03**2))

synthetic_ekg = p_wave + qrs + t_wave
ax2.plot(t_synth, synthetic_ekg * 1000, 'r-', linewidth=2)  # Convert to mV
ax2.set_title('Textbook EKG Complex (scaled to mV)', fontsize=14)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Voltage (mV)')
ax2.grid(True, alpha=0.3)

# Annotate features
ax2.annotate('P', xy=(0.1, 0.15), fontsize=12)
ax2.annotate('Q', xy=(0.22, -0.25), fontsize=12)
ax2.annotate('R', xy=(0.24, 1.4), fontsize=12)
ax2.annotate('S', xy=(0.26, -0.15), fontsize=12)
ax2.annotate('T', xy=(0.4, 0.25), fontsize=12)

plt.tight_layout()
plt.savefig('ekg_comparison.png', dpi=300, bbox_inches='tight')
plt.show()