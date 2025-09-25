import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import seaborn as sns

# Load the data files
with_charger = pd.read_csv('diagnostic_raw_20250626_172630.csv')
without_charger = pd.read_csv('diagnostic_raw_20250626_172910.csv')

print("=== EKG Signal Analysis: Charger Impact ===\n")
print(f"With charger data points: {len(with_charger)}")
print(f"Without charger data points: {len(without_charger)}")

# Basic statistics
print("\n=== ADC Value Statistics ===")
print(f"With charger - Mean: {with_charger['adc_value'].mean():.1f}, Std: {with_charger['adc_value'].std():.1f}")
print(f"Without charger - Mean: {without_charger['adc_value'].mean():.1f}, Std: {without_charger['adc_value'].std():.1f}")

print("\n=== Voltage Statistics (mV) ===")
print(f"With charger - Mean: {with_charger['voltage'].mean():.3f} V, Std: {with_charger['voltage'].std():.3f} V")
print(f"Without charger - Mean: {without_charger['voltage'].mean():.3f} V, Std: {without_charger['voltage'].std():.3f} V")

# Check for saturation
print("\n=== Saturation Analysis ===")
max_adc = 65535  # 16-bit ADC max
saturation_threshold = max_adc * 0.95  # 95% of max
with_charger_saturated = (with_charger['adc_value'] > saturation_threshold).sum()
without_charger_saturated = (without_charger['adc_value'] > saturation_threshold).sum()
print(f"With charger saturated samples: {with_charger_saturated} ({with_charger_saturated/len(with_charger)*100:.2f}%)")
print(f"Without charger saturated samples: {without_charger_saturated} ({without_charger_saturated/len(without_charger)*100:.2f}%)")

# Calculate noise metrics (using high-pass filter to remove DC and low frequencies)
def calculate_noise_metrics(data, cutoff_freq=10):
    """Calculate noise metrics after removing DC and low frequency components"""
    fs = 250  # Sample rate
    # High-pass filter to remove DC and low frequencies
    sos = signal.butter(4, cutoff_freq, 'high', fs=fs, output='sos')
    filtered = signal.sosfilt(sos, data)
    return np.std(filtered), filtered

print("\n=== Noise Analysis (High-frequency noise > 10 Hz) ===")
noise_std_with, filtered_with = calculate_noise_metrics(with_charger['voltage'].values)
noise_std_without, filtered_without = calculate_noise_metrics(without_charger['voltage'].values)
print(f"With charger noise (std): {noise_std_with*1000:.2f} mV")
print(f"Without charger noise (std): {noise_std_without*1000:.2f} mV")
print(f"Noise increase with charger: {(noise_std_with/noise_std_without - 1)*100:.1f}%")

# Frequency analysis
def perform_fft_analysis(data, fs=250):
    """Perform FFT analysis and return frequencies and power spectrum"""
    n = len(data)
    yf = fft(data - np.mean(data))  # Remove DC component
    xf = fftfreq(n, 1/fs)[:n//2]
    power = 2.0/n * np.abs(yf[:n//2])
    return xf, power

# Create visualization
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('EKG Signal Analysis: Impact of Laptop Charger', fontsize=16)

# Time domain plots
time_window = 10  # seconds
samples_to_plot = int(250 * time_window)

# Plot 1: Raw signals
ax = axes[0, 0]
ax.plot(with_charger['time_s'][:samples_to_plot], 
        with_charger['voltage'][:samples_to_plot], 
        'r-', alpha=0.7, linewidth=0.5, label='With charger')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.set_title('Raw Signal - With Charger')
ax.grid(True, alpha=0.3)
ax.legend()

ax = axes[0, 1]
ax.plot(without_charger['time_s'][:samples_to_plot], 
        without_charger['voltage'][:samples_to_plot], 
        'b-', alpha=0.7, linewidth=0.5, label='Without charger')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Voltage (V)')
ax.set_title('Raw Signal - Without Charger')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 2: Histogram of values
ax = axes[1, 0]
ax.hist(with_charger['voltage'], bins=100, color='red', alpha=0.6, label='With charger')
ax.hist(without_charger['voltage'], bins=100, color='blue', alpha=0.6, label='Without charger')
ax.set_xlabel('Voltage (V)')
ax.set_ylabel('Count')
ax.set_title('Voltage Distribution')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Power spectrum
ax = axes[1, 1]
freq_with, power_with = perform_fft_analysis(with_charger['voltage'].values)
freq_without, power_without = perform_fft_analysis(without_charger['voltage'].values)

# Focus on relevant frequencies (0-100 Hz)
freq_mask = freq_with <= 100
ax.semilogy(freq_with[freq_mask], power_with[freq_mask], 'r-', alpha=0.7, label='With charger')
ax.semilogy(freq_without[freq_mask], power_without[freq_mask], 'b-', alpha=0.7, label='Without charger')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power')
ax.set_title('Power Spectrum (0-100 Hz)')
ax.grid(True, alpha=0.3)
ax.legend()

# Plot 4: Filtered signals (high-pass to show noise)
ax = axes[2, 0]
ax.plot(with_charger['time_s'][:samples_to_plot], 
        filtered_with[:samples_to_plot] * 1000, 
        'r-', alpha=0.7, linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Noise (mV)')
ax.set_title(f'High-frequency Noise - With Charger (std: {noise_std_with*1000:.1f} mV)')
ax.grid(True, alpha=0.3)

ax = axes[2, 1]
ax.plot(without_charger['time_s'][:samples_to_plot], 
        filtered_without[:samples_to_plot] * 1000, 
        'b-', alpha=0.7, linewidth=0.5)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Noise (mV)')
ax.set_title(f'High-frequency Noise - Without Charger (std: {noise_std_without*1000:.1f} mV)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('charger_impact_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional analysis: Find specific noise frequencies
print("\n=== Dominant Noise Frequencies ===")
# Find peaks in power spectrum
from scipy.signal import find_peaks

# For with charger
peaks_with, _ = find_peaks(power_with[freq_mask], height=np.max(power_with[freq_mask])*0.1)
print("\nWith charger - dominant frequencies:")
for idx in peaks_with[:5]:  # Top 5 peaks
    print(f"  {freq_with[idx]:.1f} Hz: Power = {power_with[idx]:.4f}")

# For without charger  
peaks_without, _ = find_peaks(power_without[freq_mask], height=np.max(power_without[freq_mask])*0.1)
print("\nWithout charger - dominant frequencies:")
for idx in peaks_without[:5]:  # Top 5 peaks
    print(f"  {freq_without[idx]:.1f} Hz: Power = {power_without[idx]:.4f}")

# Look specifically for 50/60 Hz noise
for target_freq in [50, 60]:
    idx_with = np.argmin(np.abs(freq_with - target_freq))
    idx_without = np.argmin(np.abs(freq_without - target_freq))
    print(f"\n{target_freq} Hz power:")
    print(f"  With charger: {power_with[idx_with]:.6f}")
    print(f"  Without charger: {power_without[idx_without]:.6f}")
    print(f"  Ratio: {power_with[idx_with]/power_without[idx_without]:.1f}x")

# Summary
print("\n=== SUMMARY ===")
print(f"1. Saturation: With charger shows {with_charger_saturated} saturated samples")
print(f"2. Noise level: {(noise_std_with/noise_std_without - 1)*100:.0f}% higher with charger")
print(f"3. Signal quality: Without charger provides cleaner signal")
print(f"4. Recommendation: Always disconnect charger during EKG acquisition")