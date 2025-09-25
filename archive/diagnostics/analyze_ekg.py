import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt
from scipy import signal

# Read the EKG data
df = pd.read_csv('ekg_5min_20250626_161335.csv')

# Convert timestamp from microseconds to seconds
df['time_seconds'] = (df['timestamp_us'] - df['timestamp_us'].iloc[0]) / 1_000_000

# Analyze sampling rate
time_diffs = df['time_seconds'].diff().dropna()
avg_sampling_interval = time_diffs.mean()
sampling_rate = 1 / avg_sampling_interval
print(f"Average sampling rate: {sampling_rate:.2f} Hz")
print(f"Average sampling interval: {avg_sampling_interval*1000:.2f} ms")

# Extract voltage data for analysis
voltage = df['voltage'].values
time = df['time_seconds'].values

# Select a portion of the data for detailed visualization (5 seconds)
start_idx = 1000
duration_samples = int(5 * sampling_rate)  # 5 seconds of data
end_idx = start_idx + duration_samples

voltage_segment = voltage[start_idx:end_idx]
time_segment = time[start_idx:end_idx]

# Apply bandpass filter (0.5-40 Hz is typical for EKG)
nyquist = sampling_rate / 2
low_freq = 0.5 / nyquist
high_freq = 40 / nyquist

# Check if frequencies are valid
if low_freq < 1 and high_freq < 1:
    b, a = butter(4, [low_freq, high_freq], btype='band')
    filtered_voltage = filtfilt(b, a, voltage_segment)
else:
    filtered_voltage = voltage_segment
    print("Warning: Sampling rate too low for proper filtering")

# Analyze signal characteristics
voltage_range = np.max(voltage_segment) - np.min(voltage_segment)
mean_voltage = np.mean(voltage_segment)
std_voltage = np.std(voltage_segment)

print(f"\nSignal characteristics (5-second segment):")
print(f"Voltage range: {voltage_range:.3f} V")
print(f"Mean voltage: {mean_voltage:.3f} V")
print(f"Standard deviation: {std_voltage:.3f} V")

# Find peaks (potential R-waves)
# Adjust distance based on expected heart rate (40-200 bpm)
min_distance = int(sampling_rate * 0.3)  # 200 bpm max
prominence = std_voltage * 0.5  # Peaks should be at least 0.5 std above baseline

peaks, properties = find_peaks(filtered_voltage, 
                              distance=min_distance,
                              prominence=prominence,
                              height=mean_voltage + std_voltage)

# Calculate heart rate from detected peaks
if len(peaks) > 1:
    peak_intervals = np.diff(time_segment[peaks])
    avg_interval = np.mean(peak_intervals)
    heart_rate = 60 / avg_interval
    print(f"\nDetected {len(peaks)} potential R-peaks")
    print(f"Estimated heart rate: {heart_rate:.1f} bpm")
else:
    print("\nInsufficient peaks detected for heart rate calculation")

# Create visualization
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Plot 1: Raw signal
axes[0].plot(time_segment, voltage_segment, 'b-', linewidth=0.5)
axes[0].set_title('Raw EKG Signal (5 seconds)', fontsize=14)
axes[0].set_ylabel('Voltage (V)')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=mean_voltage, color='r', linestyle='--', alpha=0.5, label='Mean')
axes[0].legend()

# Plot 2: Filtered signal with detected peaks
axes[1].plot(time_segment, filtered_voltage, 'g-', linewidth=1)
if len(peaks) > 0:
    axes[1].plot(time_segment[peaks], filtered_voltage[peaks], 'ro', markersize=8, label='Detected R-peaks')
axes[1].set_title('Filtered EKG Signal (0.5-40 Hz bandpass)', fontsize=14)
axes[1].set_ylabel('Voltage (V)')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

# Plot 3: Power spectrum analysis
frequencies, power = signal.welch(voltage_segment, sampling_rate, nperseg=1024)
axes[2].semilogy(frequencies, power)
axes[2].set_title('Power Spectral Density', fontsize=14)
axes[2].set_xlabel('Frequency (Hz)')
axes[2].set_ylabel('Power (V²/Hz)')
axes[2].set_xlim(0, 50)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ekg_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Analyze signal quality indicators
print("\n=== EKG Signal Quality Analysis ===")

# 1. Check for baseline drift
baseline_trend = np.polyfit(time_segment, voltage_segment, 1)[0]
print(f"\nBaseline drift: {baseline_trend*1000:.3f} mV/s")

# 2. Signal-to-noise ratio estimation
if len(peaks) > 0:
    signal_amplitude = np.mean([filtered_voltage[p] - mean_voltage for p in peaks])
    noise_estimate = np.std(filtered_voltage[filtered_voltage < mean_voltage + std_voltage*0.5])
    snr = 20 * np.log10(signal_amplitude / noise_estimate)
    print(f"Estimated SNR: {snr:.1f} dB")

# 3. Check for typical EKG features
print("\n=== Textbook EKG Feature Assessment ===")

# Typical EKG characteristics:
# - P wave: ~0.08-0.12s duration, <0.25mV amplitude
# - QRS complex: ~0.06-0.10s duration, 0.5-2.0mV amplitude  
# - T wave: ~0.16s duration, <0.5mV amplitude
# - Normal rhythm: 60-100 bpm

# Check voltage range (typical EKG: 0.5-2.0 mV peak-to-peak)
voltage_range_mv = voltage_range * 1000  # Convert to mV
if 0.5 <= voltage_range_mv <= 5.0:
    print(f"✓ Voltage range ({voltage_range_mv:.1f} mV) is within typical EKG range")
else:
    print(f"✗ Voltage range ({voltage_range_mv:.1f} mV) is outside typical EKG range (0.5-5.0 mV)")

# Check for rhythmicity
if len(peaks) > 2:
    interval_variability = np.std(peak_intervals) / np.mean(peak_intervals)
    if interval_variability < 0.2:
        print(f"✓ Regular rhythm detected (variability: {interval_variability*100:.1f}%)")
    else:
        print(f"✗ Irregular rhythm (variability: {interval_variability*100:.1f}%)")

# Check heart rate
if len(peaks) > 1:
    if 40 <= heart_rate <= 200:
        print(f"✓ Heart rate ({heart_rate:.0f} bpm) is within physiological range")
    else:
        print(f"✗ Heart rate ({heart_rate:.0f} bpm) is outside normal range")

# Frequency content check
peak_frequency = frequencies[np.argmax(power)]
print(f"\nDominant frequency: {peak_frequency:.1f} Hz")

# Summary
print("\n=== SUMMARY ===")
if len(peaks) > 0 and 0.5 <= voltage_range_mv <= 5.0:
    print("The signal shows some characteristics of an EKG waveform:")
    print("- Periodic peaks detected")
    print("- Voltage range is plausible")
    print("\nHowever, definitive identification of P, QRS, and T waves")
    print("would require higher resolution analysis and clinical validation.")
else:
    print("The signal does NOT appear to be a proper textbook EKG:")
    print("- May be noise or poor electrode contact")
    print("- Missing characteristic EKG morphology")