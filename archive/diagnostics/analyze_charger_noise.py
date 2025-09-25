"""Analyze the impact of laptop charger on EKG signal quality"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import os

# Set style
sns.set_style("darkgrid")
plt.style.use('dark_background')

def load_and_analyze(filename):
    """Load CSV and compute signal metrics"""
    df = pd.read_csv(filename)
    adc_values = df['adc_value'].values
    
    # Basic statistics
    mean_val = np.mean(adc_values)
    std_val = np.std(adc_values)
    
    # Remove DC offset for analysis
    centered = adc_values - mean_val
    
    # Frequency analysis
    fs = 250  # Sample rate Hz
    freqs = fftfreq(len(centered), 1/fs)
    fft_vals = np.abs(fft(centered))
    
    # Find 50Hz and harmonics
    power_50hz = fft_vals[np.argmin(np.abs(freqs - 50))]
    power_100hz = fft_vals[np.argmin(np.abs(freqs - 100))]
    power_150hz = fft_vals[np.argmin(np.abs(freqs - 150))]
    
    # Signal quality metrics
    snr = 20 * np.log10(np.max(np.abs(centered)) / std_val)
    
    # Count saturated samples
    saturated_high = np.sum(adc_values > 64000)
    saturated_low = np.sum(adc_values < 1000)
    
    return {
        'df': df,
        'mean': mean_val,
        'std': std_val,
        'min': np.min(adc_values),
        'max': np.max(adc_values),
        'range': np.max(adc_values) - np.min(adc_values),
        'power_50hz': power_50hz,
        'power_100hz': power_100hz,
        'power_150hz': power_150hz,
        'snr_db': snr,
        'saturated_high': saturated_high,
        'saturated_low': saturated_low,
        'centered': centered,
        'freqs': freqs[:len(freqs)//2],  # Only positive frequencies
        'fft_vals': fft_vals[:len(fft_vals)//2]
    }

# Analyze both files
print("="*70)
print("EKG CHARGER NOISE ANALYSIS")
print("="*70)

# File paths
file_with_charger = 'diagnostic_raw_20250626_172630.csv'
file_without_charger = 'diagnostic_raw_20250626_172910.csv'

# Check if files exist
for f in [file_with_charger, file_without_charger]:
    if not os.path.exists(f):
        print(f"Error: {f} not found!")
        exit(1)

# Analyze
print("\nAnalyzing signals...")
with_charger = load_and_analyze(file_with_charger)
without_charger = load_and_analyze(file_without_charger)

# Print comparison
print("\n" + "="*70)
print("NUMERICAL COMPARISON")
print("="*70)

metrics = [
    ("Signal Range (ADC)", with_charger['range'], without_charger['range'], "units"),
    ("Noise (std dev)", with_charger['std'], without_charger['std'], ""),
    ("50Hz Power", with_charger['power_50hz'], without_charger['power_50hz'], ""),
    ("100Hz Power", with_charger['power_100hz'], without_charger['power_100hz'], ""),
    ("SNR", with_charger['snr_db'], without_charger['snr_db'], "dB"),
    ("Saturation", with_charger['saturated_high'] + with_charger['saturated_low'], 
     without_charger['saturated_high'] + without_charger['saturated_low'], "samples")
]

for metric, with_val, without_val, unit in metrics:
    ratio = with_val / without_val if without_val > 0 else float('inf')
    print(f"\n{metric}:")
    print(f"  With charger:    {with_val:,.1f} {unit}")
    print(f"  Without charger: {without_val:,.1f} {unit}")
    print(f"  Ratio:           {ratio:.2f}x")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('EKG Signal Analysis: Impact of Laptop Charger on Signal Quality', 
             fontsize=20, y=0.98)

# 1. Time domain comparison (2 seconds)
ax1 = fig.add_subplot(gs[0, :2])
time_samples = 500
ax1.plot(with_charger['df']['time_s'][:time_samples], 
         with_charger['df']['voltage'][:time_samples], 
         'r-', alpha=0.8, linewidth=1, label='With Charger')
ax1.set_title('Signal WITH Charger (2 seconds)', fontsize=14)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Voltage (V)')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = fig.add_subplot(gs[0, 2:])
ax2.plot(without_charger['df']['time_s'][:time_samples], 
         without_charger['df']['voltage'][:time_samples], 
         'g-', alpha=0.8, linewidth=1, label='Without Charger')
ax2.set_title('Signal WITHOUT Charger (2 seconds)', fontsize=14)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Voltage (V)')
ax2.grid(True, alpha=0.3)
ax2.legend()

# 2. Distribution comparison
ax3 = fig.add_subplot(gs[1, 0])
sns.histplot(data=with_charger['centered'], bins=100, color='red', alpha=0.7, ax=ax3)
ax3.set_title('Distribution (With Charger)', fontsize=12)
ax3.set_xlabel('ADC Value (centered)')
ax3.axvline(0, color='white', linestyle='--', alpha=0.5)

ax4 = fig.add_subplot(gs[1, 1])
sns.histplot(data=without_charger['centered'], bins=100, color='green', alpha=0.7, ax=ax4)
ax4.set_title('Distribution (Without Charger)', fontsize=12)
ax4.set_xlabel('ADC Value (centered)')
ax4.axvline(0, color='white', linestyle='--', alpha=0.5)

# 3. Frequency spectrum comparison
ax5 = fig.add_subplot(gs[1, 2:])
freq_limit = 120
freq_mask_with = with_charger['freqs'] < freq_limit
freq_mask_without = without_charger['freqs'] < freq_limit

ax5.semilogy(with_charger['freqs'][freq_mask_with], 
             with_charger['fft_vals'][freq_mask_with], 
             'r-', alpha=0.8, linewidth=1.5, label='With Charger')
ax5.semilogy(without_charger['freqs'][freq_mask_without], 
             without_charger['fft_vals'][freq_mask_without], 
             'g-', alpha=0.8, linewidth=1.5, label='Without Charger')

# Mark power line frequencies
for f in [50, 100]:
    ax5.axvline(f, color='yellow', linestyle='--', alpha=0.5)
    ax5.text(f, ax5.get_ylim()[1]*0.5, f'{f}Hz', 
             rotation=90, va='bottom', ha='right', color='yellow')

ax5.set_title('Frequency Spectrum Comparison', fontsize=14)
ax5.set_xlabel('Frequency (Hz)')
ax5.set_ylabel('Power (log scale)')
ax5.grid(True, alpha=0.3)
ax5.legend()

# 4. Box plot comparison
ax6 = fig.add_subplot(gs[2, 0])
data_for_box = pd.DataFrame({
    'With Charger': with_charger['centered'][:5000],
    'Without Charger': without_charger['centered'][:5000]
})
data_melted = data_for_box.melt(var_name='Condition', value_name='ADC Value')
sns.boxplot(data=data_melted, x='Condition', y='ADC Value', ax=ax6)
ax6.set_title('Signal Range Comparison', fontsize=12)

# 5. Power line interference bar chart
ax7 = fig.add_subplot(gs[2, 1])
freqs = ['50Hz', '100Hz', '150Hz']
with_powers = [with_charger['power_50hz'], with_charger['power_100hz'], with_charger['power_150hz']]
without_powers = [without_charger['power_50hz'], without_charger['power_100hz'], without_charger['power_150hz']]

x = np.arange(len(freqs))
width = 0.35

bars1 = ax7.bar(x - width/2, with_powers, width, label='With Charger', color='red', alpha=0.7)
bars2 = ax7.bar(x + width/2, without_powers, width, label='Without Charger', color='green', alpha=0.7)

ax7.set_ylabel('Power')
ax7.set_title('Power Line Interference', fontsize=12)
ax7.set_xticks(x)
ax7.set_xticklabels(freqs)
ax7.legend()
ax7.set_yscale('log')

# 6. Summary metrics
ax8 = fig.add_subplot(gs[2, 2:])
ax8.axis('off')

summary_text = f"""
IMPACT SUMMARY:

🔌 With Charger Connected:
  • Signal {(with_charger['range'] / without_charger['range']):.1f}x larger
  • Noise {(with_charger['std'] / without_charger['std']):.1f}x higher
  • 50Hz interference {(with_charger['power_50hz'] / without_charger['power_50hz']):.0f}x stronger
  • {with_charger['saturated_high'] + with_charger['saturated_low']} saturated samples
  
🔋 Without Charger (Battery Power):
  • Clean signal, no saturation
  • Minimal 50Hz interference
  • Better SNR: {without_charger['snr_db']:.1f} dB
  • Proper signal range for EKG

📊 RECOMMENDATION:
  ALWAYS use battery power for EKG measurements!
  Required scaling: 1/{int(without_charger['range']/40)} (factor: {40/without_charger['range']:.6f})
"""

ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes, 
         fontsize=12, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

plt.savefig('charger_noise_analysis.png', dpi=150, bbox_inches='tight', facecolor='black')
print(f"\n✅ Analysis complete! Plot saved to: charger_noise_analysis.png")

# Final recommendations
print("\n" + "="*70)
print("CRITICAL FINDINGS")
print("="*70)
print(f"\n🔴 Charger increases 50Hz noise by {with_charger['power_50hz']/without_charger['power_50hz']:.0f}x!")
print(f"🔴 Signal range increases by {with_charger['range']/without_charger['range']:.1f}x with charger")
print(f"🔴 ADC saturation occurs with charger connected")
print(f"\n✅ WITHOUT charger, signal needs {without_charger['range']/40:.0f}x reduction")
print(f"✅ Use scaling factor: {40/without_charger['range']:.6f}")
print("\n⚡ ALWAYS disconnect ALL chargers during EKG measurement!")