#!/usr/bin/env python3
"""
5-minute EKG data acquisition with comprehensive analysis
"""

import sys
import time
import serial
import serial.tools.list_ports
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt, welch, spectrogram
import json
from collections import deque

# Constants
ACQUISITION_TIME = 300  # 5 minutes
SAMPLE_RATE = 250  # Hz
BAUDRATE = 115200

class EKGAcquisition:
    def __init__(self):
        self.port = "COM5"
        self.connection = None
        self.samples = []
        self.start_time = None
        
    def connect(self):
        """Connect to device."""
        try:
            print(f"Connecting to {self.port}...")
            self.connection = serial.Serial(
                port=self.port,
                baudrate=BAUDRATE,
                timeout=1.0,
                write_timeout=1.0
            )
            time.sleep(2)
            self.connection.reset_input_buffer()
            self.connection.reset_output_buffer()
            print("Connected!")
            return True
        except serial.SerialException as e:
            print(f"Connection failed: {e}")
            return False
            
    def start_streaming(self):
        """Initialize device and start streaming."""
        if not self.connection:
            return False
            
        try:
            print("Initializing device...")
            # Reset
            self.connection.write(b'\x03\x03')
            time.sleep(0.5)
            self.connection.write(b'\x04')
            time.sleep(3)
            
            # Clear buffer
            while self.connection.in_waiting:
                self.connection.read(self.connection.in_waiting)
            
            # Start streaming
            self.connection.write(b'stream()\r')
            time.sleep(2)
            
            # Clear any startup messages
            while self.connection.in_waiting:
                self.connection.readline()
            
            print("Streaming started")
            return True
        except Exception as e:
            print(f"Failed to start streaming: {e}")
            return False
            
    def acquire_data(self):
        """Acquire data for 5 minutes."""
        print(f"\nStarting {ACQUISITION_TIME} second acquisition...")
        print("Keep electrodes attached and try to relax")
        print("Progress: ", end="", flush=True)
        
        self.start_time = time.time()
        self.samples = []
        error_count = 0
        last_progress = 0
        last_status_time = time.time()
        
        while time.time() - self.start_time < ACQUISITION_TIME:
            try:
                if self.connection.in_waiting:
                    line = self.connection.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Skip non-data lines
                    if not line or any(skip in line for skip in ['>>>', 'INFO:', 'FORMAT:', 'Streaming']):
                        continue
                        
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            sample = {
                                'timestamp': int(parts[0]),
                                'adc_value': int(parts[1]),
                                'heart_rate': int(parts[2]),
                                'voltage': (int(parts[1]) / 65535) * 3.3
                            }
                            self.samples.append(sample)
                            
                            # Show progress
                            progress = int((time.time() - self.start_time) / ACQUISITION_TIME * 50)
                            if progress > last_progress:
                                print("█", end="", flush=True)
                                last_progress = progress
                                
                            # Status update every 30 seconds
                            if time.time() - last_status_time > 30:
                                elapsed = int(time.time() - self.start_time)
                                remaining = ACQUISITION_TIME - elapsed
                                print(f"\n{len(self.samples)} samples, {remaining}s remaining\nProgress: ", end="", flush=True)
                                last_status_time = time.time()
                                
                        except ValueError:
                            error_count += 1
                            
            except (ValueError, UnicodeDecodeError):
                error_count += 1
                
        print(" Done!")
        print(f"\nAcquired {len(self.samples)} samples in {time.time() - self.start_time:.1f} seconds")
        print(f"Parse errors: {error_count}")
        return self.samples
        
    def save_raw_data(self):
        """Save raw data to CSV."""
        if not self.samples:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ekg_5min_{timestamp}.csv"
        
        with open(filename, 'w') as f:
            f.write("timestamp_us,adc_value,voltage,heart_rate\n")
            for s in self.samples:
                f.write(f"{s['timestamp']},{s['adc_value']},{s['voltage']:.4f},{s['heart_rate']}\n")
                
        print(f"Raw data saved to: {filename}")
        return filename
        
    def analyze_data(self):
        """Comprehensive analysis of 5-minute recording."""
        if len(self.samples) < 1000:
            print("Not enough data for analysis")
            return None
            
        print("\n=== COMPREHENSIVE EKG ANALYSIS ===")
        
        # Extract arrays
        timestamps = np.array([s['timestamp'] for s in self.samples])
        adc_values = np.array([s['adc_value'] for s in self.samples])
        voltages = np.array([s['voltage'] for s in self.samples])
        
        # Time vector
        time_diffs = np.diff(timestamps) / 1e6
        actual_rate = 1 / np.median(time_diffs)
        time_s = (timestamps - timestamps[0]) / 1e6
        
        analysis = {}
        
        # 1. Basic Statistics
        print("\n1. BASIC STATISTICS:")
        analysis['basic_stats'] = {
            'duration_s': time_s[-1],
            'samples': len(self.samples),
            'sample_rate_target': SAMPLE_RATE,
            'sample_rate_actual': actual_rate,
            'sample_rate_std': np.std(1/time_diffs),
            'voltage_mean': np.mean(voltages),
            'voltage_std': np.std(voltages),
            'voltage_range': [np.min(voltages), np.max(voltages)],
            'adc_saturation_high': np.sum(adc_values > 64000),
            'adc_saturation_low': np.sum(adc_values < 1500),
            'missing_samples': int((time_s[-1] * SAMPLE_RATE) - len(self.samples))
        }
        
        print(f"  Duration: {analysis['basic_stats']['duration_s']:.1f} seconds")
        print(f"  Samples: {analysis['basic_stats']['samples']:,}")
        print(f"  Sample rate: {analysis['basic_stats']['sample_rate_actual']:.1f} Hz (target: {SAMPLE_RATE} Hz)")
        print(f"  Voltage range: {analysis['basic_stats']['voltage_range'][0]:.3f} - {analysis['basic_stats']['voltage_range'][1]:.3f} V")
        print(f"  Missing samples: {analysis['basic_stats']['missing_samples']}")
        
        # 2. Signal Quality Metrics
        print("\n2. SIGNAL QUALITY:")
        
        # DC offset
        dc_offset = np.mean(adc_values) / 65535
        
        # Baseline wander (0.05-0.5 Hz)
        b_baseline, a_baseline = butter(4, [0.05, 0.5], 'band', fs=actual_rate)
        baseline_wander = filtfilt(b_baseline, a_baseline, adc_values)
        baseline_power = np.std(baseline_wander)
        
        # High frequency noise (>100 Hz)
        b_noise, a_noise = butter(4, 100, 'high', fs=actual_rate)
        hf_noise = filtfilt(b_noise, a_noise, adc_values)
        noise_power = np.std(hf_noise)
        
        # Power line interference (48-52 Hz)
        freqs, psd = welch(adc_values, fs=actual_rate, nperseg=2048)
        idx_50hz = (freqs >= 48) & (freqs <= 52)
        power_50hz = np.max(psd[idx_50hz]) if np.any(idx_50hz) else 0
        
        # Motion artifacts detection (sudden large changes)
        diff_signal = np.abs(np.diff(adc_values))
        motion_threshold = np.percentile(diff_signal, 99)
        motion_artifacts = np.sum(diff_signal > motion_threshold)
        
        analysis['signal_quality'] = {
            'dc_offset': dc_offset,
            'baseline_wander_std': baseline_power,
            'hf_noise_std': noise_power,
            'power_50hz_db': 10*np.log10(power_50hz) if power_50hz > 0 else -100,
            'motion_artifacts': motion_artifacts,
            'snr_estimate': np.std(adc_values) / noise_power if noise_power > 0 else 0
        }
        
        print(f"  DC offset: {dc_offset:.3f} ({dc_offset*100:.1f}% of range)")
        print(f"  Baseline wander: {baseline_power:.1f} LSB")
        print(f"  HF noise: {noise_power:.1f} LSB")
        print(f"  50Hz interference: {analysis['signal_quality']['power_50hz_db']:.1f} dB")
        print(f"  Motion artifacts: {motion_artifacts}")
        print(f"  SNR estimate: {analysis['signal_quality']['snr_estimate']:.1f}")
        
        # 3. Heart Rate Analysis
        print("\n3. HEART RATE ANALYSIS:")
        
        # Bandpass filter for QRS
        b_qrs, a_qrs = butter(4, [5, 30], 'band', fs=actual_rate)
        filtered_qrs = filtfilt(b_qrs, a_qrs, adc_values)
        
        # QRS detection
        diff_ecg = np.diff(filtered_qrs)
        squared = diff_ecg ** 2
        window_size = int(0.08 * actual_rate)
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        threshold = np.percentile(integrated, 92)
        peaks, properties = find_peaks(integrated, height=threshold, distance=int(0.3*actual_rate))
        
        if len(peaks) > 10:
            # RR intervals
            rr_intervals = np.diff(peaks) / actual_rate * 1000  # ms
            hr_values = 60000 / rr_intervals  # BPM
            
            # Remove outliers
            valid_hr = (hr_values >= 40) & (hr_values <= 180)
            hr_clean = hr_values[valid_hr]
            rr_clean = rr_intervals[valid_hr]
            
            if len(hr_clean) > 10:
                # HRV metrics
                rmssd = np.sqrt(np.mean(np.diff(rr_clean)**2))
                sdnn = np.std(rr_clean)
                pnn50 = np.sum(np.abs(np.diff(rr_clean)) > 50) / len(rr_clean) * 100
                
                analysis['heart_rate'] = {
                    'detected_beats': len(peaks),
                    'mean_hr': np.mean(hr_clean),
                    'std_hr': np.std(hr_clean),
                    'min_hr': np.min(hr_clean),
                    'max_hr': np.max(hr_clean),
                    'rmssd': rmssd,
                    'sdnn': sdnn,
                    'pnn50': pnn50,
                    'missed_beats_estimate': int((time_s[-1] / 60) * np.mean(hr_clean) - len(peaks))
                }
                
                print(f"  Detected beats: {len(peaks)}")
                print(f"  Heart rate: {analysis['heart_rate']['mean_hr']:.1f} ± {analysis['heart_rate']['std_hr']:.1f} BPM")
                print(f"  Range: {analysis['heart_rate']['min_hr']:.0f} - {analysis['heart_rate']['max_hr']:.0f} BPM")
                print(f"  RMSSD: {rmssd:.1f} ms")
                print(f"  SDNN: {sdnn:.1f} ms")
                print(f"  pNN50: {pnn50:.1f}%")
            else:
                analysis['heart_rate'] = {'error': 'Too few valid beats'}
                print("  Error: Too few valid beats detected")
        else:
            analysis['heart_rate'] = {'error': 'Insufficient QRS complexes detected'}
            print("  Error: Could not detect enough QRS complexes")
            
        # 4. Frequency Domain Analysis
        print("\n4. FREQUENCY ANALYSIS:")
        
        # Compute full spectrum
        freqs_full, psd_full = welch(adc_values, fs=actual_rate, nperseg=4096)
        
        # Find dominant frequencies
        prominent_peaks, _ = find_peaks(psd_full, height=np.percentile(psd_full, 95))
        dominant_freqs = freqs_full[prominent_peaks]
        
        analysis['frequency_domain'] = {
            'dominant_frequencies': dominant_freqs[:5].tolist(),
            'total_power': np.sum(psd_full),
            'power_0_5hz': np.sum(psd_full[freqs_full <= 0.5]),
            'power_5_15hz': np.sum(psd_full[(freqs_full > 5) & (freqs_full <= 15)]),
            'power_15_40hz': np.sum(psd_full[(freqs_full > 15) & (freqs_full <= 40)]),
            'power_above_40hz': np.sum(psd_full[freqs_full > 40])
        }
        
        print(f"  Dominant frequencies: {', '.join([f'{f:.1f}Hz' for f in dominant_freqs[:5]])}")
        print(f"  Power distribution:")
        print(f"    0-0.5 Hz: {analysis['frequency_domain']['power_0_5hz']/analysis['frequency_domain']['total_power']*100:.1f}%")
        print(f"    5-15 Hz: {analysis['frequency_domain']['power_5_15hz']/analysis['frequency_domain']['total_power']*100:.1f}%")
        print(f"    15-40 Hz: {analysis['frequency_domain']['power_15_40hz']/analysis['frequency_domain']['total_power']*100:.1f}%")
        print(f"    >40 Hz: {analysis['frequency_domain']['power_above_40hz']/analysis['frequency_domain']['total_power']*100:.1f}%")
        
        # 5. Recommendations
        print("\n5. RECOMMENDATIONS:")
        recommendations = []
        
        # Sample rate issues
        if abs(actual_rate - SAMPLE_RATE) > 10:
            recommendations.append({
                'issue': 'Significant sample rate deviation',
                'impact': 'high',
                'fix': 'Check timer configuration and CPU load'
            })
            
        # Signal quality issues
        if dc_offset < 0.3 or dc_offset > 0.7:
            recommendations.append({
                'issue': f'DC offset at {dc_offset:.1%} (should be ~50%)',
                'impact': 'medium',
                'fix': 'Adjust bias voltage or check power supply'
            })
            
        if analysis['signal_quality']['snr_estimate'] < 20:
            recommendations.append({
                'issue': f'Low SNR ({analysis["signal_quality"]["snr_estimate"]:.1f})',
                'impact': 'high',
                'fix': 'Improve shielding, check electrode contact'
            })
            
        if analysis['signal_quality']['power_50hz_db'] > -30:
            recommendations.append({
                'issue': 'Strong 50Hz interference',
                'impact': 'high',
                'fix': 'Add better filtering, improve grounding'
            })
            
        if baseline_power > 1000:
            recommendations.append({
                'issue': 'Excessive baseline wander',
                'impact': 'medium',
                'fix': 'Check electrode gel, reduce movement'
            })
            
        if analysis['basic_stats']['adc_saturation_high'] > 100:
            recommendations.append({
                'issue': f'ADC saturation detected ({analysis["basic_stats"]["adc_saturation_high"]} samples)',
                'impact': 'high',
                'fix': 'Reduce gain or adjust input offset'
            })
            
        analysis['recommendations'] = recommendations
        
        for i, rec in enumerate(recommendations, 1):
            print(f"\n  {i}. {rec['issue']}")
            print(f"     Impact: {rec['impact']}")
            print(f"     Fix: {rec['fix']}")
            
        return analysis
        
    def create_plots(self, analysis):
        """Create comprehensive visualization plots."""
        print("\n6. CREATING VISUALIZATIONS...")
        
        timestamps = np.array([s['timestamp'] for s in self.samples])
        adc_values = np.array([s['adc_value'] for s in self.samples])
        voltages = np.array([s['voltage'] for s in self.samples])
        time_s = (timestamps - timestamps[0]) / 1e6
        
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Overview - first 30 seconds
        ax1 = plt.subplot(4, 2, 1)
        idx_30s = time_s <= 30
        ax1.plot(time_s[idx_30s], voltages[idx_30s], 'b-', linewidth=0.5)
        ax1.set_title('EKG Signal - First 30 seconds', fontsize=14)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Voltage (V)')
        ax1.grid(True, alpha=0.3)
        
        # 2. 5-second detail
        ax2 = plt.subplot(4, 2, 2)
        idx_5s = (time_s >= 60) & (time_s <= 65)
        ax2.plot(time_s[idx_5s], voltages[idx_5s], 'b-', linewidth=1)
        ax2.set_title('5-Second Detail (60-65s)', fontsize=14)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Voltage (V)')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.minorticks_on()
        
        # 3. Power Spectral Density
        ax3 = plt.subplot(4, 2, 3)
        actual_rate = analysis['basic_stats']['sample_rate_actual']
        freqs, psd = welch(adc_values, fs=actual_rate, nperseg=4096)
        ax3.semilogy(freqs[freqs <= 100], psd[freqs <= 100])
        ax3.set_title('Power Spectral Density', fontsize=14)
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Power')
        ax3.grid(True, alpha=0.3)
        ax3.axvline(50, color='r', linestyle='--', alpha=0.5, label='50Hz')
        
        # 4. Spectrogram
        ax4 = plt.subplot(4, 2, 4)
        f, t, Sxx = spectrogram(adc_values[:int(60*actual_rate)], fs=actual_rate, nperseg=512)
        im = ax4.pcolormesh(t, f[f <= 100], 10*np.log10(Sxx[f <= 100]), shading='gouraud', cmap='viridis')
        ax4.set_title('Spectrogram - First 60 seconds', fontsize=14)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Frequency (Hz)')
        plt.colorbar(im, ax=ax4, label='Power (dB)')
        
        # 5. Heart Rate Over Time
        ax5 = plt.subplot(4, 2, 5)
        if 'heart_rate' in analysis and 'detected_beats' in analysis['heart_rate']:
            # Reconstruct HR over time from RR intervals
            b_qrs, a_qrs = butter(4, [5, 30], 'band', fs=actual_rate)
            filtered_qrs = filtfilt(b_qrs, a_qrs, adc_values)
            diff_ecg = np.diff(filtered_qrs)
            squared = diff_ecg ** 2
            window_size = int(0.08 * actual_rate)
            integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
            threshold = np.percentile(integrated, 92)
            peaks, _ = find_peaks(integrated, height=threshold, distance=int(0.3*actual_rate))
            
            if len(peaks) > 10:
                peak_times = time_s[peaks[:-1]]
                rr_intervals = np.diff(peaks) / actual_rate
                hr_values = 60 / rr_intervals
                valid = (hr_values >= 40) & (hr_values <= 180)
                ax5.plot(peak_times[valid], hr_values[valid], 'ro-', markersize=3)
                ax5.set_ylim(40, 120)
        ax5.set_title('Heart Rate Variability', fontsize=14)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Heart Rate (BPM)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Sample Rate Stability
        ax6 = plt.subplot(4, 2, 6)
        time_diffs = np.diff(timestamps) / 1e6
        sample_rates = 1 / time_diffs
        ax6.plot(time_s[1:1000], sample_rates[:999], 'g-', linewidth=0.5)
        ax6.axhline(SAMPLE_RATE, color='r', linestyle='--', label=f'Target: {SAMPLE_RATE} Hz')
        ax6.set_title('Sample Rate Stability', fontsize=14)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Sample Rate (Hz)')
        ax6.set_ylim(SAMPLE_RATE - 50, SAMPLE_RATE + 50)
        ax6.grid(True, alpha=0.3)
        ax6.legend()
        
        # 7. ADC Distribution
        ax7 = plt.subplot(4, 2, 7)
        ax7.hist(adc_values, bins=100, alpha=0.7, edgecolor='black')
        ax7.axvline(32768, color='r', linestyle='--', label='Center (32768)')
        ax7.set_title('ADC Value Distribution', fontsize=14)
        ax7.set_xlabel('ADC Value')
        ax7.set_ylabel('Count')
        ax7.legend()
        
        # 8. Signal Quality Metrics
        ax8 = plt.subplot(4, 2, 8)
        ax8.axis('off')
        metrics_text = f"""Signal Quality Report:
        
DC Offset: {analysis['signal_quality']['dc_offset']:.3f}
Baseline Wander: {analysis['signal_quality']['baseline_wander_std']:.1f} LSB
HF Noise: {analysis['signal_quality']['hf_noise_std']:.1f} LSB
50Hz Power: {analysis['signal_quality']['power_50hz_db']:.1f} dB
SNR Estimate: {analysis['signal_quality']['snr_estimate']:.1f}
Motion Artifacts: {analysis['signal_quality']['motion_artifacts']}

Sample Rate: {analysis['basic_stats']['sample_rate_actual']:.1f} Hz
Missing Samples: {analysis['basic_stats']['missing_samples']}
ADC Saturation: {analysis['basic_stats']['adc_saturation_high']}"""
        
        if 'heart_rate' in analysis and 'mean_hr' in analysis['heart_rate']:
            metrics_text += f"""

Heart Rate: {analysis['heart_rate']['mean_hr']:.1f} ± {analysis['heart_rate']['std_hr']:.1f} BPM
RMSSD: {analysis['heart_rate']['rmssd']:.1f} ms
pNN50: {analysis['heart_rate']['pnn50']:.1f}%"""
        
        ax8.text(0.1, 0.9, metrics_text, transform=ax8.transAxes, 
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('5-Minute EKG Recording Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"ekg_5min_analysis_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"Analysis plots saved to: {plot_filename}")
        
        plt.show()
        
        return plot_filename
        
    def save_report(self, analysis):
        """Save detailed analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON report
        json_filename = f"ekg_5min_report_{timestamp}.json"
        with open(json_filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"JSON report saved to: {json_filename}")
        
        # Text report
        txt_filename = f"ekg_5min_recommendations_{timestamp}.txt"
        with open(txt_filename, 'w') as f:
            f.write("5-MINUTE EKG RECORDING - ANALYSIS REPORT\n")
            f.write("="*60 + "\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {analysis['basic_stats']['duration_s']:.1f} seconds\n")
            f.write(f"Samples: {analysis['basic_stats']['samples']:,}\n")
            f.write("\n")
            
            f.write("SIGNAL QUALITY ISSUES:\n")
            f.write("-"*30 + "\n")
            
            if 'recommendations' in analysis:
                for i, rec in enumerate(analysis['recommendations'], 1):
                    f.write(f"\n{i}. {rec['issue']}\n")
                    f.write(f"   Impact: {rec['impact'].upper()}\n")
                    f.write(f"   Solution: {rec['fix']}\n")
            else:
                f.write("No major issues detected.\n")
                
            f.write("\n\nDETAILED METRICS:\n")
            f.write("-"*30 + "\n")
            f.write(f"Sample Rate: {analysis['basic_stats']['sample_rate_actual']:.1f} Hz (target: {SAMPLE_RATE} Hz)\n")
            f.write(f"DC Offset: {analysis['signal_quality']['dc_offset']:.3f}\n")
            f.write(f"SNR Estimate: {analysis['signal_quality']['snr_estimate']:.1f}\n")
            f.write(f"50Hz Interference: {analysis['signal_quality']['power_50hz_db']:.1f} dB\n")
            
            if 'heart_rate' in analysis and 'mean_hr' in analysis['heart_rate']:
                f.write(f"\nHeart Rate: {analysis['heart_rate']['mean_hr']:.1f} ± {analysis['heart_rate']['std_hr']:.1f} BPM\n")
                f.write(f"Heart Rate Variability (RMSSD): {analysis['heart_rate']['rmssd']:.1f} ms\n")
                
        print(f"Text report saved to: {txt_filename}")
        
        return json_filename, txt_filename
        
    def disconnect(self):
        """Disconnect from device."""
        if self.connection:
            try:
                self.connection.write(b'\x03')
                time.sleep(0.1)
                self.connection.close()
                print("Disconnected")
            except:
                pass

def main():
    print("="*60)
    print("5-MINUTE EKG DATA ACQUISITION AND ANALYSIS")
    print("="*60)
    print("\nThis will:")
    print("1. Record 5 minutes of EKG data")
    print("2. Perform comprehensive signal analysis")
    print("3. Generate detailed plots and reports")
    print("4. Provide specific recommendations for improvement")
    
    input("\nPress Enter when ready (make sure electrodes are attached)...")
    
    acq = EKGAcquisition()
    
    if not acq.connect():
        print("Failed to connect. Make sure device is on COM5")
        return
        
    if not acq.start_streaming():
        print("Failed to start streaming")
        acq.disconnect()
        return
        
    try:
        # Acquire data
        data = acq.acquire_data()
        
        if data:
            # Save raw data
            csv_file = acq.save_raw_data()
            
            # Analyze
            print("\nAnalyzing data...")
            analysis = acq.analyze_data()
            
            if analysis:
                # Create plots
                plot_file = acq.create_plots(analysis)
                
                # Save reports
                json_file, txt_file = acq.save_report(analysis)
                
                print("\n" + "="*60)
                print("ACQUISITION COMPLETE!")
                print("="*60)
                print(f"\nFiles created:")
                print(f"  - Raw data: {csv_file}")
                print(f"  - Analysis plots: {plot_file}")
                print(f"  - JSON report: {json_file}")
                print(f"  - Recommendations: {txt_file}")
                
                if analysis['recommendations']:
                    print(f"\n⚠️  Found {len(analysis['recommendations'])} issues that need attention.")
                    print(f"Check {txt_file} for detailed recommendations.")
                else:
                    print("\n✅ Signal quality looks good!")
                    
    finally:
        acq.disconnect()
        
if __name__ == "__main__":
    main()