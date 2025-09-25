#!/usr/bin/env python3
"""
EKG Data Acquisition Script
Collects EKG data for analysis and quality assessment
"""

import sys
import time
import json
import serial
import serial.tools.list_ports
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import find_peaks, butter, filtfilt, welch
from collections import deque

# Constants
ACQUISITION_TIME = 120  # 2 minutes
SAMPLE_RATE = 250  # Hz
BAUDRATE = 115200

class DataAcquisition:
    def __init__(self, port=None):
        self.port = port or self._find_device()
        self.connection = None
        self.samples = []
        self.start_time = None
        
    def _find_device(self):
        """Auto-detect RP2040 device."""
        print("Searching for RP2040...")
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            if any(id in port.description.upper() for id in ['RP2040', 'PICO', 'CDC']):
                print(f"Found device: {port.device} - {port.description}")
                return port.device
                
        if ports:
            print("Available ports:")
            for port in ports:
                print(f"  {port.device} - {port.description}")
        else:
            print("No serial ports found")
            
        return None
        
    def connect(self):
        """Connect to device."""
        if not self.port:
            print("No port specified")
            return False
            
        try:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=BAUDRATE,
                timeout=1.0,
                write_timeout=1.0
            )
            time.sleep(2)
            self.connection.reset_input_buffer()
            self.connection.reset_output_buffer()
            print(f"Connected to {self.port}")
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
            self.connection.write(b'\x03')  # Ctrl+C
            time.sleep(0.5)
            self.connection.write(b'\x04')  # Ctrl+D
            time.sleep(2)
            
            self.connection.reset_input_buffer()
            
            self.connection.write(b'init()\r')
            time.sleep(1)
            self.connection.write(b'start()\r')
            time.sleep(1)
            self.connection.write(b'stream()\r')
            time.sleep(1)
            
            self.connection.reset_input_buffer()
            print("Streaming started")
            return True
        except Exception as e:
            print(f"Failed to start streaming: {e}")
            return False
            
    def acquire_data(self):
        """Acquire data for specified duration."""
        print(f"\nStarting {ACQUISITION_TIME} second acquisition...")
        print("Progress: ", end="", flush=True)
        
        self.start_time = time.time()
        self.samples = []
        error_count = 0
        last_progress = 0
        
        while time.time() - self.start_time < ACQUISITION_TIME:
            try:
                if self.connection.in_waiting:
                    line = self.connection.readline().decode('utf-8').strip()
                    
                    # Skip non-data lines
                    if not line or any(skip in line for skip in ['===', 'FORMAT:', 'Streaming', '>>>']):
                        continue
                        
                    parts = line.split(',')
                    if len(parts) >= 3:
                        sample = {
                            'timestamp': int(parts[0]),
                            'adc_value': int(parts[1]),
                            'heart_rate': int(parts[2]),
                            'voltage': (int(parts[1]) / 65535) * 3.3
                        }
                        self.samples.append(sample)
                        
                        # Show progress
                        progress = int((time.time() - self.start_time) / ACQUISITION_TIME * 20)
                        if progress > last_progress:
                            print("█", end="", flush=True)
                            last_progress = progress
                            
            except (ValueError, UnicodeDecodeError):
                error_count += 1
                if error_count % 100 == 0:
                    print(f"\nParse errors: {error_count}")
                    
        print(" Done!")
        print(f"\nAcquired {len(self.samples)} samples in {time.time() - self.start_time:.1f} seconds")
        return self.samples
        
    def analyze_data(self):
        """Analyze acquired data for quality and characteristics."""
        if not self.samples:
            print("No data to analyze")
            return None
            
        print("\n=== DATA ANALYSIS ===")
        
        # Extract arrays
        timestamps = np.array([s['timestamp'] for s in self.samples])
        adc_values = np.array([s['adc_value'] for s in self.samples])
        voltages = np.array([s['voltage'] for s in self.samples])
        
        # Calculate actual sample rate
        time_diffs = np.diff(timestamps) / 1e6  # Convert to seconds
        actual_rate = 1 / np.median(time_diffs)
        
        # Basic statistics
        stats = {
            'samples': len(self.samples),
            'duration': (timestamps[-1] - timestamps[0]) / 1e6,
            'sample_rate': {
                'target': SAMPLE_RATE,
                'actual': actual_rate,
                'std': np.std(1/time_diffs),
                'min': np.min(1/time_diffs),
                'max': np.max(1/time_diffs)
            },
            'voltage': {
                'mean': np.mean(voltages),
                'std': np.std(voltages),
                'min': np.min(voltages),
                'max': np.max(voltages),
                'range': np.ptp(voltages)
            },
            'adc': {
                'mean': np.mean(adc_values),
                'std': np.std(adc_values),
                'min': np.min(adc_values),
                'max': np.max(adc_values),
                'saturation_count': np.sum((adc_values < 100) | (adc_values > 65435))
            }
        }
        
        # Signal quality analysis
        # 1. DC offset
        dc_offset = np.mean(adc_values) / 65535
        
        # 2. Noise analysis - high-frequency noise
        b_hp, a_hp = butter(4, 30, 'high', fs=actual_rate)
        high_freq = filtfilt(b_hp, a_hp, adc_values)
        noise_rms = np.sqrt(np.mean(high_freq**2))
        
        # 3. Power spectral density
        freqs, psd = welch(adc_values, fs=actual_rate, nperseg=1024)
        
        # Find 50Hz interference
        idx_50hz = np.argmin(np.abs(freqs - 50))
        power_50hz = psd[idx_50hz]
        
        # 4. Baseline wander - very low frequency
        b_lp, a_lp = butter(4, 0.5, 'low', fs=actual_rate)
        baseline = filtfilt(b_lp, a_lp, adc_values)
        baseline_var = np.std(baseline)
        
        # 5. QRS detection quality
        # Bandpass filter for QRS
        b_bp, a_bp = butter(4, [5, 30], 'band', fs=actual_rate)
        filtered = filtfilt(b_bp, a_bp, adc_values)
        
        # Derivative-based QRS detection
        diff_ecg = np.diff(filtered)
        squared = diff_ecg ** 2
        window_size = int(0.08 * actual_rate)  # 80ms window
        integrated = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
        
        threshold = np.percentile(integrated, 90)
        peaks, properties = find_peaks(integrated, height=threshold, distance=int(0.3*actual_rate))
        
        # Heart rate variability
        if len(peaks) > 2:
            rr_intervals = np.diff(peaks) / actual_rate * 1000  # ms
            hr_values = 60000 / rr_intervals  # BPM
            hrv_metrics = {
                'mean_hr': np.mean(hr_values),
                'std_hr': np.std(hr_values),
                'rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2)),
                'detected_beats': len(peaks)
            }
        else:
            hrv_metrics = {'detected_beats': len(peaks)}
            
        quality_metrics = {
            'dc_offset': dc_offset,
            'noise_rms': noise_rms,
            'power_50hz': power_50hz,
            'baseline_variance': baseline_var,
            'snr_estimate': np.std(filtered) / noise_rms if noise_rms > 0 else 0,
            'hrv': hrv_metrics
        }
        
        # Print analysis
        print(f"\nSample Rate:")
        print(f"  Target: {stats['sample_rate']['target']} Hz")
        print(f"  Actual: {stats['sample_rate']['actual']:.2f} Hz")
        print(f"  Jitter: {stats['sample_rate']['std']:.2f} Hz")
        
        print(f"\nSignal Characteristics:")
        print(f"  DC Offset: {quality_metrics['dc_offset']:.3f} ({quality_metrics['dc_offset']*100:.1f}%)")
        print(f"  Voltage Range: {stats['voltage']['min']:.3f} - {stats['voltage']['max']:.3f} V")
        print(f"  Signal Amplitude: {stats['voltage']['range']:.3f} V")
        print(f"  ADC Saturation: {stats['adc']['saturation_count']} samples")
        
        print(f"\nNoise Analysis:")
        print(f"  High-freq noise RMS: {quality_metrics['noise_rms']:.1f} LSB")
        print(f"  50Hz Power: {10*np.log10(quality_metrics['power_50hz']):.1f} dB")
        print(f"  Baseline Variance: {quality_metrics['baseline_variance']:.1f} LSB")
        print(f"  SNR Estimate: {quality_metrics['snr_estimate']:.1f}")
        
        if 'mean_hr' in hrv_metrics:
            print(f"\nHeart Rate:")
            print(f"  Detected Beats: {hrv_metrics['detected_beats']}")
            print(f"  Mean HR: {hrv_metrics['mean_hr']:.1f} BPM")
            print(f"  HR Variability: {hrv_metrics['std_hr']:.1f} BPM")
            print(f"  RMSSD: {hrv_metrics['rmssd']:.1f} ms")
        
        # Save detailed report
        report = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'quality_metrics': quality_metrics,
            'recommendations': self._generate_recommendations(stats, quality_metrics)
        }
        
        return report
        
    def _generate_recommendations(self, stats, quality):
        """Generate improvement recommendations."""
        recommendations = []
        
        # Sample rate issues
        rate_error = abs(stats['sample_rate']['actual'] - stats['sample_rate']['target'])
        if rate_error > 5:
            recommendations.append({
                'issue': 'Sample rate deviation',
                'severity': 'high',
                'target': 'firmware',
                'suggestion': 'Check timer configuration and interrupt priorities'
            })
            
        if stats['sample_rate']['std'] > 10:
            recommendations.append({
                'issue': 'High sample rate jitter',
                'severity': 'medium',
                'target': 'firmware',
                'suggestion': 'Use DMA for ADC reading, check for blocking operations'
            })
            
        # Signal quality issues
        if quality['dc_offset'] < 0.4 or quality['dc_offset'] > 0.6:
            recommendations.append({
                'issue': 'DC offset not centered',
                'severity': 'medium',
                'target': 'hardware',
                'suggestion': 'Adjust bias voltage in analog frontend'
            })
            
        if quality['noise_rms'] > 100:
            recommendations.append({
                'issue': 'High noise level',
                'severity': 'high',
                'target': 'hardware',
                'suggestions': [
                    'Add better power supply filtering',
                    'Improve PCB layout and grounding',
                    'Add shielding to analog section'
                ]
            })
            
        if 10*np.log10(quality['power_50hz']) > -20:
            recommendations.append({
                'issue': 'Strong 50Hz interference',
                'severity': 'high',
                'target': 'both',
                'suggestions': [
                    'Hardware: Add notch filter or improve shielding',
                    'Software: Implement adaptive notch filter'
                ]
            })
            
        if quality['baseline_variance'] > 500:
            recommendations.append({
                'issue': 'Baseline wander',
                'severity': 'medium',
                'target': 'both',
                'suggestions': [
                    'Hardware: Check electrode contact and cables',
                    'Software: Add high-pass filter (0.5Hz cutoff)'
                ]
            })
            
        if stats['adc']['saturation_count'] > 10:
            recommendations.append({
                'issue': 'ADC saturation detected',
                'severity': 'high',
                'target': 'hardware',
                'suggestion': 'Reduce gain or adjust input range'
            })
            
        if quality['snr_estimate'] < 10:
            recommendations.append({
                'issue': 'Low signal-to-noise ratio',
                'severity': 'high',
                'target': 'hardware',
                'suggestion': 'Review entire analog signal path'
            })
            
        return recommendations
        
    def plot_analysis(self):
        """Create analysis plots."""
        if not self.samples:
            return
            
        timestamps = np.array([s['timestamp'] for s in self.samples])
        adc_values = np.array([s['adc_value'] for s in self.samples])
        voltages = np.array([s['voltage'] for s in self.samples])
        
        # Time axis in seconds
        time_s = (timestamps - timestamps[0]) / 1e6
        
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle('EKG Signal Analysis', fontsize=16)
        
        # 1. Raw signal (first 10 seconds)
        idx_10s = np.where(time_s <= 10)[0]
        axes[0, 0].plot(time_s[idx_10s], voltages[idx_10s], 'b-', linewidth=0.5)
        axes[0, 0].set_title('Raw Signal (First 10s)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Voltage (V)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Full signal
        axes[0, 1].plot(time_s[::10], voltages[::10], 'b-', linewidth=0.5)
        axes[0, 1].set_title('Full Recording')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Voltage (V)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Power spectral density
        actual_rate = len(self.samples) / (time_s[-1] - time_s[0])
        freqs, psd = welch(adc_values, fs=actual_rate, nperseg=2048)
        axes[1, 0].semilogy(freqs, psd)
        axes[1, 0].set_title('Power Spectral Density')
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Power')
        axes[1, 0].set_xlim(0, 100)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axvline(50, color='r', linestyle='--', alpha=0.5, label='50Hz')
        
        # 4. Histogram
        axes[1, 1].hist(adc_values, bins=100, alpha=0.7)
        axes[1, 1].set_title('ADC Value Distribution')
        axes[1, 1].set_xlabel('ADC Value')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].axvline(32768, color='r', linestyle='--', alpha=0.5, label='Center')
        
        # 5. Sample rate variability
        time_diffs = np.diff(timestamps) / 1e6
        sample_rates = 1 / time_diffs
        axes[2, 0].plot(time_s[1:], sample_rates, 'g-', linewidth=0.5)
        axes[2, 0].set_title('Instantaneous Sample Rate')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Sample Rate (Hz)')
        axes[2, 0].axhline(SAMPLE_RATE, color='r', linestyle='--', alpha=0.5)
        axes[2, 0].grid(True, alpha=0.3)
        
        # 6. Filtered signal for QRS
        b_bp, a_bp = butter(4, [5, 30], 'band', fs=actual_rate)
        filtered = filtfilt(b_bp, a_bp, adc_values)
        idx_5s = np.where(time_s <= 5)[0]
        axes[2, 1].plot(time_s[idx_5s], filtered[idx_5s], 'r-', linewidth=0.5)
        axes[2, 1].set_title('Bandpass Filtered (5-30Hz)')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Amplitude')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 7. Baseline wander
        b_lp, a_lp = butter(4, 0.5, 'low', fs=actual_rate)
        baseline = filtfilt(b_lp, a_lp, adc_values)
        axes[3, 0].plot(time_s[::10], baseline[::10], 'orange', linewidth=1)
        axes[3, 0].set_title('Baseline Wander (<0.5Hz)')
        axes[3, 0].set_xlabel('Time (s)')
        axes[3, 0].set_ylabel('ADC Value')
        axes[3, 0].grid(True, alpha=0.3)
        
        # 8. High-frequency noise
        b_hp, a_hp = butter(4, 30, 'high', fs=actual_rate)
        high_freq = filtfilt(b_hp, a_hp, adc_values)
        axes[3, 1].plot(time_s[idx_10s], high_freq[idx_10s], 'purple', linewidth=0.5)
        axes[3, 1].set_title('High-Frequency Noise (>30Hz)')
        axes[3, 1].set_xlabel('Time (s)')
        axes[3, 1].set_ylabel('Amplitude')
        axes[3, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ekg_analysis_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nAnalysis plots saved to: {filename}")
        
        plt.show()
        
    def save_data(self):
        """Save raw data."""
        if not self.samples:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ekg_acquisition_{timestamp}.csv"
        
        with open(filename, 'w') as f:
            f.write("timestamp_us,adc_value,voltage,heart_rate\n")
            for s in self.samples:
                f.write(f"{s['timestamp']},{s['adc_value']},{s['voltage']:.4f},{s['heart_rate']}\n")
                
        print(f"Raw data saved to: {filename}")
        
    def save_report(self, report):
        """Save analysis report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ekg_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        print(f"Analysis report saved to: {filename}")
        
        # Also save recommendations in readable format
        if report['recommendations']:
            rec_file = f"recommendations_{timestamp}.txt"
            with open(rec_file, 'w') as f:
                f.write("EKG SYSTEM IMPROVEMENT RECOMMENDATIONS\n")
                f.write("="*50 + "\n\n")
                
                for i, rec in enumerate(report['recommendations'], 1):
                    f.write(f"{i}. {rec['issue']} (Severity: {rec['severity']})\n")
                    f.write(f"   Target: {rec['target']}\n")
                    
                    if 'suggestion' in rec:
                        f.write(f"   Suggestion: {rec['suggestion']}\n")
                    elif 'suggestions' in rec:
                        f.write("   Suggestions:\n")
                        for s in rec['suggestions']:
                            f.write(f"     - {s}\n")
                    f.write("\n")
                    
            print(f"Recommendations saved to: {rec_file}")
            
    def disconnect(self):
        """Disconnect from device."""
        if self.connection:
            try:
                self.connection.write(b'\x03')
                time.sleep(0.1)
                self.connection.close()
            except:
                pass
                
def main():
    print("EKG Data Acquisition and Analysis")
    print("="*50)
    
    acq = DataAcquisition()
    
    if not acq.connect():
        port = input("Enter serial port manually: ")
        if port:
            acq.port = port
            if not acq.connect():
                return
        else:
            return
            
    if not acq.start_streaming():
        print("Failed to start streaming")
        acq.disconnect()
        return
        
    try:
        # Acquire data
        acq.acquire_data()
        
        # Save raw data
        acq.save_data()
        
        # Analyze
        report = acq.analyze_data()
        
        # Save report
        if report:
            acq.save_report(report)
            
            # Show recommendations
            if report['recommendations']:
                print("\n=== RECOMMENDATIONS ===")
                for rec in report['recommendations']:
                    print(f"\n• {rec['issue']} ({rec['severity']})")
                    print(f"  Target: {rec['target']}")
                    if 'suggestion' in rec:
                        print(f"  → {rec['suggestion']}")
                    elif 'suggestions' in rec:
                        for s in rec['suggestions']:
                            print(f"  → {s}")
                            
        # Plot analysis
        acq.plot_analysis()
        
    finally:
        acq.disconnect()
        
if __name__ == "__main__":
    main()