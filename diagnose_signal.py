#!/usr/bin/env python3
"""
EKG Signal Diagnostic Tool
Helps identify and fix signal range issues
"""

import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from scipy import signal

class SignalDiagnostic:
    def __init__(self):
        self.port = "COM5"
        self.connection = None
        self.samples = deque(maxlen=25000)  # 100 seconds at 250Hz
        
    def connect(self):
        """Connect to device."""
        try:
            print(f"Connecting to {self.port}...")
            self.connection = serial.Serial(self.port, 115200, timeout=0.1)
            time.sleep(2)
            
            # Start streaming
            self.connection.write(b'\x03\x03')
            time.sleep(0.5)
            self.connection.write(b'\x04')
            time.sleep(2)
            
            # Clear buffer
            while self.connection.in_waiting:
                self.connection.read(self.connection.in_waiting)
                
            self.connection.write(b'stream()\r')
            time.sleep(1)
            
            # Clear startup messages
            start = time.time()
            while time.time() - start < 1:
                if self.connection.in_waiting:
                    self.connection.readline()
                    
            print("Connected and streaming!")
            return True
            
        except Exception as e:
            print(f"Failed to connect: {e}")
            return False
            
    def collect_samples(self, duration=90):
        """Collect samples for diagnosis."""
        print(f"\nCollecting {duration} seconds of data...")
        print("Keep still and breathe normally")
        
        start = time.time()
        error_count = 0
        
        while time.time() - start < duration:
            try:
                if self.connection.in_waiting:
                    line = self.connection.readline().decode('utf-8', errors='ignore').strip()
                    
                    if not line or any(skip in line for skip in ['>>>', 'INFO:', 'FORMAT:']):
                        continue
                        
                    parts = line.split(',')
                    if len(parts) >= 3:
                        adc_value = int(parts[1])
                        self.samples.append(adc_value)
                        
                        # Progress
                        elapsed = time.time() - start
                        progress = int(elapsed / duration * 20)
                        print(f"\rProgress: {'█' * progress}{'░' * (20-progress)} {elapsed:.1f}s", end='')
                        
            except:
                error_count += 1
                
        print(f"\n\nCollected {len(self.samples)} samples")
        return list(self.samples)
        
    def save_raw_data(self, samples):
        """Save raw diagnostic data to CSV."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"diagnostic_raw_{timestamp}.csv"
        
        with open(filename, 'w') as f:
            f.write("index,adc_value,voltage,time_s\n")
            for i, adc_val in enumerate(samples):
                voltage = adc_val * 3.3 / 65535
                time_s = i / 250.0  # Assuming 250 Hz
                f.write(f"{i},{adc_val},{voltage:.6f},{time_s:.3f}\n")
                
        print(f"\nRaw data saved to: {filename}")
        return filename
        
    def analyze_signal(self, samples):
        """Comprehensive signal analysis."""
        if len(samples) < 100:
            print("Not enough samples!")
            return None
            
        # Save raw data first
        self.save_raw_data(samples)
            
        samples_arr = np.array(samples)
        
        print("\n=== SIGNAL DIAGNOSTICS ===\n")
        
        # 1. Basic statistics
        print("1. RAW SIGNAL STATISTICS:")
        print(f"   Samples: {len(samples)}")
        print(f"   Mean: {np.mean(samples_arr):.0f}")
        print(f"   Min: {np.min(samples_arr)} ({np.min(samples_arr)/65535*100:.1f}%)")
        print(f"   Max: {np.max(samples_arr)} ({np.max(samples_arr)/65535*100:.1f}%)")
        print(f"   Range: {np.ptp(samples_arr)} ({np.ptp(samples_arr)/65535*100:.1f}% of ADC)")
        
        # Check for saturation
        saturated_high = np.sum(samples_arr > 64000)
        saturated_low = np.sum(samples_arr < 1500)
        
        if saturated_high > 0 or saturated_low > 0:
            print(f"\n   ⚠️  SATURATION DETECTED!")
            print(f"   High saturation: {saturated_high} samples")
            print(f"   Low saturation: {saturated_low} samples")
        else:
            print(f"\n   ✅ No saturation detected")
            
        # 2. Voltage conversion
        print("\n2. VOLTAGE ANALYSIS:")
        voltages = samples_arr * 3.3 / 65535
        print(f"   Mean voltage: {np.mean(voltages):.3f} V")
        print(f"   Voltage range: {np.min(voltages):.3f} - {np.max(voltages):.3f} V")
        print(f"   Peak-to-peak: {np.ptp(voltages)*1000:.1f} mV")
        
        # 3. Scaling to proper EKG range
        print("\n3. SCALED TO EKG RANGE:")
        
        # Remove DC offset
        centered = samples_arr - np.mean(samples_arr)
        
        # Current peak-to-peak in ADC units
        current_pp = np.ptp(centered)
        
        # Target: 2mV peak-to-peak (typical QRS)
        # 2mV in ADC units (assuming 3.3V reference)
        target_pp = (2e-3 / 3.3) * 65535  # ~40 ADC units
        
        scale_factor = target_pp / current_pp if current_pp > 0 else 1
        
        print(f"   Current P-P: {current_pp} ADC units ({current_pp/65535*3300:.1f} mV)")
        print(f"   Target P-P: {target_pp:.0f} ADC units (2 mV)")
        print(f"   Required scale factor: {scale_factor:.6f}")
        print(f"   Reduction needed: {1/scale_factor:.0f}x")
        
        # Apply scaling
        scaled = centered * scale_factor + 32768
        scaled_mv = (scaled - 32768) / 65535 * 3300  # Convert to mV
        
        # 4. Frequency content
        print("\n4. FREQUENCY ANALYSIS:")
        # Simple peak detection
        peaks_idx = []
        for i in range(20, len(centered)-20):
            if centered[i] > centered[i-1] and centered[i] > centered[i+1]:
                if centered[i] > np.percentile(centered, 90):
                    if not peaks_idx or i - peaks_idx[-1] > 50:  # Min 200ms between peaks
                        peaks_idx.append(i)
                        
        if len(peaks_idx) > 2:
            intervals = np.diff(peaks_idx) / 250  # Convert to seconds
            heart_rate = 60 / np.mean(intervals)
            print(f"   Detected peaks: {len(peaks_idx)}")
            print(f"   Estimated HR: {heart_rate:.0f} BPM")
        else:
            print(f"   Could not detect clear peaks")
            
        # 5. Recommendations
        print("\n5. RECOMMENDATIONS:")
        
        if scale_factor < 0.01:
            print(f"\n   🔴 CRITICAL: Signal is {1/scale_factor:.0f}x too large!")
            print(f"   This explains why you can't see EKG features.")
            print(f"\n   Solutions:")
            print(f"   1. Hardware: Add voltage divider (1:{int(1/scale_factor)} ratio)")
            print(f"   2. Software: Scale by factor {scale_factor:.6f}")
            print(f"   3. AD8232: Check if gain is set too high")
        elif scale_factor < 0.1:
            print(f"\n   🟡 WARNING: Signal is {1/scale_factor:.0f}x too large")
            print(f"   EKG features may be clipped.")
        else:
            print(f"\n   🟢 Signal amplitude is reasonable")
            
        # 6. Create plots
        self.create_diagnostic_plots(samples_arr, centered, scaled, scaled_mv)
        
        return {
            'scale_factor': scale_factor,
            'mean_adc': np.mean(samples_arr),
            'range_adc': np.ptp(samples_arr),
            'saturation': saturated_high + saturated_low > 0
        }
        
    def create_diagnostic_plots(self, raw, centered, scaled, scaled_mv):
        """Create diagnostic visualization."""
        fig, axes = plt.subplots(4, 1, figsize=(12, 10))
        time_axis = np.arange(len(raw)) / 250  # Time in seconds
        
        # 1. Raw signal
        axes[0].plot(time_axis, raw, 'b-', linewidth=0.5)
        axes[0].set_title('Raw ADC Values')
        axes[0].set_ylabel('ADC Units')
        axes[0].axhline(32768, color='r', linestyle='--', alpha=0.5, label='Center')
        axes[0].axhline(65535, color='k', linestyle='-', alpha=0.3, label='Max')
        axes[0].axhline(0, color='k', linestyle='-', alpha=0.3, label='Min')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Centered signal
        axes[1].plot(time_axis, centered, 'g-', linewidth=0.5)
        axes[1].set_title('DC Offset Removed')
        axes[1].set_ylabel('ADC Units')
        axes[1].axhline(0, color='k', linestyle='-', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        
        # 3. Scaled to proper range
        axes[2].plot(time_axis, scaled_mv, 'r-', linewidth=0.5)
        axes[2].set_title('Scaled to Proper EKG Range')
        axes[2].set_ylabel('mV')
        axes[2].set_ylim(-3, 3)  # Typical EKG range
        axes[2].axhline(0, color='k', linestyle='-', alpha=0.5)
        axes[2].grid(True, alpha=0.3)
        
        # 4. Zoomed view (2 seconds)
        zoom_idx = (time_axis >= 2) & (time_axis <= 4)
        axes[3].plot(time_axis[zoom_idx], scaled_mv[zoom_idx], 'r-', linewidth=1)
        axes[3].set_title('Zoomed View (2s) - Properly Scaled')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('mV')
        axes[3].set_ylim(-3, 3)
        axes[3].grid(True, which='both', alpha=0.3)
        axes[3].minorticks_on()
        
        plt.suptitle('EKG Signal Diagnostic Analysis', fontsize=14)
        plt.tight_layout()
        
        # Save
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"signal_diagnostic_{timestamp}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n   Diagnostic plots saved to: {filename}")
        
        plt.show()
        
    def suggest_firmware_fix(self, analysis):
        """Generate firmware fix code."""
        if not analysis:
            return
            
        print("\n\n=== SUGGESTED FIRMWARE FIX ===")
        
        scale = analysis['scale_factor']
        offset = analysis['mean_adc']
        
        print(f"""
Add this to your firmware's _process_sample() function:

```python
def _process_sample(self, raw_value):
    # Your current code...
    
    # ADD THIS SCALING FIX:
    # Remove DC offset
    centered = raw_value - {int(offset)}
    
    # Scale to proper EKG range
    scaled = int(centered * {scale:.6f})
    
    # Re-center and ensure valid range
    final_value = scaled + 32768
    final_value = max(0, min(65535, final_value))
    
    # Continue with your filtering...
```

This will reduce your signal by {1/scale:.0f}x to proper EKG range.
""")
        
    def run(self):
        """Run full diagnostic."""
        if not self.connect():
            return
            
        try:
            # Collect data
            samples = self.collect_samples(90)
            
            if samples:
                # Analyze
                analysis = self.analyze_signal(samples)
                
                # Suggest fix
                self.suggest_firmware_fix(analysis)
                
        finally:
            if self.connection:
                self.connection.close()
                

def main():
    print("="*50)
    print("EKG SIGNAL DIAGNOSTIC TOOL")
    print("="*50)
    print("\nThis will help identify why your EKG doesn't look right")
    print("and suggest specific fixes.\n")
    
    diag = SignalDiagnostic()
    diag.run()
    

if __name__ == "__main__":
    main()