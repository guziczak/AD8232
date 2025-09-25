"""
EKG Data Acquisition for RP2040-Zero + AD8232 - Improved Version
With oversampling, buffered streaming, and better timing

Hardware connections:
- GND    → GND (pin 2)
- 3.3V   → 3V3 (pin 3)
- OUTPUT → GP29/ADC3 (pin 4)
- LO-    → GP28 (pin 5)
- LO+    → GP27 (pin 6)
"""

import machine
import time
import array
import gc
import json
import sys
import _thread

# Constants
SAMPLE_RATE = 250  # Hz - output sample rate
OVERSAMPLE_FACTOR = 16  # 16x oversampling for +2 bits of resolution
ADC_SAMPLE_RATE = SAMPLE_RATE * OVERSAMPLE_FACTOR  # 4000 Hz internal
BUFFER_SIZE = 2000  # samples in main buffer
STREAM_BUFFER_SIZE = 50  # samples in streaming buffer
ADC_BITS = 12  # RP2040 has 12-bit ADC
ADC_MAX = (1 << ADC_BITS) - 1  # 4095
ADC_SCALE = 16  # MicroPython scales 12-bit to 16-bit
VOLTAGE_REF = 3.3

# Algorithm constants
HR_DERIVATIVE_WINDOW_MS = 80
HR_MIN_RR_INTERVAL_MS = 300
HR_THRESHOLD_PERCENTILE = 90
HR_MIN_BPM = 30
HR_MAX_BPM = 200

# Filter constants
NOTCH_FREQUENCY = 50.0
NOTCH_QUALITY_FACTOR = 30.0
LOWPASS_ALPHA = 0.85  # Slightly less filtering due to oversampling
BASELINE_ALPHA = 0.01

# Pin configuration
PIN_ADC = 29
PIN_LO_PLUS = 27
PIN_LO_MINUS = 28
PIN_LED = 16

# File settings
DEFAULT_FILENAME = "ekg_data.csv"
CONFIG_FILE = "config.json"


class StreamBuffer:
    """Thread-safe circular buffer for streaming."""
    
    def __init__(self, size):
        self.size = size
        self.buffer = array.array('H', [0] * size)
        self.timestamps = array.array('L', [0] * size)
        self.write_idx = 0
        self.read_idx = 0
        self.count = 0
        self.lock = _thread.allocate_lock()
        
    def write(self, value, timestamp):
        """Write sample to buffer."""
        with self.lock:
            self.buffer[self.write_idx] = value
            self.timestamps[self.write_idx] = timestamp
            self.write_idx = (self.write_idx + 1) % self.size
            if self.count < self.size:
                self.count += 1
            else:
                # Buffer full, advance read pointer
                self.read_idx = (self.read_idx + 1) % self.size
                
    def read_batch(self, max_count=10):
        """Read up to max_count samples."""
        samples = []
        with self.lock:
            while self.count > 0 and len(samples) < max_count:
                samples.append({
                    'value': self.buffer[self.read_idx],
                    'timestamp': self.timestamps[self.read_idx]
                })
                self.read_idx = (self.read_idx + 1) % self.size
                self.count -= 1
        return samples
        
    def available(self):
        """Get number of available samples."""
        with self.lock:
            return self.count


class EKGSensor:
    """Main class for EKG data acquisition with oversampling."""

    def __init__(self):
        """Initialize hardware and buffers."""
        # Hardware setup
        self.adc = self._init_adc()
        self.lo_plus = machine.Pin(PIN_LO_PLUS, machine.Pin.IN)
        self.lo_minus = machine.Pin(PIN_LO_MINUS, machine.Pin.IN)
        self.led = self._init_led()

        # Data buffers
        self.samples = array.array('H', [0] * BUFFER_SIZE)
        self.timestamps = array.array('L', [0] * BUFFER_SIZE)
        self.write_index = 0
        self.sample_count = 0
        self.last_sample_time = 0
        
        # Streaming buffer
        self.stream_buffer = StreamBuffer(STREAM_BUFFER_SIZE)
        self.streaming = False
        
        # Oversampling accumulator
        self.oversample_acc = 0
        self.oversample_count = 0
        
        # Filtering
        self.last_value = 0
        self.alpha = LOWPASS_ALPHA
        
        # Baseline tracking
        self.baseline = 32768
        self.baseline_alpha = BASELINE_ALPHA
        
        # Signal statistics for auto-gain
        self.signal_min = 65535
        self.signal_max = 0
        self.auto_gain = 1.0
        self.gain_update_counter = 0
        
        # Notch filter state
        self.notch_x1 = 0
        self.notch_x2 = 0
        self.notch_y1 = 0
        self.notch_y2 = 0
        import math
        self.notch_coeff = 2 * math.cos(2 * math.pi * NOTCH_FREQUENCY / SAMPLE_RATE)
        self.notch_gain = 0.9

        # State
        self.sampling_active = False
        self.timer = machine.Timer()
        
        # Performance monitoring
        self.missed_samples = 0
        self.total_samples = 0

        # Load configuration
        self.config = self._load_config()

    def _init_adc(self):
        """Initialize ADC with fallback options."""
        try:
            return machine.ADC(PIN_ADC)
        except:
            try:
                return machine.ADC(machine.Pin(PIN_ADC))
            except:
                return machine.ADC(3)

    def _init_led(self):
        """Initialize LED with error handling."""
        try:
            return machine.Pin(PIN_LED, machine.Pin.OUT)
        except:
            print("Warning: LED not available")
            return None

    def _load_config(self):
        """Load configuration from file or use defaults."""
        default_config = {
            "sample_rate": SAMPLE_RATE,
            "buffer_size": BUFFER_SIZE,
            "auto_save": False,
            "debug_mode": False,
            "oversample_factor": OVERSAMPLE_FACTOR
        }

        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        except:
            return default_config

    def check_electrodes(self):
        """Check electrode connection status."""
        lo_plus_disconnected = self.lo_plus.value() == 1
        lo_minus_disconnected = self.lo_minus.value() == 1

        return {
            "connected": not (lo_plus_disconnected or lo_minus_disconnected),
            "lo_plus": not lo_plus_disconnected,
            "lo_minus": not lo_minus_disconnected
        }

    def _high_speed_sample_callback(self, timer):
        """High-speed timer callback for oversampling."""
        try:
            # Read ADC
            if self.lo_plus.value() or self.lo_minus.value():
                raw_value = 32768  # Center value for disconnected
            else:
                raw_value = self.adc.read_u16()
            
            # Accumulate for oversampling
            self.oversample_acc += raw_value
            self.oversample_count += 1
            
            # When we have enough samples, process them
            if self.oversample_count >= OVERSAMPLE_FACTOR:
                # Average the oversampled values
                averaged_value = self.oversample_acc // OVERSAMPLE_FACTOR
                
                # Reset accumulator
                self.oversample_acc = 0
                self.oversample_count = 0
                
                # Process the averaged sample
                self._process_sample(averaged_value)
                
        except Exception as e:
            if self.config.get("debug_mode"):
                print(f"Sample error: {e}")
            self.missed_samples += 1

    def _process_sample(self, raw_value):
        """Process a single averaged sample."""
        # Update signal range for auto-gain
        if raw_value < self.signal_min:
            self.signal_min = raw_value
        if raw_value > self.signal_max:
            self.signal_max = raw_value
        
        # Update auto-gain every 250 samples
        self.gain_update_counter += 1
        if self.gain_update_counter >= 250:
            signal_range = self.signal_max - self.signal_min
            if signal_range > 100 and signal_range < 10000:
                # Target range is about 30000 (45% of full scale)
                self.auto_gain = min(30000.0 / signal_range, 4.0)  # Max 4x gain
            self.gain_update_counter = 0
            self.signal_min = 65535
            self.signal_max = 0
        
        # Remove DC offset by tracking baseline
        self.baseline = self.baseline_alpha * raw_value + (1 - self.baseline_alpha) * self.baseline
        centered_value = raw_value - self.baseline
        
        # Apply auto-gain and re-center
        amplified = int(centered_value * self.auto_gain + 32768)
        
        # Apply notch filter for 50Hz
        notch_out = amplified - self.notch_coeff * self.notch_x1 + self.notch_x2 + \
                   self.notch_gain * self.notch_coeff * self.notch_y1 - 0.81 * self.notch_y2
        
        # Update notch filter history
        self.notch_x2 = self.notch_x1
        self.notch_x1 = amplified
        self.notch_y2 = self.notch_y1
        self.notch_y1 = notch_out
        
        # Apply low-pass filter
        value = int(self.alpha * notch_out + (1 - self.alpha) * self.last_value)
        self.last_value = value
        
        # Ensure we stay in valid range
        value = min(max(value, 0), 65535)

        # Store sample with timestamp
        current_time = time.ticks_us()
        if current_time < self.last_sample_time:
            current_time += 0x100000000
        self.last_sample_time = current_time & 0xFFFFFFFF
        
        # Store in main buffer
        self.samples[self.write_index] = value
        self.timestamps[self.write_index] = current_time
        self.write_index = (self.write_index + 1) % BUFFER_SIZE
        self.sample_count += 1
        self.total_samples += 1

        # Also store in streaming buffer if streaming
        if self.streaming:
            self.stream_buffer.write(value, current_time)

        # Blink LED every 50 samples
        if self.led and self.sample_count % 50 == 0:
            self.led.toggle()

    def start_sampling(self):
        """Start continuous sampling with oversampling."""
        if self.sampling_active:
            print("Already sampling")
            return

        # Reset state
        self.write_index = 0
        self.sample_count = 0
        self.total_samples = 0
        self.missed_samples = 0
        self.sampling_active = True
        self.gain_update_counter = 0
        self.signal_min = 65535
        self.signal_max = 0
        self.oversample_acc = 0
        self.oversample_count = 0

        # Start high-speed timer for oversampling
        self.timer.init(
            freq=ADC_SAMPLE_RATE,
            mode=machine.Timer.PERIODIC,
            callback=self._high_speed_sample_callback
        )

        print(f"Sampling started at {SAMPLE_RATE} Hz with {OVERSAMPLE_FACTOR}x oversampling")
        print(f"Effective ADC rate: {ADC_SAMPLE_RATE} Hz")
        print(f"Auto-gain enabled, baseline tracking active")

    def stop_sampling(self):
        """Stop sampling."""
        if not self.sampling_active:
            return

        self.timer.deinit()
        self.sampling_active = False
        self.streaming = False

        if self.led:
            self.led.off()

        print(f"Sampling stopped. Collected {self.sample_count} samples")
        print(f"Missed samples: {self.missed_samples}")
        print(f"Final auto-gain: {self.auto_gain:.2f}x")

    def get_latest_samples(self, count=100):
        """Get latest samples from circular buffer."""
        if self.sample_count == 0:
            return []
            
        if count > BUFFER_SIZE:
            count = BUFFER_SIZE
            
        if count > self.sample_count:
            count = self.sample_count

        end_idx = self.write_index
        start_idx = (end_idx - count) % BUFFER_SIZE

        result = []
        idx = start_idx

        for _ in range(count):
            result.append({
                'timestamp': self.timestamps[idx],
                'value': self.samples[idx],
                'voltage': (self.samples[idx] / 65535) * VOLTAGE_REF
            })
            idx = (idx + 1) % BUFFER_SIZE

        return result

    def calculate_heart_rate(self, window_seconds=10):
        """Calculate heart rate using improved QRS detection."""
        samples_needed = int(self.config["sample_rate"] * window_seconds)
        recent_samples = self.get_latest_samples(samples_needed)

        if len(recent_samples) < samples_needed // 2:
            return None

        valid_samples = [s for s in recent_samples if s['value'] > 0]
        if len(valid_samples) < samples_needed // 2:
            return None
            
        values = [s['value'] for s in valid_samples]
        
        if max(values) - min(values) < 100:
            return None
        
        # QRS detection algorithm
        derivative = []
        for i in range(1, len(values)):
            derivative.append(values[i] - values[i-1])
        
        squared = [d * d for d in derivative]
        
        window_size = int(HR_DERIVATIVE_WINDOW_MS / 1000 * self.config["sample_rate"])
        integrated = []
        for i in range(len(squared)):
            start = max(0, i - window_size // 2)
            end = min(len(squared), i + window_size // 2)
            avg = sum(squared[start:end]) / (end - start)
            integrated.append(avg)
        
        sorted_values = sorted(integrated)
        if not sorted_values:
            return None
        threshold_index = min(int(len(sorted_values) * HR_THRESHOLD_PERCENTILE / 100), len(sorted_values) - 1)
        threshold = sorted_values[threshold_index]
        
        if threshold <= 0:
            return None
        
        min_distance = int(HR_MIN_RR_INTERVAL_MS / 1000 * self.config["sample_rate"])
        peaks = []
        last_peak = -min_distance
        
        for i in range(1, len(integrated) - 1):
            if i - last_peak >= min_distance:
                if (integrated[i] > threshold and
                    integrated[i] > integrated[i-1] and
                    integrated[i] > integrated[i+1]):
                    peaks.append(i)
                    last_peak = i

        if len(peaks) < 2:
            return None

        intervals = []
        for i in range(1, len(peaks)):
            interval_samples = peaks[i] - peaks[i-1]
            interval_seconds = interval_samples / self.config["sample_rate"]
            min_interval = 60.0 / HR_MAX_BPM
            max_interval = 60.0 / HR_MIN_BPM
            if min_interval <= interval_seconds <= max_interval:
                intervals.append(interval_seconds)

        if intervals:
            sorted_intervals = sorted(intervals)
            median_interval = sorted_intervals[len(sorted_intervals) // 2]
            bpm = int(60 / median_interval)

            if HR_MIN_BPM <= bpm <= HR_MAX_BPM:
                return bpm

        return None

    def save_to_csv(self, filename=None, duration_seconds=30):
        """Save data to CSV file."""
        if filename is None:
            filename = DEFAULT_FILENAME

        samples_to_save = int(self.config["sample_rate"] * duration_seconds)
        data = self.get_latest_samples(samples_to_save)

        if not data:
            print("No data to save")
            return False

        try:
            with open(filename, 'w') as f:
                f.write("timestamp_us,adc_value,voltage\n")
                for sample in data:
                    f.write(f"{sample['timestamp']},"
                           f"{sample['value']},"
                           f"{sample['voltage']:.4f}\n")

            print(f"Saved {len(data)} samples to {filename}")
            return True

        except Exception as e:
            print(f"Save error: {e}")
            return False

    def stream_data(self, duration_seconds=None, max_samples=None):
        """Stream data over USB serial with buffering."""
        print("Streaming mode - press Ctrl+C or send 'stop' to exit")
        print("FORMAT:timestamp_us,adc_value,heart_rate")
        print(f"INFO:auto_gain={self.auto_gain:.2f},baseline={int(self.baseline)}")
        print(f"INFO:oversample={OVERSAMPLE_FACTOR}x,effective_bits={ADC_BITS + OVERSAMPLE_FACTOR//4}")
        
        # Enable streaming
        self.streaming = True
        
        start_time = time.ticks_ms()
        last_hr_calc = time.ticks_ms()
        current_hr = 0
        samples_sent = 0
        stop_reason = "unknown"
        
        if not self.sampling_active:
            print("Warning: Sampling not active. Starting sampling...")
            self.start_sampling()
            time.sleep_ms(500)
        
        try:
            while True:
                # Check timeout
                if duration_seconds is not None:
                    elapsed = time.ticks_diff(time.ticks_ms(), start_time) / 1000
                    if elapsed >= duration_seconds:
                        stop_reason = "timeout"
                        break
                
                # Check sample limit
                if max_samples is not None and samples_sent >= max_samples:
                    stop_reason = "sample_limit"
                    break
                
                # Check for stop command
                if hasattr(sys.stdin, 'any') and sys.stdin.any():
                    try:
                        cmd = sys.stdin.readline().strip().lower()
                        if cmd == 'stop':
                            stop_reason = "user_command"
                            break
                    except:
                        pass
                
                # Get samples from stream buffer
                samples = self.stream_buffer.read_batch(1)  # Changed from 10 to 1
                
                if not samples:
                    # No samples available, wait a bit
                    time.sleep_ms(4)  # Reduced from 10ms to 4ms
                    continue
                
                # Calculate heart rate every 2 seconds
                if time.ticks_diff(time.ticks_ms(), last_hr_calc) > 2000:
                    try:
                        hr = self.calculate_heart_rate(5)
                        if hr:
                            current_hr = hr
                    except Exception as e:
                        if self.config.get("debug_mode"):
                            print(f"HR calc error: {e}")
                    last_hr_calc = time.ticks_ms()
                
                # Send data
                for sample in samples:
                    print(f"{sample['timestamp']},"
                          f"{sample['value']},"
                          f"{current_hr}")
                    samples_sent += 1
                    
                    if max_samples is not None and samples_sent >= max_samples:
                        stop_reason = "sample_limit"
                        break
                
                if stop_reason == "sample_limit":
                    break
                
        except KeyboardInterrupt:
            stop_reason = "keyboard_interrupt"
        except Exception as e:
            stop_reason = f"error: {e}"
            if self.config.get("debug_mode"):
                print(f"Unexpected error: {e}")
        
        self.streaming = False
        duration = time.ticks_diff(time.ticks_ms(), start_time) / 1000
        
        print(f"\nStreaming stopped")
        print(f"  Reason: {stop_reason}")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Samples sent: {samples_sent}")
        if samples_sent > 0:
            print(f"  Average rate: {samples_sent/duration:.1f} samples/sec")
        
        return {
            "samples_sent": samples_sent,
            "duration": duration,
            "reason": stop_reason
        }

    def run_diagnostics(self):
        """Run system diagnostics."""
        print("\n=== EKG System Diagnostics ===")

        gc.collect()
        free_mem = gc.mem_free()
        total_mem = gc.mem_alloc() + free_mem
        print(f"Memory: {free_mem}/{total_mem} bytes free "
              f"({free_mem*100//total_mem}%)")

        print("\nADC Test (10 samples):")
        adc_values = []
        for i in range(10):
            value = self.adc.read_u16()
            voltage = (value / 65535) * VOLTAGE_REF
            adc_values.append(value)
            print(f"  {i+1}: {value} ({voltage:.3f}V)")
            time.sleep_ms(100)
        
        adc_mean = sum(adc_values) / len(adc_values)
        print(f"\nADC Statistics:")
        print(f"  Mean: {adc_mean:.0f}")
        print(f"  Range: {min(adc_values)} - {max(adc_values)}")

        print("\nElectrode Status:")
        status = self.check_electrodes()
        print(f"  LO+: {'Connected' if status['lo_plus'] else 'DISCONNECTED'}")
        print(f"  LO-: {'Connected' if status['lo_minus'] else 'DISCONNECTED'}")
        print(f"  Overall: {'OK' if status['connected'] else 'CHECK CONNECTIONS'}")

        print("\nPerformance Test:")
        start = time.ticks_us()
        for _ in range(1000):
            _ = self.adc.read_u16()
        duration = time.ticks_diff(time.ticks_us(), start)
        print(f"  1000 ADC reads: {duration/1000:.1f} µs per read")
        print(f"  Max theoretical rate: {1_000_000//(duration/1000):.0f} Hz")
        
        print(f"\nOversampling Configuration:")
        print(f"  Factor: {OVERSAMPLE_FACTOR}x")
        print(f"  Effective bits: {ADC_BITS + OVERSAMPLE_FACTOR//4}")
        print(f"  Internal ADC rate: {ADC_SAMPLE_RATE} Hz")
        print(f"  Output rate: {SAMPLE_RATE} Hz")

        print("\n=== Diagnostics Complete ===\n")


# Global instance
ekg = None


def init():
    """Initialize the EKG system."""
    global ekg
    ekg = EKGSensor()
    print("EKG system initialized (Improved version)")
    print("Run 'help()' for available commands")


def help():
    """Show available commands."""
    print("""
Available commands:
  init()        - Initialize system
  start()       - Start sampling
  stop()        - Stop sampling
  hr()          - Get heart rate
  save()        - Save 30s of data
  save(60)      - Save 60s of data
  stream()      - Stream data via USB (unlimited)
  stream(30)    - Stream for 30 seconds
  stream(None, 1000) - Stream 1000 samples
  diag()        - Run diagnostics
  status()      - Check electrode status
  
Stream controls:
  - Press Ctrl+C to stop streaming
  - Type 'stop' and press Enter to stop
  - Streaming stops automatically at limits
""")


def start():
    """Start sampling."""
    if not ekg:
        init()
    ekg.start_sampling()


def stop():
    """Stop sampling."""
    if ekg:
        ekg.stop_sampling()


def hr():
    """Get current heart rate."""
    if not ekg:
        print("System not initialized. Run init() first")
        return

    if not ekg.sampling_active:
        print("Not sampling. Run start() first")
        return

    rate = ekg.calculate_heart_rate()
    if rate:
        print(f"Heart rate: {rate} BPM")
    else:
        print("Heart rate: calculating...")
    return rate


def save(duration=30):
    """Save data to file."""
    if not ekg:
        print("System not initialized. Run init() first")
        return

    ekg.save_to_csv(duration_seconds=duration)


def stream(duration=None, max_samples=None):
    """Start streaming mode."""
    if not ekg:
        init()

    if not ekg.sampling_active:
        ekg.start_sampling()

    return ekg.stream_data(duration_seconds=duration, max_samples=max_samples)


def diag():
    """Run diagnostics."""
    if not ekg:
        init()
    ekg.run_diagnostics()


def status():
    """Check electrode status."""
    if not ekg:
        init()

    status = ekg.check_electrodes()
    if status['connected']:
        print("✅ All electrodes connected")
    else:
        print("⚠️  Electrode problem:")
        if not status['lo_plus']:
            print("  - LO+ disconnected")
        if not status['lo_minus']:
            print("  - LO- disconnected")


# Auto-initialize on import
if __name__ == "__main__":
    init()
    diag()