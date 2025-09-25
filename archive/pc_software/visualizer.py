"""
Real-time EKG Visualizer for RP2040 EKG System
Author: [Your Name]
Version: 1.0.0
License: MIT

Requirements:
- pyserial>=3.5
- matplotlib>=3.7.0
- numpy>=1.24.0
"""

# Standard library
import sys
import time
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Deque

# Third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
from matplotlib.ticker import MultipleLocator
import serial
import serial.tools.list_ports
from scipy import signal
from scipy.signal import find_peaks


# Constants
DEFAULT_SAMPLE_RATE = 250  # Hz
MAX_RECORD_DURATION = 300  # 5 minutes in seconds
NOTCH_FREQ = 50.0  # Hz
NOTCH_Q = 30.0  # Quality factor

# Algorithm parameters
HR_DERIVATIVE_WINDOW_MS = 80
HR_MIN_RR_INTERVAL_MS = 300
HR_THRESHOLD_PERCENTILE = 90
HR_MIN_BPM = 30
HR_MAX_BPM = 200

# UI parameters
PLOT_WINDOW_SECONDS = 3.0  # Show 3 seconds of EKG
ANIMATION_FPS = 25
Y_SCALE_PERCENTILE_LOW = 5
Y_SCALE_PERCENTILE_HIGH = 95

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable matplotlib debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)


@dataclass
class EKGSample:
    """Single EKG data sample."""
    timestamp: int
    adc_value: int
    heart_rate: int
    voltage: float = 0.0

    def __post_init__(self):
        """Calculate voltage from ADC value."""
        if self.voltage == 0.0:
            self.voltage = (self.adc_value / 65535) * 3.3


class SerialManager:
    """Manages serial connection to RP2040."""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        self.port = port or self._find_device()
        self.baudrate = baudrate
        self.connection: Optional[serial.Serial] = None
        self.error_count = 0
    
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    @staticmethod
    def _find_device() -> Optional[str]:
        """Auto-detect RP2040 device."""
        logger.info("Searching for RP2040...")

        ports = serial.tools.list_ports.comports()

        # First try COM5 as default
        for port in ports:
            if port.device == "COM5":
                logger.info(f"Using default COM5: {port.description}")
                return "COM5"

        # Look for RP2040 identifiers
        for port in ports:
            if any(id in port.description.upper() for id in ['RP2040', 'PICO', 'CDC', 'USB']):
                logger.info(f"Found device: {port.device} - {port.description}")
                return port.device

        # Show all available ports
        if ports:
            logger.warning("RP2040 not found. Available ports:")
            for port in ports:
                logger.warning(f"  {port.device} - {port.description}")
        else:
            logger.error("No serial ports found")

        return None

    def connect(self) -> bool:
        """Establish serial connection."""
        if not self.port:
            logger.error("No port specified")
            return False

        try:
            self.connection = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1.0,
                write_timeout=1.0
            )

            # Wait for device to initialize
            time.sleep(2)

            # Clear buffers
            self.connection.reset_input_buffer()
            self.connection.reset_output_buffer()

            logger.info(f"Connected to {self.port} at {self.baudrate} baud")
            return True

        except serial.SerialException as e:
            logger.error(f"Connection failed: {e}")
            return False

    def disconnect(self):
        """Close serial connection."""
        if self.connection and self.connection.is_open:
            try:
                # Send Ctrl+C to interrupt streaming
                self.connection.write(b'\x03')
                time.sleep(0.1)
                self.connection.close()
                logger.info("Disconnected")
            except:
                pass

    def start_streaming(self):
        """Initialize device and start streaming."""
        if not self.connection or not self.connection.is_open:
            return False

        try:
            # Enter REPL mode
            logger.info("Entering REPL mode...")
            self.connection.write(b'\x03')  # Ctrl+C to interrupt
            time.sleep(0.5)
            self.connection.write(b'\x04')  # Ctrl+D to soft reset  
            time.sleep(2)
            
            # Clear any pending data
            self.connection.reset_input_buffer()
            
            # Send commands in REPL
            logger.info("Initializing device...")
            self.connection.write(b'init()\r')
            time.sleep(1)
            
            logger.info("Starting sampling...")
            self.connection.write(b'start()\r')
            time.sleep(1)
            
            logger.info("Starting stream...")
            self.connection.write(b'stream()\r')
            time.sleep(1)
            
            # Clear any REPL prompts
            self.connection.reset_input_buffer()
            
            logger.info("Device initialized and streaming")
            return True
        except serial.SerialException as e:
            logger.error(f"Failed to start streaming: {e}")
            return False

    def read_sample(self) -> Optional[EKGSample]:
        """Read single sample from serial."""
        if not self.connection or not self.connection.is_open:
            return None

        try:
            # Check if connection is still alive
            if not self.connection.is_open:
                raise serial.SerialException("Connection lost")
            if self.connection.in_waiting:
                line = self.connection.readline().decode('utf-8').strip()

                # Log raw data for debugging
                if logger.level <= logging.DEBUG:
                    logger.debug(f"Raw data: {line}")
                
                # Skip debug messages and empty lines
                skip_patterns = [
                    '===', 'FORMAT:', 'Streaming', 'EKG system', 'Run', 'Available',
                    'initialized', 'Sampling started', 'Auto-gain',
                    'Heart rate:', 'Samples:', 'Rate:', 'Buffer:',
                    'help()', 'init()', 'start()', 'stop()', 'hr()', 'save()',
                    'stream()', 'diag()', 'status()', 'Memory:', 'ADC Test',
                    'Electrode Status', 'Performance Test', 'Diagnostics',
                    'Oversampling', 'Effective', 'Internal ADC',
                    '>>>'  # Skip REPL prompt
                ]
                
                # Handle INFO messages separately
                if line.startswith('INFO:'):
                    logger.debug(f"Info message: {line}")
                    return None
                if not line or any(skip in line for skip in skip_patterns):
                    logger.debug(f"Skipping: {line}")
                    return None

                # Parse CSV data
                parts = line.split(',')
                if len(parts) >= 3:
                    sample = EKGSample(
                        timestamp=int(parts[0]),
                        adc_value=int(parts[1]),
                        heart_rate=int(parts[2])
                    )
                    self.error_count = 0
                    return sample

        except (ValueError, UnicodeDecodeError) as e:
            self.error_count += 1
            if self.error_count % 10 == 0:
                logger.warning(f"Parse errors: {self.error_count}, last error: {e}")
            if logger.level <= logging.DEBUG:
                logger.debug(f"Parse error: {e}, line: {line if 'line' in locals() else 'N/A'}")
        except serial.SerialException as e:
            logger.error(f"Serial connection error: {e}")
            self.connection = None  # Mark connection as lost
            raise  # Re-raise to handle reconnection

        return None


class DataBuffer:
    """Manages data buffering and analysis."""

    def __init__(self, window_size: int = 2500, sample_rate: int = DEFAULT_SAMPLE_RATE):
        self.window_size = window_size
        self.sample_rate = sample_rate

        # Data buffers
        self.timestamps: Deque[int] = deque(maxlen=window_size)
        self.values: Deque[int] = deque(maxlen=window_size)
        self.heart_rates: Deque[int] = deque(maxlen=100)
        self.voltages: Deque[float] = deque(maxlen=window_size)
        self.filtered_values: Deque[float] = deque(maxlen=window_size)

        # Statistics
        self.total_samples = 0
        self.start_time = time.time()
        self.last_timestamp = 0
        self.timestamp_offset = 0
        
        # Design notch filter
        self.b_notch, self.a_notch = signal.iirnotch(NOTCH_FREQ, NOTCH_Q, fs=self.sample_rate)
        # Initialize filter state
        self.notch_state = signal.lfilter_zi(self.b_notch, self.a_notch) * 0

    def add_sample(self, sample: EKGSample):
        """Add new sample to buffers."""
        # Handle timestamp overflow/jumps
        if self.last_timestamp > 0 and sample.timestamp < self.last_timestamp:
            # Timestamp jumped backwards (overflow or reset)
            if self.last_timestamp - sample.timestamp > 0x80000000:
                # Likely an overflow
                self.timestamp_offset += 0x100000000
            else:
                # Reset detected, adjust offset
                self.timestamp_offset = self.last_timestamp
        
        adjusted_timestamp = sample.timestamp + self.timestamp_offset
        self.last_timestamp = sample.timestamp
        
        self.timestamps.append(adjusted_timestamp)
        self.values.append(sample.adc_value)
        self.voltages.append(sample.voltage)
        
        # Apply notch filter to remove 50Hz noise
        filtered_val, self.notch_state = signal.lfilter(
            self.b_notch, self.a_notch, [sample.adc_value], zi=self.notch_state
        )
        self.filtered_values.append(filtered_val[0])

        if sample.heart_rate > 0:
            self.heart_rates.append(sample.heart_rate)

        self.total_samples += 1

    def get_plot_data(self) -> Tuple[List[float], List[float]]:
        """Get data for plotting (time in seconds, voltage)."""
        if not self.timestamps:
            return [], []

        # Convert to relative time in seconds
        t0 = self.timestamps[0] if self.timestamps else 0
        times = [(t - t0) / 1_000_000 for t in self.timestamps]

        return times, list(self.voltages)

    def get_current_hr(self) -> Optional[int]:
        """Get most recent heart rate."""
        return self.heart_rates[-1] if self.heart_rates else None
    
    def calculate_heart_rate(self, window_seconds: float = 5.0) -> Optional[int]:
        """Calculate heart rate using improved QRS detection on filtered signal.
        
        Args:
            window_seconds: Duration of analysis window
            
        Returns:
            Heart rate in BPM or None if unable to calculate
        """
        min_samples = int(self.sample_rate * 2)  # Need at least 2 seconds
        if len(self.filtered_values) < min_samples:
            return None
            
        # Get samples for analysis
        samples_needed = min(
            int(self.sample_rate * window_seconds),
            len(self.filtered_values)
        )
        
        values = np.array(list(self.filtered_values))[-samples_needed:]
        
        # Check for flat signal (disconnected or no variation)
        if np.ptp(values) < 100:  # peak-to-peak range
            return None
        
        # Step 1: Compute derivative
        diff_ecg = np.diff(values)
        
        # Step 2: Square the derivative
        squared = diff_ecg ** 2
        
        # Step 3: Moving average integration
        window_size = int(HR_DERIVATIVE_WINDOW_MS / 1000 * self.sample_rate)
        kernel = np.ones(window_size) / window_size
        integrated = np.convolve(squared, kernel, mode='same')
        
        # Step 4: Adaptive threshold with validation
        threshold = np.percentile(integrated, HR_THRESHOLD_PERCENTILE)
        if threshold <= 0:
            return None
        
        # Step 5: Find peaks with minimum distance
        min_distance = int(HR_MIN_RR_INTERVAL_MS / 1000 * self.sample_rate)
        
        peaks, _ = find_peaks(integrated, 
                             height=threshold,
                             distance=min_distance)
        
        if len(peaks) < 2:
            return None
            
        # Calculate RR intervals
        rr_intervals = np.diff(peaks) / self.sample_rate  # seconds
        
        # Filter valid intervals
        valid_mask = (rr_intervals >= 60/HR_MAX_BPM) & (rr_intervals <= 60/HR_MIN_BPM)
        valid_intervals = rr_intervals[valid_mask]
        
        if len(valid_intervals) == 0:
            return None
            
        # Use median for robustness
        median_interval = np.median(valid_intervals)
        bpm = int(60 / median_interval)
        
        # Final validation
        if HR_MIN_BPM <= bpm <= HR_MAX_BPM:
            return bpm
            
        return None

    def get_statistics(self) -> dict:
        """Calculate current statistics."""
        elapsed = time.time() - self.start_time

        stats = {
            'total_samples': self.total_samples,
            'elapsed_time': elapsed,
            'sample_rate': self.total_samples / elapsed if elapsed > 0 else 0,
            'current_hr': self.get_current_hr(),
            'buffer_fill': len(self.values) / self.window_size * 100
        }

        if self.voltages:
            stats.update({
                'voltage_min': min(self.voltages),
                'voltage_max': max(self.voltages),
                'voltage_mean': sum(self.voltages) / len(self.voltages)
            })

        return stats


class EKGVisualizer:
    """Main visualizer application."""

    def __init__(self, port: Optional[str] = None, debug: bool = False):
        self.serial_manager = SerialManager(port)
        self.data_buffer = DataBuffer()
        self.recording = False
        # Use deque with maxlen for automatic circular buffer
        max_samples = DEFAULT_SAMPLE_RATE * MAX_RECORD_DURATION
        self.record_buffer: Deque[EKGSample] = deque(maxlen=max_samples)
        self.debug = debug

        # Setup plot
        self._setup_plot()
        
        # Connection retry state
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 3

    def _setup_plot(self):
        """Initialize matplotlib figure and axes."""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle('EKG Monitor - RP2040', fontsize=16)

        # Create grid layout
        gs = self.fig.add_gridspec(3, 3, height_ratios=[3, 1, 0.5],
                                   width_ratios=[3, 1, 1])

        # Main EKG plot
        self.ax_ekg = self.fig.add_subplot(gs[0, :])
        self.line_ekg, = self.ax_ekg.plot([], [], 'g-', linewidth=2.0)  # Bright green line, thicker
        self.ax_ekg.set_ylabel('Voltage (V)', fontsize=12)
        self.ax_ekg.set_ylim(0, 3.3)
        
        # EKG-style grid (major grid every 0.2s and 0.5mV)
        self.ax_ekg.grid(True, which='major', color='red', alpha=0.4, linewidth=0.8)
        self.ax_ekg.grid(True, which='minor', color='red', alpha=0.2, linewidth=0.4)
        self.ax_ekg.minorticks_on()
        
        # Set major ticks every 0.2 seconds (5 small squares = 1 big square)
        self.ax_ekg.xaxis.set_major_locator(MultipleLocator(0.2))
        self.ax_ekg.xaxis.set_minor_locator(MultipleLocator(0.04))  # 40ms minor ticks
        
        self.ax_ekg.set_title('EKG Signal - 25 mm/s', fontsize=14)

        # Heart rate plot
        self.ax_hr = self.fig.add_subplot(gs[1, :2])
        self.line_hr, = self.ax_hr.plot([], [], 'r-', linewidth=2)
        self.ax_hr.set_ylabel('BPM', fontsize=12)
        self.ax_hr.set_xlabel('Time', fontsize=12)
        self.ax_hr.set_ylim(40, 180)
        self.ax_hr.grid(True, alpha=0.3)
        self.ax_hr.set_title('Heart Rate', fontsize=14)

        # Info panel
        self.ax_info = self.fig.add_subplot(gs[1, 2])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(
            0.1, 0.5, '', fontsize=12,
            verticalalignment='center',
            fontfamily='monospace'
        )

        # Control buttons
        self.ax_btn_record = self.fig.add_subplot(gs[2, 0])
        self.btn_record = Button(self.ax_btn_record, 'Record', color='green')
        self.btn_record.on_clicked(self._toggle_recording)

        self.ax_btn_save = self.fig.add_subplot(gs[2, 1])
        self.btn_save = Button(self.ax_btn_save, 'Save', color='blue')
        self.btn_save.on_clicked(self._save_data)

        self.ax_btn_clear = self.fig.add_subplot(gs[2, 2])
        self.btn_clear = Button(self.ax_btn_clear, 'Clear', color='orange')
        self.btn_clear.on_clicked(self._clear_data)

        plt.tight_layout()

    def _toggle_recording(self, event):
        """Toggle data recording."""
        self.recording = not self.recording

        if self.recording:
            self.record_buffer.clear()
            self.btn_record.label.set_text('Stop')
            self.btn_record.ax.set_facecolor('red')
            logger.info("Recording started")
        else:
            self.btn_record.label.set_text('Record')
            self.btn_record.ax.set_facecolor('green')
            logger.info(f"Recording stopped. {len(self.record_buffer)} samples")

    def _save_data(self, event):
        """Save data to file."""
        # Save from main buffer if no recording, otherwise save recording
        if self.record_buffer:
            data_to_save = self.record_buffer
            source = "recording"
        else:
            # Create samples from buffer data
            data_to_save = []
            for i in range(len(self.data_buffer.timestamps)):
                data_to_save.append(EKGSample(
                    timestamp=self.data_buffer.timestamps[i],
                    adc_value=self.data_buffer.values[i],
                    heart_rate=self.data_buffer.heart_rates[i] if i < len(self.data_buffer.heart_rates) else 0,
                    voltage=self.data_buffer.voltages[i]
                ))
            source = "buffer"
        
        if not data_to_save:
            logger.warning("No data to save")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ekg_{source}_{timestamp}.csv"

        try:
            with open(filename, 'w') as f:
                f.write("timestamp_us,adc_value,voltage,heart_rate\n")
                for sample in data_to_save:
                    f.write(f"{sample.timestamp},{sample.adc_value},"
                           f"{sample.voltage:.4f},{sample.heart_rate}\n")

            logger.info(f"Saved {len(data_to_save)} samples from {source} to {filename}")

        except IOError as e:
            logger.error(f"Save failed: {e}")

    def _clear_data(self, event):
        """Clear all buffers."""
        self.data_buffer = DataBuffer()
        self.record_buffer.clear()
        logger.info("Buffers cleared")

    def _update_plot(self, frame):
        """Animation update function."""
        # Check connection status
        if not self.serial_manager.connection:
            self._handle_disconnection()
            return self.line_ekg, self.line_hr, self.info_text
            
        # Read new samples - at 250Hz and 25 FPS, we expect ~10 samples per frame
        # Read a bit more to account for timing variations
        buffer_fill = self.data_buffer.get_statistics()['buffer_fill']
        samples_to_read = 15 if buffer_fill > 80 else 12
        
        samples_read = 0
        for _ in range(samples_to_read):  # Read more samples if buffer is getting full
            try:
                sample = self.serial_manager.read_sample()
            except serial.SerialException:
                self._handle_disconnection()
                return self.line_ekg, self.line_hr, self.info_text
                
            if sample:
                self.data_buffer.add_sample(sample)
                samples_read += 1
                
                # Debug: print first few samples
                if self.debug and self.data_buffer.total_samples < 10:
                    print(f"Sample {self.data_buffer.total_samples}: ADC={sample.adc_value}, V={sample.voltage:.3f}")

                if self.recording:
                    self.record_buffer.append(sample)

        # Update EKG plot with filtered data
        times, voltages = self.data_buffer.get_plot_data()
        if times:
            # Use filtered values if available, otherwise use raw voltages
            if len(self.data_buffer.filtered_values) > 0:
                # Make sure we have the same number of filtered values as times
                num_samples = min(len(times), len(self.data_buffer.filtered_values))
                plot_times = times[-num_samples:]
                filtered_voltages = [(v / 65535) * 3.3 for v in list(self.data_buffer.filtered_values)[-num_samples:]]
                self.line_ekg.set_data(plot_times, filtered_voltages)
            else:
                # Fallback to raw voltages if no filtered data yet
                self.line_ekg.set_data(times, voltages)

            # Adjust x-axis to show configured window
            if times[-1] > PLOT_WINDOW_SECONDS:
                self.ax_ekg.set_xlim(times[-1] - PLOT_WINDOW_SECONDS, times[-1])
            else:
                self.ax_ekg.set_xlim(0, PLOT_WINDOW_SECONDS)

            # Auto-scale y-axis based on percentiles for better visualization
            if 'filtered_voltages' in locals() and filtered_voltages:
                v_5 = np.percentile(filtered_voltages, Y_SCALE_PERCENTILE_LOW)
                v_95 = np.percentile(filtered_voltages, Y_SCALE_PERCENTILE_HIGH)
                
                # If signal is flat (disconnected), show fixed range centered on signal
                if abs(v_95 - v_5) < 0.01:
                    center = (v_5 + v_95) / 2
                    self.ax_ekg.set_ylim(center - 0.5, center + 0.5)
                else:
                    # Increase margin to prevent clipping
                    margin = (v_95 - v_5) * 0.3  # Increased from 0.1 to 0.3
                    # Also check absolute limits to prevent going outside ADC range
                    y_min = max(0, v_5 - margin)
                    y_max = min(3.3, v_95 + margin)
                    self.ax_ekg.set_ylim(y_min, y_max)

        # Calculate heart rate locally using improved algorithm
        calculated_hr = self.data_buffer.calculate_heart_rate()
        if calculated_hr:
            # Add to heart rates buffer for plotting
            self.data_buffer.heart_rates.append(calculated_hr)
        
        # Update heart rate plot
        if self.data_buffer.heart_rates:
            hr_times = list(range(len(self.data_buffer.heart_rates)))
            self.line_hr.set_data(hr_times, list(self.data_buffer.heart_rates))
            self.ax_hr.set_xlim(0, max(100, len(hr_times)))

        # Update info panel
        stats = self.data_buffer.get_statistics()
        # Use locally calculated HR instead of firmware HR
        current_hr = calculated_hr if calculated_hr else stats.get('current_hr', '--')
        info_text = (
            f"Heart Rate: {current_hr} BPM\n"
            f"Samples: {stats['total_samples']:,}\n"
            f"Rate: {stats['sample_rate']:.1f} Hz\n"
            f"Buffer: {stats['buffer_fill']:.0f}%\n"
            f"Read: {samples_read}/frame\n"
        )

        if self.recording:
            info_text += f"\n[REC] {len(self.record_buffer)}"

        self.info_text.set_text(info_text)

        # Update title with connection status
        if self.serial_manager.error_count > 10:
            self.fig.suptitle('EKG Monitor - RP2040 [Connection Issues]',
                             fontsize=16, color='yellow')

        return self.line_ekg, self.line_hr, self.info_text

    def run(self):
        """Start the visualizer."""
        # Connect to device
        if not self.serial_manager.connect():
            logger.error("Failed to connect to device")

            # Ask for manual port selection
            port = input("Enter serial port manually (or press Enter to exit): ")
            if port:
                self.serial_manager.port = port
                if not self.serial_manager.connect():
                    return
            else:
                return

        # Start streaming
        if not self.serial_manager.start_streaming():
            logger.error("Failed to start streaming")
            return

        # Start animation
        try:
            self.ani = animation.FuncAnimation(
                self.fig, self._update_plot,
                interval=1000 // ANIMATION_FPS,  # Convert FPS to interval
                blit=True,
                cache_frame_data=False
            )

            plt.show()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.serial_manager.disconnect()
    
    def _handle_disconnection(self):
        """Handle device disconnection."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self.fig.suptitle('EKG Monitor - DISCONNECTED', fontsize=16, color='red')
            return
            
        self.reconnect_attempts += 1
        logger.warning(f"Device disconnected. Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        self.fig.suptitle(f'EKG Monitor - Reconnecting... ({self.reconnect_attempts}/{self.max_reconnect_attempts})', 
                         fontsize=16, color='yellow')
        
        # Try to reconnect
        time.sleep(1)
        if self.serial_manager.connect() and self.serial_manager.start_streaming():
            logger.info("Reconnection successful")
            self.reconnect_attempts = 0
            self.fig.suptitle('EKG Monitor - RP2040', fontsize=16)


def main():
    """Main entry point."""
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description='EKG Visualizer for RP2040')
    parser.add_argument('--port', '-p', help='Serial port (auto-detect if not specified)')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print header
    print("=" * 50)
    print("EKG Visualizer for RP2040")
    print("=" * 50)
    print()

    # Check dependencies
    try:
        import matplotlib
        import serial
        import numpy
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pyserial matplotlib numpy")
        sys.exit(1)

    # Run visualizer
    visualizer = EKGVisualizer(port=args.port, debug=args.debug)
    visualizer.run()


if __name__ == "__main__":
    main()