#!/usr/bin/env python3
"""
Real-time EKG Visualizer - Version for scaled firmware
Displays properly scaled EKG signal with correct voltage calculations
"""

import sys
import time
import logging
from dataclasses import dataclass
from typing import Optional, List, Tuple, Deque
from collections import deque
import threading
import queue
import statistics

import serial
import serial.tools.list_ports
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT = "COM5"
DEFAULT_BAUDRATE = 115200
DEFAULT_SAMPLE_RATE = 250  # Hz
PLOT_WINDOW = 10  # seconds
PLOT_UPDATE_INTERVAL = 100  # ms
DISPLAY_UPDATE_INTERVAL = 100  # ms
GRID_COLOR = '#2a2a2a'
NOTCH_FREQ = 50  # Hz (European power line frequency)
NOTCH_Q = 30  # Quality factor

# Signal display range
SIGNAL_MIN_MV = -2.0  # millivolts
SIGNAL_MAX_MV = 2.0   # millivolts

# QRS Detection parameters
QRS_WINDOW_SIZE = int(DEFAULT_SAMPLE_RATE * 0.15)  # 150ms window
QRS_THRESHOLD_FACTOR = 0.6  # 60% of recent max

# Heart Rate calculation
HR_WINDOW_SIZE = 10  # Number of R-R intervals to average

# Buffer settings
BUFFER_SIZE = DEFAULT_SAMPLE_RATE * 30  # 30 seconds of data

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
        """Calculate voltage from scaled ADC value."""
        if self.voltage == 0.0:
            # For scaled firmware: values are centered around 32768
            # Convert to millivolts directly (already scaled to ±2mV range)
            centered = self.adc_value - 32768
            self.voltage = (centered / 32768) * 2.0  # Convert to mV range


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

    def _find_device(self) -> Optional[str]:
        """Find RP2040 device automatically."""
        logger.info("Searching for RP2040 device...")
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
                    'initialized', 'Sampling started', 'Auto-gain', 'Signal scaled',
                    'Heart rate:', 'Samples:', 'Rate:', 'Buffer:', 'Scale factor:',
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
        
        # Apply notch filter to voltage values (not ADC values)
        filtered_val, self.notch_state = signal.lfilter(
            self.b_notch, self.a_notch, [sample.voltage], zi=self.notch_state
        )
        self.filtered_values.append(filtered_val[0])

        if sample.heart_rate > 0:
            self.heart_rates.append(sample.heart_rate)

        self.total_samples += 1

    def get_plot_data(self) -> Tuple[List[float], List[float]]:
        """Get data for plotting (time in seconds, voltage in mV)."""
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
        recent_values = list(self.filtered_values)[-samples_needed:]
        
        # Find QRS complexes using voltage threshold
        # For scaled signal, typical QRS peak is 0.5-1.5 mV
        threshold = 0.3  # mV threshold for QRS detection
        
        # Find peaks
        peaks = []
        for i in range(1, len(recent_values) - 1):
            if (recent_values[i] > recent_values[i-1] and 
                recent_values[i] > recent_values[i+1] and
                recent_values[i] > threshold):
                peaks.append(i)
        
        # Remove peaks too close together (< 200ms)
        min_distance = int(0.2 * self.sample_rate)
        filtered_peaks = []
        last_peak = -min_distance
        
        for peak in peaks:
            if peak - last_peak >= min_distance:
                filtered_peaks.append(peak)
                last_peak = peak
        
        # Need at least 2 peaks for HR calculation
        if len(filtered_peaks) < 2:
            return None
            
        # Calculate R-R intervals
        intervals = []
        for i in range(1, len(filtered_peaks)):
            interval_samples = filtered_peaks[i] - filtered_peaks[i-1]
            interval_ms = (interval_samples / self.sample_rate) * 1000
            intervals.append(interval_ms)
        
        if not intervals:
            return None
            
        # Calculate heart rate from average interval
        avg_interval_ms = statistics.mean(intervals)
        hr = int(60000 / avg_interval_ms)
        
        # Sanity check
        if 40 <= hr <= 200:
            return hr
        return None


class EKGVisualizer:
    """Real-time EKG visualization with matplotlib."""

    def __init__(self, buffer: DataBuffer):
        self.buffer = buffer
        self.fig, self.ax = self._setup_plot()
        self.line, = self.ax.plot([], [], 'lime', linewidth=1.5)
        self.background = None
        self._is_connected = True

    def _setup_plot(self) -> Tuple[plt.Figure, plt.Axes]:
        """Configure the plot."""
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Configure axes for EKG display
        ax.set_xlim(0, PLOT_WINDOW)
        ax.set_ylim(SIGNAL_MIN_MV - 0.5, SIGNAL_MAX_MV + 0.5)
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Voltage (mV)', fontsize=12)
        ax.set_title('Real-time EKG Monitor (Scaled Signal)', fontsize=14, pad=20)

        # Configure grid
        ax.grid(True, which='both', color=GRID_COLOR, alpha=0.5)
        ax.grid(True, which='major', color=GRID_COLOR, alpha=0.8, linewidth=1)
        
        # Add horizontal reference lines
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        ax.axhline(y=-1, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)

        # Configure ticks
        ax.set_xticks(np.arange(0, PLOT_WINDOW + 1, 1))
        ax.set_yticks(np.arange(SIGNAL_MIN_MV, SIGNAL_MAX_MV + 0.5, 0.5))
        ax.tick_params(axis='both', which='major', labelsize=10)

        # Add text displays
        self.hr_text = ax.text(0.02, 0.95, '', transform=ax.transAxes,
                               fontsize=16, color='lime', weight='bold',
                               verticalalignment='top')
        self.stats_text = ax.text(0.02, 0.85, '', transform=ax.transAxes,
                                  fontsize=10, color='white',
                                  verticalalignment='top')
        self.connection_text = ax.text(0.98, 0.95, '● Connected', 
                                       transform=ax.transAxes,
                                       fontsize=12, color='lime',
                                       verticalalignment='top',
                                       horizontalalignment='right')

        # Add legend
        legend_elements = [
            mpatches.Patch(color='lime', label='EKG Signal'),
            mpatches.Patch(color='gray', label='0 mV Reference')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        return fig, ax

    def update_plot(self, frame):
        """Update plot with new data."""
        times, voltages = self.buffer.get_plot_data()
        
        if times and voltages:
            # Adjust time window
            if times[-1] > PLOT_WINDOW:
                self.ax.set_xlim(times[-1] - PLOT_WINDOW, times[-1])
            
            # Update line data
            self.line.set_data(times, voltages)
            
            # Update heart rate display
            hr = self.buffer.calculate_heart_rate()
            if hr:
                self.hr_text.set_text(f'Heart Rate: {hr} BPM')
                self.hr_text.set_color('lime' if 60 <= hr <= 100 else 'yellow')
            else:
                hr_from_device = self.buffer.get_current_hr()
                if hr_from_device and hr_from_device > 0:
                    self.hr_text.set_text(f'Heart Rate: {hr_from_device} BPM*')
                    self.hr_text.set_color('cyan')
                else:
                    self.hr_text.set_text('Heart Rate: --')
                    self.hr_text.set_color('gray')
            
            # Update statistics
            if len(voltages) > 10:
                recent_voltages = voltages[-250:]  # Last second
                stats_text = (
                    f'Samples: {self.buffer.total_samples:,}\n'
                    f'Rate: {self.buffer.total_samples / (time.time() - self.buffer.start_time):.1f} Hz\n'
                    f'Range: {min(recent_voltages):.2f} - {max(recent_voltages):.2f} mV\n'
                    f'Signal: {np.std(recent_voltages):.3f} mV std'
                )
                self.stats_text.set_text(stats_text)
            
            # Update connection status
            if self._is_connected:
                self.connection_text.set_text('● Connected')
                self.connection_text.set_color('lime')
            else:
                self.connection_text.set_text('● Disconnected')
                self.connection_text.set_color('red')

    def set_connection_status(self, connected: bool):
        """Update connection status display."""
        self._is_connected = connected

    def start(self):
        """Start the animation."""
        self.ani = FuncAnimation(
            self.fig, self.update_plot,
            interval=PLOT_UPDATE_INTERVAL,
            blit=False,
            cache_frame_data=False
        )
        plt.show()


def data_acquisition_thread(serial_mgr: SerialManager, data_queue: queue.Queue, stop_event: threading.Event):
    """Thread function for continuous data acquisition."""
    logger.info("Data acquisition thread started")
    reconnect_delay = 2.0
    
    while not stop_event.is_set():
        try:
            # Ensure connection
            if not serial_mgr.connection or not serial_mgr.connection.is_open:
                logger.info("Attempting to connect...")
                if not serial_mgr.connect():
                    time.sleep(reconnect_delay)
                    continue
                
                # Start streaming
                if not serial_mgr.start_streaming():
                    logger.error("Failed to start streaming")
                    time.sleep(reconnect_delay)
                    continue
                
                data_queue.put(('connected', True))
            
            # Read samples
            sample = serial_mgr.read_sample()
            if sample:
                data_queue.put(('sample', sample))
            
        except serial.SerialException:
            logger.warning("Connection lost, attempting reconnect...")
            data_queue.put(('connected', False))
            serial_mgr.connection = None
            time.sleep(reconnect_delay)
        except Exception as e:
            logger.error(f"Unexpected error in acquisition thread: {e}")
            time.sleep(0.1)
    
    logger.info("Data acquisition thread stopped")


def main():
    """Main application entry point."""
    logger.info("Starting EKG Visualizer (Scaled Version)...")
    
    # Check for command line arguments
    port = sys.argv[1] if len(sys.argv) > 1 else None
    
    # Create components
    buffer = DataBuffer(window_size=BUFFER_SIZE)
    visualizer = EKGVisualizer(buffer)
    data_queue = queue.Queue(maxsize=1000)
    stop_event = threading.Event()
    
    # Start serial connection in separate thread
    serial_mgr = SerialManager(port=port)
    acq_thread = threading.Thread(
        target=data_acquisition_thread,
        args=(serial_mgr, data_queue, stop_event),
        daemon=True
    )
    acq_thread.start()
    
    # Process data queue
    def process_queue():
        """Process items from the data queue."""
        while not data_queue.empty():
            try:
                msg_type, data = data_queue.get_nowait()
                if msg_type == 'sample':
                    buffer.add_sample(data)
                elif msg_type == 'connected':
                    visualizer.set_connection_status(data)
            except queue.Empty:
                break
    
    # Timer for queue processing
    def timer_callback():
        """Timer callback for processing queue."""
        process_queue()
        visualizer.fig.canvas.draw_idle()
    
    # Setup timer
    timer = visualizer.fig.canvas.new_timer(interval=DISPLAY_UPDATE_INTERVAL)
    timer.add_callback(timer_callback)
    timer.start()
    
    try:
        # Start visualization
        visualizer.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Cleanup
        stop_event.set()
        timer.stop()
        serial_mgr.disconnect()
        acq_thread.join(timeout=2.0)
        plt.close('all')
    
    logger.info("EKG Visualizer stopped")


if __name__ == "__main__":
    main()