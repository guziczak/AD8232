#!/usr/bin/env python3
"""
Real-time Spectrum Analyzer Visualizer for RP2040
Displays spectrum data from the spectrum analyzer firmware
"""

import sys
import time
import struct
import logging
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import serial
import serial.tools.list_ports

# Constants
PACKET_HEADER = 0xAABB
PACKET_SIZE = 132  # 2 header + 128 data + 2 checksum
NUM_BINS = 128
SAMPLE_RATE = 40000  # 40 kHz
MAX_FREQ = SAMPLE_RATE // 2  # Nyquist frequency

# Display settings
ANIMATION_FPS = 30
WATERFALL_HEIGHT = 100  # Number of time steps to show

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpectrumAnalyzer:
    """Main spectrum analyzer application"""
    
    def __init__(self, port=None, baudrate=115200):
        self.port = port or self._find_device()
        self.baudrate = baudrate
        self.serial = None
        
        # Data buffers
        self.spectrum_data = np.zeros(NUM_BINS)
        self.waterfall_data = np.zeros((WATERFALL_HEIGHT, NUM_BINS))
        self.packet_times = deque(maxlen=100)
        
        # Statistics
        self.packet_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Serial buffer
        self.buffer = bytearray()
        
        # Setup plot
        self._setup_plot()
    
    @staticmethod
    def _find_device():
        """Auto-detect RP2040 device"""
        logger.info("Searching for RP2040...")
        ports = serial.tools.list_ports.comports()
        
        for port in ports:
            if any(id in port.description.upper() for id in ['RP2040', 'PICO', 'CDC']):
                logger.info(f"Found device: {port.device}")
                return port.device
        
        # List all ports if RP2040 not found
        if ports:
            logger.warning("RP2040 not found. Available ports:")
            for port in ports:
                logger.warning(f"  {port.device} - {port.description}")
        
        return None
    
    def _setup_plot(self):
        """Initialize matplotlib figure"""
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.suptitle('Spectrum Analyzer - RP2040', fontsize=16)
        
        # Create subplots
        gs = self.fig.add_gridspec(3, 2, height_ratios=[2, 2, 1], width_ratios=[5, 1])
        
        # Spectrum plot
        self.ax_spectrum = self.fig.add_subplot(gs[0, 0])
        self.line_spectrum, = self.ax_spectrum.plot([], [], 'g-', linewidth=2)
        self.ax_spectrum.set_xlabel('Frequency (Hz)', fontsize=12)
        self.ax_spectrum.set_ylabel('Magnitude (dB)', fontsize=12)
        self.ax_spectrum.set_xlim(0, MAX_FREQ)
        self.ax_spectrum.set_ylim(-60, 0)
        self.ax_spectrum.grid(True, alpha=0.3)
        self.ax_spectrum.set_title('Spectrum', fontsize=14)
        
        # Waterfall plot
        self.ax_waterfall = self.fig.add_subplot(gs[1, 0])
        self.im_waterfall = self.ax_waterfall.imshow(
            self.waterfall_data,
            aspect='auto',
            origin='lower',
            cmap='viridis',
            extent=[0, MAX_FREQ, 0, WATERFALL_HEIGHT],
            vmin=-60, vmax=0
        )
        self.ax_waterfall.set_xlabel('Frequency (Hz)', fontsize=12)
        self.ax_waterfall.set_ylabel('Time (frames)', fontsize=12)
        self.ax_waterfall.set_title('Waterfall Display', fontsize=14)
        
        # Colorbar for waterfall
        self.ax_colorbar = self.fig.add_subplot(gs[1, 1])
        self.colorbar = plt.colorbar(self.im_waterfall, cax=self.ax_colorbar)
        self.colorbar.set_label('Magnitude (dB)', fontsize=12)
        
        # Info panel
        self.ax_info = self.fig.add_subplot(gs[2, :])
        self.ax_info.axis('off')
        self.info_text = self.ax_info.text(
            0.5, 0.5, '', 
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=12,
            fontfamily='monospace',
            transform=self.ax_info.transAxes
        )
        
        # Add frequency markers
        self._add_frequency_markers()
        
        plt.tight_layout()
    
    def _add_frequency_markers(self):
        """Add vertical lines at important frequencies"""
        # Common frequencies to mark
        frequencies = {
            50: 'AC Mains',
            1000: '1kHz',
            5000: '5kHz',
            10000: '10kHz',
            15000: '15kHz'
        }
        
        for freq, label in frequencies.items():
            if freq < MAX_FREQ:
                self.ax_spectrum.axvline(freq, color='yellow', alpha=0.3, linestyle='--')
                self.ax_spectrum.text(freq, -55, label, rotation=90, 
                                    verticalalignment='bottom', fontsize=8)
    
    def connect(self):
        """Connect to serial device"""
        if not self.port:
            logger.error("No port specified")
            return False
        
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=0.1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            
            # Clear buffers
            time.sleep(2)
            self.serial.reset_input_buffer()
            
            logger.info(f"Connected to {self.port} at {self.baudrate} baud")
            return True
            
        except serial.SerialException as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Close serial connection"""
        if self.serial and self.serial.is_open:
            self.serial.close()
            logger.info("Disconnected")
    
    def calculate_checksum(self, data):
        """Calculate packet checksum"""
        return sum(data) & 0xFFFF
    
    def process_packet(self, packet):
        """Process a received packet"""
        # Verify header
        header = struct.unpack('<H', packet[0:2])[0]
        if header != PACKET_HEADER:
            self.error_count += 1
            return False
        
        # Extract data
        spectrum_data = packet[2:130]
        
        # Verify checksum
        checksum = struct.unpack('<H', packet[130:132])[0]
        calc_checksum = self.calculate_checksum(spectrum_data)
        
        if checksum != calc_checksum:
            self.error_count += 1
            logger.warning(f"Checksum error: expected {calc_checksum}, got {checksum}")
            return False
        
        # Convert to numpy array and dB scale
        # Values are 0-255, where 255 = 0dB, 0 = -60dB
        self.spectrum_data = np.array(list(spectrum_data))
        self.spectrum_data = (self.spectrum_data - 255) * 60 / 255  # Convert to dB
        
        # Update waterfall
        self.waterfall_data = np.roll(self.waterfall_data, 1, axis=0)
        self.waterfall_data[0, :] = self.spectrum_data
        
        # Update statistics
        self.packet_count += 1
        self.packet_times.append(time.time())
        
        return True
    
    def read_data(self):
        """Read and process serial data"""
        if not self.serial or not self.serial.is_open:
            return
        
        # Read available data
        if self.serial.in_waiting:
            new_data = self.serial.read(self.serial.in_waiting)
            self.buffer.extend(new_data)
            
            # Process complete packets
            while len(self.buffer) >= PACKET_SIZE:
                # Look for header
                header_pos = -1
                for i in range(len(self.buffer) - 1):
                    if self.buffer[i] == 0xBB and self.buffer[i+1] == 0xAA:
                        header_pos = i
                        break
                
                if header_pos == -1:
                    # No header found, keep last byte
                    self.buffer = self.buffer[-1:]
                    break
                
                # Remove data before header
                if header_pos > 0:
                    self.buffer = self.buffer[header_pos:]
                
                # Extract packet
                if len(self.buffer) >= PACKET_SIZE:
                    packet = self.buffer[:PACKET_SIZE]
                    self.process_packet(packet)
                    self.buffer = self.buffer[PACKET_SIZE:]
    
    def calculate_packet_rate(self):
        """Calculate current packet rate"""
        if len(self.packet_times) < 2:
            return 0
        
        # Use recent packets for rate calculation
        recent_times = list(self.packet_times)[-10:]
        if len(recent_times) >= 2:
            time_span = recent_times[-1] - recent_times[0]
            if time_span > 0:
                return (len(recent_times) - 1) / time_span
        
        return 0
    
    def _update_plot(self, frame):
        """Animation update function"""
        # Read new data
        self.read_data()
        
        # Update spectrum plot
        freqs = np.linspace(0, MAX_FREQ, NUM_BINS)
        self.line_spectrum.set_data(freqs, self.spectrum_data)
        
        # Update waterfall
        self.im_waterfall.set_data(self.waterfall_data)
        
        # Update info text
        elapsed = time.time() - self.start_time
        packet_rate = self.calculate_packet_rate()
        
        # Find peak frequency
        if np.any(self.spectrum_data > -60):
            peak_bin = np.argmax(self.spectrum_data)
            peak_freq = peak_bin * (MAX_FREQ / NUM_BINS)
            peak_db = self.spectrum_data[peak_bin]
        else:
            peak_freq = 0
            peak_db = -60
        
        info = (
            f"Packets: {self.packet_count:,} | "
            f"Rate: {packet_rate:.1f} Hz | "
            f"Errors: {self.error_count} | "
            f"Runtime: {elapsed:.1f}s | "
            f"Peak: {peak_freq:.0f} Hz @ {peak_db:.1f} dB"
        )
        
        self.info_text.set_text(info)
        
        return self.line_spectrum, self.im_waterfall, self.info_text
    
    def run(self):
        """Start the visualizer"""
        # Connect to device
        if not self.connect():
            logger.error("Failed to connect")
            
            # Manual port selection
            port = input("Enter serial port manually (or Enter to exit): ")
            if port:
                self.port = port
                if not self.connect():
                    return
            else:
                return
        
        # Start animation
        try:
            self.ani = animation.FuncAnimation(
                self.fig, self._update_plot,
                interval=1000 // ANIMATION_FPS,
                blit=True,
                cache_frame_data=False
            )
            
            plt.show()
            
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        finally:
            self.disconnect()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Spectrum Analyzer Visualizer')
    parser.add_argument('--port', '-p', help='Serial port')
    parser.add_argument('--baud', '-b', type=int, default=115200, help='Baud rate')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Spectrum Analyzer Visualizer for RP2040")
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
    analyzer = SpectrumAnalyzer(port=args.port, baudrate=args.baud)
    analyzer.run()


if __name__ == "__main__":
    main()