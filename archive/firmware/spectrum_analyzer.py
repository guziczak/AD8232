"""
Spectrum Analyzer Firmware for RP2040
Sends spectrum data over USB serial in binary packet format

Packet format:
- Header: 0xAABB (2 bytes, little-endian)
- Data: 128 bytes (spectrum magnitude for each bin)
- Checksum: 2 bytes (sum of data bytes & 0xFFFF)
Total: 132 bytes per packet
"""

import machine
import array
import struct
import time
import gc
import math

# Constants
SAMPLE_RATE = 40000  # 40 kHz sampling rate
FFT_SIZE = 256       # FFT size (must be power of 2)
NUM_BINS = 128       # Number of spectrum bins to send
PACKET_HEADER = 0xAABB
UPDATE_RATE = 30     # Target updates per second

# Hardware setup
ADC_PIN = 28         # GP28/ADC2
LED_PIN = 16         # Onboard LED for RP2040-Zero

# Initialize hardware
adc = machine.ADC(ADC_PIN)
led = machine.Pin(LED_PIN, machine.Pin.OUT)

# Pre-allocate buffers
samples = array.array('h', [0] * FFT_SIZE)  # 16-bit signed samples
real = array.array('f', [0.0] * FFT_SIZE)   # Real part for FFT
imag = array.array('f', [0.0] * FFT_SIZE)   # Imaginary part for FFT
magnitude = array.array('B', [0] * NUM_BINS)  # 8-bit magnitude output

# Timing variables
last_update = 0
update_interval = 1000000 // UPDATE_RATE  # microseconds

def apply_window(samples, real_out):
    """Apply Hann window to samples and convert to float"""
    n = len(samples)
    for i in range(n):
        # Hann window: 0.5 - 0.5 * cos(2*pi*i/(n-1))
        window = 0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1))
        # Convert from ADC range to signed, apply window
        real_out[i] = (samples[i] - 32768) * window / 32768.0

def bit_reverse(x, bits):
    """Bit reverse for FFT"""
    result = 0
    for i in range(bits):
        if x & (1 << i):
            result |= 1 << (bits - 1 - i)
    return result

def fft_radix2(real, imag):
    """In-place radix-2 FFT"""
    n = len(real)
    levels = int(math.log2(n))
    
    # Bit reversal
    for i in range(n):
        j = bit_reverse(i, levels)
        if i < j:
            real[i], real[j] = real[j], real[i]
            imag[i], imag[j] = imag[j], imag[i]
    
    # FFT computation
    for level in range(1, levels + 1):
        m = 1 << level
        m2 = m // 2
        w_real = math.cos(-2 * math.pi / m)
        w_imag = math.sin(-2 * math.pi / m)
        
        for j in range(0, n, m):
            wr = 1.0
            wi = 0.0
            
            for i in range(m2):
                idx1 = j + i
                idx2 = idx1 + m2
                
                # Butterfly computation
                tr = wr * real[idx2] - wi * imag[idx2]
                ti = wr * imag[idx2] + wi * real[idx2]
                
                real[idx2] = real[idx1] - tr
                imag[idx2] = imag[idx1] - ti
                real[idx1] = real[idx1] + tr
                imag[idx1] = imag[idx1] + ti
                
                # Update twiddle factor
                wr_temp = wr
                wr = wr * w_real - wi * w_imag
                wi = wr_temp * w_imag + wi * w_real

def compute_magnitude(real, imag, magnitude):
    """Compute magnitude spectrum and scale to 8-bit"""
    max_mag = 0.0
    
    # First pass: find maximum magnitude (skip DC bin)
    for i in range(1, NUM_BINS):
        mag = math.sqrt(real[i] * real[i] + imag[i] * imag[i])
        if mag > max_mag:
            max_mag = mag
    
    # Second pass: scale to 8-bit with logarithmic scaling
    if max_mag > 0:
        for i in range(NUM_BINS):
            mag = math.sqrt(real[i] * real[i] + imag[i] * imag[i])
            # Logarithmic scaling for better visualization
            if mag > 0:
                db = 20 * math.log10(mag / max_mag)  # Convert to dB
                # Map -60dB to 0dB range to 0-255
                scaled = int((db + 60) * 255 / 60)
                magnitude[i] = max(0, min(255, scaled))
            else:
                magnitude[i] = 0
    else:
        # No signal
        for i in range(NUM_BINS):
            magnitude[i] = 0

def calculate_checksum(data):
    """Calculate simple checksum"""
    return sum(data) & 0xFFFF

def send_packet(magnitude):
    """Send spectrum data packet over serial"""
    # Build packet
    packet = bytearray(132)
    
    # Header (little-endian)
    struct.pack_into('<H', packet, 0, PACKET_HEADER)
    
    # Data
    packet[2:130] = magnitude
    
    # Checksum
    checksum = calculate_checksum(magnitude)
    struct.pack_into('<H', packet, 130, checksum)
    
    # Send packet
    sys.stdout.buffer.write(packet)

def collect_samples():
    """Collect ADC samples with precise timing"""
    sample_interval = 1000000 // SAMPLE_RATE  # microseconds
    start_time = time.ticks_us()
    
    for i in range(FFT_SIZE):
        # Wait for next sample time
        target_time = start_time + i * sample_interval
        while time.ticks_diff(target_time, time.ticks_us()) > 0:
            pass
        
        # Read ADC
        samples[i] = adc.read_u16()

def main():
    """Main spectrum analyzer loop"""
    print("Spectrum Analyzer Starting...")
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"FFT size: {FFT_SIZE}")
    print(f"Frequency resolution: {SAMPLE_RATE/FFT_SIZE:.1f} Hz/bin")
    print(f"Max frequency: {SAMPLE_RATE/2} Hz")
    
    # Heartbeat counter for LED
    heartbeat = 0
    
    while True:
        try:
            current_time = time.ticks_us()
            
            # Check if it's time for an update
            if time.ticks_diff(current_time, last_update) >= update_interval:
                last_update = current_time
                
                # Collect samples
                collect_samples()
                
                # Apply window and prepare for FFT
                apply_window(samples, real)
                
                # Clear imaginary part
                for i in range(FFT_SIZE):
                    imag[i] = 0.0
                
                # Compute FFT
                fft_radix2(real, imag)
                
                # Compute magnitude spectrum
                compute_magnitude(real, imag, magnitude)
                
                # Send packet
                send_packet(magnitude)
                
                # Update heartbeat LED
                heartbeat += 1
                if heartbeat % UPDATE_RATE == 0:  # Toggle every second
                    led.toggle()
                
                # Garbage collection
                if heartbeat % (UPDATE_RATE * 10) == 0:  # Every 10 seconds
                    gc.collect()
                
        except KeyboardInterrupt:
            print("\nStopping spectrum analyzer...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(0.1)
    
    # Cleanup
    led.off()
    print("Spectrum analyzer stopped")

# Run if this is the main file
if __name__ == "__main__":
    main()