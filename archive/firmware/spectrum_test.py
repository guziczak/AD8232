"""
Test Spectrum Analyzer Firmware for RP2040
Generates test patterns for verifying communication

Test patterns:
1. Ramp: 0-255 gradient
2. Single peak: Peak that moves across spectrum
3. Random: Random noise
"""

import machine
import array
import struct
import time
import sys
import random

# Constants
PACKET_HEADER = 0xAABB
NUM_BINS = 128
UPDATE_RATE = 10  # 10 Hz for easier debugging

# Hardware setup
LED_PIN = 16  # Onboard LED for RP2040-Zero
led = machine.Pin(LED_PIN, machine.Pin.OUT)

# Pre-allocate magnitude buffer
magnitude = array.array('B', [0] * NUM_BINS)

# Pattern state
pattern_type = 0  # 0=ramp, 1=peak, 2=random
peak_position = 0
pattern_counter = 0

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

def generate_ramp_pattern():
    """Generate a ramp pattern (0-255)"""
    for i in range(NUM_BINS):
        magnitude[i] = (i * 255) // (NUM_BINS - 1)

def generate_peak_pattern(position):
    """Generate a single peak that moves"""
    # Clear spectrum
    for i in range(NUM_BINS):
        magnitude[i] = 10  # Noise floor
    
    # Add peak with Gaussian shape
    for i in range(NUM_BINS):
        distance = abs(i - position)
        if distance < 10:
            # Gaussian-like peak
            level = int(255 * math.exp(-distance * distance / 10))
            magnitude[i] = min(255, magnitude[i] + level)

def generate_random_pattern():
    """Generate random noise pattern"""
    for i in range(NUM_BINS):
        magnitude[i] = random.randint(0, 255)

def generate_sine_pattern(freq_bin):
    """Generate pattern with sine wave at specific frequency bin"""
    # Clear spectrum
    for i in range(NUM_BINS):
        magnitude[i] = 5  # Noise floor
    
    # Add main frequency
    if 0 <= freq_bin < NUM_BINS:
        magnitude[freq_bin] = 200
        
        # Add some harmonics
        if freq_bin * 2 < NUM_BINS:
            magnitude[freq_bin * 2] = 100  # 2nd harmonic
        if freq_bin * 3 < NUM_BINS:
            magnitude[freq_bin * 3] = 50   # 3rd harmonic

def main():
    """Main test loop"""
    global pattern_type, peak_position, pattern_counter
    
    print("Spectrum Analyzer Test Mode")
    print(f"Sending {NUM_BINS}-bin spectrum at {UPDATE_RATE} Hz")
    print("Patterns: Ramp -> Moving Peak -> Random -> Sine")
    print("Binary packets: Header(2) + Data(128) + Checksum(2) = 132 bytes")
    
    # Import math for peak pattern
    import math
    
    update_counter = 0
    last_update = time.ticks_ms()
    
    while True:
        try:
            current_time = time.ticks_ms()
            
            # Update at specified rate
            if time.ticks_diff(current_time, last_update) >= (1000 // UPDATE_RATE):
                last_update = current_time
                
                # Generate pattern based on current mode
                if pattern_type == 0:
                    # Ramp pattern
                    generate_ramp_pattern()
                elif pattern_type == 1:
                    # Moving peak
                    generate_peak_pattern(peak_position)
                    peak_position = (peak_position + 1) % NUM_BINS
                elif pattern_type == 2:
                    # Random pattern
                    generate_random_pattern()
                elif pattern_type == 3:
                    # Sine at specific frequency
                    # 1kHz if sample rate is 40kHz and 128 bins = bin 6
                    generate_sine_pattern(6)
                
                # Send packet
                send_packet(magnitude)
                
                # Update counters
                update_counter += 1
                pattern_counter += 1
                
                # Change pattern every 5 seconds
                if pattern_counter >= UPDATE_RATE * 5:
                    pattern_counter = 0
                    pattern_type = (pattern_type + 1) % 4
                    peak_position = 0
                    
                    # Print pattern change
                    patterns = ["Ramp", "Moving Peak", "Random", "1kHz Sine"]
                    print(f"\nSwitched to pattern: {patterns[pattern_type]}")
                
                # Heartbeat LED
                if update_counter % UPDATE_RATE == 0:
                    led.toggle()
                
                # Status every second
                if update_counter % UPDATE_RATE == 0:
                    print(".", end="")
                
        except KeyboardInterrupt:
            print("\nTest stopped")
            break
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(0.1)
    
    # Cleanup
    led.off()

# Run if this is the main file
if __name__ == "__main__":
    main()