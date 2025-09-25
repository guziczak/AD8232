#!/usr/bin/env python3
"""
Simple test script to check if data is coming from RP2040
"""

import serial
import time

# Connect to device
port = input("Enter COM port (e.g. COM5): ")
ser = serial.Serial(port, 115200, timeout=1)
time.sleep(2)  # Wait for connection

# Enter REPL mode
print("Entering REPL mode...")
ser.write(b'\x03')  # Ctrl+C to interrupt any running code
time.sleep(0.5)
ser.write(b'\x04')  # Ctrl+D to soft reset
time.sleep(2)

# Clear buffer
ser.reset_input_buffer()

# Now send commands in REPL
print("Sending commands...")
ser.write(b'init()\r')
time.sleep(1)
response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
print("Init response:", response)

ser.write(b'start()\r')
time.sleep(1)
response = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
print("Start response:", response)

ser.write(b'stream()\r')
time.sleep(1)

print("\nReading data for 10 seconds...")
print("Expected format: timestamp,adc_value,heart_rate")
print("-" * 50)

# Wait a bit for data to start flowing
time.sleep(2)

# Read for 10 seconds
start_time = time.time()
line_count = 0
data_lines = 0

while time.time() - start_time < 10:
    if ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(line)
            line_count += 1
            
            # Count actual data lines (not info messages)
            if ',' in line and not any(skip in line for skip in ['FORMAT:', 'INFO:', '>>>']):
                data_lines += 1
            
            # Stop after 20 lines to not flood console
            if line_count >= 20:
                print("... (more data available)")
                break

print("-" * 50)
print(f"Received {line_count} lines total, {data_lines} data lines")

# Stop streaming
ser.write(b'stop\n')
time.sleep(0.1)
ser.close()

print("\nDone!")