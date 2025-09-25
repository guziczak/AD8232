#!/usr/bin/env python3
"""Direct communication test"""

import serial
import time

print("Direct test on COM5...")

ser = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2)

# Clear everything
ser.write(b'\x03\x03')  # Ctrl+C twice
time.sleep(0.5)

# Read and print everything for 5 seconds
print("\n--- RAW OUTPUT FOR 5 SECONDS ---")
start = time.time()
while time.time() - start < 5:
    if ser.in_waiting:
        data = ser.read(ser.in_waiting)
        print(data.decode('utf-8', errors='ignore'), end='')

print("\n\n--- SENDING COMMANDS ---")

# Now try commands one by one
commands = [
    (b'\x04', "Soft reset", 3),
    (b'init()\r', "Init", 2),
    (b'start()\r', "Start sampling", 2),
    (b'stream()\r', "Start streaming", 2)
]

for cmd, desc, wait in commands:
    print(f"\nSending: {desc}")
    ser.write(cmd)
    time.sleep(wait)
    
    # Read response
    while ser.in_waiting:
        data = ser.read(ser.in_waiting)
        print(data.decode('utf-8', errors='ignore'), end='')

print("\n\n--- READING STREAM FOR 10 SECONDS ---")
start = time.time()
line_count = 0
while time.time() - start < 10:
    if ser.in_waiting:
        line = ser.readline()
        try:
            decoded = line.decode('utf-8').strip()
            if decoded:
                print(f"{line_count}: {decoded}")
                line_count += 1
        except:
            print(f"{line_count}: RAW={line.hex()}")
            line_count += 1

print(f"\nGot {line_count} lines in 10 seconds")
ser.close()