"""Quick serial test to see raw data from scaled firmware"""
import serial
import time

# Connect to device
ser = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2)

# Send commands
ser.write(b'\x03')  # Ctrl+C
time.sleep(0.5)
ser.write(b'\x04')  # Ctrl+D
time.sleep(2)

ser.reset_input_buffer()

# Start streaming
ser.write(b'stream()\r')
time.sleep(1)

print("Reading 20 lines of raw data:")
print("-" * 60)

for i in range(20):
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        print(line)

ser.close()
print("-" * 60)
print("Test complete")