"""Test what the scaled firmware is actually outputting"""
import serial
import time

ser = serial.Serial('COM5', 115200, timeout=1)
print("Connecting...")
time.sleep(2)

# Reset and start streaming
ser.write(b'\x03')  # Ctrl+C
time.sleep(0.5)
ser.reset_input_buffer()

print("\nReading raw output from scaled firmware:")
print("-" * 60)

# Read raw lines
for i in range(30):
    if ser.in_waiting:
        line = ser.readline().decode('utf-8').strip()
        if line and ',' in line:
            try:
                parts = line.split(',')
                if len(parts) >= 3:
                    timestamp = parts[0]
                    adc_value = int(parts[1])
                    hr = parts[2]
                    
                    # Calculate what voltage this would be
                    centered = adc_value - 32768
                    voltage_mv = (centered / 32768) * 2.0
                    
                    print(f"ADC: {adc_value:5d} | Centered: {centered:6d} | Voltage: {voltage_mv:7.3f} mV | HR: {hr}")
            except:
                print(f"Raw: {line}")
        else:
            print(f"Other: {line}")

ser.close()
print("-" * 60)
print("\nIf all ADC values are around 32768, the signal is being over-scaled!")