"""Debug why we see flat line when LED shows heartbeat"""
import serial
import time

print("Debugging flat line issue...")
print("LED shows heartbeat but visualizer shows flat line\n")

ser = serial.Serial('COM5', 115200, timeout=1)
time.sleep(2)

# Reset device
ser.write(b'\x03')  # Ctrl+C
time.sleep(0.5)
ser.write(b'\x04')  # Ctrl+D  
time.sleep(2)

# Clear buffer and start streaming
ser.reset_input_buffer()
ser.write(b'stream()\r')
time.sleep(1)

print("Reading 30 lines of raw data from device:")
print("-" * 80)
print("timestamp,adc_value,heart_rate | Analysis")
print("-" * 80)

valid_samples = 0
all_values = []

for i in range(50):  # Read more lines to skip any startup messages
    if ser.in_waiting:
        try:
            line = ser.readline().decode('utf-8').strip()
            
            # Skip non-data lines
            if not line or any(skip in line for skip in ['FORMAT', 'Streaming', 'EKG', '>>>', 'Scale']):
                if line:
                    print(f"[SKIP] {line}")
                continue
                
            # Try to parse as CSV
            if ',' in line:
                parts = line.split(',')
                if len(parts) >= 3:
                    try:
                        timestamp = int(parts[0])
                        adc_value = int(parts[1])
                        hr = int(parts[2])
                        
                        all_values.append(adc_value)
                        
                        # Analyze the value
                        distance_from_center = abs(adc_value - 32768)
                        voltage_mv = (adc_value - 32768) / 32768 * 2.0
                        
                        analysis = ""
                        if distance_from_center < 10:
                            analysis = "⚠️  FLAT (too close to center)"
                        elif distance_from_center < 100:
                            analysis = "📊 Small signal"
                        else:
                            analysis = "✅ Good signal"
                        
                        print(f"{timestamp},{adc_value},{hr} | ADC={adc_value:5d}, Offset={distance_from_center:5d}, V={voltage_mv:+.3f}mV {analysis}")
                        
                        valid_samples += 1
                        if valid_samples >= 20:
                            break
                    except ValueError:
                        print(f"[ERROR] Can't parse: {line}")
        except:
            pass

ser.close()

print("-" * 80)
print("\nANALYSIS:")
if all_values:
    min_val = min(all_values)
    max_val = max(all_values)
    avg_val = sum(all_values) / len(all_values)
    
    print(f"ADC values: min={min_val}, max={max_val}, avg={avg_val:.1f}")
    print(f"Range: {max_val - min_val} (should be ~40 for proper EKG)")
    
    if max_val - min_val < 10:
        print("\n🔴 PROBLEM: Signal is completely flat!")
        print("   Possible causes:")
        print("   1. Scaling factor is way too aggressive (signal crushed to 0)")
        print("   2. Electrodes not connected properly")
        print("   3. Firmware bug in scaling calculation")
    elif max_val - min_val < 40:
        print("\n⚠️  Signal is too small, increase scaling factor")
    else:
        print("\n✅ Signal range looks good")
        
    # Check if all values are stuck at center
    if all(abs(v - 32768) < 5 for v in all_values):
        print("\n🔴 All values are stuck at 32768 (center)!")
        print("   The scaling is crushing the signal to zero!")