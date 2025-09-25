#!/usr/bin/env python3
"""Test improved firmware streaming"""

import serial
import time

def main():
    print("Testing improved firmware on COM5...")
    
    try:
        ser = serial.Serial('COM5', 115200, timeout=0.1)
        time.sleep(2)
        
        # Reset and init
        print("Resetting device...")
        ser.write(b'\x03\x03')  # Double Ctrl+C
        time.sleep(0.5)
        ser.write(b'\x04')  # Ctrl+D
        time.sleep(3)
        
        # Clear any output
        while ser.in_waiting:
            ser.read(ser.in_waiting)
            
        print("Starting stream...")
        ser.write(b'stream()\r')
        time.sleep(1)
        
        print("\nReading data for 10 seconds...")
        start = time.time()
        sample_count = 0
        
        while time.time() - start < 10:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line and not any(skip in line for skip in ['>>>', 'INFO:', 'FORMAT:', 'Streaming']):
                    try:
                        parts = line.split(',')
                        if len(parts) == 3:
                            print(f"Sample {sample_count}: {line}")
                            sample_count += 1
                            if sample_count >= 10:
                                print(f"\n... (continuing, got {sample_count} samples so far)")
                                sample_count = 11  # Don't spam
                    except:
                        pass
                        
        print(f"\nTest complete. Received samples in 10 seconds.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    main()