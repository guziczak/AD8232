#!/usr/bin/env python3
"""Debug serial communication from RP2040"""

import serial
import serial.tools.list_ports
import time

def find_device():
    """Find RP2040 device."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if any(id in port.description.upper() for id in ['RP2040', 'PICO', 'CDC']):
            return port.device
    return None

def main():
    # Try default COM5 first
    port = "COM5"
    
    # Check if COM5 exists
    available_ports = [p.device for p in serial.tools.list_ports.comports()]
    
    if port not in available_ports:
        print(f"COM5 not found. Available ports: {', '.join(available_ports)}")
        port = find_device()
        
        if not port and available_ports:
            port = input(f"Enter COM port (available: {', '.join(available_ports)}): ")
            if not port:
                return
        
    print(f"Connecting to {port}...")
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)  # Wait for connection
        
        print("Reading data (press Ctrl+C to stop):")
        print("-" * 50)
        
        # Clear buffer
        ser.reset_input_buffer()
        
        # Send commands to start streaming
        print("Sending init commands...")
        ser.write(b'\x03')  # Ctrl+C
        time.sleep(0.5)
        ser.write(b'\x04')  # Ctrl+D  
        time.sleep(2)
        
        print("Initializing EKG...")
        ser.write(b'init()\r')
        time.sleep(1)
        
        print("Starting sampling...")
        ser.write(b'start()\r')
        time.sleep(1)
        
        print("Starting stream...")
        ser.write(b'stream()\r')
        time.sleep(1)
        
        print("\nWaiting for data...")
        no_data_counter = 0
        
        # Read data
        while True:
            if ser.in_waiting:
                data = ser.readline()
                try:
                    decoded = data.decode('utf-8').strip()
                    if decoded:  # Only print non-empty lines
                        print(f"TEXT: {decoded}")
                        no_data_counter = 0
                except:
                    print(f"RAW: {data.hex()}")
                    no_data_counter = 0
            else:
                time.sleep(0.1)
                no_data_counter += 1
                if no_data_counter == 50:  # 5 seconds no data
                    print("\nNo data received for 5 seconds. Trying manual mode...")
                    print("Type commands manually (init, start, stream):")
                    break
                    
    except KeyboardInterrupt:
        print("\nStopped")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'ser' in locals():
            ser.close()

if __name__ == "__main__":
    main()