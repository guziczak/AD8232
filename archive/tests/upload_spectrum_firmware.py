#!/usr/bin/env python3
"""
Helper script to upload spectrum analyzer firmware to RP2040
"""

import os
import sys
import time
import serial
import serial.tools.list_ports

def find_rp2040():
    """Find RP2040 device"""
    print("Searching for RP2040...")
    ports = serial.tools.list_ports.comports()
    
    for port in ports:
        if any(id in port.description.upper() for id in ['RP2040', 'PICO', 'CDC']):
            print(f"Found: {port.device} - {port.description}")
            return port.device
    
    if ports:
        print("\nAvailable ports:")
        for i, port in enumerate(ports):
            print(f"{i}: {port.device} - {port.description}")
        
        try:
            idx = int(input("\nSelect port number: "))
            return ports[idx].device
        except:
            pass
    
    return None

def upload_file(port, filename):
    """Upload a file to RP2040 using REPL"""
    print(f"\nUploading {filename} to {port}...")
    
    try:
        # Connect
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        
        # Enter raw REPL mode
        ser.write(b'\x03')  # Ctrl+C
        time.sleep(0.5)
        ser.write(b'\x01')  # Ctrl+A (raw REPL)
        time.sleep(0.5)
        
        # Read and discard any pending data
        ser.read(ser.in_waiting)
        
        # Read file
        with open(filename, 'rb') as f:
            content = f.read()
        
        # Create file on device
        target = os.path.basename(filename)
        if target.endswith('_test.py'):
            target = 'main.py'  # Upload test as main.py
        elif target == 'spectrum_analyzer.py':
            target = 'main.py'  # Upload as main.py
        
        cmd = f"f = open('{target}', 'wb')\r\n"
        ser.write(cmd.encode())
        time.sleep(0.1)
        
        # Write content in chunks
        chunk_size = 256
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i+chunk_size]
            # Escape any special bytes
            escaped = repr(chunk)[2:-1].replace("'", "\\'")
            cmd = f"f.write(b'{escaped}')\r\n"
            ser.write(cmd.encode())
            time.sleep(0.05)
            
            # Progress
            progress = (i + len(chunk)) * 100 // len(content)
            print(f"\rProgress: {progress}%", end='')
        
        # Close file
        ser.write(b"f.close()\r\n")
        time.sleep(0.1)
        
        # Exit raw REPL
        ser.write(b'\x04')  # Ctrl+D
        time.sleep(0.5)
        
        # Soft reset to run new code
        ser.write(b'\x04')  # Ctrl+D
        time.sleep(2)
        
        print(f"\n\nUpload complete! File saved as '{target}'")
        print("Device is now running the new firmware.")
        
        ser.close()
        return True
        
    except Exception as e:
        print(f"\nError: {e}")
        return False

def main():
    """Main function"""
    print("=" * 50)
    print("RP2040 Spectrum Analyzer Firmware Uploader")
    print("=" * 50)
    
    # Find device
    port = find_rp2040()
    if not port:
        print("No RP2040 found!")
        return
    
    # Select firmware
    print("\nAvailable firmware:")
    print("1. spectrum_test.py    - Test patterns (recommended for first test)")
    print("2. spectrum_analyzer.py - Full spectrum analyzer")
    
    try:
        choice = int(input("\nSelect firmware (1 or 2): "))
        if choice == 1:
            firmware = "firmware/spectrum_test.py"
        elif choice == 2:
            firmware = "firmware/spectrum_analyzer.py"
        else:
            print("Invalid choice")
            return
    except:
        print("Invalid input")
        return
    
    # Check if file exists
    if not os.path.exists(firmware):
        print(f"File not found: {firmware}")
        return
    
    # Upload
    if upload_file(port, firmware):
        print("\nNext steps:")
        print("1. The RP2040 should now be running the spectrum analyzer firmware")
        print("2. The onboard LED should be blinking as a heartbeat")
        print("3. Run one of these tools to verify:")
        print("   - ./debug_serial.py      - See raw serial data")
        print("   - ./debug_packets.py     - See parsed packets") 
        print("   - ./test_spectrum_protocol.py - Detailed packet analysis")
        print("   - ./pc_software/spectrum_visualizer.py - Full visualizer")

if __name__ == "__main__":
    main()