#!/usr/bin/env python3
"""
Test script to verify spectrum analyzer protocol
Expected packet format:
- Header: 0xAABB (2 bytes)
- Data: 128 bytes (spectrum bins)
- Checksum: 2 bytes
Total: 132 bytes per packet
"""

import serial
import serial.tools.list_ports
import struct
import time
import sys

PACKET_HEADER = 0xAABB
PACKET_SIZE = 132  # 2 header + 128 data + 2 checksum

def list_serial_ports():
    """List all available serial ports"""
    ports = serial.tools.list_ports.comports()
    print("Available serial ports:")
    for i, port in enumerate(ports):
        print(f"{i}: {port.device} - {port.description}")
    return ports

def calculate_checksum(data):
    """Calculate simple checksum"""
    return sum(data) & 0xFFFF

def main():
    # List available ports
    ports = list_serial_ports()
    
    if not ports:
        print("No serial ports found!")
        return
    
    # Select port
    if len(ports) == 1:
        port_idx = 0
        print(f"\nUsing only available port: {ports[0].device}")
    else:
        try:
            port_idx = int(input("\nSelect port number: "))
        except (ValueError, IndexError):
            print("Invalid selection")
            return
    
    port_name = ports[port_idx].device
    
    # Configure serial connection
    try:
        ser = serial.Serial(
            port=port_name,
            baudrate=115200,
            timeout=0.1,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )
        
        print(f"\nConnected to {port_name} at 115200 baud")
        print("Monitoring for spectrum analyzer packets...")
        print("Expected: Header (0xAABB) + 128 data bytes + checksum")
        print("-" * 60)
        
        # Clear any existing data
        ser.reset_input_buffer()
        
        # Statistics
        packet_count = 0
        error_count = 0
        last_packet_time = time.time()
        start_time = time.time()
        
        # Buffer for accumulating data
        buffer = bytearray()
        
        while True:
            # Read available data
            if ser.in_waiting:
                new_data = ser.read(ser.in_waiting)
                buffer.extend(new_data)
                
                # Process buffer
                while len(buffer) >= 2:
                    # Look for header
                    header_found = False
                    for i in range(len(buffer) - 1):
                        if buffer[i] == 0xBB and buffer[i+1] == 0xAA:
                            if i > 0:
                                print(f"Discarded {i} bytes before header")
                            buffer = buffer[i:]
                            header_found = True
                            break
                    
                    if not header_found:
                        # Keep last byte in case it's part of header
                        if len(buffer) > 1:
                            buffer = buffer[-1:]
                        break
                    
                    # Check if we have complete packet
                    if len(buffer) >= PACKET_SIZE:
                        packet = buffer[:PACKET_SIZE]
                        
                        # Verify header
                        header = struct.unpack('<H', packet[0:2])[0]
                        if header == PACKET_HEADER:
                            # Extract data
                            spectrum_data = list(packet[2:130])
                            
                            # Verify checksum
                            checksum = struct.unpack('<H', packet[130:132])[0]
                            calc_checksum = calculate_checksum(packet[2:130])
                            
                            if checksum == calc_checksum:
                                # Valid packet!
                                packet_count += 1
                                current_time = time.time()
                                
                                # Calculate rate
                                if packet_count > 1:
                                    packet_rate = 1.0 / (current_time - last_packet_time)
                                else:
                                    packet_rate = 0
                                
                                last_packet_time = current_time
                                
                                # Display packet info
                                print(f"\nPacket #{packet_count} received at {packet_rate:.1f} Hz")
                                
                                # Show spectrum summary
                                avg = sum(spectrum_data) / len(spectrum_data)
                                max_val = max(spectrum_data)
                                min_val = min(spectrum_data)
                                
                                print(f"Spectrum stats: avg={avg:.1f}, max={max_val}, min={min_val}")
                                
                                # Find peak frequency bin
                                peak_bin = spectrum_data.index(max_val)
                                # Assuming 0-20kHz range over 128 bins
                                peak_freq = peak_bin * (20000 / 128)
                                print(f"Peak at bin {peak_bin} (~{peak_freq:.0f} Hz)")
                                
                                # Show mini spectrum
                                print("Spectrum visualization:")
                                for i in range(0, 128, 8):
                                    bar_val = spectrum_data[i]
                                    bar_height = int(bar_val / 16)  # Scale to 0-15
                                    bar = '█' * min(bar_height, 10)
                                    freq = i * (20000 / 128)
                                    print(f"{freq:5.0f}Hz: {bar}")
                                
                            else:
                                error_count += 1
                                print(f"Checksum error! Expected {calc_checksum}, got {checksum}")
                        else:
                            error_count += 1
                            print(f"Invalid header: {header:04X}")
                        
                        # Remove processed packet
                        buffer = buffer[PACKET_SIZE:]
                    else:
                        # Need more data
                        break
            
            # Status update every 5 seconds
            if packet_count == 0 and time.time() - start_time > 5:
                elapsed = int(time.time() - start_time)
                print(f"\rNo packets received after {elapsed}s. Buffer: {len(buffer)} bytes", end='')
                
            time.sleep(0.001)
            
    except serial.SerialException as e:
        print(f"\nSerial error: {e}")
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\n\nStopped monitoring.")
        print(f"Statistics:")
        print(f"  Total packets: {packet_count}")
        print(f"  Errors: {error_count}")
        print(f"  Duration: {elapsed:.1f}s")
        if packet_count > 0:
            print(f"  Average rate: {packet_count/elapsed:.1f} packets/s")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")

if __name__ == "__main__":
    main()