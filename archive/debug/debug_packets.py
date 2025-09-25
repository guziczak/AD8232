#!/usr/bin/env python3
"""
Debug script to parse and display spectrum analyzer packets
"""

import serial
import serial.tools.list_ports
import struct
import time
import sys

PACKET_HEADER = 0xAABB
PACKET_SIZE = 132  # 2 header + 2 checksum + 128 data bytes

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

def parse_packet(data):
    """Parse a complete packet"""
    if len(data) < PACKET_SIZE:
        return None
    
    # Check header
    header = struct.unpack('<H', data[0:2])[0]
    if header != PACKET_HEADER:
        return None
    
    # Extract spectrum data
    spectrum_data = list(data[2:130])
    
    # Check checksum
    checksum = struct.unpack('<H', data[130:132])[0]
    calc_checksum = calculate_checksum(data[2:130])
    
    if checksum != calc_checksum:
        print(f"Checksum mismatch: expected {calc_checksum}, got {checksum}")
        return None
    
    return spectrum_data

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
        print("Looking for spectrum packets (Ctrl+C to stop)...")
        print("-" * 50)
        
        # Clear any existing data
        ser.reset_input_buffer()
        
        buffer = bytearray()
        packet_count = 0
        last_packet_time = time.time()
        
        # Monitor serial data
        while True:
            # Read available data
            if ser.in_waiting:
                data = ser.read(ser.in_waiting)
                buffer.extend(data)
                
                # Look for packet header
                while len(buffer) >= PACKET_SIZE:
                    # Find header
                    header_pos = -1
                    for i in range(len(buffer) - 1):
                        if buffer[i] == 0xBB and buffer[i+1] == 0xAA:
                            header_pos = i
                            break
                    
                    if header_pos == -1:
                        # No header found, keep last byte in case it's part of header
                        buffer = buffer[-1:]
                        break
                    
                    # Remove data before header
                    if header_pos > 0:
                        print(f"Discarded {header_pos} bytes before header")
                        buffer = buffer[header_pos:]
                    
                    # Check if we have a complete packet
                    if len(buffer) >= PACKET_SIZE:
                        packet_data = buffer[:PACKET_SIZE]
                        spectrum = parse_packet(packet_data)
                        
                        if spectrum:
                            packet_count += 1
                            current_time = time.time()
                            packet_rate = 1.0 / (current_time - last_packet_time) if last_packet_time else 0
                            last_packet_time = current_time
                            
                            # Display packet info
                            print(f"\nPacket #{packet_count} ({packet_rate:.1f} Hz)")
                            
                            # Show spectrum preview (first 16 values)
                            preview = ' '.join(f'{v:3d}' for v in spectrum[:16])
                            print(f"Data preview: {preview}...")
                            
                            # Calculate and show statistics
                            avg = sum(spectrum) / len(spectrum)
                            max_val = max(spectrum)
                            min_val = min(spectrum)
                            print(f"Stats: avg={avg:.1f}, max={max_val}, min={min_val}")
                            
                            # Show simple ASCII visualization
                            print("Spectrum: ", end='')
                            for i in range(0, len(spectrum), 4):  # Show every 4th bin
                                height = int(spectrum[i] / 16)  # Scale to 0-15
                                bar = '█' * min(height, 8)  # Cap at 8 chars
                                print(bar.ljust(8), end='')
                            print()
                        else:
                            print("Invalid packet detected")
                        
                        # Remove processed packet
                        buffer = buffer[PACKET_SIZE:]
                    else:
                        break
            
            # Show status every 5 seconds if no packets
            if packet_count == 0 and time.time() - last_packet_time > 5:
                print(".", end='', flush=True)
                last_packet_time = time.time()
            
            time.sleep(0.001)
            
    except serial.SerialException as e:
        print(f"Serial error: {e}")
    except KeyboardInterrupt:
        print(f"\n\nStopped monitoring. Received {packet_count} packets")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("Serial port closed")

if __name__ == "__main__":
    main()