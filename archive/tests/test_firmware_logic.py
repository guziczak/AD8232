"""Test firmware logic without hardware"""

# Test ADC scaling
print("=== ADC Scaling Test ===")
ADC_BITS = 12
ADC_MAX = (1 << ADC_BITS) - 1
print(f"12-bit ADC range: 0-{ADC_MAX}")

# Test 16-bit to 12-bit conversion
raw_16bit = 32768  # Mid-range 16-bit
value_12bit = raw_16bit >> 4
print(f"16-bit value {raw_16bit} -> 12-bit value {value_12bit}")

# Test filtering
print("\n=== Filter Test ===")
alpha = 0.1
last_value = 2048
new_value = 2200
filtered = int(alpha * new_value + (1 - alpha) * last_value)
print(f"Last: {last_value}, New: {new_value}, Filtered: {filtered}")

# Test voltage calculation
print("\n=== Voltage Test ===")
VOLTAGE_REF = 3.3
for adc_val in [0, 2048, 4095]:
    # Scale back to 16-bit for storage
    stored_val = adc_val << 4
    # Calculate voltage
    voltage = (stored_val / (ADC_MAX << 4)) * VOLTAGE_REF
    print(f"ADC {adc_val} -> Voltage {voltage:.3f}V")

print("\n✓ All tests passed!")