"""
EKG RAW TEST - Bez filtrowania, bez auto-gain
Testowy firmware do diagnozy problemu
"""

import machine
import time

# Konfiguracja pinów
ADC_PIN = 29
LO_PLUS_PIN = 27
LO_MINUS_PIN = 28
LED_PIN = 16

# Inicjalizacja
print("\n=== EKG RAW TEST MODE ===")
print("Bez filtrowania, bez auto-gain")
print("Wysyła surowe dane ADC\n")

# Hardware
adc = machine.ADC(ADC_PIN)
lo_plus = machine.Pin(LO_PLUS_PIN, machine.Pin.IN)
lo_minus = machine.Pin(LO_MINUS_PIN, machine.Pin.IN)

try:
    led = machine.Pin(LED_PIN, machine.Pin.OUT)
except:
    led = None
    print("LED niedostępny")

def test_continuous():
    """Test ciągły - wysyła surowe wartości"""
    print("Test ciągły - Ctrl+C aby zatrzymać")
    print("Format: timestamp_us,raw_adc,lo_plus,lo_minus")
    
    counter = 0
    while True:
        # Odczyt surowy
        raw_value = adc.read_u16()
        timestamp = time.ticks_us()
        lo_p = lo_plus.value()
        lo_m = lo_minus.value()
        
        # Wyślij dane
        print(f"{timestamp},{raw_value},{lo_p},{lo_m}")
        
        # LED co 50 próbek
        counter += 1
        if led and counter % 50 == 0:
            led.toggle()
        
        # Delay dla ~250 Hz
        time.sleep_ms(4)

def test_stats(duration=5):
    """Test ze statystykami"""
    print(f"Zbieranie danych przez {duration} sekund...")
    
    values = []
    start = time.ticks_ms()
    
    while time.ticks_diff(time.ticks_ms(), start) < duration * 1000:
        raw = adc.read_u16()
        values.append(raw)
        time.sleep_ms(4)
    
    # Statystyki
    min_val = min(values)
    max_val = max(values)
    avg_val = sum(values) / len(values)
    
    print(f"\n=== STATYSTYKI ({len(values)} próbek) ===")
    print(f"MIN: {min_val} ({min_val/65535*3.3:.3f}V)")
    print(f"MAX: {max_val} ({max_val/65535*3.3:.3f}V)")
    print(f"AVG: {avg_val:.0f} ({avg_val/65535*3.3:.3f}V)")
    print(f"ZAKRES: {max_val - min_val}")
    
    # Sprawdź elektrody
    print(f"\nSTAN ELEKTROD:")
    print(f"LO+ (pin {LO_PLUS_PIN}): {'ODŁĄCZONA!' if lo_plus.value() else 'OK'}")
    print(f"LO- (pin {LO_MINUS_PIN}): {'ODŁĄCZONA!' if lo_minus.value() else 'OK'}")
    
    return values

def test_noise():
    """Test szumu - dotknij palcem OUTPUT"""
    print("\n=== TEST SZUMU ===")
    print("1. Odłącz elektrody")
    print("2. Dotknij palcem pinu OUTPUT")
    print("3. Powinieneś zobaczyć szum 50Hz\n")
    
    input("Naciśnij Enter gdy będziesz gotowy...")
    
    print("Rejestrowanie przez 2 sekundy...")
    values = []
    timestamps = []
    
    start = time.ticks_us()
    while time.ticks_diff(time.ticks_us(), start) < 2_000_000:
        values.append(adc.read_u16())
        timestamps.append(time.ticks_us())
    
    # Analiza
    changes = []
    for i in range(1, len(values)):
        changes.append(abs(values[i] - values[i-1]))
    
    avg_change = sum(changes) / len(changes)
    max_change = max(changes)
    
    print(f"\nZebrano {len(values)} próbek")
    print(f"Średnia zmiana: {avg_change:.0f}")
    print(f"Max zmiana: {max_change}")
    
    if max_change > 5000:
        print("✅ Wykryto duże zmiany - układ reaguje na dotyk")
    else:
        print("❌ Brak reakcji - sprawdź połączenia!")

def help():
    """Pokaż dostępne komendy"""
    print("""
DOSTĘPNE TESTY:
  test_continuous() - Ciągły stream surowych danych
  test_stats(5)     - Statystyki 5-sekundowe
  test_noise()      - Test szumu 50Hz
  check_pins()      - Sprawdź połączenia
  help()            - Ta pomoc
    """)

def check_pins():
    """Sprawdź wszystkie połączenia"""
    print("\n=== SPRAWDZANIE POŁĄCZEŃ ===")
    
    # Test ADC
    print(f"\n1. TEST ADC (pin GP{ADC_PIN}):")
    for i in range(5):
        val = adc.read_u16()
        print(f"   Odczyt {i+1}: {val} ({val/65535*3.3:.3f}V)")
        time.sleep_ms(200)
    
    # Test LO+/-
    print(f"\n2. TEST ELEKTROD:")
    for i in range(5):
        print(f"   LO+: {'HIGH (brak elektrody)' if lo_plus.value() else 'LOW (OK)'}")
        print(f"   LO-: {'HIGH (brak elektrody)' if lo_minus.value() else 'LOW (OK)'}")
        print()
        time.sleep_ms(500)
    
    # Test LED
    print(f"\n3. TEST LED:")
    if led:
        for i in range(5):
            led.toggle()
            print(f"   LED {'ON' if led.value() else 'OFF'}")
            time.sleep_ms(200)
    else:
        print("   LED niedostępny")

# Automatyczne sprawdzenie przy starcie
print("Uruchamianie testów...")
check_pins()
print("\nWpisz help() aby zobaczyć dostępne komendy")
print("Lub test_continuous() aby rozpocząć stream danych")