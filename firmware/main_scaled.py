"""
EKG Data Acquisition - Properly scaled version
Scales the signal by 1/475 to get proper 2mV EKG range
"""

import machine
import time
import gc

# Constants
SAMPLE_RATE = 250  # Hz
PIN_ADC = 29
PIN_LO_PLUS = 27
PIN_LO_MINUS = 28
PIN_LED = 16

# Scaling factor - optimized for ±1-2 mV range
# Current signal is ±0.25 mV, need it 4-8x larger
# So reducing current scaling by factor of 6
SCALE_FACTOR = 1.8  # Minimal scaling to get proper amplitude

class EKGSensor:
    def __init__(self):
        # Hardware
        self.adc = machine.ADC(PIN_ADC)
        self.lo_plus = machine.Pin(PIN_LO_PLUS, machine.Pin.IN)
        self.lo_minus = machine.Pin(PIN_LO_MINUS, machine.Pin.IN)
        try:
            self.led = machine.Pin(PIN_LED, machine.Pin.OUT)
        except:
            self.led = None
            
        # State
        self.sampling_active = False
        self.streaming = False
        self.timer = machine.Timer()
        self.sample_count = 0
        self.current_hr = 0
        
        # Simple filtering
        self.last_value = 32768
        self.baseline = 32768
        
    def _sample_callback(self, timer):
        """Timer callback - just sets flag"""
        self.sample_ready = True
        
    def _scale_value(self, raw_value):
        """Scale the raw ADC value to proper EKG range"""
        # Remove DC offset (center around 0)
        centered = raw_value - 32768
        
        # Apply scaling factor
        scaled = int(centered * SCALE_FACTOR)
        
        # Re-center at 32768 and clamp to valid range
        final_value = scaled + 32768
        return max(0, min(65535, final_value))
        
    def start_sampling(self):
        """Start sampling"""
        if self.sampling_active:
            print("Already sampling")
            return
            
        self.sampling_active = True
        self.sample_count = 0
        self.sample_ready = False
        
        # Start timer
        self.timer.init(
            freq=SAMPLE_RATE,
            mode=machine.Timer.PERIODIC,
            callback=self._sample_callback
        )
        
        print(f"Sampling started at {SAMPLE_RATE} Hz (scaled by {SCALE_FACTOR})")
        
    def stop_sampling(self):
        """Stop sampling"""
        if not self.sampling_active:
            return
            
        self.timer.deinit()
        self.sampling_active = False
        self.streaming = False
        
        if self.led:
            self.led.off()
            
        print(f"Sampling stopped. Collected {self.sample_count} samples")
        
    def stream_data(self):
        """Simple streaming with proper scaling"""
        print("Streaming mode - press Ctrl+C to exit")
        print("FORMAT:timestamp_us,adc_value,heart_rate")
        print(f"Signal scaled by {SCALE_FACTOR} for proper EKG range")
        
        if not self.sampling_active:
            self.start_sampling()
            time.sleep_ms(100)
            
        self.streaming = True
        self.sample_ready = False
        
        try:
            while self.streaming:
                if self.sample_ready:
                    self.sample_ready = False
                    
                    # Read ADC
                    if self.lo_plus.value() or self.lo_minus.value():
                        value = 32768  # Center value when disconnected
                    else:
                        raw_value = self.adc.read_u16()
                        
                        # Apply scaling to get proper EKG range
                        value = self._scale_value(raw_value)
                        
                        # Simple baseline tracking (on scaled value)
                        self.baseline = int(0.995 * self.baseline + 0.005 * value)
                        
                        # Simple smoothing
                        value = int(0.7 * value + 0.3 * self.last_value)
                        self.last_value = value
                    
                    # Get timestamp
                    timestamp = time.ticks_us()
                    
                    # Send data immediately
                    print(f"{timestamp},{value},{self.current_hr}")
                    
                    self.sample_count += 1
                    
                    # Blink LED
                    if self.led and self.sample_count % 50 == 0:
                        self.led.toggle()
                        
                # Small yield to prevent watchdog
                time.sleep_us(100)
                
        except KeyboardInterrupt:
            pass
        finally:
            self.streaming = False
            print("\nStreaming stopped")

# Global instance
ekg = None

def init():
    global ekg
    ekg = EKGSensor()
    print("EKG system initialized (Scaled version)")
    print(f"Scale factor: {SCALE_FACTOR} (reduces signal by {int(1/SCALE_FACTOR)}x)")
    
def start():
    if not ekg:
        init()
    ekg.start_sampling()
    
def stop():
    if ekg:
        ekg.stop_sampling()
        
def stream():
    if not ekg:
        init()
    ekg.stream_data()
    
def test():
    """Quick test with scaling info"""
    print("Testing ADC with scaling...")
    if not ekg:
        init()
        
    print("\nRaw values:")
    for i in range(5):
        raw = ekg.adc.read_u16()
        scaled = ekg._scale_value(raw)
        print(f"{i}: Raw={raw} ({raw/65535*3.3:.3f}V) -> Scaled={scaled}")
        time.sleep_ms(100)
        
    print("\nElectrodes:")
    print(f"LO+: {'Disconnected' if ekg.lo_plus.value() else 'Connected'}")
    print(f"LO-: {'Disconnected' if ekg.lo_minus.value() else 'Connected'}")

# Auto-init
if __name__ == "__main__":
    init()
    test()