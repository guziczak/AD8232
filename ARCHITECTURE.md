# ECG Edge AI - Kompaktowa Architektura Systemu

**Autor:** Łukasz Guziczak
**Hardware:** RP2040-Zero + AD8232
**Koszt:** ~45 PLN

---

## KONCEPCJA GŁÓWNA

**Semantic Compression kierowana AI**: System hybrydowy (edge + cloud) kompresuje dane EKG
inteligentnie - normalne rytmy zapisuje jako statystyki (1000x kompresja), anomalie jako
pełny sygnał (0x kompresja). Dzięki temu 24h nagrania mieści się w 2MB Flash.

---

## SPECYFIKACJA HARDWARE

```
RP2040-Zero:
├─ CPU: Dual Cortex-M0+ @ 133MHz
├─ RAM: 264KB
├─ Flash: 2MB
└─ ADC: 12-bit (oversampled do 16-bit)

AD8232:
└─ Analog front-end dla EKG, filtr 0.5-40Hz, output: analog

Elektrody: 3x zatrzaskowe (RA, LA, RL)
```

---

## ARCHITEKTURA 3-WARSTWOWA

### 1. EDGE LAYER (RP2040)
```
AD8232 → ADC @ 250Hz
    ↓
Digital Filtering (bandpass 0.5-40Hz, notch 50Hz)
    ↓
Pan-Tompkins QRS Detection
    ↓
Feature Extraction (RR intervals, HRV: RMSSD/SDNN)
    ↓
TinyML Classifier (TFLite INT8, <500KB)
    ↓
Compression Decision (Tier 1/2/3)
    ↓
Flash Storage (2MB) + UART Stream → PC
```

**Core 0 (Real-time):** Sampling, filtering, QRS detection, UART
**Core 1 (Background):** TinyML inference, compression, Flash writes

**Memory Budget:**
- Sample buffer: 30KB
- Model activations: 80KB
- Variables: 20KB
- Stack: 30KB
- **Free RAM: ~100KB**

---

### 2. CLOUD LAYER (PC)

```
UART Serial ← RP2040
    ↓
Protocol Decoder (binary packets)
    ↓
Live Visualization (Matplotlib/PyQt)
    ↓
Deep Learning Inference (CNN/LSTM opcjonalnie)
    ↓
Long-term Storage (CSV/HDF5)
```

**Training Pipeline:**
```
MIT-BIH/PTB-XL datasets
    ↓
Preprocessing (normalizacja, augmentacja)
    ↓
Model Training (CNN: 4 Conv1D blocks, ~300K params)
    ↓
Evaluation (accuracy, precision, recall, F1)
    ↓
Quantization (FP32 → INT8 via TFLite)
    ↓
Deploy to RP2040 (upload przez USB)
```

---

### 3. STORAGE LAYER (3-TIER SEMANTIC COMPRESSION)

**Tier 1 - Stats Only (50 B/min, ~1000x compression)**
- Warunek: Normal rhythm + confidence >95%
- Przechowuje: HR mean/std, HRV metrics, QRS count
- Capacity: 500KB → ~7 dni

**Tier 2 - Features (500 B/min, ~100x compression)**
- Warunek: Normal rhythm + confidence >85%
- Przechowuje: RR intervals, QRS peaks, features
- Capacity: 400KB → ~13h

**Tier 3 - Raw Data (15 KB/event, 1x compression)**
- Warunek: Anomaly OR confidence <85%
- Przechowuje: Full raw signal ±30s context
- Capacity: 600KB → ~40 events

**Ring Buffer z Priority:**
- Tier 3 (anomalies): NEVER overwrite
- Tier 2 (features): Protect recent
- Tier 1 (stats): Overwrite oldest first

**Przykład - Normal Day (90% normal):**
- 21.6h normal → Tier 1 → 64.8 KB
- 2.4h anomalies → Tier 3 → 450 KB
- **Total: 515 KB (65x compression)**

**Przykład - Problematic Day (30% anomalies):**
- 16.8h normal → Tier 1 → 50.4 KB
- 7.2h anomalies → Tier 3 → 1.5 MB
- **Total: 1.55 MB (18x compression, 77% Flash)**

---

## ALGORYTMY KLUCZOWE

### Pan-Tompkins QRS Detection
```python
# IEEE Trans BME (1985)
1. Derivative filter: y[n] = (2x[n] + x[n-1] - x[n-3] - 2x[n-4]) / 8
2. Squaring: x²
3. Moving Window Integration (150ms window)
4. Adaptive thresholding (learning rate 0.125)
5. RR interval tracking → Heart Rate + HRV
```

### CNN Model (TensorFlow)
```
Input: (750, 1) = 3s @ 250Hz

Conv1D(32, k=5) + BN + MaxPool(2) + Dropout(0.2)
Conv1D(64, k=5) + BN + MaxPool(2) + Dropout(0.3)
Conv1D(128, k=3) + BN + MaxPool(2) + Dropout(0.3)
Conv1D(128, k=3) + BN + GlobalAvgPool
Dense(64) + Dropout(0.5)
Dense(5, softmax)

Output: [Normal, AF, PVC, Brady, Tachy]
Params: ~300K → INT8: ~300KB
```

### Quantization (TFLite)
```python
# FP32 → INT8 (4x reduction)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Representative dataset dla calibration (100 samples)
# Result: 300KB model, <5% accuracy drop
```

---

## PROTOKÓŁ KOMUNIKACJI (UART)

**Baud rate:** 115200
**Format:** Binary packets

**Packet Types:**
1. **HEARTBEAT** (5s interval) - system alive
2. **RAW_SAMPLE** - pojedynczy sample (2 bytes)
3. **QRS_EVENT** - detected QRS (timestamp, RR, amplitude)
4. **FEATURES** - extracted features (HR, HRV, etc.)
5. **CLASSIFICATION** - ML result (class_id, confidence)
6. **ANOMALY** - full raw segment (raw_buffer + metadata)

---

## VALIDATION METODOLOGIA

### Metryki Target
- **Sensitivity (Recall):** >95% dla AF/VT/VF (critical)
- **False Negative Rate:** <5% (cannot miss arrhythmias!)
- **Specificity:** >70% (false alarms OK)
- **F1 Score:** >0.90
- **Compression loss rate:** <3%

### 3-Path Validation
1. **Offline:** MIT-BIH/PTB-XL datasets, symulacja kompresji
2. **Real-world:** Dual stream (full + compressed), porównanie completeness
3. **Clinical:** Blinded doctor assessment, diagnostic agreement

### Per-Class Requirements
- 🔴 Critical (AF, VT, VF): Sensitivity >95%
- 🟡 Important (PVC, PAC): Sensitivity >90%
- 🟢 Benign (Brady, Tachy): Sensitivity >85%
- ⚪ Normal: Specificity >70%

---

## STRUKTURA PROJEKTU

```
ecg-edge-ai/
│
├── firmware/                  # RP2040 (MicroPython)
│   ├── src/
│   │   ├── main.py           # Entry point
│   │   ├── config.py         # Constants (pins, thresholds, etc.)
│   │   ├── acquisition/
│   │   │   ├── ad8232.py     # AD8232 driver
│   │   │   └── sampler.py    # 250Hz sampling
│   │   ├── processing/
│   │   │   ├── filters.py    # Bandpass, notch
│   │   │   ├── pan_tompkins.py  # QRS detection
│   │   │   └── features.py   # RR, HRV extraction
│   │   ├── ml/
│   │   │   ├── tflite_model.py  # TFLite Micro inference
│   │   │   └── classifier.py    # Decision logic
│   │   ├── storage/
│   │   │   ├── flash_manager.py  # 2MB Flash
│   │   │   ├── compression.py    # 3-tier compression
│   │   │   └── ring_buffer.py    # Priority overwrite
│   │   └── communication/
│   │       ├── protocol.py       # Binary protocol
│   │       └── uart_handler.py   # UART TX/RX
│   ├── models/
│   │   └── model_quantized.tflite  # Deployed model
│   └── tests/
│
├── pc_software/              # PC/Laptop (Python)
│   ├── src/
│   │   ├── data/
│   │   │   ├── datasets.py       # MIT-BIH, PTB-XL loaders
│   │   │   └── preprocessing.py  # Normalizacja, augmentacja
│   │   ├── models/
│   │   │   ├── cnn.py           # CNN architecture
│   │   │   └── lstm.py          # LSTM architecture
│   │   ├── training/
│   │   │   └── trainer.py       # Training loop
│   │   ├── optimization/
│   │   │   ├── quantization.py  # FP32→INT8
│   │   │   └── converter.py     # TFLite conversion
│   │   ├── evaluation/
│   │   │   ├── metrics.py       # Accuracy, F1, sensitivity
│   │   │   └── validator.py     # Clinical validation
│   │   ├── real_time/
│   │   │   ├── serial_reader.py  # Read from RP2040
│   │   │   ├── protocol_decoder.py
│   │   │   └── dashboard.py      # Live visualization
│   │   └── deployment/
│   │       └── flash_uploader.py # Upload model to RP2040
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_model_training.ipynb
│   │   ├── 03_optimization.ipynb
│   │   └── 04_validation.ipynb
│   └── scripts/
│       ├── train_model.py        # CLI: python train_model.py --dataset mitbih --quantize
│       ├── deploy_to_device.py
│       └── run_dashboard.py
│
├── data/                     # Datasets (gitignored)
│   ├── raw/mitbih/
│   ├── raw/ptb-xl/
│   └── processed/
│
└── models/                   # Trained models
    ├── checkpoints/
    └── production/
```

---

## OPERATIONAL MODES

**1. Training Mode (PC)**
- Load MIT-BIH/PTB-XL
- Train CNN/LSTM model
- Quantize to INT8
- Validate accuracy

**2. Deployment Mode (PC → RP2040)**
- Package model.tflite
- Upload via USB
- Store in Flash

**3. Real-time Mode (RP2040 + PC)**
- Edge: Sampling, QRS, classification
- UART: Stream to PC
- PC: Live visualization, long-term storage

**4. Standalone Holter (RP2040 only)**
- Autonomous 24h recording
- Flash: Local 3-tier storage
- USB: Download later

---

## KLUCZOWE PARAMETRY

**Signal Processing:**
- Sample rate: 250 Hz
- ADC resolution: 16-bit (oversampled)
- Bandpass filter: 0.5-40 Hz
- Notch filter: 50 Hz (EU) / 60 Hz (US)

**QRS Detection:**
- Min distance: 150ms (400 BPM max)
- Window: 25 samples peak detection
- Adaptive threshold: 0.6 multiplier

**Machine Learning:**
- Model: CNN1D
- Input: 750 samples (3s @ 250Hz)
- Output: 5 classes
- Confidence thresholds:
  - Tier 1: >0.95
  - Tier 2: >0.85
  - Tier 3: <0.85

**Storage:**
- Flash total: 2MB
- Firmware + Model: 600KB
- Data storage: 1.4MB (3-tier allocation)

**Clinical Thresholds:**
- Bradycardia: HR <60 BPM
- Tachycardia: HR >100 BPM
- HR range: 30-250 BPM

---

## PERFORMANCE TARGETS

✅ **Accuracy:** >90% on MIT-BIH
✅ **Sensitivity:** >95% dla AF/VT/VF
✅ **Latency:** <100ms dla edge classification
✅ **Model Size:** <500KB (INT8)
✅ **Storage:** 24h recording w 2MB Flash
✅ **False Negative Rate:** <5%
✅ **Compression:** 20-50x (zależnie od arrhythmia frequency)

---

## WORKFLOW DEVELOPERSKI

### Setup Development Environment
```bash
# PC Software
cd pc_software
python -m venv venv
source venv/bin/activate
pip install tensorflow numpy scipy wfdb matplotlib click

# Firmware (MicroPython on RP2040)
# Flash MicroPython: https://micropython.org/download/rp2-pico/
# Upload firmware/ files via Thonny or rshell
```

### Train Model
```bash
python pc_software/scripts/train_model.py \
  --dataset mitbih \
  --model cnn \
  --epochs 100 \
  --batch-size 32 \
  --quantize \
  --validate
```

### Deploy to Device
```bash
python pc_software/scripts/deploy_to_device.py \
  --model models/production/cnn_mitbih_quantized.tflite \
  --port /dev/ttyACM0
```

### Run Live Dashboard
```bash
python pc_software/scripts/run_dashboard.py \
  --port /dev/ttyACM0 \
  --baudrate 115200
```

---

## BEZPIECZEŃSTWO I OGRANICZENIA

⚠️ **NIE JEST TO URZĄDZENIE MEDYCZNE!**

Projekt edukacyjny/badawczy. Dla użytku klinicznego wymagane:
- Certyfikacja FDA/CE
- ISO 13485 (medical devices)
- IEC 60601-2-47 (ECG ambulatory monitors)
- Clinical trials
- Regulatory approval

**Ograniczenia:**
- Brak izolacji galwanicznej (nie spełnia norm bezpieczeństwa medycznego)
- Single-lead EKG (commercial to 12-lead)
- Brak automatycznego wykrywania wszystkich arytmii (tylko 5 klas)
- Flash memory ograniczona (commercial używa SD card GB)

---

## EXTENSIONS/TODO

**Hardware:**
- [ ] Dodać izolację galwaniczną (AD8232 → ADuM1401 → RP2040)
- [ ] MicroSD card slot dla extended storage
- [ ] Battery + power management (TP4056 + boost converter)
- [ ] 3-lead lub 12-lead support

**Firmware:**
- [ ] More arrhythmia classes (SVT, PAC, artifact detection)
- [ ] Adaptive sampling rate (250Hz → 500Hz dla VT/VF)
- [ ] Bluetooth LE streaming (zamiast UART)

**PC Software:**
- [ ] LSTM model evaluation
- [ ] Ensemble voting (CNN + LSTM)
- [ ] Real-time cloud upload (Firebase/AWS IoT)
- [ ] Mobile app (React Native + BLE)

**Validation:**
- [ ] Real patient trials (IRB approval needed)
- [ ] PTB-XL full validation
- [ ] Inter-rater reliability study
- [ ] Compare vs commercial Holter

---

## REFERENCES

**Papers:**
- Pan & Tompkins (1985) - "A Real-Time QRS Detection Algorithm", IEEE Trans BME
- Rajpurkar et al. (2017) - "Cardiologist-Level Arrhythmia Detection", Stanford
- Hannun et al. (2019) - "Cardiologist-level arrhythmia detection", Nature Medicine

**Datasets:**
- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/
- PTB-XL: https://physionet.org/content/ptb-xl/

**Tools:**
- TensorFlow Lite: https://www.tensorflow.org/lite
- MicroPython: https://micropython.org
- WFDB Python: https://github.com/MIT-LCP/wfdb-python

---

**Last updated:** 2025-11-11
**Version:** 1.0
**License:** MIT (educational use only, not for clinical deployment)
