# System Monitorowania EKG

Projekt systemu do monitorowania elektrokardiogramu w czasie rzeczywistym, stworzony w ramach pracy magisterskiej.

**Autor**: Łukasz Guziczak

## O projekcie

System EKG oparty na mikrokontrolerze RP2040-Zero i module AD8232, umożliwiający rejestrację i wizualizację sygnału elektrokardiograficznego. Projekt demonstruje pełny przepływ danych od akwizycji sygnału biomedycznego po jego przetwarzanie i wyświetlanie w czasie rzeczywistym.

## Co zawiera projekt?

- **Sprzęt**: Mikrokontroler RP2040-Zero + moduł EKG AD8232
- **Firmware**: Program w MicroPythonie zbierający dane z czujnika
- **Wizualizator**: Aplikacja na PC pokazująca wykres EKG w czasie rzeczywistym
- **Konfiguracja**: System 3-elektrodowy (standardowe odprowadzenie)

## Możliwości systemu

- Pomiar EKG z częstotliwością 250 Hz (250 próbek na sekundę)
- Skalowanie sygnału do standardowego zakresu medycznego ±2mV
- Wyświetlanie w czasie rzeczywistym:
  - Krzywej EKG z wszystkimi charakterystycznymi elementami
  - Aktualnego tętna (BPM)
  - Detekcji zespołów QRS
  - Jakości połączenia elektrod
- Filtrowanie zakłóceń sieciowych 50Hz
- Zapis danych do analizy

## Czego potrzebujesz?

### Komponenty elektroniczne
- **Mikrokontroler**: Raspberry Pi Pico RP2040-Zero (lub zwykły Pico)
- **Moduł EKG**: AD8232 (tani moduł dostępny w sklepach elektronicznych)
- **Elektrody**: 3 sztuki elektrod EKG na zatrzaski
- **Kabelki**: Do połączenia modułu z mikrokontrolerem

### Połączenia
Podłącz moduł AD8232 do RP2040 według schematu:
- Wyjście sygnału → GPIO29 (ADC3)
- LO+ → GPIO27
- LO- → GPIO28
- Zasilanie → 3.3V i GND

## Gdzie przykleić elektrody?

Używam standardowego układu 3-elektrodowego:
- **Czerwona (RA)**: Prawa strona klatki piersiowej, pod obojczykiem
- **Żółta (LA)**: Lewa strona klatki piersiowej, pod obojczykiem
- **Zielona (LL)**: Lewa strona brzucha, poniżej żeber

Ważne: oczyść skórę wacikiem ze spirytusem przed naklejeniem elektrod. Lepszy kontakt = czystszy sygnał.

## Instalacja

### Krok 1: Wgranie firmware

1. Otwórz Thonny IDE i podłącz RP2040 przez USB
2. Jeśli nie masz MicroPythona, Thonny zaproponuje instalację
3. Otwórz plik `firmware/main_scaled.py`
4. Zapisz go jako `main.py` na urządzeniu

### Krok 2: Instalacja programu na PC

```bash
# Zainstaluj wymagane biblioteki Pythona
pip install numpy scipy matplotlib pyserial

# Na Windowsie może być też potrzebne:
pip install windows-curses
```

## Jak używać?

### 1. Uruchomienie firmware
Po podłączeniu RP2040 do komputera, w terminalu MicroPythona wpisz:
```python
import main_scaled
main_scaled.stream()
```

### 2. Uruchomienie wizualizatora
```bash
# Domyślnie używa portu COM5
python pc_software/visualizer_scaled.py

# Jeśli masz inny port (sprawdź w Menedżerze urządzeń)
python pc_software/visualizer_scaled.py --port COM7

# Z zapisem danych
python pc_software/visualizer_scaled.py --record pomiar_ekg.csv
```

## Sterowanie programem

Podczas działania wizualizatora możesz używać klawiszy:
- **Spacja**: Zatrzymaj/wznów wyświetlanie
- **R**: Resetuj linię bazową (jeśli "uciekła")
- **F**: Włącz/wyłącz filtr 50Hz
- **S**: Zapisz zrzut ekranu
- **Q lub ESC**: Wyjście

## Parametry techniczne

- **Częstotliwość próbkowania**: 250 Hz
- **Rozdzielczość**: 12-bit (przeskalowane do 16-bit)
- **Zakres napięć**: ±2 mV (standard medyczny)
- **Pasmo przenoszenia**: 0.5-40 Hz (typowe dla EKG)
- **Format danych**: `timestamp,wartość_adc,tętno`

## Rozwiązywanie problemów

### Nie widzę sygnału / płaska linia
- Sprawdź czy elektrody są dobrze przyklejone
- Zobacz wskaźniki LO+ i LO- (powinny pokazywać "Connected")
- Upewnij się że skóra jest czysta i sucha
- Spróbuj lekko zwilżyć elektrody

### Sygnał jest bardzo zaszumiony
- Oddal się od urządzeń elektrycznych (zasilacze, monitory)
- Włącz filtr 50Hz klawiszem F
- Sprawdź czy kable nie są uszkodzone
- Upewnij się że elektrody dobrze przylegają

### Nie mogę się połączyć
- Sprawdź numer portu COM w Menedżerze urządzeń
- Upewnij się że używasz prędkości 115200 baud
- Spróbuj odłączyć i podłączyć USB ponownie

## Uwagi o bezpieczeństwie

System jest zasilany niskim napięciem 3.3V z portu USB, co jest bezpieczne. Moduł AD8232 nie posiada izolacji galwanicznej, ale przy zasilaniu z baterii lub portu USB (które same w sobie są izolowane od sieci elektrycznej) nie stanowi to zagrożenia.