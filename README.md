# Atemfrequenz-Monitoring mit RGB-Kamera

Kontaktlose Echtzeit-Messung der Atemfrequenz mittels Optical Flow.

## Überblick

Diese Pipeline analysiert die Thorax-Bewegung in Videobildern und extrahiert daraus die Atemfrequenz. Basierend auf dem Review-Paper "Remote Respiration Measurement with RGB Cameras" (ACM Computing Surveys, 2025).

## Installation

```bash
# Repository klonen / Dateien kopieren


# Virtuelle Umgebung erstellen (empfohlen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# oder: venv\Scripts\activate  # Windows

# Abhängigkeiten installieren
pip install ultralytics

```

## Verwendung

```bash
python respiration_pipeline.py
```

### Optionen

| Parameter | Beschreibung | Default |
|-----------|--------------|---------|
| `--source` | Kamera-ID oder Videodatei | 0 |
| `--fps` | Ziel-Framerate für Verarbeitung | 10 |
| `--buffer` | Puffergrösse in Sekunden | 30 |
| `--no-raft` | Farnebäck statt RAFT verwenden | False |

### Beispiele

```bash
# Webcam mit 10 FPS
python simple_monitor.py --source 0 --fps 10

# Videodatei analysieren
python respiration_pipeline.py --source video.mp4

# Ohne RAFT (schneller, weniger genau)
python respiration_pipeline.py --no-raft
```

## Tastenbelegung

- `q` - Beenden
- `s` - Display / Hide Skeleton
- `r` - ROI zurücksetzen (nur simple_monitor.py)

## Architektur

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Kamera    │ ──▶ │  ROI-Erkennung   │ ──▶ │  Optical Flow   │
│  (30 FPS)   │     │       (YOLO)     │     │  (RAFT/Farneb.) │
└─────────────┘     └──────────────────┘     └─────────────────┘
                                                      │
                                                      ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Anzeige    │ ◀── │  RR-Berechnung   │ ◀── │ Signal-Filter   │
│  (RR, Plot) │     │  (Welch PSD)     │     │  (Butterworth)  │
└─────────────┘     └──────────────────┘     └─────────────────┘
```

## Signal-Verarbeitung

1. **ROI-Extraktion**: Thorax-Region via YOLO Pose Landmarks
2. **Motion-Extraktion**: Vertikale Komponente des Optical Flow (Median)
3. **Filterung**: Butterworth Bandpass 0.1-0.5 Hz (6-30 BPM)
4. **RR-Schätzung**: Peak im Welch-Periodogramm

## Konfiguration

Die wichtigsten Parameter in `Config`:

```python
@dataclass
class Config:
    target_fps: int = 10        # Verarbeitungs-Framerate
    buffer_seconds: int = 30    # Sekunden für RR-Berechnung
    filter_low: float = 0.1     # Hz (6 BPM)
    filter_high: float = 0.5    # Hz (30 BPM)
    use_raft_small: bool = True # Schnellere RAFT-Variante
```

## Genauigkeit

Basierend auf dem Benchmark aus dem Paper:

| Methode | MAE (BPM) | PCC | Zeit/Frame |
|---------|-----------|-----|------------|
| RAFT | ~1.4 | ~0.66 | ~10ms |
| Farnebäck | ~2.0 | ~0.51 | ~1ms |

## Einschränkungen

- Patient sollte relativ still sitzen/liegen
- Thorax muss im Bild sichtbar sein
- Mindestens 5 Sekunden für erste Messung
- Starke Bewegungen verfälschen das Signal

## Troubleshooting

### "RAFT nicht verfügbar"
- PyTorch/Torchvision installieren
- GPU-Treiber prüfen
- Fallback auf `--no-raft`

### Keine Pose erkannt
- Beleuchtung verbessern
- Abstand zur Kamera anpassen
- Oberkörper vollständig im Bild

### Ungenaue Messungen
- Buffer-Zeit erhöhen (`--buffer 60`)
- Bewegungen minimieren
- Kleidung mit Kontrast/Textur hilft

## Referenzen

- Boccignone et al. (2025): "Remote Respiration Measurement with RGB Cameras: A Review and Benchmark"
- Teed & Deng (2020): "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
- resPyre: https://github.com/phuselab/resPyre

## Lizenz

MIT License - Für Forschung und nicht-kommerzielle Anwendungen.
Für klinische Anwendungen entsprechende Zertifizierungen beachten.
