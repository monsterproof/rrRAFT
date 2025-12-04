# Respiration Monitor

Real-time contactless respiratory rate monitoring using RGB camera, YOLOv8 Pose detection, and optical flow analysis.

## Overview

This system measures breathing rate by detecting subtle chest movements through a standard webcam. It combines deep learning-based pose estimation with optical flow algorithms to extract respiratory signals from video frames.

## Installation

### Requirements

- Python 3.9+
- Webcam or video file

### Dependencies

```bash
pip install numpy opencv-python torch torchvision ultralytics scipy
```

For Apple Silicon (M1/M2/M3/M4), PyTorch with MPS acceleration is used automatically.

### Download YOLO Model

The YOLOv8 pose model downloads automatically on first run, or manually:

```bash
# Nano model (fastest, ~6MB)
yolo export model=yolov8n-pose.pt

# Small model (better accuracy, ~23MB)
yolo export model=yolov8s-pose.pt
```

## Usage

### Basic Usage

```bash
cd respiration_monitor
python main.py
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--source` | `0` | Video source: camera ID or file path |
| `--fps` | `10` | Target processing FPS |
| `--buffer` | `40` | Signal buffer size in seconds |
| `--yolo-model` | `yolov8n-pose.pt` | YOLO pose model |
| `--roi-mode` | `full` | ROI mode: `full`, `chest`, `jugulum`, `abdomen`, `shoulders` |
| `--no-raft` | - | Use Farnebäck instead of RAFT optical flow |
| `--no-skeleton` | - | Hide skeleton overlay |
| `--no-flow` | - | Hide optical flow visualization |

### Examples

```bash
# Use external webcam
python main.py --source 1

# Analyze video file
python main.py --source breathing_video.mp4

# Focus on upper chest region
python main.py --roi-mode chest

# Use classical optical flow (faster on CPU)
python main.py --no-raft

# Longer buffer for more stable readings
python main.py --buffer 60
```

### Keyboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `r` | Reset signal buffer |
| `s` | Toggle skeleton display |
| `f` | Toggle optical flow visualization |
| `1` | ROI: Full Thorax (shoulders to hips) |
| `2` | ROI: Upper Chest (shoulders to mid-torso) |
| `3` | ROI: Jugulum (décolletage area) |
| `4` | ROI: Abdomen (belly region) |
| `5` | ROI: Shoulders (shoulder line only) |
| `Space` | Start/stop recording |

### ROI Modes

Different body regions can be tracked depending on clothing, camera angle, and breathing pattern:

- **Full Thorax**: Best for general use, tracks entire upper body
- **Upper Chest**: Good for seated positions, focuses on ribcage expansion
- **Jugulum**: Sensitive area near the throat, works well with low-cut clothing
- **Abdomen**: Tracks diaphragmatic breathing, useful for relaxation monitoring
- **Shoulders**: Tracks shoulder rise/fall, works through most clothing

## Technical Implementation

### Processing Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Camera    │ -> │  YOLO Pose  │ -> │  ROI Crop   │ -> │ Optical Flow│
│   Frame     │    │  Detection  │    │  Extraction │    │  (RAFT)     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                │
                                                                v
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Display    │ <- │  Smoothing  │ <- │   Welch     │ <- │  Bandpass   │
│  Result     │    │  (3-stage)  │    │   PSD       │    │  Filter     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 1. Pose Detection (YOLOv8)

YOLOv8-Pose detects 17 COCO keypoints per person. The relevant keypoints for respiratory monitoring are:

- Keypoints 5, 6: Left/Right Shoulder
- Keypoints 11, 12: Left/Right Hip
- Keypoint 0: Nose (for jugulum mode reference)

The ROI is computed from these keypoints with configurable padding and smoothed over multiple frames using median filtering to reduce jitter.

### 2. Optical Flow (RAFT / Farnebäck)

Optical flow measures pixel displacement between consecutive frames within the ROI.

**RAFT (Recurrent All-Pairs Field Transforms)**:
- Deep learning-based optical flow
- More accurate, especially for small movements
- Requires GPU/MPS for real-time performance
- Minimum ROI size: 128x128 pixels

**Farnebäck**:
- Classical polynomial expansion method
- Faster on CPU
- Less accurate for subtle movements
- No minimum size requirement

The vertical component (y-axis motion) is extracted as the respiratory signal, using the median value across all pixels in the ROI to reduce noise.

### 3. Signal Processing

#### Preprocessing
- DC removal (subtract mean)
- Median filter (kernel size 3) to remove impulse noise
- Outlier rejection: samples exceeding 5× MAD (median absolute deviation) are discarded

#### Bandpass Filter
- Butterworth filter, 4th order
- Passband: 0.1 Hz to 0.75 Hz (6 to 45 breaths/min)
- Applied with `filtfilt` for zero phase distortion

#### Spectral Analysis (Welch's Method)
- Window length: min(buffer_length, 20 seconds × sample_rate)
- 50% overlap between segments
- Hanning window
- Peak detection within passband

#### Confidence Estimation
- Primary: Peak power / total power in passband
- Secondary: Signal-to-noise ratio (peak / median power)
- Confidence reduced by 50% if SNR < 2

### 4. Output Smoothing (3-Stage)

To provide stable readings despite noisy input:

1. **Median Filter**: Rolling median over last 5 peak frequencies
2. **Exponential Smoothing**: `RR_exp = 0.2 × RR_new + 0.8 × RR_exp`
3. **Adaptive Low-Pass**: 
   - High confidence (>25%): `β = 0.15`
   - Low confidence: `β = 0.05`
   - `RR_smooth = RR_smooth + β × (RR_exp - RR_smooth)`

### 5. Calibration

A 5-second calibration period at startup allows the signal to stabilize before measurements begin. During this time, no samples are added to the buffer.

## Data Recording

Press `Space` to start/stop recording. Data is saved to the `recordings/` directory:

| File | Content |
|------|---------|
| `*_raw.csv` | Timestamp, vertical motion, ROI size, confidence, ROI mode |
| `*_analysis.csv` | Timestamp, respiratory rate, confidence, actual sample rate |
| `*_signals.npz` | NumPy archive with raw and filtered signals |
| `*_summary.txt` | Session summary with statistics |

## Module Structure

```
respiration_monitor/
├── main.py              # Main application, CLI, keyboard handling
├── config.py            # Configuration dataclass
├── roi_detection.py     # YOLOv8 pose detection, ROI extraction
├── optical_flow.py      # RAFT and Farnebäck optical flow
├── signal_analysis.py   # Bandpass filter, Welch PSD, smoothing
├── visualization.py     # OpenCV drawing utilities
├── data_recorder.py     # CSV/NPZ data export
└── __init__.py          # Package exports
```

## Performance Tips

- **Reduce target FPS** (`--fps 5`) for slower machines
- **Use Farnebäck** (`--no-raft`) on CPU-only systems
- **Choose smaller ROI** (`--roi-mode jugulum`) for faster processing
- **Use nano model** (`--yolo-model yolov8n-pose.pt`) for speed
- **Ensure good lighting** for better pose detection
- **Minimize subject movement** for cleaner signal
- **Wear fitted clothing** or use jugulum mode for best results

## Limitations

- Single person tracking only (uses first detected person)
- Requires relatively stable camera position
- Performance depends on lighting conditions
- Large movements (walking, gesturing) corrupt the signal
- Loose clothing reduces accuracy


## References

- Boccignone et al. (2025): "Remote Respiration Measurement with RGB Cameras: A Review and Benchmark"
- Teed & Deng (2020): "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow"
- resPyre: https://github.com/phuselab/resPyre


## License

MIT License