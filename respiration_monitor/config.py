"""
Configuration
=============
Central configuration parameters for respiration monitoring.
"""
from dataclasses import dataclass
from roi_detection import ROIMode


@dataclass
class Config:
    """Configuration parameters for the pipeline"""

    # Camera
    camera_id: int = 0
    camera_fps: int = 30
    target_fps: int = 10

    # YOLO
    yolo_model: str = "yolov8n-pose.pt"
    yolo_confidence: float = 0.5

    # ROI
    roi_mode: ROIMode = ROIMode.UPPER_CHEST
    roi_padding: float = 0.15
    roi_smoothing: int = 5

    # Signal Processing
    buffer_seconds: int = 40
    min_seconds: int = 5
    filter_low: float = 0.1   # Hz (6 BPM)
    filter_high: float = 0.75  # Hz (45 BPM)
    filter_order: int = 4

    # Optical Flow
    use_raft: bool = True
    use_raft_small: bool = True

    # Display
    show_skeleton: bool = True
    show_flow: bool = True
    show_signal: bool = True
    signal_plot_width: int = 400
    signal_plot_height: int = 120