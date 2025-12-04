"""
Respiration Monitor
===================
Real-time respiratory rate monitoring with YOLO Pose + Optical Flow.
Contactless measurement of respiratory rate via RGB camera.
Uses YOLOv8 Pose for thorax detection and RAFT/Farnebäck for motion analysis.
"""

from config import Config
from roi_detection import ROIMode, YOLOThoraxDetector
from optical_flow import OpticalFlowProcessor
from signal_analysis import RespirationAnalyzer
from visualization import Visualizer
from data_recorder import DataRecorder

__version__ = "1.0.0"
__all__ = [
    "Config",
    "ROIMode",
    "YOLOThoraxDetector",
    "OpticalFlowProcessor",
    "RespirationAnalyzer",
    "Visualizer",
    "DataRecorder",
]
