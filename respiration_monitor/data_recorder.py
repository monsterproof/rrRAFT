"""
Data Recording
==============

Records measurements for later analysis.
Stores raw data, analysis results, and filtered signals.
"""

import os
import time
import numpy as np
from typing import List, Tuple, Optional
from datetime import datetime


class DataRecorder:
    """Records all measurements for later analysis"""
    
    def __init__(self, output_dir: str = "recordings"):
        """
        Args:
            output_dir: Directory for recordings
        """
        self.output_dir = output_dir
        self.recording = False
        self.start_time: Optional[float] = None
        self.session_id: Optional[str] = None
        
        # Raw data
        self.timestamps: List[float] = []
        self.vertical_motion_raw: List[float] = []
        self.roi_sizes: List[Tuple[int, int]] = []
        self.confidences: List[float] = []
        self.roi_modes: List[str] = []
        
        # Analyzed data
        self.analysis_timestamps: List[float] = []
        self.respiration_rates: List[float] = []
        self.rr_confidences: List[float] = []
        self.actual_fs_values: List[float] = []
        
        # Filtered signals
        self.filtered_signals: List[np.ndarray] = []
        self.filtered_signal_timestamps: List[float] = []
    
    def start_recording(self):
        """Starts a new recording"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.recording = True
        
        # Clear buffers
        self._clear_buffers()
        
        print(f"Recording started: {self.session_id}")
    
    def _clear_buffers(self):
        """Clears all data buffers"""
        self.timestamps.clear()
        self.vertical_motion_raw.clear()
        self.roi_sizes.clear()
        self.confidences.clear()
        self.roi_modes.clear()
        self.analysis_timestamps.clear()
        self.respiration_rates.clear()
        self.rr_confidences.clear()
        self.actual_fs_values.clear()
        self.filtered_signals.clear()
        self.filtered_signal_timestamps.clear()
    
    def stop_recording(self) -> str:
        """
        Stops the recording and saves all data.
        
        Returns:
            Path to the saved recording
        """
        if not self.recording:
            return ""
        
        self.recording = False
        filepath = self._save_data()
        print(f"Recording saved: {filepath}")
        return filepath
    
    def add_sample(self, timestamp: float, vertical_motion: float, 
                   roi_size: Tuple[int, int], confidence: float, roi_mode: str):
        """Adds a raw data sample"""
        if not self.recording:
            return
        
        self.timestamps.append(timestamp)
        self.vertical_motion_raw.append(vertical_motion)
        self.roi_sizes.append(roi_size)
        self.confidences.append(confidence)
        self.roi_modes.append(roi_mode)
    
    def add_analysis(self, timestamp: float, rr: float, confidence: float, 
                     actual_fs: float, filtered_signal: np.ndarray):
        """Adds analysis results"""
        if not self.recording:
            return
        
        self.analysis_timestamps.append(timestamp)
        self.respiration_rates.append(rr)
        self.rr_confidences.append(confidence)
        self.actual_fs_values.append(actual_fs)
        
        self.filtered_signals.append(filtered_signal.copy())
        self.filtered_signal_timestamps.append(timestamp)
    
    def _save_data(self) -> str:
        """Saves all data to CSV and NPZ files"""
        base_path = os.path.join(self.output_dir, self.session_id)
        
        # 1. Raw data as CSV
        raw_csv_path = f"{base_path}_raw.csv"
        with open(raw_csv_path, 'w') as f:
            f.write("timestamp,vertical_motion,roi_width,roi_height,confidence,roi_mode\n")
            for i in range(len(self.timestamps)):
                roi_w, roi_h = self.roi_sizes[i] if i < len(self.roi_sizes) else (0, 0)
                f.write(f"{self.timestamps[i]:.4f},"
                       f"{self.vertical_motion_raw[i]:.6f},"
                       f"{roi_w},{roi_h},"
                       f"{self.confidences[i]:.4f},"
                       f"{self.roi_modes[i]}\n")
        
        # 2. Analysis results as CSV
        analysis_csv_path = f"{base_path}_analysis.csv"
        with open(analysis_csv_path, 'w') as f:
            f.write("timestamp,respiration_rate,confidence,actual_fs\n")
            for i in range(len(self.analysis_timestamps)):
                f.write(f"{self.analysis_timestamps[i]:.4f},"
                       f"{self.respiration_rates[i]:.2f},"
                       f"{self.rr_confidences[i]:.4f},"
                       f"{self.actual_fs_values[i]:.2f}\n")
        
        # 3. Signals as NPZ
        npz_path = f"{base_path}_signals.npz"
        np.savez_compressed(
            npz_path,
            timestamps=np.array(self.timestamps),
            vertical_motion_raw=np.array(self.vertical_motion_raw),
            analysis_timestamps=np.array(self.analysis_timestamps),
            respiration_rates=np.array(self.respiration_rates),
            filtered_signal=self.filtered_signals[-1] if self.filtered_signals else np.array([]),
            session_id=self.session_id,
            duration=self.timestamps[-1] if self.timestamps else 0
        )
        
        # 4. Summary
        summary_path = f"{base_path}_summary.txt"
        with open(summary_path, 'w') as f:
            duration = self.timestamps[-1] if self.timestamps else 0
            avg_rr = np.mean(self.respiration_rates) if self.respiration_rates else 0
            std_rr = np.std(self.respiration_rates) if self.respiration_rates else 0
            avg_fs = np.mean(self.actual_fs_values) if self.actual_fs_values else 0
            
            f.write(f"Session: {self.session_id}\n")
            f.write(f"Duration: {duration:.1f} seconds\n")
            f.write(f"Samples: {len(self.timestamps)}\n")
            f.write(f"Average sample rate: {avg_fs:.2f} Hz\n")
            f.write(f"Respiratory rate: {avg_rr:.1f} ± {std_rr:.1f} /min\n")
            f.write(f"\nFiles:\n")
            f.write(f"  - {raw_csv_path}\n")
            f.write(f"  - {analysis_csv_path}\n")
            f.write(f"  - {npz_path}\n")
        
        return base_path
    
    def is_recording(self) -> bool:
        """Returns whether recording is in progress"""
        return self.recording
    
    def get_duration(self) -> float:
        """Returns the current recording duration"""
        if not self.recording or not self.timestamps:
            return 0.0
        return self.timestamps[-1]