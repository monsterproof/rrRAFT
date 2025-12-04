"""
Visualization
=============

Draws ROI, signal plot, info box, and flow display.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

from roi_detection import ROIMode


class Visualizer:
    """Visualization of monitoring results"""
    
    # ROI colors for different modes
    ROI_COLORS = {
        ROIMode.FULL_THORAX: (0, 255, 0),
        ROIMode.UPPER_CHEST: (255, 200, 0),
        ROIMode.JUGULUM: (0, 165, 255),
        ROIMode.ABDOMEN: (255, 0, 255),
        ROIMode.SHOULDERS: (255, 255, 0),
    }
    
    def __init__(self, signal_plot_width: int = 400, signal_plot_height: int = 120):
        """
        Args:
            signal_plot_width: Width of signal plot
            signal_plot_height: Height of signal plot
        """
        self.signal_plot_width = signal_plot_width
        self.signal_plot_height = signal_plot_height
    
    @staticmethod
    def draw_roi(frame: np.ndarray, roi: Tuple[int, int, int, int], 
                 color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Draws ROI rectangle"""
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        return frame
    
    def draw_signal_plot(self, signal: np.ndarray,
                         color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draws signal plot.
        
        Args:
            signal: 1D signal to plot
            color: Line color (BGR)
            
        Returns:
            Plot as BGR image
        """
        width = self.signal_plot_width
        height = self.signal_plot_height
        
        plot = np.zeros((height, width, 3), dtype=np.uint8)
        plot[:] = (30, 30, 30)
        
        if len(signal) < 2:
            return plot
        
        # Normalize signal
        signal_centered = signal - np.mean(signal)
        max_val = np.max(np.abs(signal_centered))
        if max_val > 0:
            signal_norm = signal_centered / max_val
        else:
            signal_norm = signal_centered
        
        # Calculate points
        margin = 10
        x_points = np.linspace(margin, width - margin, len(signal_norm)).astype(int)
        y_points = ((1 - signal_norm) * (height - 2 * margin) / 2 + margin).astype(int)
        y_points = np.clip(y_points, margin, height - margin)
        
        # Draw line
        points = np.column_stack((x_points, y_points))
        cv2.polylines(plot, [points], False, color, 2, cv2.LINE_AA)
        
        # Center line
        cv2.line(plot, (margin, height // 2), (width - margin, height // 2), 
                (80, 80, 80), 1)
        
        # Border
        cv2.rectangle(plot, (0, 0), (width - 1, height - 1), (100, 100, 100), 1)
        
        return plot
    
    @staticmethod
    def draw_info_box(frame: np.ndarray, rr: float, confidence: float,
                      fps: int, progress: float = 1.0,
                      median_rr: Optional[float] = None) -> np.ndarray:
        """
        Draws info box with RR, median RR, and confidence.
        
        Args:
            frame: Display image
            rr: Respiratory rate (/min)
            confidence: Confidence (0-1)
            fps: Current FPS
            progress: Data collection progress (0-1)
            median_rr: Median RR over time window (optional)
        """
        # Expanded background for median display
        cv2.rectangle(frame, (10, 10), (380, 145), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (380, 145), (100, 100, 100), 1)
        
        # Title
        cv2.putText(frame, "Respiratory Rate Monitor", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if progress < 1.0:
            # Still collecting
            cv2.putText(frame, "Collecting data...", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Progress bar
            bar_width = 280
            bar_height = 15
            filled = int(bar_width * progress)
            cv2.rectangle(frame, (20, 90), (20 + bar_width, 90 + bar_height), 
                         (80, 80, 80), -1)
            cv2.rectangle(frame, (20, 90), (20 + filled, 90 + bar_height), 
                         (0, 200, 0), -1)
            cv2.putText(frame, f"{progress:.0%}", (310 - 40, 103),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            if confidence < 0.05:
                rr = 0

            if rr > 0:
                # Color based on confidence
                if confidence > 0.35:
                    color = (0, 255, 0)
                elif confidence > 0.2:
                    color = (0, 255, 255)
                else:
                    color = (0, 165, 255)
                
                # Current RR
                cv2.putText(frame, f"{rr:.0f}", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
                cv2.putText(frame, "/min", (140, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                # Median RR (20s window)
                if median_rr is not None:
                    cv2.putText(frame, f"Median (20s): {median_rr:.0f} /min", (200, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
                
                # Confidence bar
                conf_width = int(150 * confidence)
                cv2.rectangle(frame, (20, 100), (170, 115), (80, 80, 80), -1)
                cv2.rectangle(frame, (20, 100), (20 + conf_width, 115), color, -1)
                cv2.putText(frame, f"Confidence: {confidence:.0%}", (180, 112),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "No measurement", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps}", (280, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def draw_roi_mode_legend(self, frame: np.ndarray, 
                             current_mode: ROIMode) -> np.ndarray:
        """
        Draws the ROI mode legend.
        
        Args:
            frame: Display image
            current_mode: Currently active mode
        """
        fh, fw = frame.shape[:2]
        
        legend_x = fw - 180
        legend_y = 270
        legend_h = 130
        legend_w = 170
        
        cv2.rectangle(frame, (legend_x, legend_y), 
                     (legend_x + legend_w, legend_y + legend_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x, legend_y), 
                     (legend_x + legend_w, legend_y + legend_h), (100, 100, 100), 1)
        
        cv2.putText(frame, "ROI Mode (1-5):", (legend_x + 5, legend_y + 18),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        
        modes = [
            ("1: Full Thorax", ROIMode.FULL_THORAX),
            ("2: Upper Chest", ROIMode.UPPER_CHEST),
            ("3: Jugulum", ROIMode.JUGULUM),
            ("4: Abdomen", ROIMode.ABDOMEN),
            ("5: Shoulders", ROIMode.SHOULDERS),
        ]
        
        for i, (text, mode) in enumerate(modes):
            y_offset = legend_y + 38 + i * 18
            color = self.ROI_COLORS[mode]
            
            if mode == current_mode:
                cv2.putText(frame, ">", (legend_x + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(frame, text, (legend_x + 18, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return frame
    
    @staticmethod
    def draw_recording_indicator(frame: np.ndarray, duration: float) -> np.ndarray:
        """Draws recording indicator"""
        fh, fw = frame.shape[:2]
        
        cv2.circle(frame, (fw - 30, 30), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (fw - 70, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"{duration:.1f}s", (fw - 70, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame