"""
ROI Detection using YOLOv8 Pose
===============================

Detects various body regions for breathing analysis:
- FULL_THORAX: Shoulders to hips
- UPPER_CHEST: Upper chest area
- JUGULUM: Jugulum/décolletage area
- ABDOMEN: Abdominal region
- SHOULDERS: Shoulder area
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
from ultralytics import YOLO


class ROIMode(Enum):
    """Different ROI modes for breathing analysis"""
    FULL_THORAX = "full"
    UPPER_CHEST = "chest"
    JUGULUM = "jugulum"
    ABDOMEN = "abdomen"
    SHOULDERS = "shoulders"


class YOLOThoraxDetector:
    """
    Detects various body regions using YOLOv8 Pose.
    
    COCO Keypoint indices:
    0: Nose, 1-2: Eyes, 3-4: Ears
    5: L_Shoulder, 6: R_Shoulder
    7: L_Elbow, 8: R_Elbow
    9: L_Wrist, 10: R_Wrist
    11: L_Hip, 12: R_Hip
    13: L_Knee, 14: R_Knee
    15: L_Ankle, 16: R_Ankle
    """
    
    # Keypoint indices
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16
    
    # For skeleton drawing
    SKELETON_CONNECTIONS = [
        (5, 6),   # Shoulders
        (5, 7),   # L Shoulder - Elbow
        (7, 9),   # L Elbow - Wrist
        (6, 8),   # R Shoulder - Elbow
        (8, 10),  # R Elbow - Wrist
        (5, 11),  # L Shoulder - Hip
        (6, 12),  # R Shoulder - Hip
        (11, 12), # Hips
        (11, 13), # L Hip - Knee
        (13, 15), # L Knee - Ankle
        (12, 14), # R Hip - Knee
        (14, 16), # R Knee - Ankle
    ]
    
    # ROI colors for different modes
    ROI_COLORS = {
        ROIMode.FULL_THORAX: (0, 255, 0),    # Green
        ROIMode.UPPER_CHEST: (255, 200, 0),  # Cyan
        ROIMode.JUGULUM: (0, 165, 255),      # Orange
        ROIMode.ABDOMEN: (255, 0, 255),      # Magenta
        ROIMode.SHOULDERS: (255, 255, 0),    # Yellow
    }
    
    def __init__(self, model_path: str = "yolov8n-pose.pt", 
                 confidence: float = 0.5,
                 roi_mode: ROIMode = ROIMode.FULL_THORAX,
                 roi_padding: float = 0.15,
                 roi_smoothing: int = 5):
        """
        Args:
            model_path: Path to YOLO Pose model
            confidence: Minimum confidence for detection
            roi_mode: Initial ROI mode
            roi_padding: Padding around ROI (relative)
            roi_smoothing: Number of frames for smoothing
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.roi_mode = roi_mode
        self.roi_padding = roi_padding
        
        # ROI smoothing
        self.roi_history: deque = deque(maxlen=roi_smoothing)
        self.last_valid_roi: Optional[Tuple[int, int, int, int]] = None
        
    def set_roi_mode(self, mode: ROIMode):
        """Changes ROI mode and resets the history buffer"""
        if mode != self.roi_mode:
            self.roi_mode = mode
            self.roi_history.clear()
            self.last_valid_roi = None
            print(f"ROI mode changed to: {mode.value}")
    
    def get_roi_color(self) -> Tuple[int, int, int]:
        """Returns the color for the current ROI mode"""
        return self.ROI_COLORS.get(self.roi_mode, (0, 255, 0))
        
    def detect(self, frame: np.ndarray) -> dict:
        """
        Detects pose and extracts ROI based on current mode.
        
        Returns:
            dict with 'roi', 'keypoints', 'confidence', 'roi_mode'
        """
        results = self.model(frame, verbose=False, conf=self.confidence)
        
        result = {
            'roi': None,
            'keypoints': None,
            'confidence': 0.0,
            'all_keypoints': None,
            'roi_mode': self.roi_mode
        }
        
        # No person detected
        if len(results[0].keypoints) == 0:
            if self.last_valid_roi is not None:
                result['roi'] = self.last_valid_roi
            return result
        
        # Take first person
        keypoints = results[0].keypoints
        
        if keypoints.xy is None or len(keypoints.xy) == 0:
            if self.last_valid_roi is not None:
                result['roi'] = self.last_valid_roi
            return result
        
        kpts = keypoints.xy[0].cpu().numpy()
        conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(17)
        
        result['all_keypoints'] = kpts
        result['keypoints'] = {
            'nose': kpts[self.NOSE],
            'left_shoulder': kpts[self.LEFT_SHOULDER],
            'right_shoulder': kpts[self.RIGHT_SHOULDER],
            'left_hip': kpts[self.LEFT_HIP],
            'right_hip': kpts[self.RIGHT_HIP]
        }
        
        # Calculate ROI based on mode
        roi, roi_conf = self._compute_roi(kpts, conf, frame.shape)
        
        if roi is None:
            if self.last_valid_roi is not None:
                result['roi'] = self.last_valid_roi
            return result
        
        result['confidence'] = roi_conf
        
        # Smooth ROI
        self.roi_history.append(roi)
        smoothed_roi = self._smooth_roi()
        
        result['roi'] = smoothed_roi
        self.last_valid_roi = smoothed_roi
        
        return result
    
    def _compute_roi(self, kpts: np.ndarray, conf: np.ndarray, 
                     frame_shape: Tuple[int, ...]) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Calculates ROI based on current mode."""
        h, w = frame_shape[:2]
        
        # Base keypoints
        left_shoulder = kpts[self.LEFT_SHOULDER]
        right_shoulder = kpts[self.RIGHT_SHOULDER]
        left_hip = kpts[self.LEFT_HIP]
        right_hip = kpts[self.RIGHT_HIP]
        nose = kpts[self.NOSE]
        
        # Confidence check for shoulders
        shoulder_valid = (conf[self.LEFT_SHOULDER] > 0.3 and left_shoulder[0] > 0 and
                         conf[self.RIGHT_SHOULDER] > 0.3 and right_shoulder[0] > 0)
        
        if not shoulder_valid:
            return None, 0.0
        
        # Shoulder center and width
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        
        # Hip check
        hip_valid = (conf[self.LEFT_HIP] > 0.3 and left_hip[0] > 0 and
                    conf[self.RIGHT_HIP] > 0.3 and right_hip[0] > 0)
        
        if hip_valid:
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            torso_height = hip_center_y - shoulder_center_y
        else:
            hip_center_x = shoulder_center_x
            torso_height = shoulder_width * 1.5
            hip_center_y = shoulder_center_y + torso_height
        
        torso_center_x = (shoulder_center_x + hip_center_x) / 2
        
        # ROI based on mode
        x_min, x_max, y_min, y_max, used_conf = self._get_roi_bounds(
            shoulder_center_x, shoulder_center_y, shoulder_width,
            hip_center_x, hip_center_y, torso_height, torso_center_x,
            nose, conf, hip_valid
        )
        
        if x_min is None:
            return None, 0.0
        
        # Ensure min < max
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        
        # Add padding
        width = x_max - x_min
        height = y_max - y_min
        
        if width <= 0 or height <= 0:
            return None, 0.0
            
        pad_x = int(width * self.roi_padding)
        pad_y = int(height * self.roi_padding)
        
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        
        roi = (x_min, y_min, x_max - x_min, y_max - y_min)
        avg_conf = float(np.mean(used_conf))
        
        return roi, avg_conf
    
    def _get_roi_bounds(self, shoulder_center_x, shoulder_center_y, shoulder_width,
                        hip_center_x, hip_center_y, torso_height, torso_center_x,
                        nose, conf, hip_valid):
        """Calculates ROI bounds for the current mode."""
        
        if self.roi_mode == ROIMode.FULL_THORAX:
            roi_width = shoulder_width * 1.0
            x_min = int(torso_center_x - roi_width / 2)
            x_max = int(torso_center_x + roi_width / 2)
            y_min = int(shoulder_center_y)
            y_max = int(hip_center_y)
            used_conf = [conf[self.LEFT_SHOULDER], conf[self.RIGHT_SHOULDER]]
            if hip_valid:
                used_conf.extend([conf[self.LEFT_HIP], conf[self.RIGHT_HIP]])
                
        elif self.roi_mode == ROIMode.UPPER_CHEST:
            roi_width = shoulder_width * 0.8
            x_min = int(shoulder_center_x - roi_width / 2)
            x_max = int(shoulder_center_x + roi_width / 2)
            y_min = int(shoulder_center_y)
            y_max = int(shoulder_center_y + torso_height * 0.5)
            used_conf = [conf[self.LEFT_SHOULDER], conf[self.RIGHT_SHOULDER]]
            
        elif self.roi_mode == ROIMode.JUGULUM:
            roi_width = shoulder_width * 0.5
            jugulum_height = torso_height * 0.25
            x_min = int(shoulder_center_x - roi_width / 2)
            x_max = int(shoulder_center_x + roi_width / 2)
            
            if conf[self.NOSE] > 0.3 and nose[1] > 0:
                y_min = int(nose[1] + (shoulder_center_y - nose[1]) * 0.5)
            else:
                y_min = int(shoulder_center_y - jugulum_height * 1.5)
            
            y_max = int(shoulder_center_y + jugulum_height * 0.5)
            used_conf = [conf[self.LEFT_SHOULDER], conf[self.RIGHT_SHOULDER]]
            
        elif self.roi_mode == ROIMode.ABDOMEN:
            roi_width = shoulder_width * 0.7
            abdomen_center_x = (shoulder_center_x + hip_center_x) / 2
            x_min = int(abdomen_center_x - roi_width / 2)
            x_max = int(abdomen_center_x + roi_width / 2)
            y_min = int(shoulder_center_y + torso_height * 0.4)
            y_max = int(hip_center_y)
            used_conf = [conf[self.LEFT_SHOULDER], conf[self.RIGHT_SHOULDER]]
            if hip_valid:
                used_conf.extend([conf[self.LEFT_HIP], conf[self.RIGHT_HIP]])
                
        elif self.roi_mode == ROIMode.SHOULDERS:
            roi_width = shoulder_width * 1.2
            shoulder_region_height = torso_height * 0.2
            x_min = int(shoulder_center_x - roi_width / 2)
            x_max = int(shoulder_center_x + roi_width / 2)
            y_min = int(shoulder_center_y - shoulder_region_height)
            y_max = int(shoulder_center_y + shoulder_region_height)
            used_conf = [conf[self.LEFT_SHOULDER], conf[self.RIGHT_SHOULDER]]
            
        else:
            return None, None, None, None, []
        
        return x_min, x_max, y_min, y_max, used_conf
    
    def _smooth_roi(self) -> Tuple[int, int, int, int]:
        """Smooths ROI over multiple frames"""
        if len(self.roi_history) == 0:
            return self.last_valid_roi
        
        rois = np.array(list(self.roi_history))
        smoothed = np.median(rois, axis=0).astype(int)
        
        return tuple(smoothed)
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Draws skeleton on the frame"""
        if keypoints is None:
            return frame
        
        frame = frame.copy()
        
        # Draw connections
        for connection in self.SKELETON_CONNECTIONS:
            pt1 = keypoints[connection[0]]
            pt2 = keypoints[connection[1]]
            
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(frame, 
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        (0, 255, 255), 2)
        
        # Draw points
        for i, pt in enumerate(keypoints):
            if pt[0] > 0 and pt[1] > 0:
                if i in [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, 
                        self.LEFT_HIP, self.RIGHT_HIP]:
                    color = (0, 255, 0)
                    radius = 6
                else:
                    color = (0, 165, 255)
                    radius = 4
                cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)
        
        return frame