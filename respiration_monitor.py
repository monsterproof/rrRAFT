"""
Echtzeit-Atemfrequenz-Monitoring mit YOLO Pose + Optical Flow
=============================================================

Pipeline zur kontaktlosen Messung der Atemfrequenz über RGB-Kamera.
Verwendet YOLOv8 Pose für Thorax-Detektion und RAFT/Farnebäck für Bewegungsanalyse.

"""

import cv2
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum
import time
from scipy.signal import butter, filtfilt, welch
from scipy.ndimage import median_filter
from ultralytics import YOLO


class ROIMode(Enum):
    """Verschiedene ROI-Modi für die Atemanalyse"""
    FULL_THORAX = "full"          # Schultern bis Hüften (Original)
    UPPER_CHEST = "chest"         # Nur oberer Brustkorb (Schultern bis Mitte)
    JUGULUM = "jugulum"           # Jugulum/Dekolleté-Bereich
    ABDOMEN = "abdomen"           # Bauchregion (unter Brust bis Hüften)
    SHOULDERS = "shoulders"       # Nur Schulterbereich


@dataclass
class Config:
    """Konfigurationsparameter für die Pipeline"""
    # Kamera
    camera_id: int = 0
    camera_fps: int = 30
    target_fps: int = 10  # Reduzierte FPS für Verarbeitung
    
    # YOLO
    yolo_model: str = "yolov8n-pose.pt"  # n=nano, s=small, m=medium
    yolo_confidence: float = 0.5
    
    # ROI
    roi_mode: ROIMode = ROIMode.FULL_THORAX  # ROI-Modus
    roi_padding: float = 0.15  # Padding um Thorax-ROI
    roi_smoothing: int = 5     # Frames für ROI-Glättung
    
    # Signal Processing
    buffer_seconds: int = 40   # Sekunden für RR-Berechnung
    min_seconds: int = 5       # Minimum für erste Schätzung
    filter_low: float = 0.1    # Hz (6 BPM)
    filter_high: float = 0.75   # Hz (45 BPM)
    filter_order: int = 4
    
    # Optical Flow
    use_raft: bool = True      # RAFT oder Farnebäck
    use_raft_small: bool = True
    
    # Display
    show_skeleton: bool = True
    show_flow: bool = True
    show_signal: bool = True
    signal_plot_width: int = 400
    signal_plot_height: int = 120


class YOLOThoraxDetector:
    """
    Erkennt verschiedene Körperregionen mittels YOLOv8 Pose.
    
    COCO Keypoint-Indizes:
    0: Nase, 1-2: Augen, 3-4: Ohren
    5: L_Schulter, 6: R_Schulter
    7: L_Ellbogen, 8: R_Ellbogen
    9: L_Handgelenk, 10: R_Handgelenk
    11: L_Hüfte, 12: R_Hüfte
    13: L_Knie, 14: R_Knie
    15: L_Knöchel, 16: R_Knöchel
    
    ROI-Modi:
    - FULL_THORAX: Schultern bis Hüften
    - UPPER_CHEST: Oberer Brustkorb (Schultern bis Mitte Thorax)
    - JUGULUM: Jugulum/Dekolleté (zwischen Hals und Schultern)
    - ABDOMEN: Bauchregion (Mitte bis Hüften)
    - SHOULDERS: Nur Schulterbereich
    """
    
    # Keypoint-Indizes
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
    
    # Für Skeleton-Zeichnung
    SKELETON_CONNECTIONS = [
        (5, 6),   # Schultern
        (5, 7),   # L Schulter - Ellbogen
        (7, 9),   # L Ellbogen - Handgelenk
        (6, 8),   # R Schulter - Ellbogen
        (8, 10),  # R Ellbogen - Handgelenk
        (5, 11),  # L Schulter - Hüfte
        (6, 12),  # R Schulter - Hüfte
        (11, 12), # Hüften
        (11, 13), # L Hüfte - Knie
        (13, 15), # L Knie - Knöchel
        (12, 14), # R Hüfte - Knie
        (14, 16), # R Knie - Knöchel
    ]
    
    # ROI-Farben für verschiedene Modi
    ROI_COLORS = {
        ROIMode.FULL_THORAX: (0, 255, 0),    # Grün
        ROIMode.UPPER_CHEST: (255, 200, 0),  # Cyan
        ROIMode.JUGULUM: (0, 165, 255),      # Orange
        ROIMode.ABDOMEN: (255, 0, 255),      # Magenta
        ROIMode.SHOULDERS: (255, 255, 0),    # Gelb
    }
    
    def __init__(self, config: Config):
        self.config = config
        self.model = YOLO(config.yolo_model)
        self.roi_mode = config.roi_mode
        
        # ROI-Glättung
        self.roi_history: deque = deque(maxlen=config.roi_smoothing)
        self.last_valid_roi: Optional[Tuple[int, int, int, int]] = None
        
    def set_roi_mode(self, mode: ROIMode):
        """Ändert den ROI-Modus und resettet den History-Buffer"""
        if mode != self.roi_mode:
            self.roi_mode = mode
            self.roi_history.clear()
            self.last_valid_roi = None
            print(f"ROI-Modus geändert zu: {mode.value}")
    
    def get_roi_color(self) -> Tuple[int, int, int]:
        """Gibt die Farbe für den aktuellen ROI-Modus zurück"""
        return self.ROI_COLORS.get(self.roi_mode, (0, 255, 0))
        
    def detect(self, frame: np.ndarray) -> dict:
        """
        Erkennt Pose und extrahiert ROI basierend auf aktuellem Modus.
        
        Returns:
            dict mit 'roi', 'keypoints', 'confidence', 'roi_mode'
        """
        results = self.model(
            frame, 
            verbose=False, 
            conf=self.config.yolo_confidence
        )
        
        result = {
            'roi': None,
            'keypoints': None,
            'confidence': 0.0,
            'all_keypoints': None,
            'roi_mode': self.roi_mode
        }
        
        # Keine Person erkannt
        if len(results[0].keypoints) == 0:
            if self.last_valid_roi is not None:
                result['roi'] = self.last_valid_roi
            return result
        
        # Erste Person nehmen
        keypoints = results[0].keypoints
        
        if keypoints.xy is None or len(keypoints.xy) == 0:
            if self.last_valid_roi is not None:
                result['roi'] = self.last_valid_roi
            return result
        
        kpts = keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
        conf = keypoints.conf[0].cpu().numpy() if keypoints.conf is not None else np.ones(17)
        
        result['all_keypoints'] = kpts
        result['keypoints'] = {
            'nose': kpts[self.NOSE],
            'left_shoulder': kpts[self.LEFT_SHOULDER],
            'right_shoulder': kpts[self.RIGHT_SHOULDER],
            'left_hip': kpts[self.LEFT_HIP],
            'right_hip': kpts[self.RIGHT_HIP]
        }
        
        # ROI basierend auf Modus berechnen
        roi, roi_conf = self._compute_roi(kpts, conf, frame.shape)
        
        if roi is None:
            if self.last_valid_roi is not None:
                result['roi'] = self.last_valid_roi
            return result
        
        result['confidence'] = roi_conf
        
        # ROI glätten
        self.roi_history.append(roi)
        smoothed_roi = self._smooth_roi()
        
        result['roi'] = smoothed_roi
        self.last_valid_roi = smoothed_roi
        
        return result
    
    def _compute_roi(self, kpts: np.ndarray, conf: np.ndarray, 
                 frame_shape: Tuple[int, ...]) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """
        Berechnet ROI basierend auf aktuellem Modus.
        
        Returns:
            (roi_tuple, confidence) oder (None, 0.0)
        """
        h, w = frame_shape[:2]
        
        # Basis-Keypoints holen
        left_shoulder = kpts[self.LEFT_SHOULDER]
        right_shoulder = kpts[self.RIGHT_SHOULDER]
        left_hip = kpts[self.LEFT_HIP]
        right_hip = kpts[self.RIGHT_HIP]
        nose = kpts[self.NOSE]
        
        # Konfidenz-Check für Schultern (immer benötigt)
        shoulder_valid = (conf[self.LEFT_SHOULDER] > 0.3 and left_shoulder[0] > 0 and
                        conf[self.RIGHT_SHOULDER] > 0.3 and right_shoulder[0] > 0)
        
        if not shoulder_valid:
            return None, 0.0
        
        # Schulter-Mitte und -Breite berechnen
        shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        shoulder_width = abs(right_shoulder[0] - left_shoulder[0])
        
        # Hüften-Check
        hip_valid = (conf[self.LEFT_HIP] > 0.3 and left_hip[0] > 0 and
                    conf[self.RIGHT_HIP] > 0.3 and right_hip[0] > 0)
        
        if hip_valid:
            hip_center_x = (left_hip[0] + right_hip[0]) / 2
            hip_center_y = (left_hip[1] + right_hip[1]) / 2
            torso_height = hip_center_y - shoulder_center_y
        else:
            # Schätze Torso-Höhe basierend auf Schulterbreite
            hip_center_x = shoulder_center_x
            torso_height = shoulder_width * 1.5
            hip_center_y = shoulder_center_y + torso_height
        
        # Torso-Mitte (zwischen Schultern und Hüften)
        torso_center_x = (shoulder_center_x + hip_center_x) / 2
        
        # ROI basierend auf Modus - IMMER von der Mitte aus berechnet
        if self.roi_mode == ROIMode.FULL_THORAX:
            # Schultern bis Hüften, volle Breite
            roi_width = shoulder_width * 1.0
            x_min = int(torso_center_x - roi_width / 2)
            x_max = int(torso_center_x + roi_width / 2)
            y_min = int(shoulder_center_y)
            y_max = int(hip_center_y)
            used_conf = [conf[self.LEFT_SHOULDER], conf[self.RIGHT_SHOULDER]]
            if hip_valid:
                used_conf.extend([conf[self.LEFT_HIP], conf[self.RIGHT_HIP]])
                
        elif self.roi_mode == ROIMode.UPPER_CHEST:
            # Oberer Brustkorb: 80% Schulterbreite, Schultern bis Mitte Torso
            roi_width = shoulder_width * 0.8
            x_min = int(shoulder_center_x - roi_width / 2)
            x_max = int(shoulder_center_x + roi_width / 2)
            y_min = int(shoulder_center_y)
            y_max = int(shoulder_center_y + torso_height * 0.5)
            used_conf = [conf[self.LEFT_SHOULDER], conf[self.RIGHT_SHOULDER]]
            
        elif self.roi_mode == ROIMode.JUGULUM:
            # Jugulum/Dekolleté: schmaler Bereich zwischen Hals und Schultern
            roi_width = shoulder_width * 0.5
            jugulum_height = torso_height * 0.25
            
            x_min = int(shoulder_center_x - roi_width / 2)
            x_max = int(shoulder_center_x + roi_width / 2)
            
            # Oberhalb der Schultern, aber unterhalb der Nase
            if conf[self.NOSE] > 0.3 and nose[1] > 0:
                y_min = int(nose[1] + (shoulder_center_y - nose[1]) * 0.5)
            else:
                y_min = int(shoulder_center_y - jugulum_height * 1.5)
            
            y_max = int(shoulder_center_y + jugulum_height * 0.5)
            used_conf = [conf[self.LEFT_SHOULDER], conf[self.RIGHT_SHOULDER]]
            
        elif self.roi_mode == ROIMode.ABDOMEN:
            # Bauchregion: 70% Schulterbreite, Mitte bis Hüften
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
            # Schulterbereich: volle Breite + 10%, schmale Höhe
            roi_width = shoulder_width * 1.2
            shoulder_region_height = torso_height * 0.2
            
            x_min = int(shoulder_center_x - roi_width / 2)
            x_max = int(shoulder_center_x + roi_width / 2)
            y_min = int(shoulder_center_y - shoulder_region_height)
            y_max = int(shoulder_center_y + shoulder_region_height)
            used_conf = [conf[self.LEFT_SHOULDER], conf[self.RIGHT_SHOULDER]]
            
        else:
            return None, 0.0
        
        # Sicherstellen, dass min < max (Fallback falls doch was schiefgeht)
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        
        # Padding hinzufügen
        width = x_max - x_min
        height = y_max - y_min
        
        if width <= 0 or height <= 0:
            return None, 0.0
            
        pad_x = int(width * self.config.roi_padding)
        pad_y = int(height * self.config.roi_padding)
        
        x_min = max(0, x_min - pad_x)
        x_max = min(w, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(h, y_max + pad_y)
        
        roi = (x_min, y_min, x_max - x_min, y_max - y_min)
        avg_conf = float(np.mean(used_conf))
        
        return roi, avg_conf
    
    def _smooth_roi(self) -> Tuple[int, int, int, int]:
        """Glättet ROI über mehrere Frames"""
        if len(self.roi_history) == 0:
            return self.last_valid_roi
        
        rois = np.array(list(self.roi_history))
        smoothed = np.median(rois, axis=0).astype(int)
        
        return tuple(smoothed)
    
    def draw_skeleton(self, frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        """Zeichnet Skeleton auf das Frame"""
        if keypoints is None:
            return frame
        
        frame = frame.copy()
        
        # Verbindungen zeichnen
        for connection in self.SKELETON_CONNECTIONS:
            pt1 = keypoints[connection[0]]
            pt2 = keypoints[connection[1]]
            
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(frame, 
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        (0, 255, 255), 2)
        
        # Punkte zeichnen
        for i, pt in enumerate(keypoints):
            if pt[0] > 0 and pt[1] > 0:
                # Thorax-Punkte hervorheben
                if i in [self.LEFT_SHOULDER, self.RIGHT_SHOULDER, 
                        self.LEFT_HIP, self.RIGHT_HIP]:
                    color = (0, 255, 0)
                    radius = 6
                else:
                    color = (0, 165, 255)
                    radius = 4
                cv2.circle(frame, (int(pt[0]), int(pt[1])), radius, color, -1)
        
        return frame


class OpticalFlowProcessor:
    """Berechnet Optical Flow mit RAFT oder Farnebäck"""
    
    def __init__(self, config: Config):
        self.config = config
        
        # Device-Auswahl: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        if config.use_raft:
            try:
                self._init_raft()
                self.use_raft = True
                print(f"RAFT initialisiert auf {self.device}")
            except Exception as e:
                print(f"RAFT nicht verfügbar ({e}), verwende Farnebäck")
                self.use_raft = False
        else:
            self.use_raft = False
            print("Verwende Farnebäck Optical Flow")
        
        # Farnebäck Parameter
        self.farneback_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
    
    def _init_raft(self):
        """Initialisiert RAFT Modell"""
        if self.config.use_raft_small:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights)
        else:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights)
        
        self.model = self.model.to(self.device).eval()
    
    def compute(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        Berechnet Optical Flow.
        
        Returns:
            Flow array mit Shape (H, W, 2) - [vx, vy]
        """
        if self.use_raft:
            return self._compute_raft(frame1, frame2)
        else:
            return self._compute_farneback(frame1, frame2)
    
    def _compute_raft(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """RAFT Optical Flow"""
        # BGR zu RGB
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Zu Tensor und normalisieren (0-255 -> 0-1)
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        
        # Batch dimension hinzufügen
        img1_batch = img1_tensor.unsqueeze(0).to(self.device)
        img2_batch = img2_tensor.unsqueeze(0).to(self.device)
        
        # Bilder müssen durch 8 teilbar sein für RAFT
        _, _, h, w = img1_batch.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img1_batch = torch.nn.functional.pad(img1_batch, [0, pad_w, 0, pad_h])
            img2_batch = torch.nn.functional.pad(img2_batch, [0, pad_w, 0, pad_h])
        
        with torch.no_grad():
            # Neuere torchvision API: gibt Liste von Flow-Tensoren zurück
            flow_list = self.model(img1_batch, img2_batch)
            
            # Letztes Element ist der finale Flow
            if isinstance(flow_list, list):
                flow = flow_list[-1]
            else:
                flow = flow_list
        
        # Padding entfernen falls nötig
        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :h, :w]
        
        # (1, 2, H, W) -> (H, W, 2)
        return flow[0].permute(1, 2, 0).cpu().numpy()
    
    def _compute_farneback(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Farnebäck Optical Flow"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, **self.farneback_params
        )
        
        return flow  # Shape: (H, W, 2)
    
    @staticmethod
    def extract_vertical_motion(flow: np.ndarray, roi_mask: Optional[np.ndarray] = None) -> float:
        """
        Extrahiert vertikale Bewegungskomponente.
        
        Args:
            flow: Shape (H, W, 2)
            roi_mask: Optionale Maske
            
        Returns:
            Median der vertikalen Bewegung
        """
        vy = flow[:, :, 1]
        
        if roi_mask is not None:
            vy = vy[roi_mask > 0]
        
        if len(vy) == 0:
            return 0.0
        
        return float(np.median(vy))
    
    @staticmethod
    def flow_to_color(flow: np.ndarray, max_mag: float = None) -> np.ndarray:
        """Konvertiert Flow zu HSV-Farbdarstellung"""
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        
        if max_mag is None:
            max_mag = np.percentile(mag, 99) + 1e-5
        
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[:, :, 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class RespirationAnalyzer:
    """Analysiert das Atmungssignal und berechnet die Atemfrequenz"""

    def __init__(self, config: Config):
        self.config = config
        self.buffer_size = config.buffer_seconds * config.target_fps
        self.min_samples = config.min_seconds * config.target_fps

        self.signal_buffer: deque = deque(maxlen=self.buffer_size)
        self.time_buffer: deque = deque(maxlen=self.buffer_size)

        self.filter_order = config.filter_order
        self.filter_low = config.filter_low
        self.filter_high = config.filter_high

        self.current_rr: float = 0.0
        self.current_confidence: float = 0.0
        self.filtered_signal: np.ndarray = np.array([])
        self.actual_fs: float = config.target_fps

        # neue SMOOTHING-Variablen
        self.freq_history = deque(maxlen=20)
        self.rr_exp = None
        self.rr_smooth = None

        # Kalibrierung
        self.calibration_seconds = 5.0
        self.calibration_complete = False
        self._calibration_start = None


    def reset(self):
        self.signal_buffer.clear()
        self.time_buffer.clear()
        self.current_rr = 0.0
        self.current_confidence = 0.0
        self.filtered_signal = np.array([])
        self.actual_fs = self.config.target_fps
        self.calibration_complete = False
        self._calibration_start = None

        # SMOOTHING-Zustände resetten
        self.freq_history.clear()
        self.rr_exp = None
        self.rr_smooth = None


    def add_sample(self, value: float, timestamp: float):

        if not self.calibration_complete:
            if self._calibration_start is None:
                self._calibration_start = timestamp
            if timestamp - self._calibration_start < self.calibration_seconds:
                return
            self.calibration_complete = True

        # Ausreißer-Schutz
        if len(self.signal_buffer) > 10:
            recent = list(self.signal_buffer)[-10:]
            median = np.median(recent)
            mad = np.median(np.abs(np.array(recent) - median))
            threshold = 5 * (mad + 0.1)
            if abs(value - median) > threshold:
                return

        self.signal_buffer.append(value)
        self.time_buffer.append(timestamp)


    def _compute_actual_fs(self):
        if len(self.time_buffer) < 2:
            return self.config.target_fps

        t = np.array(self.time_buffer)
        duration = t[-1] - t[0]
        if duration <= 0:
            return self.config.target_fps

        fs = (len(t) - 1) / duration
        if fs < 1 or fs > self.config.target_fps * 2:
            return self.config.target_fps
        return fs


    def analyze(self):

        if len(self.signal_buffer) < self.min_samples:
            return 0.0, 0.0

        signal = np.array(self.signal_buffer)
        self.actual_fs = self._compute_actual_fs()

        # Preprocessing
        signal = signal - np.mean(signal)
        signal = median_filter(signal, size=3)

        # Filter
        if len(signal) > 3 * self.filter_order:
            try:
                b, a = self._compute_filter_coefficients(self.actual_fs)
                self.filtered_signal = filtfilt(b, a, signal)
            except Exception:
                self.filtered_signal = signal
        else:
            self.filtered_signal = signal

        # Welch
        nperseg = min(len(self.filtered_signal), int(self.actual_fs * 20))
        try:
            freq, psd = welch(
                self.filtered_signal,
                fs=self.actual_fs,
                nperseg=nperseg,
                noverlap=nperseg // 2
            )
        except Exception:
            return self.current_rr, self.current_confidence

        mask = (freq >= self.filter_low) & (freq <= self.filter_high)
        freq_range = freq[mask]
        psd_range = psd[mask]
        if len(psd_range) == 0:
            return self.current_rr, self.current_confidence

        # Peak
        peak_idx = np.argmax(psd_range)
        peak_freq = freq_range[peak_idx]
        peak_power = psd_range[peak_idx]

        # -------------------------
        # KONFIDENZ – VOR SMOOTHING!
        # -------------------------
        total_power = np.sum(psd_range)
        if total_power > 0:
            confidence = peak_power / total_power
        else:
            confidence = 0.0

        noise_floor = np.median(psd_range)
        if noise_floor > 0:
            snr = peak_power / noise_floor
            if snr < 2:
                confidence *= 0.5

        # -------------------------
        # 3-STUFIGE GLÄTTUNG
        # -------------------------
        peak_bpm = peak_freq * 60

        # (1) Median über letzte Peaks
        self.freq_history.append(peak_bpm)
        if len(self.freq_history) >= 5:
            peak_bpm = float(np.median(list(self.freq_history)[-5:]))

        # (2) Exponentielle Glättung
        if self.rr_exp is None:
            self.rr_exp = peak_bpm
        self.rr_exp = 0.2 * peak_bpm + 0.8 * self.rr_exp

        # (3) Low-Pass je nach confidence
        if self.rr_smooth is None:
            self.rr_smooth = self.rr_exp

        beta = 0.15 if confidence > 0.25 else 0.05
        self.rr_smooth = self.rr_smooth + beta * (self.rr_exp - self.rr_smooth)

        # Ergebnis
        self.current_rr = float(self.rr_smooth)
        self.current_confidence = float(min(confidence, 1.0))

        return self.current_rr, self.current_confidence

    
    def get_signal_for_plot(self) -> np.ndarray:
        """Gibt das gefilterte Signal für Visualisierung zurück"""
        return self.filtered_signal
    
    def get_actual_fs(self) -> float:
        """Gibt die tatsächliche Sample-Rate zurück"""
        return self.actual_fs
    
    def reset(self):
        """Setzt die Buffer zurück"""
        self.signal_buffer.clear()
        self.time_buffer.clear()
        self.current_rr = 0.0
        self.current_confidence = 0.0
        self.filtered_signal = np.array([])
        self.actual_fs = self.config.target_fps
        self.calibration_complete = False


class DataRecorder:
    """Zeichnet alle Messwerte für spätere Analyse auf"""
    
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = output_dir
        self.recording = False
        self.start_time: Optional[float] = None
        self.session_id: Optional[str] = None
        
        # Rohdaten
        self.timestamps: List[float] = []
        self.vertical_motion_raw: List[float] = []
        self.roi_sizes: List[Tuple[int, int]] = []  # (width, height)
        self.confidences: List[float] = []
        self.roi_modes: List[str] = []
        
        # Analysierte Daten (werden periodisch gespeichert)
        self.analysis_timestamps: List[float] = []
        self.respiration_rates: List[float] = []
        self.rr_confidences: List[float] = []
        self.actual_fs_values: List[float] = []
        
        # Gefilterte Signale (gespeichert bei Analyse)
        self.filtered_signals: List[np.ndarray] = []
        self.filtered_signal_timestamps: List[float] = []
    
    def start_recording(self):
        """Startet eine neue Aufzeichnung"""
        import os
        from datetime import datetime
        
        # Verzeichnis erstellen
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Session ID aus Timestamp
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = time.time()
        self.recording = True
        
        # Buffer leeren
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
        
        print(f"Aufzeichnung gestartet: {self.session_id}")
    
    def stop_recording(self) -> str:
        """Stoppt die Aufzeichnung und speichert alle Daten"""
        if not self.recording:
            return ""
        
        self.recording = False
        filepath = self._save_data()
        print(f"Aufzeichnung gespeichert: {filepath}")
        return filepath
    
    def add_sample(self, timestamp: float, vertical_motion: float, 
                   roi_size: Tuple[int, int], confidence: float, roi_mode: str):
        """Fügt einen Rohdaten-Sample hinzu"""
        if not self.recording:
            return
        
        self.timestamps.append(timestamp)
        self.vertical_motion_raw.append(vertical_motion)
        self.roi_sizes.append(roi_size)
        self.confidences.append(confidence)
        self.roi_modes.append(roi_mode)
    
    def add_analysis(self, timestamp: float, rr: float, confidence: float, 
                     actual_fs: float, filtered_signal: np.ndarray):
        """Fügt Analyse-Ergebnisse hinzu"""
        if not self.recording:
            return
        
        self.analysis_timestamps.append(timestamp)
        self.respiration_rates.append(rr)
        self.rr_confidences.append(confidence)
        self.actual_fs_values.append(actual_fs)
        
        # Gefiltertes Signal kopieren
        self.filtered_signals.append(filtered_signal.copy())
        self.filtered_signal_timestamps.append(timestamp)
    
    def _save_data(self) -> str:
        """Speichert alle Daten in CSV und NPZ Dateien"""
        import os
        
        base_path = os.path.join(self.output_dir, self.session_id)
        
        # 1. Rohdaten als CSV
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
        
        # 2. Analyse-Ergebnisse als CSV
        analysis_csv_path = f"{base_path}_analysis.csv"
        with open(analysis_csv_path, 'w') as f:
            f.write("timestamp,respiration_rate,confidence,actual_fs\n")
            for i in range(len(self.analysis_timestamps)):
                f.write(f"{self.analysis_timestamps[i]:.4f},"
                       f"{self.respiration_rates[i]:.2f},"
                       f"{self.rr_confidences[i]:.4f},"
                       f"{self.actual_fs_values[i]:.2f}\n")
        
        # 3. Signale als NPZ (komprimiertes NumPy Format)
        npz_path = f"{base_path}_signals.npz"
        np.savez_compressed(
            npz_path,
            timestamps=np.array(self.timestamps),
            vertical_motion_raw=np.array(self.vertical_motion_raw),
            analysis_timestamps=np.array(self.analysis_timestamps),
            respiration_rates=np.array(self.respiration_rates),
            # Letztes gefiltertes Signal (vollständig)
            filtered_signal=self.filtered_signals[-1] if self.filtered_signals else np.array([]),
            # Metadata
            session_id=self.session_id,
            duration=self.timestamps[-1] if self.timestamps else 0
        )
        
        # 4. Zusammenfassung
        summary_path = f"{base_path}_summary.txt"
        with open(summary_path, 'w') as f:
            duration = self.timestamps[-1] if self.timestamps else 0
            avg_rr = np.mean(self.respiration_rates) if self.respiration_rates else 0
            std_rr = np.std(self.respiration_rates) if self.respiration_rates else 0
            avg_fs = np.mean(self.actual_fs_values) if self.actual_fs_values else 0
            
            f.write(f"Session: {self.session_id}\n")
            f.write(f"Dauer: {duration:.1f} Sekunden\n")
            f.write(f"Samples: {len(self.timestamps)}\n")
            f.write(f"Durchschnittliche Sample-Rate: {avg_fs:.2f} Hz\n")
            f.write(f"Atemfrequenz: {avg_rr:.1f} ± {std_rr:.1f} /min\n")
            f.write(f"\nDateien:\n")
            f.write(f"  - {raw_csv_path}\n")
            f.write(f"  - {analysis_csv_path}\n")
            f.write(f"  - {npz_path}\n")
        
        return base_path
    
    def is_recording(self) -> bool:
        return self.recording
    
    def get_duration(self) -> float:
        if not self.recording or not self.timestamps:
            return 0.0
        return self.timestamps[-1]

class Visualizer:
    """Visualisierung der Ergebnisse"""
    
    @staticmethod
    def draw_roi(frame: np.ndarray, roi: Tuple[int, int, int, int], 
                 color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Zeichnet ROI-Rechteck"""
        x, y, w, h = roi
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        return frame
    
    @staticmethod
    def draw_signal_plot(signal: np.ndarray, width: int = 400, height: int = 120,
                         color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """Zeichnet Signal-Plot"""
        plot = np.zeros((height, width, 3), dtype=np.uint8)
        plot[:] = (30, 30, 30)
        
        if len(signal) < 2:
            return plot
        
        # Signal normalisieren
        signal_centered = signal - np.mean(signal)
        max_val = np.max(np.abs(signal_centered))
        if max_val > 0:
            signal_norm = signal_centered / max_val
        else:
            signal_norm = signal_centered
        
        # Punkte berechnen
        margin = 10
        x_points = np.linspace(margin, width - margin, len(signal_norm)).astype(int)
        y_points = ((1 - signal_norm) * (height - 2 * margin) / 2 + margin).astype(int)
        y_points = np.clip(y_points, margin, height - margin)
        
        # Linie zeichnen
        points = np.column_stack((x_points, y_points))
        cv2.polylines(plot, [points], False, color, 2, cv2.LINE_AA)
        
        # Mittellinie
        cv2.line(plot, (margin, height // 2), (width - margin, height // 2), 
                (80, 80, 80), 1)
        
        # Rahmen
        cv2.rectangle(plot, (0, 0), (width - 1, height - 1), (100, 100, 100), 1)
        
        return plot
    
    @staticmethod
    def draw_info_box(frame: np.ndarray, rr: float, confidence: float,
                      fps: int, progress: float = 1.0) -> np.ndarray:
        """Zeichnet Info-Box mit RR und Konfidenz"""
        # Hintergrund
        cv2.rectangle(frame, (10, 10), (340, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (340, 130), (100, 100, 100), 1)
        
        # Titel
        cv2.putText(frame, "Atemfrequenz-Monitor", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        if progress < 1.0:
            # Noch am Sammeln
            cv2.putText(frame, "Sammle Daten...", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            
            # Fortschrittsbalken
            bar_width = 280
            bar_height = 15
            filled = int(bar_width * progress)
            cv2.rectangle(frame, (20, 90), (20 + bar_width, 90 + bar_height), (80, 80, 80), -1)
            cv2.rectangle(frame, (20, 90), (20 + filled, 90 + bar_height), (0, 200, 0), -1)
            cv2.putText(frame, f"{progress:.0%}", (310 - 40, 103),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            if confidence < 0.1:
                rr = 0

            # RR anzeigen
            if rr > 0:
                # Farbe basierend auf Konfidenz
                if confidence > 0.40:
                    color = (0, 255, 0)  # Grün
                elif confidence > 0.2:
                    color = (0, 255, 255)  # Gelb
                else:
                    color = (0, 165, 255)  # Orange
                
                cv2.putText(frame, f"{rr:.0f}", (20, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
                cv2.putText(frame, "/min", (140, 85),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                # Konfidenz-Balken
                conf_width = int(150 * confidence)
                cv2.rectangle(frame, (20, 100), (170, 115), (80, 80, 80), -1)
                cv2.rectangle(frame, (20, 100), (20 + conf_width, 115), color, -1)
                cv2.putText(frame, f"Konfidenz: {confidence:.0%}", (180, 112),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            else:
                cv2.putText(frame, "Keine Messung", (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        
        # FPS
        cv2.putText(frame, f"FPS: {fps}", (280, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame


class RespirationMonitor:

    RAFT_MIN_W = 128
    RAFT_MIN_H = 128

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.detector = YOLOThoraxDetector(self.config)
        self.optical_flow = OpticalFlowProcessor(self.config)
        self.analyzer = RespirationAnalyzer(self.config)
        self.visualizer = Visualizer()
        self.recorder = DataRecorder()

        self.prev_roi_frame = None
        self.start_time = None
        self.frame_count = 0

        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0


    def process_frame(self, frame):

        if self.start_time is None:
            self.start_time = time.time()

        timestamp = time.time() - self.start_time

        result = {
            "frame": frame,
            "timestamp": timestamp,
            "roi": None,
            "keypoints": None,
            "flow": None,
            "vertical_motion": 0.0,
            "respiration_rate": 0.0,
            "confidence": 0.0,
            "signal": np.array([]),
            "progress": 0.0
        }

        # Detection
        detection = self.detector.detect(frame)
        roi = detection["roi"]
        result["roi"] = roi
        result["keypoints"] = detection["all_keypoints"]

        if roi is None:
            self.prev_roi_frame = None
            return result

        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]

        # ------------------------------------------------
        # RAFT MINDESTGRÖSSE SCHUTZ
        # ------------------------------------------------
        if w < self.RAFT_MIN_W or h < self.RAFT_MIN_H:
            self.prev_roi_frame = None
            result["vertical_motion"] = 0.0
            return result

        # Optical Flow
        if self.prev_roi_frame is not None:

            prev_h, prev_w = self.prev_roi_frame.shape[:2]
            curr_h, curr_w = roi_frame.shape[:2]

            # Mindestgrößen sicherstellen
            if (prev_w >= self.RAFT_MIN_W and prev_h >= self.RAFT_MIN_H and
                curr_w >= self.RAFT_MIN_W and curr_h >= self.RAFT_MIN_H):

                # auf gemeinsame Größe normalisieren
                tw = min(prev_w, curr_w)
                th = min(prev_h, curr_h)

                prev_res = cv2.resize(self.prev_roi_frame, (tw, th))
                curr_res = cv2.resize(roi_frame, (tw, th))

                try:
                    flow = self.optical_flow.compute(prev_res, curr_res)
                    result["flow"] = flow

                    vertical_motion = self.optical_flow.extract_vertical_motion(flow)
                    result["vertical_motion"] = vertical_motion

                    self.analyzer.add_sample(vertical_motion, timestamp)

                except Exception:
                    # Fallback: keine Bewegung
                    result["vertical_motion"] = 0.0

        # Save for next iteration
        self.prev_roi_frame = roi_frame.copy()

        # Analyse
        rr, conf = self.analyzer.analyze()
        result["respiration_rate"] = rr
        result["confidence"] = conf
        result["signal"] = self.analyzer.get_signal_for_plot()

        # Recording
        if result["roi"] is not None:
            if result["vertical_motion"] != 0:
                self.recorder.add_sample(
                    timestamp=timestamp,
                    vertical_motion=result["vertical_motion"],
                    roi_size=(w, h),
                    confidence=detection.get("confidence", 0),
                    roi_mode=self.detector.roi_mode.value
                )

        if rr > 0:
            self.recorder.add_analysis(
                timestamp=timestamp,
                rr=rr,
                confidence=conf,
                actual_fs=self.analyzer.get_actual_fs(),
                filtered_signal=self.analyzer.get_signal_for_plot()
            )

        # Fortschritt
        progress = len(self.analyzer.signal_buffer) / self.analyzer.min_samples
        result["progress"] = min(progress, 1.0)

        return result
    
    def render(self, result: dict) -> np.ndarray:
        """Rendert die Visualisierung"""
        frame = result['frame'].copy()
        
        # Skeleton zeichnen
        if self.config.show_skeleton and result['keypoints'] is not None:
            frame = self.detector.draw_skeleton(frame, result['keypoints'])
        
        # ROI zeichnen mit Modus-spezifischer Farbe
        if result['roi'] is not None:
            color = self.detector.get_roi_color()
            frame = self.visualizer.draw_roi(frame, result['roi'], color)
            
            # ROI-Modus anzeigen
            x, y, w, h = result['roi']
            mode_name = self.detector.roi_mode.value.upper()
            cv2.putText(frame, mode_name, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Info-Box
        frame = self.visualizer.draw_info_box(
            frame,
            result['respiration_rate'],
            result['confidence'],
            self.current_fps,
            result['progress']
        )
        
        # ROI-Modus Legende (rechts oben)
        self._draw_roi_mode_legend(frame)
        
        # Signal-Plot
        if self.config.show_signal and len(result['signal']) > 0:
            plot = self.visualizer.draw_signal_plot(
                result['signal'],
                self.config.signal_plot_width,
                self.config.signal_plot_height
            )
            
            # Plot positionieren (unten links)
            fh, fw = frame.shape[:2]
            ph, pw = plot.shape[:2]
            y_pos = fh - ph - 10
            frame[y_pos:y_pos + ph, 10:10 + pw] = plot
            
            cv2.putText(frame, "Atemsignal", (15, y_pos - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Flow Visualisierung
        if self.config.show_flow and result['flow'] is not None:
            flow_vis = self.optical_flow.flow_to_color(result['flow'])
            flow_vis = cv2.resize(flow_vis, (160, 120))
            
            fh, fw = frame.shape[:2]
            frame[140:260, fw - 170:fw - 10] = flow_vis
            cv2.putText(frame, "Optical Flow", (fw - 165, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Aufzeichnungsstatus
        if self.recorder.is_recording():
            # Roter Punkt + "REC"
            cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (frame.shape[1] - 70, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Aufzeichnungsdauer
            duration = self.recorder.get_duration()
            cv2.putText(frame, f"{duration:.1f}s", (frame.shape[1] - 70, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        return frame
    
    def _draw_roi_mode_legend(self, frame: np.ndarray):
        """Zeichnet die ROI-Modus Legende"""
        fh, fw = frame.shape[:2]
        
        # Hintergrund
        legend_x = fw - 180
        legend_y = 270
        legend_h = 130
        legend_w = 170
        
        cv2.rectangle(frame, (legend_x, legend_y), 
                     (legend_x + legend_w, legend_y + legend_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (legend_x, legend_y), 
                     (legend_x + legend_w, legend_y + legend_h), (100, 100, 100), 1)
        
        cv2.putText(frame, "ROI-Modus (1-5):", (legend_x + 5, legend_y + 18),
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
            color = self.detector.ROI_COLORS[mode]
            
            # Aktuellen Modus hervorheben
            if mode == self.detector.roi_mode:
                cv2.putText(frame, ">", (legend_x + 5, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            cv2.putText(frame, text, (legend_x + 18, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def update_fps(self):
        """Aktualisiert FPS-Zähler"""
        self.fps_counter += 1
        if time.time() - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = time.time()
    
    def run(self, video_source=0):
        """
        Startet das Echtzeit-Monitoring.
        
        Args:
            video_source: Kamera-ID (int) oder Videodatei-Pfad (str)
        """
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Fehler: Konnte Videoquelle '{video_source}' nicht öffnen")
            return
        
        # Kamera-Eigenschaften
        camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {frame_width}x{frame_height} @ {camera_fps} FPS")
        print(f"Ziel-FPS: {self.config.target_fps}")
        
        frame_skip = max(1, int(camera_fps / self.config.target_fps))
        print(f"Frame-Skip: {frame_skip}")
        
        print("\nTasten:")
        print("  q - Beenden")
        print("  r - Reset (Signal zurücksetzen)")
        print("  s - Skeleton ein/aus")
        print("  f - Flow-Anzeige ein/aus")
        print("  1 - ROI: Full Thorax (Schultern-Hüften)")
        print("  2 - ROI: Upper Chest (Oberer Brustkorb)")
        print("  3 - ROI: Jugulum (Dekolleté)")
        print("  4 - ROI: Abdomen (Bauch)")
        print("  5 - ROI: Shoulders (Schultern)")
        print("  SPACE - Aufzeichnung starten/stoppen") 
        print()
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                # Bei Videodatei: Loop oder Ende
                if isinstance(video_source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            frame_idx += 1
            
            # Frame-Skipping
            if frame_idx % frame_skip != 0:
                continue
            
            # Verarbeiten
            result = self.process_frame(frame)
            
            # Rendern
            display = self.render(result)
            
            # FPS aktualisieren
            self.update_fps()
            
            # Anzeigen
            cv2.imshow("Atemfrequenz-Monitor (YOLO + Optical Flow)", display)
            
            # Tastatureingabe
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if self.recorder.is_recording():
                    self.recorder.stop_recording()
                break
            elif key == ord(' '):  # SPACE
                if self.recorder.is_recording():
                    self.recorder.stop_recording()
                else:
                    self.recorder.start_recording()
            elif key == ord('r'):
                self.analyzer.reset()
                self.prev_roi_frame = None
                print("Signal zurückgesetzt")
            elif key == ord('s'):
                self.config.show_skeleton = not self.config.show_skeleton
                print(f"Skeleton: {'an' if self.config.show_skeleton else 'aus'}")
            elif key == ord('f'):
                self.config.show_flow = not self.config.show_flow
                print(f"Flow-Anzeige: {'an' if self.config.show_flow else 'aus'}")
            # ROI-Modus Tasten
            elif key == ord('1'):
                self._change_roi_mode(ROIMode.FULL_THORAX)
            elif key == ord('2'):
                self._change_roi_mode(ROIMode.UPPER_CHEST)
            elif key == ord('3'):
                self._change_roi_mode(ROIMode.JUGULUM)
            elif key == ord('4'):
                self._change_roi_mode(ROIMode.ABDOMEN)
            elif key == ord('5'):
                self._change_roi_mode(ROIMode.SHOULDERS)
        
        cap.release()
        cv2.destroyAllWindows()
        print("Monitoring beendet.")
    
    def _change_roi_mode(self, mode: ROIMode):
        """Wechselt den ROI-Modus und resettet das Signal"""
        self.detector.set_roi_mode(mode)
        self.analyzer.reset()
        self.prev_roi_frame = None


def main():
    """Hauptfunktion"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Atemfrequenz-Monitoring mit YOLO Pose + Optical Flow"
    )
    parser.add_argument('--source', type=str, default='0',
                       help='Videoquelle: Kamera-ID oder Dateipfad')
    parser.add_argument('--fps', type=int, default=10,
                       help='Ziel-FPS für Verarbeitung (default: 10)')
    parser.add_argument('--buffer', type=int, default=40,
                       help='Puffergrösse in Sekunden (default: 30)')
    parser.add_argument('--yolo-model', type=str, default='yolov8n-pose.pt',
                       help='YOLO Pose Modell (default: yolov8n-pose.pt)')
    parser.add_argument('--roi-mode', type=str, default='chest',
                       choices=['full', 'chest', 'jugulum', 'abdomen', 'shoulders'],
                       help='ROI-Modus: full, chest, jugulum, abdomen, shoulders')
    parser.add_argument('--no-raft', action='store_true',
                       help='Farnebäck statt RAFT verwenden')
    parser.add_argument('--no-skeleton', action='store_true',
                       help='Skeleton-Anzeige deaktivieren')
    parser.add_argument('--no-flow', action='store_true',
                       help='Flow-Visualisierung deaktivieren')
    
    args = parser.parse_args()
    
    # ROI-Modus parsen
    roi_mode_map = {
        'full': ROIMode.FULL_THORAX,
        'chest': ROIMode.UPPER_CHEST,
        'jugulum': ROIMode.JUGULUM,
        'abdomen': ROIMode.ABDOMEN,
        'shoulders': ROIMode.SHOULDERS,
    }
    roi_mode = roi_mode_map.get(args.roi_mode, ROIMode.FULL_THORAX)
    
    # Konfiguration
    config = Config(
        target_fps=args.fps,
        buffer_seconds=args.buffer,
        yolo_model=args.yolo_model,
        roi_mode=roi_mode,
        use_raft=not args.no_raft,
        show_skeleton=not args.no_skeleton,
        show_flow=not args.no_flow
    )
    
    # Videoquelle
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Monitor starten
    monitor = RespirationMonitor(config=config)
    monitor.run(video_source=source)


if __name__ == "__main__":
    main()