"""
Optical Flow Processor
======================

Computes optical flow for motion analysis.
Supports RAFT (deep learning) and Farnebäck (classical).
"""

import cv2
import numpy as np
import torch
from typing import Optional


class OpticalFlowProcessor:
    """Computes optical flow using RAFT or Farnebäck"""
    
    def __init__(self, use_raft: bool = True, use_raft_small: bool = True):
        """
        Args:
            use_raft: Use RAFT (if available)
            use_raft_small: Use RAFT Small instead of Large
        """
        # Device selection: CUDA > MPS (Apple Silicon) > CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        self.use_raft_small = use_raft_small
        
        if use_raft:
            try:
                self._init_raft()
                self.use_raft = True
                print(f"RAFT initialized on {self.device}")
            except Exception as e:
                print(f"RAFT not available ({e}), using Farnebäck")
                self.use_raft = False
        else:
            self.use_raft = False
            print("Using Farnebäck optical flow")
        
        # Farnebäck parameters
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
        """Initializes RAFT model"""
        if self.use_raft_small:
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
        Computes optical flow.
        
        Args:
            frame1: First frame (BGR)
            frame2: Second frame (BGR)
            
        Returns:
            Flow array with shape (H, W, 2) - [vx, vy]
        """
        if self.use_raft:
            return self._compute_raft(frame1, frame2)
        else:
            return self._compute_farneback(frame1, frame2)
    
    def _compute_raft(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """RAFT optical flow"""
        # BGR to RGB
        img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize
        img1_tensor = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0
        img2_tensor = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
        
        # Batch dimension
        img1_batch = img1_tensor.unsqueeze(0).to(self.device)
        img2_batch = img2_tensor.unsqueeze(0).to(self.device)
        
        # RAFT requires dimensions divisible by 8
        _, _, h, w = img1_batch.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        
        if pad_h > 0 or pad_w > 0:
            img1_batch = torch.nn.functional.pad(img1_batch, [0, pad_w, 0, pad_h])
            img2_batch = torch.nn.functional.pad(img2_batch, [0, pad_w, 0, pad_h])
        
        with torch.no_grad():
            flow_list = self.model(img1_batch, img2_batch)
            flow = flow_list[-1] if isinstance(flow_list, list) else flow_list
        
        # Remove padding
        if pad_h > 0 or pad_w > 0:
            flow = flow[:, :, :h, :w]
        
        # (1, 2, H, W) -> (H, W, 2)
        return flow[0].permute(1, 2, 0).cpu().numpy()
    
    def _compute_farneback(self, frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """Farnebäck optical flow"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, **self.farneback_params
        )
        
        return flow
    
    @staticmethod
    def extract_vertical_motion(flow: np.ndarray, 
                                roi_mask: Optional[np.ndarray] = None) -> float:
        """
        Extracts vertical motion component.
        
        Args:
            flow: Shape (H, W, 2)
            roi_mask: Optional mask
            
        Returns:
            Median of vertical motion
        """
        vy = flow[:, :, 1]
        
        if roi_mask is not None:
            vy = vy[roi_mask > 0]
        
        if len(vy) == 0:
            return 0.0
        
        return float(np.median(vy))
    
    @staticmethod
    def flow_to_color(flow: np.ndarray, max_mag: float = None) -> np.ndarray:
        """
        Converts flow to HSV color representation.
        
        Args:
            flow: Shape (H, W, 2)
            max_mag: Maximum magnitude for normalization
            
        Returns:
            BGR color image
        """
        mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1])
        
        if max_mag is None:
            max_mag = np.percentile(mag, 99) + 1e-5
        
        hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
        hsv[:, :, 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)