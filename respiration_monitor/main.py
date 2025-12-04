#!/usr/bin/env python3
"""
Respiration Monitor - Main Program
==================================

Real-time respiratory rate monitoring with YOLO Pose + Optical Flow.

Usage:
    python main.py                    # Default camera
    python main.py --source 1         # Different camera
    python main.py --source video.mp4 # Video file
    python main.py --roi-mode chest   # Upper chest only
    python main.py --no-raft          # Farnebäck instead of RAFT

Keys:
    q     - Quit
    r     - Reset signal
    s     - Toggle skeleton
    f     - Toggle flow display
    1-5   - Change ROI mode
    SPACE - Start/stop recording
"""

import argparse
import time
import cv2
import numpy as np
from typing import Optional, Union

from config import Config
from roi_detection import ROIMode, YOLOThoraxDetector
from optical_flow import OpticalFlowProcessor
from signal_analysis import RespirationAnalyzer
from visualization import Visualizer
from data_recorder import DataRecorder


class RespirationMonitor:
    """Main class for respiratory rate monitoring"""
    
    # Minimum ROI size for RAFT
    RAFT_MIN_W = 128
    RAFT_MIN_H = 128

    def __init__(self, config: Optional[Config] = None):
        """
        Args:
            config: Configuration parameters (or default)
        """
        self.config = config or Config()
        
        # Initialize components
        self.detector = YOLOThoraxDetector(
            model_path=self.config.yolo_model,
            confidence=self.config.yolo_confidence,
            roi_mode=self.config.roi_mode,
            roi_padding=self.config.roi_padding,
            roi_smoothing=self.config.roi_smoothing
        )
        
        self.optical_flow = OpticalFlowProcessor(
            use_raft=self.config.use_raft,
            use_raft_small=self.config.use_raft_small
        )
        
        self.analyzer = RespirationAnalyzer(
            target_fps=self.config.target_fps,
            buffer_seconds=self.config.buffer_seconds,
            min_seconds=self.config.min_seconds,
            filter_low=self.config.filter_low,
            filter_high=self.config.filter_high,
            filter_order=self.config.filter_order
        )
        
        self.visualizer = Visualizer(
            signal_plot_width=self.config.signal_plot_width,
            signal_plot_height=self.config.signal_plot_height
        )
        
        self.recorder = DataRecorder()

        # State
        self.prev_roi_frame = None
        self.start_time = None
        self.frame_count = 0
        
        # FPS tracking
        self.fps_counter = 0
        self.fps_time = time.time()
        self.current_fps = 0

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Processes a single frame.
        
        Args:
            frame: BGR image from camera
            
        Returns:
            dict with all results
        """
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

        # ROI detection
        detection = self.detector.detect(frame)
        roi = detection["roi"]
        result["roi"] = roi
        result["keypoints"] = detection["all_keypoints"]

        if roi is None:
            self.prev_roi_frame = None
            return result

        x, y, w, h = roi
        roi_frame = frame[y:y+h, x:x+w]

        # Check minimum size for RAFT
        if w < self.RAFT_MIN_W or h < self.RAFT_MIN_H:
            self.prev_roi_frame = None
            return result

        # Calculate optical flow
        if self.prev_roi_frame is not None:
            prev_h, prev_w = self.prev_roi_frame.shape[:2]
            curr_h, curr_w = roi_frame.shape[:2]

            if (prev_w >= self.RAFT_MIN_W and prev_h >= self.RAFT_MIN_H and
                curr_w >= self.RAFT_MIN_W and curr_h >= self.RAFT_MIN_H):

                # Normalize to common size
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
                    pass

        self.prev_roi_frame = roi_frame.copy()

        # Analyze signal
        rr, conf = self.analyzer.analyze()
        result["respiration_rate"] = rr
        result["confidence"] = conf
        result["signal"] = self.analyzer.get_signal_for_plot()

        # Recording
        if result["roi"] is not None and result["vertical_motion"] != 0:
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

        # Progress
        result["progress"] = min(self.analyzer.get_buffer_fill(), 1.0)

        return result
    
    def render(self, result: dict) -> np.ndarray:
        """
        Renders the visualization.
        
        Args:
            result: Result from process_frame()
            
        Returns:
            Display image
        """
        frame = result['frame'].copy()
        
        # Skeleton
        if self.config.show_skeleton and result['keypoints'] is not None:
            frame = self.detector.draw_skeleton(frame, result['keypoints'])
        
        # ROI with mode color
        if result['roi'] is not None:
            color = self.detector.get_roi_color()
            frame = self.visualizer.draw_roi(frame, result['roi'], color)
            
            x, y, w, h = result['roi']
            mode_name = self.detector.roi_mode.value.upper()
            cv2.putText(frame, mode_name, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Info box

        frame = self.visualizer.draw_info_box(
            frame,
            result['respiration_rate'],
            result['confidence'],
            self.current_fps,
            result['progress'],
            median_rr=self.analyzer.get_median_rr()
        )
        
        # ROI mode legend
        frame = self.visualizer.draw_roi_mode_legend(frame, self.detector.roi_mode)
        
        # Signal plot
        if self.config.show_signal and len(result['signal']) > 0:
            plot = self.visualizer.draw_signal_plot(result['signal'])
            
            fh, fw = frame.shape[:2]
            ph, pw = plot.shape[:2]
            y_pos = fh - ph - 10
            frame[y_pos:y_pos + ph, 10:10 + pw] = plot
            
            cv2.putText(frame, "Breathing Signal", (15, y_pos - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Flow visualization
        if self.config.show_flow and result['flow'] is not None:
            flow_vis = self.optical_flow.flow_to_color(result['flow'])
            flow_vis = cv2.resize(flow_vis, (160, 120))
            
            fh, fw = frame.shape[:2]
            frame[140:260, fw - 170:fw - 10] = flow_vis
            cv2.putText(frame, "Optical Flow", (fw - 165, 135),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Recording indicator
        if self.recorder.is_recording():
            frame = self.visualizer.draw_recording_indicator(
                frame, self.recorder.get_duration()
            )
        
        return frame
    
    def update_fps(self):
        """Updates FPS counter"""
        self.fps_counter += 1
        if time.time() - self.fps_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_time = time.time()
    
    def _change_roi_mode(self, mode: ROIMode):
        """Changes ROI mode and resets signal"""
        self.detector.set_roi_mode(mode)
        self.analyzer.reset()
        self.prev_roi_frame = None
    
    def run(self, video_source: Union[int, str] = 0):
        """
        Starts real-time monitoring.
        
        Args:
            video_source: Camera ID (int) or video file path (str)
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print(f"Error: Could not open video source '{video_source}'")
            return
        
        # Camera properties
        camera_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video: {frame_width}x{frame_height} @ {camera_fps} FPS")
        print(f"Target FPS: {self.config.target_fps}")
        
        frame_skip = max(1, int(camera_fps / self.config.target_fps))
        print(f"Frame skip: {frame_skip}")
        
        self._print_help()
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(video_source, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            frame_idx += 1
            
            # Frame skipping
            if frame_idx % frame_skip != 0:
                continue
            
            # Process and render
            result = self.process_frame(frame)
            display = self.render(result)
            
            self.update_fps()
            
            cv2.imshow("Respiratory Rate Monitor (YOLO + Optical Flow)", display)
            
            # Process keyboard input
            if self._handle_key_input(cv2.waitKey(1) & 0xFF):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Monitoring ended.")
    
    def _handle_key_input(self, key: int) -> bool:
        """
        Processes keyboard input.
        
        Returns:
            True if program should exit
        """
        if key == ord('q'):
            if self.recorder.is_recording():
                self.recorder.stop_recording()
            return True
        
        elif key == ord(' '):
            if self.recorder.is_recording():
                self.recorder.stop_recording()
            else:
                self.recorder.start_recording()
        
        elif key == ord('r'):
            self.analyzer.reset()
            self.prev_roi_frame = None
            print("Signal reset")
        
        elif key == ord('s'):
            self.config.show_skeleton = not self.config.show_skeleton
            print(f"Skeleton: {'on' if self.config.show_skeleton else 'off'}")
        
        elif key == ord('f'):
            self.config.show_flow = not self.config.show_flow
            print(f"Flow display: {'on' if self.config.show_flow else 'off'}")
        
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
        
        return False
    
    def _print_help(self):
        """Prints help text"""
        print("\nKeys:")
        print("  q     - Quit")
        print("  r     - Reset (clear signal)")
        print("  s     - Toggle skeleton")
        print("  f     - Toggle flow display")
        print("  1     - ROI: Full Thorax (shoulders-hips)")
        print("  2     - ROI: Upper Chest")
        print("  3     - ROI: Jugulum (décolletage)")
        print("  4     - ROI: Abdomen")
        print("  5     - ROI: Shoulders")
        print("  SPACE - Start/stop recording")
        print()


def parse_args():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(
        description="Respiratory rate monitoring with YOLO Pose + Optical Flow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Default camera
  python main.py --source 1         # Different camera
  python main.py --source video.mp4 # Video file
  python main.py --roi-mode chest   # Upper chest only
  python main.py --no-raft          # Farnebäck instead of RAFT
        """
    )
    
    parser.add_argument('--source', type=str, default='0',
                       help='Video source: camera ID or file path (default: 0)')
    
    parser.add_argument('--fps', type=int, default=10,
                       help='Target FPS for processing (default: 10)')
    
    parser.add_argument('--buffer', type=int, default=40,
                       help='Buffer size in seconds (default: 40)')
    
    parser.add_argument('--yolo-model', type=str, default='yolov8n-pose.pt',
                       help='YOLO Pose model (default: yolov8n-pose.pt)')
    
    parser.add_argument('--roi-mode', type=str, default='chest',
                       choices=['full', 'chest', 'jugulum', 'abdomen', 'shoulders'],
                       help='ROI mode (default: chest)')
    
    parser.add_argument('--no-raft', action='store_true',
                       help='Use Farnebäck instead of RAFT')
    
    parser.add_argument('--no-skeleton', action='store_true',
                       help='Disable skeleton display')
    
    parser.add_argument('--no-flow', action='store_true',
                       help='Disable flow visualization')
    
    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()
    
    # Parse ROI mode
    roi_mode_map = {
        'full': ROIMode.FULL_THORAX,
        'chest': ROIMode.UPPER_CHEST,
        'jugulum': ROIMode.JUGULUM,
        'abdomen': ROIMode.ABDOMEN,
        'shoulders': ROIMode.SHOULDERS,
    }
    roi_mode = roi_mode_map.get(args.roi_mode, ROIMode.FULL_THORAX)
    
    # Create configuration
    config = Config(
        target_fps=args.fps,
        buffer_seconds=args.buffer,
        yolo_model=args.yolo_model,
        roi_mode=roi_mode,
        use_raft=not args.no_raft,
        show_skeleton=not args.no_skeleton,
        show_flow=not args.no_flow
    )
    
    # Parse video source
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Start monitor
    monitor = RespirationMonitor(config=config)
    monitor.run(video_source=source)


if __name__ == "__main__":
    main()