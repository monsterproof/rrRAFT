"""
Respiration Signal Analysis
===========================

Analyzes the breathing signal and calculates respiratory rate
using bandpass filtering and Welch spectral analysis.
"""

import numpy as np
from collections import deque
from typing import Optional
from scipy.signal import butter, filtfilt, welch
from scipy.ndimage import median_filter


class RespirationAnalyzer:
    """Analyzes the breathing signal and calculates respiratory rate"""

    def __init__(self, 
                 target_fps: int = 10,
                 buffer_seconds: int = 40,
                 min_seconds: int = 5,
                 filter_low: float = 0.1,
                 filter_high: float = 0.75,
                 filter_order: int = 4,
                 calibration_seconds: float = 5.0,
                 median_window_seconds: float = 20.0):
        """
        Args:
            target_fps: Expected sample rate
            buffer_seconds: Buffer size for RR calculation
            min_seconds: Minimum for first estimate
            filter_low: Lower cutoff frequency (Hz)
            filter_high: Upper cutoff frequency (Hz)
            filter_order: Butterworth filter order
            calibration_seconds: Calibration time at start
            median_window_seconds: Time window for median RR calculation
        """
        self.target_fps = target_fps
        self.buffer_size = buffer_seconds * target_fps
        self.min_samples = min_seconds * target_fps
        
        self.filter_order = filter_order
        self.filter_low = filter_low
        self.filter_high = filter_high

        self.signal_buffer: deque = deque(maxlen=self.buffer_size)
        self.time_buffer: deque = deque(maxlen=self.buffer_size)

        self.current_rr: float = 0.0
        self.current_confidence: float = 0.0
        self.filtered_signal: np.ndarray = np.array([])
        self.actual_fs: float = target_fps

        # Smoothing variables
        self.freq_history = deque(maxlen=20)
        self.rr_exp = None
        self.rr_smooth = None

        # Calibration
        self.calibration_seconds = calibration_seconds
        self.calibration_complete = False
        self._calibration_start = None

        # Median RR tracking (stores (timestamp, rr) tuples)
        self.median_window_seconds = median_window_seconds
        self.rr_history: deque = deque()

    def reset(self):
        """Resets all buffers and states"""
        self.signal_buffer.clear()
        self.time_buffer.clear()
        self.current_rr = 0.0
        self.current_confidence = 0.0
        self.filtered_signal = np.array([])
        self.actual_fs = self.target_fps
        self.calibration_complete = False
        self._calibration_start = None

        self.freq_history.clear()
        self.rr_exp = None
        self.rr_smooth = None
        
        self.rr_history.clear()

    def add_sample(self, value: float, timestamp: float):
        """
        Adds a new sample.
        
        Args:
            value: Vertical motion
            timestamp: Timestamp
        """
        # Calibration phase
        if not self.calibration_complete:
            if self._calibration_start is None:
                self._calibration_start = timestamp
            if timestamp - self._calibration_start < self.calibration_seconds:
                return
            self.calibration_complete = True

        # Outlier protection
        if len(self.signal_buffer) > 10:
            recent = list(self.signal_buffer)[-10:]
            median = np.median(recent)
            mad = np.median(np.abs(np.array(recent) - median))
            threshold = 5 * (mad + 0.1)
            if abs(value - median) > threshold:
                return

        self.signal_buffer.append(value)
        self.time_buffer.append(timestamp)

    def _compute_actual_fs(self) -> float:
        """Calculates the actual sample rate"""
        if len(self.time_buffer) < 2:
            return self.target_fps

        t = np.array(self.time_buffer)
        duration = t[-1] - t[0]
        if duration <= 0:
            return self.target_fps

        fs = (len(t) - 1) / duration
        if fs < 1 or fs > self.target_fps * 2:
            return self.target_fps
        return fs

    def _compute_filter_coefficients(self, fs: float):
        """Calculates Butterworth filter coefficients"""
        nyq = fs / 2
        low = max(self.filter_low / nyq, 0.01)
        high = min(self.filter_high / nyq, 0.99)
        
        if low >= high:
            low = 0.01
            high = 0.99
            
        return butter(self.filter_order, [low, high], btype='band')

    def _update_rr_history(self, rr: float, timestamp: float):
        """Updates the RR history for median calculation"""
        if rr > 0:
            self.rr_history.append((timestamp, rr))
            
            # Remove samples outside the time window
            cutoff = timestamp - self.median_window_seconds
            while self.rr_history and self.rr_history[0][0] < cutoff:
                self.rr_history.popleft()

    def analyze(self) -> tuple[float, float]:
        """
        Analyzes the signal and calculates respiratory rate.
        
        Returns:
            (respiration_rate, confidence)
        """
        if len(self.signal_buffer) < self.min_samples:
            return 0.0, 0.0

        signal = np.array(self.signal_buffer)
        self.actual_fs = self._compute_actual_fs()

        # Preprocessing
        signal = signal - np.mean(signal)
        signal = median_filter(signal, size=3)

        # Bandpass filter
        if len(signal) > 3 * self.filter_order:
            try:
                b, a = self._compute_filter_coefficients(self.actual_fs)
                self.filtered_signal = filtfilt(b, a, signal)
            except Exception:
                self.filtered_signal = signal
        else:
            self.filtered_signal = signal

        # Welch spectral analysis
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

        # Filter frequency range
        mask = (freq >= self.filter_low) & (freq <= self.filter_high)
        freq_range = freq[mask]
        psd_range = psd[mask]
        
        if len(psd_range) == 0:
            return self.current_rr, self.current_confidence

        # Find peak
        peak_idx = np.argmax(psd_range)
        peak_freq = freq_range[peak_idx]
        peak_power = psd_range[peak_idx]

        # Calculate confidence
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

        # 3-stage smoothing
        peak_bpm = peak_freq * 60

        # 1. Median over recent peaks
        self.freq_history.append(peak_bpm)
        if len(self.freq_history) >= 5:
            peak_bpm = float(np.median(list(self.freq_history)[-5:]))

        # 2. Exponential smoothing
        if self.rr_exp is None:
            self.rr_exp = peak_bpm
        self.rr_exp = 0.2 * peak_bpm + 0.8 * self.rr_exp

        # 3. Low-pass depending on confidence
        if self.rr_smooth is None:
            self.rr_smooth = self.rr_exp

        beta = 0.15 if confidence > 0.25 else 0.05
        self.rr_smooth = self.rr_smooth + beta * (self.rr_exp - self.rr_smooth)

        self.current_rr = float(self.rr_smooth)
        self.current_confidence = float(min(confidence, 1.0))

        # Update RR history for median calculation
        if self.time_buffer:
            self._update_rr_history(self.current_rr, self.time_buffer[-1])

        return self.current_rr, self.current_confidence
    
    def get_median_rr(self) -> Optional[float]:
        """
        Returns the median RR over the time window.
        
        Returns:
            Median RR or None if insufficient data
        """
        if len(self.rr_history) < 3:
            return None
        
        rr_values = [rr for _, rr in self.rr_history]
        return float(np.median(rr_values))
    
    def get_rr_sample_count(self) -> int:
        """Returns the number of RR samples in the median window"""
        return len(self.rr_history)
    
    def get_signal_for_plot(self) -> np.ndarray:
        """Returns the filtered signal for visualization"""
        return self.filtered_signal
    
    def get_actual_fs(self) -> float:
        """Returns the actual sample rate"""
        return self.actual_fs
    
    def get_buffer_fill(self) -> float:
        """Returns the buffer fill level (0-1)"""
        return len(self.signal_buffer) / self.min_samples