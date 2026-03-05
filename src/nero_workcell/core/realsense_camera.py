"""
Intel RealSense D435i camera implementation.
Supports RGB + depth streams.
"""
import logging
import time
from typing import Any, Dict

import numpy as np
import pyrealsense2 as rs

logger = logging.getLogger(__name__)


class RealSenseCamera:
    """Intel RealSense D435i camera."""
    
    def __init__(self,
                 width: int = 640,
                 height: int = 480,
                 fps: int = 30,
                 *,
                 serial_number: str):
        """
        Initialize the RealSense camera.

        Args:
            width: Image width.
            height: Image height.
            fps: Frame rate.
            serial_number: Camera serial number (required).
        """
        if not isinstance(serial_number, str) or not serial_number.strip():
            raise ValueError("serial_number 是必填参数，不能为空")

        self.width = width
        self.height = height
        self.fps = fps
        self._is_opened = False

        self.frames_captured = 0
        self.failed_reads = 0
        self.start_time = 0

        self.serial_number = serial_number.strip()
        
        self.pipeline = None
        self.config = None
        self.align = None
        self.profile = None
        self.depth_scale = 1.0
    
    def start(self) -> bool:
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            # Bind to the target device by serial number.
            self.config.enable_device(self.serial_number)
            
            # Always enable depth and color streams.
            self.config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps
            )
            self.config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps
            )
            
            # Start the RealSense pipeline.
            self.profile = self.pipeline.start(self.config)
            
            # Always align depth to color.
            self.align = rs.align(rs.stream.color)

            # Read depth scale.
            depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            # Read device info.
            device = self.profile.get_device()
            device_name = device.get_info(rs.camera_info.name)
            device_serial = device.get_info(rs.camera_info.serial_number)
            
            logger.info("[RealSenseCamera] Camera started")
            logger.info("  Device: %s", device_name)
            logger.info("  Serial: %s", device_serial)
            logger.info("  Resolution: %sx%s", self.width, self.height)
            logger.info("  FPS: %s", self.fps)
            logger.info("  Depth scale: %s", self.depth_scale)
            
            # Warm up camera frames.
            logger.info("[RealSenseCamera] Warming up...")
            for _ in range(30):
                self.pipeline.wait_for_frames()
            
            self.start_time = time.time()
            self._is_opened = True
            logger.info("[RealSenseCamera] Warm-up completed")
            return True
            
        except Exception as e:
            logger.error("[RealSenseCamera] Start failed: %s", e)
            try:
                if self.pipeline is not None:
                    self.pipeline.stop()
            except Exception as stop_error:
                logger.debug("[RealSenseCamera] Stop after start failure failed: %s", stop_error)
            self.pipeline = None
            self.config = None
            self.align = None
            self.profile = None
            self._is_opened = False
            return False

    def read_frame(self) -> Dict[str, Any]:
        """
        Read one frame and return a unified payload.

        Returns:
            dict: {
                'color': np.ndarray | None,  # BGR image (H, W, 3)
                'depth': np.ndarray | None,  # Depth map (H, W), in meters
                'timestamp': float,          # Unix timestamp
            }

        Raises:
            RuntimeError: Raised when camera is not started.
        """
        if not self._is_opened:
            raise RuntimeError("Camera is not opened. Call start() first.")

        try:
            result = self._read_frame_raw()
            result["timestamp"] = time.time()

            if result.get("color") is not None or result.get("depth") is not None:
                self.frames_captured += 1
            else:
                self.failed_reads += 1
            return result
        except Exception as e:
            self.failed_reads += 1
            logger.error("[RealSenseCamera] Read frame failed: %s", e)
            return {"color": None, "depth": None, "timestamp": time.time()}
    
    def _read_frame_raw(self) -> Dict[str, Any]:
        if self.pipeline is None:
            return {'color': None, 'depth': None}
        
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            
            # Align depth frame to color frame.
            frames = self.align.process(frames)
            
            result = {'color': None, 'depth': None}
            
            # Read color frame.
            color_frame = frames.get_color_frame()
            if color_frame:
                result['color'] = np.asanyarray(color_frame.get_data())

            # Read depth frame and convert to meters.
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth_image = np.asanyarray(depth_frame.get_data())
                result['depth'] = depth_image.astype(np.float32) * self.depth_scale
            
            return result
            
        except Exception as e:
            logger.error("[RealSenseCamera] Read raw frame failed: %s", e)
            return {'color': None, 'depth': None}
    
    @property
    def is_opened(self) -> bool:
        return self._is_opened

    def get_stats(self) -> Dict[str, Any]:
        total = self.frames_captured + self.failed_reads
        success_rate = (self.frames_captured / total * 100) if total > 0 else 0
        return {
            "frames_captured": self.frames_captured,
            "failed_reads": self.failed_reads,
            "success_rate": success_rate,
        }

    def print_stats(self):
        stats = self.get_stats()
        logger.info("[Camera] Statistics")
        logger.info("  Captured frames: %s", stats["frames_captured"])
        logger.info("  Failed reads: %s", stats["failed_reads"])
        logger.info("  Success rate: %.1f%%", stats["success_rate"])
    
    def get_intrinsics(self) -> Dict[str, Any]:
        if not self._is_opened or self.profile is None:
            return {}
        
        try:
            stream = self.profile.get_stream(rs.stream.color)
            intrinsics = stream.as_video_stream_profile().get_intrinsics()
            return {
                'width': intrinsics.width,
                'height': intrinsics.height,
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'cx': intrinsics.ppx,
                'cy': intrinsics.ppy,
                'distortion_model': str(intrinsics.model),
                'distortion_coeffs': list(intrinsics.coeffs),
            }
        except Exception as e:
            logger.error("[RealSenseCamera] Get intrinsics failed: %s", e)
            return {}
    
    def stop(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
        
        self._is_opened = False
        self.print_stats()
        logger.info("[RealSenseCamera] Stopped")
