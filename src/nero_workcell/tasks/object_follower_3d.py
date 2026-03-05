#!/usr/bin/env python3
# coding=utf-8
"""
Object following task (visual servoing).
Uses RealSense and YOLO to control the robot arm to follow a target object.

Usage:
    python -m nero_workcell.tasks.object_follower --target bottle --conf 0.5
"""

import logging
import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

from nero_workcell.core import NeroController, PIDController, RealSenseCamera
from nero_workcell.core.target_object import TargetObject

logger = logging.getLogger(__name__)


class ObjectFollower:
    def __init__(self, 
                 target_class: str, 
                 robot_channel: str = "can0",
                 model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.5,
                 target_distance: float = 0.3):
        
        self.target_class = target_class
        self.conf_threshold = conf_threshold
        self.target_distance = target_distance
        
        # Initialize camera.
        self.width = 640
        self.height = 480
        self.camera = None
        self.fx = 0.0
        self.fy = 0.0
        self.cx = 0.0
        self.cy = 0.0
        
        # Initialize YOLO.
        logger.info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        self.pid_x = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05) # 这里的单位是 m/s
        self.pid_y = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)
        self.pid_z = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)
        
        # Robot controller instance.
        self.robot = NeroController(robot_channel, "nero")
        self.is_running = False

    def setup_camera(self):
        """Auto-discover and connect to the first available RealSense camera."""
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未找到 RealSense 相机")
        
        camera_serial = devices[0].get_info(rs.camera_info.serial_number)
        logger.info(f"Using camera: {camera_serial}")
        self.camera = RealSenseCamera(width=self.width, height=self.height, fps=30, serial_number=camera_serial) 
        
        if not self.camera.start():
            raise RuntimeError("相机启动失败")
        
        # Read and store intrinsics.
        intrinsics = self.camera.get_intrinsics()
        self.fx = intrinsics.get('fx', 0)
        self.fy = intrinsics.get('fy', 0)
        self.cx = intrinsics.get('cx', 0)
        self.cy = intrinsics.get('cy', 0)
        logger.info(f"Camera intrinsics: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def _yolo_detect(self, color: np.ndarray, depth: np.ndarray) -> List[TargetObject]:
        """
        Run object detection and project valid detections into the camera frame.

        This method performs YOLO inference on the color image, filters detections
        by confidence and target classes, reads local depth around each box center,
        and converts the pixel center into a 3D point in camera
        coordinates (`position = [x, y, z]`, `frame="camera"`).

        Args:
            color: RGB/BGR color image used for YOLO inference.
            depth: Depth image aligned with `color`.

        Returns:
            A list of camera-frame detections. Each item follows `TargetObject`
            and includes class info, 2D box/center, confidence, and 3D position.
        """
        results = self.model(color, verbose=False)
        h, w = depth.shape
        target_objects: List[TargetObject] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                conf = float(box.conf[0])

                if conf < self.conf_threshold:
                    logger.debug(f"Skipping {cls_name}: conf {conf:.2f} < {self.conf_threshold}")
                    continue
                if cls_name != self.target_class:
                    logger.debug(f"Skipping {cls_name}: not target {self.target_class}")
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cu, cv_pt = (x1 + x2) // 2, (y1 + y2) // 2

                # Estimate depth with a local median window for robustness.
                region = depth[max(0, cv_pt - 5):min(h, cv_pt + 5), max(0, cu - 5):min(w, cu + 5)]
                valid = region[region > 0]
                d = np.median(valid) if len(valid) > 0 else 0
                if d <= 0:
                    logger.debug(f"Skipping {cls_name}: invalid depth {d}")
                    continue

                p_cam = np.array([(cu - self.cx) * d / self.fx, (cv_pt - self.cy) * d / self.fy, d])
                target_objects.append(
                    TargetObject(
                        name=cls_name,
                        class_id=cls_id,
                        bbox=(x1, y1, x2, y2),
                        center=(cu, cv_pt),
                        position=p_cam,
                        conf=conf,
                        frame="camera",
                    )
                )

        logger.debug("VisionDetector.detect: %d camera objects detected", len(target_objects))
        return target_objects

    def _pick_best_target(self, detected_targets: List[TargetObject]) -> Optional[TargetObject]:
        """Select the highest-confidence target among candidates of the same class."""
        candidates = [
            obj for obj in detected_targets
            if obj.name == self.target_class
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda obj: float(obj.conf))

    def detect_object(self):
        """
        Capture one frame and detect the target in base coordinates.

        Returns:
            dict | None: A payload with color image and base-frame target object.
            Returns None when frame read fails, target is missing, or pose read fails.
        """
        # Read image and depth frame.
        frame = self.camera.read_frame()
        color, depth = frame["color"], frame["depth"]
        if color is None or depth is None:
            logger.warning("[detect] Frame read failed")
            return None
        # Run YOLO detection.
        detected_targets = self._yolo_detect(color, depth)
        best_target = self._pick_best_target(detected_targets)
        if best_target is None:
            logger.warning("[detect] No target detected")
            return {"color": color, "target": None}

        # Get current flange pose.
        T_gripper2base = self.robot.get_current_pose()
        if T_gripper2base is None:
            logger.warning("[detect] Failed to get flange pose: no data")
            return {"color": color, "target": None}

        T_cam2base = T_gripper2base @ self.T_cam2gripper
        
        # p_base = T_cam2base * p_cam
        p_cam_homo = np.append(best_target.position, 1.0)
        p_base = (T_cam2base @ p_cam_homo)[:3]
        detected_target = TargetObject(
            name=best_target.name, 
            class_id=best_target.class_id, 
            bbox=best_target.bbox,
            center=best_target.center, 
            position=p_base, 
            conf=best_target.conf, 
            frame="base"
        )

        logger.info("[detect] Target detected: %s", detected_target)

        return {
            "color": color,
            "target": detected_target,
        }

    def follow_target(self, target: TargetObject):
        """
        Follow a target point in base coordinates (no grasping).

        Constraints:
        - target.position must be in the base frame.
        - Desired TCP is self.target_distance meters above the target point.
        """
        if not self.robot.is_connected():
            return

        if target.frame != "base":
            logger.warning("[follow] Target frame is not base, skipping frame: %s", target.frame)
            return

        tcp_pose = self.robot.get_tcp_pose()
        if tcp_pose is None:
            logger.warning("[follow] Failed to get TCP pose")
            return

        tcp_pos = np.array(tcp_pose[:3], dtype=float)
        target_pos = np.array(target.position, dtype=float)

        desired_pos = target_pos.copy()
        desired_pos[2] += self.target_distance

        error = desired_pos - tcp_pos
        err_x, err_y, err_z = error.tolist()

        deadband = 0.005  # 5mm
        if abs(err_x) < deadband and abs(err_y) < deadband and abs(err_z) < deadband:
            return

        # Reuse PID gains from 2D follower; scale error to millimeters.
        err_scale = 1000.0
        cmd_x = float(self.pid_x.compute(err_x * err_scale))
        cmd_y = float(self.pid_y.compute(err_y * err_scale))
        cmd_z = float(self.pid_z.compute(err_z * err_scale))

        max_step_z = 0.03
        cmd_z = float(np.clip(cmd_z, -max_step_z, max_step_z))

        self.robot.move_relative(dx=cmd_x, dy=cmd_y, dz=cmd_z)

    def run(self):
        # 1. Load eye-in-hand calibration.
        calib_file = Path("configs/eye_in_hand_calibration.json")
        if not calib_file.exists():
             # Fallback or check current directory
             calib_file = Path("eye_in_hand_calibration.json")

        try:
            with open(calib_file, 'r') as f:
                calib_data = json.load(f)
            self.T_cam2gripper = np.array(calib_data["homogeneous_matrix"])
        except Exception as e:
            logger.error(f"Failed to load calibration file: {e}")
            return

        # 2. Start camera and load intrinsics.
        self.setup_camera()

        # 3. Connect robot.
        if not self.robot.connect():
            logger.error("Robot connection failed, task aborted")
            return
        self.is_running = True
        
        logger.info(f"Starting follow task, target: {self.target_class}")
        logger.info("Press 's' to start following after target is detected")
        logger.info("Press 'q' to exit")

        center_x, center_y = self.width // 2, self.height // 2
        follow_enabled = False

        try:
            while self.is_running:
                result = self.detect_object()
                if result is None:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    continue
                
                display_img = result["color"].copy()
                target = result["target"]
                
                if target is None:
                    self.pid_x.reset()
                    self.pid_y.reset()
                    self.pid_z.reset()
                    status = "SEARCHING"
                    status_color = (0, 255, 255)
                else:
                    if follow_enabled:
                        # Execute follow control after manual trigger.
                        self.follow_target(target)
                        status = "FOLLOWING"
                        status_color = (0, 255, 0)
                    else:
                        status = "DETECTED: press 's' to follow"
                        status_color = (0, 255, 255)

                # Draw image-center crosshair.
                cv2.line(display_img, (center_x-20, center_y), (center_x+20, center_y), (255, 0, 0), 1)
                cv2.line(display_img, (center_x, center_y-20), (center_x, center_y+20), (255, 0, 0), 1)
                cv2.putText(display_img, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

                cv2.imshow("Object Follower 3D", display_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                if key == ord('s') and not follow_enabled and target is not None:
                    follow_enabled = True
                    logger.info("Follow triggered by key 's'")
                    self.follow_target(target)

        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            logger.info("Task finished")


def main():
    parser = argparse.ArgumentParser(description="Nero Workcell - Object Following Task")
    parser.add_argument("--target", type=str, default="bottle", help="Target object class name (e.g., bottle, cup)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Configure logging.
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )

    follower = ObjectFollower(
        target_class=args.target,
        robot_channel="can0",
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    follower.run()


if __name__ == "__main__":
    main()
