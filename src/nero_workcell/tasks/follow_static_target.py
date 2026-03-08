#!/usr/bin/env python3
# coding=utf-8
"""
Static-target following task (visual servoing).
Uses RealSense and YOLO to control the robot arm to approach a locked target object.

Usage:
    python -m nero_workcell.tasks.follow_static_target --target cup --conf 0.5
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from nero_workcell.core import (
    DifferentialIKFollower,
    KinematicsModel,
    RealSenseCamera,
    YOLODetector,
)
from nero_workcell.core.target_object import TargetObject
from nero_workcell.utils.common import load_eye_in_hand_calibration, transform_to_base

logger = logging.getLogger(__name__)


def build_follower(
    *,
    robot_channel: str,
    target_distance: float,
    urdf_path: str,
    tcp_frame: str,
    approach_direction: tuple[float, float, float],
) -> DifferentialIKFollower:
    """Construct the Pinocchio-based follower."""
    model = KinematicsModel(
        urdf_path=urdf_path,
        tcp_frame=tcp_frame,
    )
    return DifferentialIKFollower(
        model=model,
        robot_channel=robot_channel,
        standoff_distance=target_distance,
        approach_direction=np.array(approach_direction, dtype=float),
    )

def detect_object(
    camera: RealSenseCamera,
    detector: YOLODetector,
    follower: DifferentialIKFollower,
    T_cam2gripper,
) -> dict | None:
    frame = camera.read_frame()
    color = frame["color"]
    depth = frame["depth"]
    if color is None or depth is None:
        logger.warning("[detect] Frame read failed")
        return None

    best_target = detector.detect_object(color, depth)
    if best_target is None:
        logger.warning("[detect] No target detected")
        return {"color": color, "target": None}

    T_gripper2base = follower.robot.get_current_pose()
    if T_gripper2base is None:
        logger.warning("[detect] Failed to get flange pose: no data")
        return {"color": color, "target": None}

    T_cam2base = T_gripper2base @ T_cam2gripper
    detected_target: TargetObject = transform_to_base([best_target], T_cam2base)[0]
    logger.info("[detect] Target detected: %s", detected_target)
    return {"color": color, "target": detected_target}


def run(
    target_class: str,
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.2,
    camera_serial_number: str | None = None,
    robot_channel: str = "can0",
    target_distance: float = 0.3,
    urdf_path: str | None = None,
    tcp_frame: str = KinematicsModel.DEFAULT_TCP_FRAME,
    approach_direction: tuple[float, float, float] = (0.0, 0.0, -1.0),
) -> None:
    if not urdf_path:
        raise ValueError("urdf_path is required")

    follower = build_follower(
        robot_channel=robot_channel,
        target_distance=target_distance,
        urdf_path=urdf_path,
        tcp_frame=tcp_frame,
        approach_direction=approach_direction,
    )
    detector = YOLODetector(
        target_class=target_class,
        model_path=model_path,
        conf_threshold=conf_threshold,
    )

    # 1. 加载手眼标定矩阵
    calib_file = Path("configs/eye_in_hand_calibration.json")
    T_cam2gripper = load_eye_in_hand_calibration(str(calib_file))

    # 2. 启动相机并加载相机内参
    width = 640
    height = 480
    camera = RealSenseCamera.setup(
        width=width,
        height=height,
        fps=30,
        serial_number=camera_serial_number,
    )
    detector.set_intrinsics(
        fx=camera.fx,
        fy=camera.fy,
        cx=camera.cx,
        cy=camera.cy,
    )

    # 3. 连接机械臂
    if not follower.robot.connect():
        logger.error("Robot connection failed, task aborted")
        return
    follower.robot.set_normal_mode()

    logger.info("Starting follow task, target: %s", target_class)
    logger.info("Press 's' to lock the current target and follow it")
    logger.info("Press 'c' to clear the locked target")
    logger.info("Press 'q' to exit")

    center_x, center_y = width // 2, height // 2
    follow_enabled = False

    try:
        while True:
            # 4. 目标识别与检测
            result = detect_object(camera, detector, follower, T_cam2gripper)
            if result is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            display_img = result["color"].copy()
            detected_target = result["target"]
            active_target = follower.get_follow_target(
                detected_target,
                follow_enabled=follow_enabled,
            )

            if active_target is None:
                follower.reset_follow_state()
                status = "SEARCHING"
                status_color = (0, 255, 255)
            else:
                if follow_enabled:
                    step = follower.follow_target(active_target)
                    if step.reached_target:
                        status = "LOCKED TARGET REACHED"
                        status_color = (0, 255, 0)
                    elif step.phase == "staging":
                        status = "APPROACHING STAGING POSE"
                        status_color = (0, 255, 0)
                    else:
                        status = "FINE ALIGNMENT"
                        status_color = (0, 255, 0)
                elif detected_target is not None:
                    status = "DETECTED: press 's' to lock"
                    status_color = (0, 255, 0)
                else:
                    status = "SEARCHING"
                    status_color = (0, 255, 255)

            cv2.line(display_img, (center_x - 20, center_y), (center_x + 20, center_y), (255, 0, 0), 1)
            cv2.line(display_img, (center_x, center_y - 20), (center_x, center_y + 20), (255, 0, 0), 1)
            cv2.putText(display_img, status, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

            cv2.imshow("Follow Static Target", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                follow_enabled = False
                follower.clear_locked_target()
                logger.info("Locked target cleared by key 'c'")
            if key == ord("s") and detected_target is not None:
                follow_enabled = True
                follower.lock_target(detected_target)
                logger.info("Follow triggered by key 's'")
                locked_target = follower.locked_target
                if locked_target is not None:
                    follower.follow_target(locked_target)

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        logger.info("Task finished")


def main():
    parser = argparse.ArgumentParser(description="Nero Workcell - Follow Static Target Task")
    parser.add_argument("--target", type=str, default="cup", help="Target object class name (e.g., cup, bottle)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.2, help="Confidence threshold")
    parser.add_argument("--camera-serial", type=str, default=None, help="RealSense camera serial number")
    parser.add_argument(
        "--target-distance",
        type=float,
        default=0.3,
        help="Desired standoff distance from the target point in meters",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        required=True,
        help="Path to the robot URDF file",
    )
    parser.add_argument(
        "--tcp-frame",
        type=str,
        default=KinematicsModel.DEFAULT_TCP_FRAME,
        help="Pinocchio TCP frame name inside the robot URDF",
    )
    parser.add_argument(
        "--approach-dir",
        type=float,
        nargs=3,
        metavar=("AX", "AY", "AZ"),
        default=(0.0, 0.0, -1.0),
        help="Approach direction vector used by the Pinocchio follower",
    )
    args = parser.parse_args()
    
    # Configure logging.
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )

    run(
        target_class=args.target,
        model_path=args.model,
        conf_threshold=args.conf,
        camera_serial_number=args.camera_serial,
        robot_channel="can0",
        target_distance=args.target_distance,
        urdf_path=args.urdf,
        tcp_frame=args.tcp_frame,
        approach_direction=tuple(float(v) for v in args.approach_dir),
    )


if __name__ == "__main__":
    main()
