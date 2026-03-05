#!/usr/bin/env python3
# coding=utf-8

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from nero_workcell.core.target_object import TargetObject

logger = logging.getLogger(__name__)

def load_eye_in_hand_calibration(calib_file: str) -> Optional[np.ndarray]:
    """
    Load eye-in-hand calibration and build the T_cam2gripper transform matrix.

    Args:
        calib_file (str): Path to the eye-in-hand calibration JSON file.

    Returns:
        np.ndarray | None:
            4x4 homogeneous transform matrix on success;
            None when the calibration file does not exist.
    """
    calib_file = Path(calib_file)
    if not calib_file.exists():
        logger.error(
            f"Calibration file not found: {calib_file}. Run eye-in-hand calibration first."
        )
        return None

    with open(calib_file, "r") as f:
        calib = json.load(f)

    if calib.get("calibration_type") != "eye_in_hand":
        logger.error(
            f"Calibration type is {calib.get('calibration_type')}; expected eye_in_hand"
        )

    T_cam2gripper = np.eye(4)
    T_cam2gripper[:3, :3] = np.array(calib["rotation_matrix"])
    T_cam2gripper[:3, 3] = np.array(calib["translation_vector"])
    logger.info(
        f"Eye-in-hand calibration loaded: {calib_file}, T_cam2gripper:\n{T_cam2gripper}"
    )
    return T_cam2gripper

def transform_to_base(
    target_objects_camera: List[TargetObject],
    T_cam2base: np.ndarray,
) -> List[TargetObject]:
    """
    Transform detected objects from the camera coordinate frame to the base coordinate frame.

    Args:
        target_objects_camera (List[TargetObject]): List of objects detected in the camera frame.
        T_cam2base (np.ndarray): 4x4 homogeneous transformation matrix from camera to base frame.

    Returns:
        List[TargetObject]: List of objects transformed into the base frame.
    """
    target_objects_base: List[TargetObject] = []

    for camera_object in target_objects_camera:
        if camera_object.frame != "camera":
            raise ValueError(
                f"Expected camera-frame object, but got frame='{camera_object.frame}'"
            )

        p_cam = np.append(camera_object.position, 1.0)
        p_base = (T_cam2base @ p_cam)[:3]
        target_objects_base.append(
            TargetObject(
                name=camera_object.name,
                class_id=camera_object.class_id,
                bbox=camera_object.bbox,
                center=camera_object.center,
                position=p_base,
                conf=camera_object.conf,
                frame="base",
            )
        )

    return target_objects_base