#!/usr/bin/env python3
"""Get the checkerboard position in the robot base frame from a RealSense capture."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from nero_workcell.core import RealSenseCamera
from nero_workcell.utils.common import load_eye_to_hand_calibration


logger = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CALIBRATION_FILE = REPO_ROOT / "configs" / "eye_to_hand_calibration.json"
DEFAULT_CAPTURE_CONFIG_FILE = REPO_ROOT / "src" / "nero_workcell" / "eye_to_hand" / "config.json"
MAX_INTRINSICS_DELTA_PX = 50.0
PREVIEW_WINDOW_NAME = "Checkerboard Capture"


def load_json(path: Path) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to load JSON file: {path}") from exc


def normalize_corner_order(corners: np.ndarray) -> np.ndarray:
    """Keep checkerboard corner ordering consistent with calibration."""
    first_point = corners[0][0]
    last_point = corners[-1][0]

    first_score = first_point[1] - first_point[0]
    last_score = last_point[1] - last_point[0]
    if last_score > first_score:
        return corners[::-1].copy()
    return corners


def build_checkerboard_points(
    corner_long: int,
    corner_short: int,
    corner_size: float,
) -> np.ndarray:
    object_points = np.zeros((corner_long * corner_short, 3), dtype=np.float32)
    object_points[:, :2] = np.mgrid[0:corner_long, 0:corner_short].T.reshape(-1, 2)
    object_points *= float(corner_size)
    return object_points


def make_homogeneous_transform(rotation_matrix: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np.asarray(translation, dtype=float).reshape(3)
    return transform


def estimate_checkerboard_pose(
    color_image: np.ndarray,
    *,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    corner_long: int,
    corner_short: int,
    corner_size: float,
) -> dict[str, Any] | None:
    """从单张彩色图像估计棋盘格相对于相机坐标系的位姿。

    参数:
        color_image: RealSense 采集到的 BGR 彩色图像。
        camera_matrix: 3x3 针孔相机内参矩阵。
        dist_coeffs: OpenCV PnP 使用的镜头畸变参数。
        corner_long: 棋盘格长边方向的内角点数量。
        corner_short: 棋盘格短边方向的内角点数量。
        corner_size: 相邻内角点之间的物理间距，单位米。

    返回:
        成功时返回一个字典，包含亚像素角点、OpenCV 的 ``rvec/tvec``、
        4x4 齐次变换矩阵 ``T_board2cam``，以及绘制了调试信息的图像。
        如果未检测到棋盘格，或者 ``solvePnP`` 失败，则返回 ``None``。

    示例:
        最小调用示例::

            result = estimate_checkerboard_pose(
                color_image=frame["color"],
                camera_matrix=np.array(calibration["camera_matrix"], dtype=np.float64),
                dist_coeffs=np.array(calibration["dist_coeffs"], dtype=np.float64),
                corner_long=9,
                corner_short=13,
                corner_size=0.04,
            )

        将结果继续转换到基坐标系::

            if result is not None:
                T_board2cam = result["T_board2cam"]
                T_board2base = T_cam2base @ T_board2cam
                board_origin_in_base = T_board2base[:3, 3]
    """
    gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    # 先在图像平面上检测棋盘格角点的粗略位置。
    found, corners = cv2.findChessboardCorners(gray, (corner_long, corner_short), flags)
    if not found:
        return None

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    # 将角点优化到亚像素精度，提高后续位姿估计的稳定性。
    corners_subpix = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    # 统一角点顺序，避免棋盘格翻转后原点定义发生变化。
    corners_subpix = normalize_corner_order(corners_subpix)

    object_points = build_checkerboard_points(corner_long, corner_short, corner_size)
    # 根据棋盘格 3D 点和图像中的 2D 角点求解 board -> camera 位姿。
    success, rvec, tvec = cv2.solvePnP(
        object_points,
        corners_subpix,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    transform_board_to_camera = make_homogeneous_transform(rotation_matrix, tvec)

    # 生成调试图，叠加显示角点和恢复出的棋盘格坐标轴。
    annotated = color_image.copy()
    cv2.drawChessboardCorners(
        annotated,
        (corner_long, corner_short),
        corners_subpix,
        found,
    )
    cv2.drawFrameAxes(
        annotated,
        camera_matrix,
        dist_coeffs,
        rvec,
        tvec,
        length=max(corner_size * 3.0, 0.05),
        thickness=2,
    )

    origin = tuple(corners_subpix[0][0].astype(int))
    cv2.circle(annotated, origin, 8, (0, 255, 0), 2)
    cv2.putText(
        annotated,
        "BOARD ORIGIN",
        (origin[0] + 12, origin[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    board_center = np.array(
        [[(corner_long - 1) * corner_size * 0.5, (corner_short - 1) * corner_size * 0.5, 0.0]],
        dtype=np.float32,
    )
    center_image_points, _ = cv2.projectPoints(
        board_center,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    center = tuple(np.round(center_image_points[0, 0]).astype(int))
    cv2.circle(annotated, center, 8, (255, 255, 0), 2)
    cv2.putText(
        annotated,
        "BOARD CENTER",
        (center[0] + 12, center[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 0),
        2,
    )

    return {
        "corners": corners_subpix,
        "rvec": rvec,
        "tvec": tvec,
        "T_board2cam": transform_board_to_camera,
        "annotated_image": annotated,
    }


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image: {path}")


def validate_intrinsics_delta(intrinsics_delta: np.ndarray) -> None:
    max_abs_delta = float(np.max(np.abs(intrinsics_delta)))
    if max_abs_delta > MAX_INTRINSICS_DELTA_PX:
        raise RuntimeError(
            "Live camera intrinsics differ too much from calibration: "
            f"delta={np.array2string(intrinsics_delta, precision=4)}, "
            f"threshold={MAX_INTRINSICS_DELTA_PX:.1f}px"
        )


def build_output_payload(
    *,
    calibration_file: Path,
    camera_serial: str,
    width: int,
    height: int,
    corner_long: int,
    corner_short: int,
    corner_size: float,
    T_board2base: np.ndarray,
    T_board2cam: np.ndarray,
) -> dict[str, Any]:
    rotation = R.from_matrix(T_board2base[:3, :3])
    quaternion = rotation.as_quat()
    euler_deg = rotation.as_euler("xyz", degrees=True)
    checkerboard_center_board = np.array(
        [
            (corner_long - 1) * corner_size * 0.5,
            (corner_short - 1) * corner_size * 0.5,
            0.0,
            1.0,
        ],
        dtype=float,
    )
    checkerboard_center_camera = T_board2cam @ checkerboard_center_board
    checkerboard_center_base = T_board2base @ checkerboard_center_board

    return {
        "calibration_file": str(calibration_file),
        "camera_serial": camera_serial,
        "resolution": {
            "width": int(width),
            "height": int(height),
        },
        "checkerboard_origin_camera_m": {
            "x": float(T_board2cam[0, 3]),
            "y": float(T_board2cam[1, 3]),
            "z": float(T_board2cam[2, 3]),
        },
        "checkerboard_center_camera_m": {
            "x": float(checkerboard_center_camera[0]),
            "y": float(checkerboard_center_camera[1]),
            "z": float(checkerboard_center_camera[2]),
        },
        "checkerboard_origin_base_m": {
            "x": float(T_board2base[0, 3]),
            "y": float(T_board2base[1, 3]),
            "z": float(T_board2base[2, 3]),
        },
        "checkerboard_center_base_m": {
            "x": float(checkerboard_center_base[0]),
            "y": float(checkerboard_center_base[1]),
            "z": float(checkerboard_center_base[2]),
        },
        "checkerboard_quaternion_base": {
            "x": float(quaternion[0]),
            "y": float(quaternion[1]),
            "z": float(quaternion[2]),
            "w": float(quaternion[3]),
        },
        "checkerboard_euler_base_deg": {
            "rx": float(euler_deg[0]),
            "ry": float(euler_deg[1]),
            "rz": float(euler_deg[2]),
        },
        "T_board2base": T_board2base.tolist(),
        "T_board2cam": T_board2cam.tolist(),
    }


def build_preview_image(
    color_image: np.ndarray,
    pose_result: dict[str, Any] | None,
) -> np.ndarray:
    if pose_result is not None:
        preview = pose_result["annotated_image"].copy()
        status_text = "Checkerboard detected"
        status_color = (0, 255, 0)
    else:
        preview = color_image.copy()
        status_text = "Checkerboard not detected"
        status_color = (0, 0, 255)

    cv2.putText(
        preview,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2,
    )
    cv2.putText(
        preview,
        "Press 's' to capture, 'q' to quit",
        (10, preview.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return preview


def build_unavailable_preview(width: int, height: int) -> np.ndarray:
    preview = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        preview,
        "Color frame unavailable",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        preview,
        "Press 'q' to quit",
        (10, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return preview


def print_human_readable(payload: dict[str, Any]) -> None:
    origin_camera = payload["checkerboard_origin_camera_m"]
    center_camera = payload["checkerboard_center_camera_m"]
    origin_base = payload["checkerboard_origin_base_m"]
    center_base = payload["checkerboard_center_base_m"]

    print("checkerboard center in camera frame [m]:")
    print(f"  x: {center_camera['x']:.6f}")
    print(f"  y: {center_camera['y']:.6f}")
    print(f"  z: {center_camera['z']:.6f}")
    print("checkerboard origin in camera frame [m]:")
    print(f"  x: {origin_camera['x']:.6f}")
    print(f"  y: {origin_camera['y']:.6f}")
    print(f"  z: {origin_camera['z']:.6f}")
    print("checkerboard center in base frame [m]:")
    print(f"  x: {center_base['x']:.6f}")
    print(f"  y: {center_base['y']:.6f}")
    print(f"  z: {center_base['z']:.6f}")
    print("checkerboard origin in base frame [m]:")
    print(f"  x: {origin_base['x']:.6f}")
    print(f"  y: {origin_base['y']:.6f}")
    print(f"  z: {origin_base['z']:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Get the checkerboard position in the robot base frame with RealSense.",
    )
    parser.add_argument(
        "--calib-file",
        type=Path,
        default=DEFAULT_CALIBRATION_FILE,
        help="Path to eye-to-hand calibration JSON.",
    )
    parser.add_argument(
        "--capture-config",
        type=Path,
        default=DEFAULT_CAPTURE_CONFIG_FILE,
        help="Path to eye-to-hand camera/checkerboard config JSON.",
    )
    parser.add_argument(
        "--camera-serial",
        type=str,
        default=None,
        help="RealSense serial number. Defaults to the first discovered camera.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Maximum consecutive failed frame reads before giving up.",
    )
    parser.add_argument(
        "--output-image",
        type=Path,
        default=None,
        help="Optional path to save the annotated capture image.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print result as JSON.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.max_frames < 1:
        print("--max-frames must be at least 1", file=sys.stderr)
        return 2

    try:
        capture_config = load_json(args.capture_config)
        calibration = load_json(args.calib_file)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    width = int(capture_config["camera"]["width"])
    height = int(capture_config["camera"]["height"])
    fps = int(capture_config["camera"]["fps"])

    checkerboard = calibration.get("checkerboard") or capture_config["checkerboard"]
    corner_long = int(checkerboard["corner_point_long"])
    corner_short = int(checkerboard["corner_point_short"])
    corner_size = float(checkerboard["corner_point_size"])

    try:
        camera_matrix = np.array(calibration["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.array(calibration["dist_coeffs"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        print(f"Invalid camera intrinsics in calibration file: {exc}", file=sys.stderr)
        return 1

    try:
        T_cam2base = load_eye_to_hand_calibration(str(args.calib_file))
    except SystemExit:
        return 1

    logger.info(
        "Starting capture: resolution=%sx%s fps=%s checkerboard=%sx%s size=%.4fm",
        width,
        height,
        fps,
        corner_long,
        corner_short,
        corner_size,
    )

    camera = None
    pose_result = None

    try:
        camera = RealSenseCamera.setup(
            width=width,
            height=height,
            fps=fps,
            serial_number=args.camera_serial,
        )

        intrinsics_delta = np.array(
            [
                camera.fx - camera_matrix[0, 0],
                camera.fy - camera_matrix[1, 1],
                camera.cx - camera_matrix[0, 2],
                camera.cy - camera_matrix[1, 2],
            ],
            dtype=float,
        )
        logger.info(
            "Live intrinsics delta vs calibration [fx, fy, cx, cy]: %s",
            np.array2string(intrinsics_delta, precision=4),
        )
        validate_intrinsics_delta(intrinsics_delta)

        cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)

        failed_frame_reads = 0
        while True:
            frame = camera.read_frame()
            color_image = frame["color"]
            if color_image is None:
                failed_frame_reads += 1
                logger.warning(
                    "Color image unavailable (%s/%s)",
                    failed_frame_reads,
                    args.max_frames,
                )
                cv2.imshow(
                    PREVIEW_WINDOW_NAME,
                    build_unavailable_preview(width, height),
                )
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    logger.info("User requested exit.")
                    return 0
                if failed_frame_reads >= args.max_frames:
                    print(
                        "Failed to read a color image from the camera.",
                        file=sys.stderr,
                    )
                    return 1
                continue

            failed_frame_reads = 0
            preview_pose_result = estimate_checkerboard_pose(
                color_image,
                camera_matrix=camera_matrix,
                dist_coeffs=dist_coeffs,
                corner_long=corner_long,
                corner_short=corner_short,
                corner_size=corner_size,
            )
            cv2.imshow(
                PREVIEW_WINDOW_NAME,
                build_preview_image(color_image, preview_pose_result),
            )

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                logger.info("User requested exit.")
                return 0
            if key not in (ord("s"), ord("S")):
                continue

            if preview_pose_result is None:
                logger.warning(
                    "Checkerboard not detected in current frame; capture ignored."
                )
                continue

            pose_result = preview_pose_result
            logger.info("Checkerboard captured successfully.")
            break

        assert pose_result is not None
        T_board2cam = pose_result["T_board2cam"]
        T_board2base = T_cam2base @ T_board2cam
        payload = build_output_payload(
            calibration_file=args.calib_file,
            camera_serial=camera.serial_number,
            width=width,
            height=height,
            corner_long=corner_long,
            corner_short=corner_short,
            corner_size=corner_size,
            T_board2base=T_board2base,
            T_board2cam=T_board2cam,
        )

        if args.output_image is not None:
            save_image(args.output_image, pose_result["annotated_image"])
            logger.info("Annotated image saved to %s", args.output_image)

        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            print_human_readable(payload)
        return 0
    except Exception as exc:
        logger.exception("Capture failed: %s", exc)
        print(f"Capture failed: {exc}", file=sys.stderr)
        return 1
    finally:
        cv2.destroyAllWindows()
        if camera is not None and camera.is_opened:
            camera.stop()


if __name__ == "__main__":
    raise SystemExit(main())
