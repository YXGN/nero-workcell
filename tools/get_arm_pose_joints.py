#!/usr/bin/env python3
"""Read the robot's current pose and joint angles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
POSE_LABELS = ("x", "y", "z", "roll", "pitch", "yaw")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _pose_to_mapping(values: list[float]) -> dict[str, float]:
    return {label: float(value) for label, value in zip(POSE_LABELS, values, strict=True)}


def _joint_to_mapping(names: list[str], values: list[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(names, values, strict=True)}


def read_arm_state(
    channel: str,
    speed: int = 5,
    pose_frame: str = "both",
) -> dict[str, Any]:
    from nero_workcell.core import ArmController

    controller = ArmController(channel=channel)

    controller.connect(speed_percent=speed)
    try:
        joint_angles = controller.get_joint_angles()
        if joint_angles is None:
            raise RuntimeError("Failed to read joint angles from the robot")

        result: dict[str, Any] = {
            "channel": channel,
            "joint_angles": _joint_to_mapping(controller.joint_names, joint_angles),
        }

        if pose_frame in {"flange", "both"}:
            flange_pose = controller.get_flange_pose()
            if flange_pose is None:
                raise RuntimeError("Failed to read flange pose from the robot")
            result["flange_pose"] = _pose_to_mapping(flange_pose)

        if pose_frame in {"tcp", "both"}:
            tcp_pose = controller.get_tcp_pose()
            if tcp_pose is None:
                raise RuntimeError("Failed to read TCP pose from the robot")
            result["tcp_pose"] = _pose_to_mapping(tcp_pose)

        return result
    finally:
        if controller.is_connected():
            controller.disconnect()


def _print_mapping(title: str, values: dict[str, float]) -> None:
    print(title)
    for key, value in values.items():
        print(f"  {key}: {value:.6f}")


def print_human_readable(state: dict[str, Any]) -> None:
    print(f"channel: {state['channel']}")
    _print_mapping("joint_angles (rad):", state["joint_angles"])

    if "flange_pose" in state:
        _print_mapping("flange_pose [x, y, z, roll, pitch, yaw]:", state["flange_pose"])

    if "tcp_pose" in state:
        _print_mapping("tcp_pose [x, y, z, roll, pitch, yaw]:", state["tcp_pose"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Get the robot's current pose and joint angles."
    )
    parser.add_argument("--channel", default="can0", help="CAN channel")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the result as JSON",
    )
    args = parser.parse_args()

    try:
        state = read_arm_state(
            channel=args.channel,
        )
    except Exception as exc:
        print(f"Failed to read robot state: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(state, ensure_ascii=False))
    else:
        print_human_readable(state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
