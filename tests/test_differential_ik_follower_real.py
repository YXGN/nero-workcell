#!/usr/bin/env python3
# coding=utf-8
"""
DifferentialIKFollower integration tests using a real Nero robot arm and the real URDF.

Requirements:
- Run `pip install -e .` in the repository root first.
- `pinocchio` must be installed.
- `pyAgxArm` must be installed.
- A real Nero robot arm must be connected and powered on.

Optional environment variables:
- `NERO_ARM_CHANNEL`: CAN channel to use. Default: `can0`.
- `NERO_ROBOT_TYPE`: robot type passed to `ArmController`. Default: `nero`.
- `NERO_URDF_PATH`: URDF file path. Default: checked-in Nero URDF.
- `NERO_TCP_FRAME`: TCP frame name in the URDF. Default: `end_effector`.

Safety:
- `test_04_follow_target_second_tick_generates_bounded_live_joint_command`
  sends one tiny non-blocking joint command. With the default test parameters,
  each joint delta is limited to at most 0.001 rad in a single control step.
"""

import logging
import os
import time
from pathlib import Path
import unittest

import numpy as np

from nero_workcell.core.arm_controller import ArmController
from nero_workcell.core.differential_ik_follower import DifferentialIKFollower
from nero_workcell.core.kinematics_model import KinematicsModel
from nero_workcell.core.target_object import TargetObject


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestDifferentialIKFollowerReal")


class TestDifferentialIKFollowerReal(unittest.TestCase):
    """Integration tests that exercise DifferentialIKFollower on a live robot state."""

    @classmethod
    def _skip_class(cls, message: str):
        logger.warning("Skipping DifferentialIKFollower real integration tests: %s", message)
        raise unittest.SkipTest(message)

    def _skip_test(self, message: str):
        logger.warning("Skipping %s: %s", self.id().split(".")[-1], message)
        self.skipTest(message)

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.urdf_path = Path(
            os.environ.get(
                "NERO_URDF_PATH",
                repo_root / "nero_description" / "urdf" / "nero_description.urdf",
            )
        ).resolve()
        if not cls.urdf_path.exists():
            cls._skip_class(f"URDF file not found: {cls.urdf_path}")

        cls.robot_channel = os.environ.get("NERO_ARM_CHANNEL", "can0")
        cls.robot_type = os.environ.get("NERO_ROBOT_TYPE", "nero")
        cls.tcp_frame = os.environ.get("NERO_TCP_FRAME", "end_effector")

        cls.model = KinematicsModel(
            urdf_path=cls.urdf_path,
            tcp_frame=cls.tcp_frame,
        )
        cls.controller = ArmController(
            channel=cls.robot_channel,
            robot_type=cls.robot_type,
        )
        try:
            connected = cls.controller.connect()
        except Exception as exc:
            cls._skip_class(f"Cannot connect to robot arm: {exc}")

        if not connected:
            cls._skip_class("Cannot connect to robot arm")

        try:
            cls.controller.set_normal_mode()
        except Exception as exc:
            logger.warning("Failed to switch robot to normal mode: %s", exc)

        logger.info(
            "Using robot channel=%s, robot_type=%s, urdf=%s, tcp_frame=%s",
            cls.robot_channel,
            cls.robot_type,
            cls.urdf_path,
            cls.tcp_frame,
        )

    @classmethod
    def tearDownClass(cls):
        controller = getattr(cls, "controller", None)
        if controller is not None and controller.is_connected():
            controller.disconnect()

    def _make_follower(self, **kwargs) -> DifferentialIKFollower:
        return DifferentialIKFollower(
            model=self.model,
            robot=self.controller,
            **kwargs,
        )

    def _read_live_joint_configuration(self) -> np.ndarray:
        joint_angles = self.controller.get_joint_angles()
        if joint_angles is None:
            self._skip_test("Robot did not return live joint angles")

        q = np.array(joint_angles, dtype=float)
        self.assertEqual(
            q.shape,
            (self.model.nq,),
            msg=f"Expected live joint vector shape {(self.model.nq,)}, got {q.shape}",
        )
        self.assertTrue(np.isfinite(q).all(), msg="Live joint vector must be finite")
        return q

    def _read_live_tcp_position(self) -> np.ndarray:
        q = self._read_live_joint_configuration()
        tcp_position = self.model.forward_tcp_position(q)
        self.assertEqual(tcp_position.shape, (3,))
        self.assertTrue(np.isfinite(tcp_position).all(), msg="Live TCP position must be finite")
        return tcp_position

    def _make_base_target(self, position: np.ndarray) -> TargetObject:
        return TargetObject(
            name="live-test-target",
            class_id=0,
            bbox=(0, 0, 1, 1),
            center=(0, 0),
            position=np.array(position, dtype=float),
            conf=1.0,
            frame="base",
        )

    def test_01_current_joint_configuration_reads_live_robot_state(self):
        """验证 follower 能直接读取真机关节角，并返回与模型匹配的配置向量。"""
        logger.info("=== Test 01: Current Joint Configuration ===")
        follower = self._make_follower()

        q = follower._current_joint_configuration()

        self.assertIsNotNone(q)
        self.assertEqual(q.shape, (self.model.nq,))
        self.assertTrue(np.isfinite(q).all())

    def test_02_lock_target_and_clear_target_keep_base_frame_state(self):
        """验证锁定/清除目标流程在真机 follower 上行为正确。"""
        logger.info("=== Test 02: Lock And Clear Target ===")
        follower = self._make_follower()
        live_tcp_position = self._read_live_tcp_position()
        target = self._make_base_target(live_tcp_position + np.array([0.0, 0.0, -0.02]))

        follower.lock_target(target)
        active_target = follower.get_follow_target(None, follow_enabled=True)

        self.assertIsNotNone(active_target)
        self.assertIsNot(active_target, target)
        np.testing.assert_allclose(active_target.position, target.position, atol=0.0, rtol=0.0)

        target.position[:] = 123.0
        np.testing.assert_allclose(
            follower.locked_target.position,
            live_tcp_position + np.array([0.0, 0.0, -0.02]),
            atol=0.0,
            rtol=0.0,
        )

        follower.clear_locked_target()
        self.assertIsNone(follower.locked_target)
        self.assertEqual(follower.follow_phase, "idle")
        self.assertIsNone(follower.get_follow_target(None, follow_enabled=True))

    def test_03_follow_target_reports_reached_when_live_tcp_is_already_at_goal(self):
        """验证当前 TCP 已在目标位置时，follower 会直接返回 reached_target=True。"""
        logger.info("=== Test 03: Reached Goal Without Motion ===")
        live_tcp_position = self._read_live_tcp_position()
        follower = self._make_follower(
            standoff_distance=0.0,
            pre_standoff_offset=0.0,
        )
        target = self._make_base_target(live_tcp_position)

        step = follower.follow_target(target, now=100.0)

        self.assertTrue(step.reached_target)
        self.assertEqual(step.phase, "fine")
        self.assertEqual(follower.follow_phase, "fine")
        self.assertIsNotNone(step.commanded_joints)
        self.assertEqual(step.commanded_joints.shape, (self.model.nq,))
        self.assertTrue(np.isfinite(step.commanded_joints).all())
        np.testing.assert_allclose(step.target_position, target.position, atol=1e-9, rtol=0.0)
        np.testing.assert_allclose(step.tracking_error, np.zeros(3, dtype=float), atol=0.0, rtol=0.0)

    def test_04_follow_target_second_tick_generates_bounded_live_joint_command(self):
        """验证第二个控制 tick 会在真机状态下生成受限且有限的关节命令。"""
        logger.info("=== Test 04: Second Tick Produces Small Joint Command ===")
        live_tcp_position = self._read_live_tcp_position()
        follower = self._make_follower(
            standoff_distance=0.0,
            pre_standoff_offset=0.005,
            control_period=0.01,
            max_cartesian_speed=0.02,
            max_joint_speed=0.1,
            position_gain=1.0,
        )
        target = self._make_base_target(live_tcp_position)
        t0 = 200.0

        first_step = follower.follow_target(target, now=t0)
        second_step = follower.follow_target(target, now=t0 + 0.25)

        self.assertFalse(first_step.reached_target)
        self.assertEqual(first_step.phase, "staging")
        self.assertFalse(second_step.reached_target)
        self.assertEqual(second_step.phase, "staging")
        self.assertIsNotNone(first_step.commanded_joints)
        self.assertIsNotNone(second_step.commanded_joints)
        self.assertTrue(np.isfinite(second_step.commanded_joints).all())
        self.assertTrue(np.isfinite(second_step.tracking_error).all())
        self.assertGreater(np.linalg.norm(second_step.tracking_error), 0.0)
        self.assertFalse(
            np.allclose(
                second_step.commanded_joints,
                first_step.commanded_joints,
                atol=1e-8,
                rtol=0.0,
            ),
            msg="The second control tick should produce a non-zero joint-space update",
        )

        joint_delta = second_step.commanded_joints - first_step.commanded_joints
        self.assertLessEqual(
            float(np.max(np.abs(joint_delta))),
            follower.max_joint_speed * follower.control_period + 1e-9,
        )
        np.testing.assert_array_less(
            second_step.commanded_joints,
            self.model.upper_position_limits + 1e-9,
        )
        np.testing.assert_array_less(
            self.model.lower_position_limits - 1e-9,
            second_step.commanded_joints,
        )
        np.testing.assert_allclose(
            second_step.target_position,
            follower._active_plan.pre_standoff_position,
            atol=1e-9,
            rtol=0.0,
        )

        time.sleep(0.3)


if __name__ == "__main__":
    unittest.main()
