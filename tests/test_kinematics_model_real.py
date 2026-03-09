#!/usr/bin/env python3
# coding=utf-8
"""
KinematicsModel integration tests using a real Nero robot arm and the real URDF.

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
"""

import logging
import os
from pathlib import Path
import unittest

import numpy as np
from nero_workcell.core.arm_controller import ArmController
from nero_workcell.core.kinematics_model import KinematicsModel


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestKinematicsModelReal")


class TestKinematicsModelReal(unittest.TestCase):
    """Integration tests that exercise KinematicsModel on a live robot state."""

    @classmethod
    def setUpClass(cls):
        repo_root = Path(__file__).resolve().parents[1]
        cls.urdf_path = Path(
            os.environ.get(
                "NERO_URDF_PATH",
                repo_root / "third_party" / "nero_description" / "urdf" / "nero_description.urdf",
            )
        ).resolve()
        if not cls.urdf_path.exists():
            raise unittest.SkipTest(f"URDF file not found: {cls.urdf_path}")

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
            raise unittest.SkipTest(f"Cannot connect to robot arm: {exc}") from exc

        if not connected:
            raise unittest.SkipTest("Cannot connect to robot arm")

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

    def _read_live_joint_configuration(self) -> np.ndarray:
        joint_angles = self.controller.get_joint_angles()
        if joint_angles is None:
            self.skipTest("Robot did not return live joint angles")

        q = np.array(joint_angles, dtype=float)
        self.assertEqual(
            q.shape,
            (self.model.nq,),
            msg=f"Expected live joint vector shape {(self.model.nq,)}, got {q.shape}",
        )
        self.assertTrue(np.isfinite(q).all(), msg="Live joint vector must be finite")
        return q

    def _build_small_test_velocity(self, q: np.ndarray) -> np.ndarray:
        dq = np.zeros(self.model.nv, dtype=float)
        lower = self.model.lower_position_limits
        upper = self.model.upper_position_limits
        step_margin = 0.01
        test_speed = 0.05

        for idx in range(self.model.nv):
            if np.isfinite(upper[idx]) and upper[idx] - q[idx] > step_margin:
                dq[idx] = test_speed
                return dq
            if np.isfinite(lower[idx]) and q[idx] - lower[idx] > step_margin:
                dq[idx] = -test_speed
                return dq

        self.skipTest("No joint has enough clearance for a small integration step")

    def test_01_live_joint_count_matches_model_dofs(self):
        """验证真机返回的关节数与 URDF 模型的自由度定义一致。"""
        logger.info("=== Test 01: Live Joint Count Matches Model DOFs ===")
        q = self._read_live_joint_configuration()

        self.assertEqual(self.model.nq, self.model.nv)
        self.assertEqual(len(q), self.model.nq)
        self.assertEqual(len(self.model.joint_names), self.model.nq)

    def test_02_neutral_configuration_is_finite_and_dimensionally_correct(self):
        """验证模型生成的 neutral configuration 维度正确且数值有效。"""
        logger.info("=== Test 02: Neutral Configuration ===")
        q_neutral = self.model.neutral_configuration()

        self.assertEqual(q_neutral.shape, (self.model.nq,))
        self.assertTrue(np.issubdtype(q_neutral.dtype, np.floating))
        self.assertTrue(np.isfinite(q_neutral).all())

    def test_03_live_forward_kinematics_and_jacobian_are_finite(self):
        """验证真机关节角输入下，正运动学和位置雅可比都能稳定返回有限值。"""
        logger.info("=== Test 03: Live Forward Kinematics and Jacobian ===")
        q = self._read_live_joint_configuration()

        tcp_position = self.model.forward_tcp_position(q)
        jacobian = self.model.compute_tcp_position_jacobian(q)

        self.assertEqual(tcp_position.shape, (3,))
        self.assertEqual(jacobian.shape, (3, self.model.nv))
        self.assertTrue(np.isfinite(tcp_position).all(), msg="TCP position must be finite")
        self.assertTrue(np.isfinite(jacobian).all(), msg="Jacobian must be finite")

    def test_04_zero_velocity_integration_keeps_live_configuration(self):
        """验证零关节速度积分不会改变当前关节配置。"""
        logger.info("=== Test 04: Zero-Velocity Integration ===")
        q = self._read_live_joint_configuration()
        dq = np.zeros(self.model.nv, dtype=float)

        q_next = self.model.integrate_configuration(q, dq, dt=0.1)

        np.testing.assert_allclose(q_next, q, atol=1e-12, rtol=0.0)

    def test_05_live_jacobian_matches_small_step_linearization(self):
        """验证实时姿态下的小步长 TCP 位移与雅可比线性近似一致。"""
        logger.info("=== Test 05: Live Jacobian Linearization ===")
        q = self._read_live_joint_configuration()
        dq = self._build_small_test_velocity(q)
        dt = 1e-3

        tcp_position = self.model.forward_tcp_position(q)
        jacobian = self.model.compute_tcp_position_jacobian(q)
        q_next = self.model.integrate_configuration(q, dq, dt=dt)
        tcp_position_next = self.model.forward_tcp_position(q_next)

        actual_delta = tcp_position_next - tcp_position
        predicted_delta = jacobian @ (dq * dt)

        np.testing.assert_allclose(actual_delta, predicted_delta, atol=1e-4, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
