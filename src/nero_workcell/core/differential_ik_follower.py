"""
Model-based static target follower using differential IK.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .approach_planner import ApproachPlan, ApproachPlanner
from .arm_controller import ArmController
from .cartesian_trajectory import CartesianTrajectory
from .kinematics_model import KinematicsModel
from .target_object import TargetObject

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FollowStep:
    """Single control-step result."""

    reached_target: bool
    phase: str
    target_position: Optional[np.ndarray] = None
    commanded_joints: Optional[np.ndarray] = None
    tracking_error: Optional[np.ndarray] = None


class DifferentialIKFollower:
    """Drive the arm toward a static 3D target using joint-space commands."""

    def __init__(
        self,
        model: KinematicsModel,
        robot: Optional[ArmController] = None,
        *,
        robot_channel: str = "can0",
        standoff_distance: float = 0.3,
        pre_standoff_offset: float = 0.08,
        approach_direction: np.ndarray | tuple[float, float, float] = (0.0, 0.0, -1.0),
        control_period: float = 0.05,
        max_cartesian_speed: float = 0.08,
        max_joint_speed: float = 0.6,
        position_gain: float = 1.5,
        damping: float = 1e-3,
        nullspace_gain: float = 0.05,
        standoff_tolerance: float = 0.005,
        pre_standoff_lateral_tolerance: float = 0.015,
        pre_standoff_axial_tolerance: float = 0.01,
    ):
        self.model = model
        self.robot = robot or ArmController(robot_channel)
        self.planner = ApproachPlanner(
            standoff_distance=standoff_distance,
            pre_standoff_offset=pre_standoff_offset,
            approach_direction=approach_direction,
        )

        self.control_period = float(control_period)
        self.max_cartesian_speed = float(max_cartesian_speed)
        self.max_joint_speed = float(max_joint_speed)
        self.position_gain = float(position_gain)
        self.damping = float(damping)
        self.nullspace_gain = float(nullspace_gain)
        self.standoff_tolerance = float(standoff_tolerance)
        self.pre_standoff_lateral_tolerance = float(pre_standoff_lateral_tolerance)
        self.pre_standoff_axial_tolerance = float(pre_standoff_axial_tolerance)

        self.locked_target: Optional[TargetObject] = None
        self.follow_phase = "idle"
        self._active_plan: Optional[ApproachPlan] = None
        self._active_trajectory: Optional[CartesianTrajectory] = None
        self._reference_configuration: Optional[np.ndarray] = None

    def reset_follow_state(self):
        self.follow_phase = "idle"
        self._active_plan = None
        self._active_trajectory = None
        self._reference_configuration = None

    def _set_follow_phase(self, phase: str):
        if phase == self.follow_phase:
            return
        logger.info("[ik-follow] Phase -> %s", phase.upper())
        self.follow_phase = phase

    def lock_target(self, target: TargetObject):
        if target.frame != "base":
            raise ValueError(f"Expected base-frame target, got frame='{target.frame}'")

        self.locked_target = TargetObject(
            name=target.name,
            class_id=target.class_id,
            bbox=target.bbox,
            center=target.center,
            position=np.array(target.position, dtype=float).copy(),
            conf=target.conf,
            frame=target.frame,
        )
        self.reset_follow_state()

    def clear_locked_target(self):
        self.locked_target = None
        self.reset_follow_state()

    def get_follow_target(
        self,
        detected_target: Optional[TargetObject],
        *,
        follow_enabled: bool,
    ) -> Optional[TargetObject]:
        if follow_enabled and self.locked_target is not None:
            return self.locked_target
        return detected_target

    def _current_joint_configuration(self) -> Optional[np.ndarray]:
        """读取机器人当前关节角，并返回与模型兼容的关节配置向量。

        示例：
            如果机器人返回 ``[0.0, -1.57, 1.2, 0.0, 0.8, 0.0, 0.0]``，且
            ``self.model.nq == 7``，则返回值是形状为 ``(7,)`` 的浮点 NumPy 数组。
            如果机器人没有返回关节状态，则返回 ``None``。
        """
        joint_angles = self.robot.get_joint_angles()
        if joint_angles is None:
            return None
        q = np.array(joint_angles, dtype=float)
        if q.shape != (self.model.nq,):
            raise ValueError(
                f"Expected {self.model.nq} joints from robot, got shape {q.shape}"
            )
        return q

    def _clip_cartesian_velocity(self, velocity: np.ndarray) -> np.ndarray:
        velocity = np.array(velocity, dtype=float)
        speed = float(np.linalg.norm(velocity))
        if speed <= self.max_cartesian_speed:
            return velocity
        return velocity * (self.max_cartesian_speed / speed)

    def _clip_joint_velocity(self, dq: np.ndarray) -> np.ndarray:
        dq = np.array(dq, dtype=float)
        return np.clip(dq, -self.max_joint_speed, self.max_joint_speed)

    def _solve_joint_velocity(
        self,
        q: np.ndarray,
        desired_velocity: np.ndarray,
    ) -> np.ndarray:
        """使用阻尼最小二乘法将期望 TCP 线速度映射为关节速度，并抑制冗余姿态漂移。"""
        # 用位置雅可比的阻尼最小二乘（Damped Least Squares, DLS）伪逆求解主任务。
        jacobian = self.model.compute_tcp_position_jacobian(q)
        # 在 J J^T 上加入阻尼项，提高近奇异位形下的数值稳定性。
        damping_matrix = (self.damping ** 2) * np.eye(jacobian.shape[0])
        # 计算位置雅可比的阻尼伪逆，将 TCP 线速度映射到关节速度空间。
        jacobian_pinv = jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping_matrix)
        # 求解主任务关节速度，使 J(q) @ dq 尽量接近期望的 TCP 线速度。
        dq = jacobian_pinv @ desired_velocity

        if self._reference_configuration is not None:
            # 在零空间中加入次任务：把关节配置轻微拉回参考姿态，同时尽量不影响 TCP 位置跟踪。
            nullspace = np.eye(self.model.nv) - jacobian_pinv @ jacobian
            dq += nullspace @ (
                self.nullspace_gain * (self._reference_configuration - q)
            )

        # 对关节速度逐轴限幅，避免解出来的速度过激。
        return self._clip_joint_velocity(dq)

    def _step_toward(
        self,
        q: np.ndarray,
        tcp_position: np.ndarray,
        stage_goal_position: np.ndarray,
        *,
        now: float,
    ) -> FollowStep:
        """执行一次朝参考目标点前进的控制步。

        参数：
            q: 当前关节配置向量。
            tcp_position: 当前 TCP 在基坐标系下的三维位置。
            stage_goal_position: 当前阶段希望逼近的笛卡尔目标位置，例如 ``pre-standoff position`` 或 ``standoff position``。
            now: 当前控制时刻，通常使用单调时钟时间。

        返回：
            FollowStep: 当前控制步的执行结果，包含当前阶段、参考位置、下发的目标关节角以及当前位置误差。

        说明：
            该函数会先对当前局部轨迹进行采样，得到参考位置和参考速度，
            再结合 TCP 位置误差构造期望笛卡尔速度，随后通过 differential IK
            将末端速度映射为关节速度，并积分得到下一拍的目标关节角发送给机器人。
            该函数只负责“向目标迈进一步”，是否真正到达目标由外层流程判断。
        """
        # 创建当前局部笛卡尔轨迹，并取出这一拍的参考位置和参考速度。
        if self._active_trajectory is None:
            self._active_trajectory = CartesianTrajectory.from_distance(
                start_position=tcp_position,
                goal_position=stage_goal_position,
                start_time=now,
                max_speed=self.max_cartesian_speed,
            )
        sample = self._active_trajectory.sample(now)
        # 计算 TCP 相对当前参考点的三维位置误差。
        error = sample.position - tcp_position
        # 将轨迹前馈速度与位置误差反馈叠加，形成期望末端速度。
        desired_velocity = self._clip_cartesian_velocity(
            sample.velocity + self.position_gain * error
        )
        # 用 differential IK 将期望末端速度映射为关节速度。
        dq = self._solve_joint_velocity(q, desired_velocity)
        # 将关节速度积分为下一控制周期的目标关节角。
        q_target = self.model.integrate_configuration(q, dq, self.control_period)
        q_target = self.model.clamp_to_joint_limits(q_target)
        self.robot.move_j(q_target.tolist(), blocking=False)
        return FollowStep(
            reached_target=False,
            phase=self.follow_phase,
            target_position=sample.position,
            commanded_joints=q_target,
            tracking_error=error,
        )

    def follow_target(self, target: TargetObject, *, now: Optional[float] = None) -> FollowStep:
        if target.frame != "base":
            raise ValueError(f"Expected base-frame target, got frame='{target.frame}'")
        # 1. 读取机械臂当前关节角，通过正运动学计算TCP在基坐标系下的三维位置 [x, y, z]
        now = time.monotonic() if now is None else float(now)
        q = self._current_joint_configuration() # 关节角
        if q is None:
            return FollowStep(reached_target=False, phase=self.follow_phase)

        if self._reference_configuration is None:
            self._reference_configuration = q.copy()

        tcp_position = self.model.forward_tcp_position(q) # 末端在哪里


        if self._active_plan is None:
            self._active_plan = self.planner.make_plan(target.position)
            self._set_follow_phase("staging")
            self._active_trajectory = None

        if self.follow_phase == "staging":
            if self.planner.is_pre_standoff_reached(
                tcp_position,
                self._active_plan,
                lateral_tolerance=self.pre_standoff_lateral_tolerance,
                axial_tolerance=self.pre_standoff_axial_tolerance,
            ):
                self._set_follow_phase("fine")
                self._active_trajectory = None
            else:
                return self._step_toward(
                    q,
                    tcp_position,
                    self._active_plan.pre_standoff_position,
                    now=now,
                )

        if self.planner.is_standoff_reached(
            tcp_position,
            self._active_plan,
            position_tolerance=self.standoff_tolerance,
        ):
            return FollowStep(
                reached_target=True,
                phase=self.follow_phase,
                target_position=self._active_plan.standoff_position.copy(),
                commanded_joints=q.copy(),
                tracking_error=np.zeros(3, dtype=float),
            )

        return self._step_toward(
            q,
            tcp_position,
            self._active_plan.standoff_position,
            now=now,
        )
