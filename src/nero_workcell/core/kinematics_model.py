"""
Pinocchio-backed kinematics model for follow tasks.
"""

import logging
from pathlib import Path

import numpy as np
import pinocchio as pin

logger = logging.getLogger(__name__)


class KinematicsModel:
    """Thin wrapper around Pinocchio for the TCP frame used by follow tasks."""

    DEFAULT_TCP_FRAME = "end_effector"

    def __init__(
        self,
        urdf_path: str | Path,
        *,
        tcp_frame: str = DEFAULT_TCP_FRAME,
    ):
        self.urdf_path = Path(urdf_path)
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.tcp_frame = tcp_frame
        self.tcp_frame_id = self.model.getFrameId(tcp_frame)
        if self.tcp_frame_id >= len(self.model.frames):
            raise ValueError(f"TCP frame not found in URDF: {tcp_frame}")
        logger.info(
            "Loaded kinematics model from %s with tcp_frame=%s, nq=%d, nv=%d",
            self.urdf_path,
            self.tcp_frame,
            self.nq,
            self.nv,
        )

    @property
    def joint_names(self) -> list[str]:
        return list(self.model.names[1:])

    @property
    def nq(self) -> int:
        return int(self.model.nq)

    @property
    def nv(self) -> int:
        return int(self.model.nv)

    @property
    def lower_position_limits(self) -> np.ndarray:
        return np.array(self.model.lowerPositionLimit, dtype=float)

    @property
    def upper_position_limits(self) -> np.ndarray:
        return np.array(self.model.upperPositionLimit, dtype=float)

    def neutral_configuration(self) -> np.ndarray:
        return np.array(pin.neutral(self.model), dtype=float)

    def _assert_configuration_vector(self, q: np.ndarray) -> None:
        assert isinstance(q, np.ndarray), (
            f"Expected q to be np.ndarray, got {type(q).__name__}"
        )
        assert q.shape == (self.nq,), f"Expected q shape {(self.nq,)}, got {q.shape}"
        assert np.issubdtype(q.dtype, np.floating), (
            f"Expected q dtype to be floating, got {q.dtype}"
        )

    def _assert_velocity_vector(self, dq: np.ndarray) -> None:
        assert isinstance(dq, np.ndarray), (
            f"Expected dq to be np.ndarray, got {type(dq).__name__}"
        )
        assert dq.shape == (self.nv,), f"Expected dq shape {(self.nv,)}, got {dq.shape}"
        assert np.issubdtype(dq.dtype, np.floating), (
            f"Expected dq dtype to be floating, got {dq.dtype}"
        )

    def clamp_to_joint_limits(self, q: np.ndarray) -> np.ndarray:
        self._assert_configuration_vector(q)
        lower = self.lower_position_limits
        upper = self.upper_position_limits
        finite_lower = np.where(np.isfinite(lower), lower, q)
        finite_upper = np.where(np.isfinite(upper), upper, q)
        return np.clip(q, finite_lower, finite_upper)

    def forward_tcp_position(self, q: np.ndarray) -> np.ndarray:
        """根据关节配置向量 q，通过正运动学计算 TCP 在机器人基坐标系下的位置向量 [x, y, z]。

        示例：
            当 ``q`` 的形状为 ``(self.nq,)`` 时，返回值是形状为 ``(3,)`` 的
            NumPy 浮点数组，例如 ``[0.42, -0.08, 0.31]``。
        """
        self._assert_configuration_vector(q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.tcp_frame_id)
        return np.array(self.data.oMf[self.tcp_frame_id].translation, dtype=float).copy()

    def compute_tcp_position_jacobian(self, q: np.ndarray) -> np.ndarray:
        """计算 TCP 在当前关节配置下的位置雅可比矩阵。

        参数：
            q: 当前关节配置向量，形状为 ``(self.nq,)``。

        返回：
            np.ndarray: 形状为 ``(3, self.nv)`` 的浮点矩阵，表示 TCP 线速度与关节速度之间的
                一阶线性映射关系，即 ``v_tcp ≈ J(q) @ dq``。

        示例：
            当 ``q`` 的形状为 ``(self.nq,)``，且机器人有 7 个速度自由度时，
            返回值是形状为 ``(3, 7)`` 的 NumPy 浮点矩阵，例如：

            ``[[ 0.00, -0.21, -0.18,  0.00,  0.05,  0.00,  0.00],
               [ 0.32,  0.00,  0.00, -0.07,  0.00,  0.02,  0.00],
               [ 0.00,  0.31,  0.12,  0.00, -0.03,  0.00,  0.00]]``

        说明：
        """
        self._assert_configuration_vector(q)
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacement(self.model, self.data, self.tcp_frame_id)
        jacobian = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.tcp_frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )
        return np.array(jacobian[:3, :], dtype=float)

    def integrate_configuration(self, q: np.ndarray, dq: np.ndarray, dt: float) -> np.ndarray:
        """根据当前关节配置和关节速度积分得到下一时刻的关节配置。

        参数：
            q: 当前关节配置向量，形状为 ``(self.nq,)``。
            dq: 当前关节速度向量，形状为 ``(self.nv,)``。
            dt: 积分时间步长，单位为秒。

        返回：
            np.ndarray: 形状为 ``(self.nq,)`` 的浮点数组，表示沿模型流形执行
                ``q_next = integrate(q, dq * dt)`` 后得到的新关节配置。
        """
        self._assert_configuration_vector(q)
        self._assert_velocity_vector(dq)
        return np.array(pin.integrate(self.model, q, dq * dt), dtype=float)
