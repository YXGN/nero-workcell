"""
Geometric helpers for approaching a static 3D target point.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ApproachPlan:
    """单个目标点对应的两段式接近规划结果。

    字段说明：
        target_position: 真实目标点在基坐标系下的位置。
        standoff_position: 末端最终希望到达的停靠点，与目标点沿接近方向保持固定距离。
        pre_standoff_position: 到达停靠点之前的过渡点，用于先完成粗接近，再执行精细接近。
        approach_direction: 接近方向的单位向量，表示末端朝目标推进时的运动方向。
    """

    target_position: np.ndarray
    standoff_position: np.ndarray
    pre_standoff_position: np.ndarray
    approach_direction: np.ndarray


@dataclass(frozen=True)
class ErrorComponents:
    """Error decomposition along and orthogonal to the approach axis."""

    axial_error: np.ndarray
    lateral_error: np.ndarray


class ApproachPlanner:
    """
    围绕目标点生成两段式接近参考点的几何规划器。 该类不直接控制关节，也不求解动力学；
    它只在笛卡尔空间中根据目标点生成两个关键参考位置：
    1. ``standoff_position``：末端最终希望停留的安全工作点。
        该点通常与真实目标保持一段固定距离，避免末端直接撞到目标。
    2. ``pre_standoff_position``：进入最终停靠点之前的过渡点。
        机械臂会先到这个点，再沿接近方向逼近 ``standoff_position``，
        从而让接近过程更稳定、更可控。

    示例：
        采用自上而下的接近方式 ``approach_direction = [0, 0, -1]``，
        目标点为 ``[0.2, 0.1, 0.0]``，停靠距离 ``0.30 m``，
        预停靠偏移 ``0.08 m``，则：

        - ``standoff_position = [0.2, 0.1, 0.3]``
        - ``pre_standoff_position = [0.2, 0.1, 0.38]``

        这表示机械臂末端会先移动到目标上方 0.38 米处，
        再进一步下降到目标上方 0.30 米处。
    """

    def __init__(
        self,
        standoff_distance: float = 0.3,
        pre_standoff_offset: float = 0.08,
        approach_direction: np.ndarray | tuple[float, float, float] = (0.0, 0.0, -1.0),
    ):
        if standoff_distance < 0.0:
            raise ValueError("standoff_distance must be non-negative")
        if pre_standoff_offset < 0.0:
            raise ValueError("pre_standoff_offset must be non-negative")

        self.standoff_distance = float(standoff_distance)
        self.pre_standoff_offset = float(pre_standoff_offset)
        self.approach_direction = self._normalize(np.array(approach_direction, dtype=float))

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape != (3,):
            raise ValueError(f"Expected a 3D vector, got shape {vector.shape}")
        norm = float(np.linalg.norm(vector))
        if norm <= 0.0:
            raise ValueError("approach_direction must be non-zero")
        return vector / norm

    def with_approach_direction(
        self,
        approach_direction: np.ndarray | tuple[float, float, float],
    ) -> "ApproachPlanner":
        """Create a copy with a different approach axis."""
        return ApproachPlanner(
            standoff_distance=self.standoff_distance,
            pre_standoff_offset=self.pre_standoff_offset,
            approach_direction=approach_direction,
        )

    def make_plan(self, target_position: np.ndarray) -> ApproachPlan:
        """Compute the pre-standoff point and final standoff point."""
        target_position = np.array(target_position, dtype=float)
        standoff_position = target_position - self.approach_direction * self.standoff_distance
        pre_standoff_position = target_position - self.approach_direction * (
            self.standoff_distance + self.pre_standoff_offset
        )
        return ApproachPlan(
            target_position=target_position,
            standoff_position=standoff_position,
            pre_standoff_position=pre_standoff_position,
            approach_direction=self.approach_direction.copy(),
        )

    def decompose_error(
        self,
        tcp_position: np.ndarray,
        reference_position: np.ndarray,
    ) -> ErrorComponents:
        """Split reference tracking error into axial and lateral components."""
        tcp_position = np.array(tcp_position, dtype=float)
        reference_position = np.array(reference_position, dtype=float)
        error = reference_position - tcp_position
        axial_error = np.dot(error, self.approach_direction) * self.approach_direction
        lateral_error = error - axial_error
        return ErrorComponents(axial_error=axial_error, lateral_error=lateral_error)

    def is_pre_standoff_reached(
        self,
        tcp_position: np.ndarray,
        plan: ApproachPlan,
        *,
        lateral_tolerance: float,
        axial_tolerance: float,
    ) -> bool:
        """Check whether the TCP is close enough to the pre-standoff waypoint."""
        components = self.decompose_error(tcp_position, plan.pre_standoff_position)
        return (
            np.linalg.norm(components.lateral_error) <= lateral_tolerance
            and np.linalg.norm(components.axial_error) <= axial_tolerance
        )

    def is_standoff_reached(
        self,
        tcp_position: np.ndarray,
        plan: ApproachPlan,
        *,
        position_tolerance: float,
    ) -> bool:
        """Check whether the TCP has reached the standoff goal."""
        tcp_position = np.array(tcp_position, dtype=float)
        return np.linalg.norm(plan.standoff_position - tcp_position) <= position_tolerance
