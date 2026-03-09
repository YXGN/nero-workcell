"""
Geometric helpers for approaching a static 3D target point.
"""

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


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
class OffsetComponents:
    """参考位置偏移在接近轴方向和横向方向上的分解结果。

    字段说明：
        axial_offset: 沿 ``approach_direction`` 的偏移分量，表示末端在接近方向上还差多少。
        lateral_offset: 垂直于 ``approach_direction`` 的偏移分量，表示末端偏离接近轴多少。

    示例：
        若 ``approach_direction = [0, 0, -1]``，且
        ``target_position - tcp_position = [0.02, -0.02, -0.10]``，则：

        - ``axial_offset = [0.00, 0.00, -0.10]``
        - ``lateral_offset = [0.02, -0.02, 0.00]``

        这表示末端还需要沿接近方向推进 10 cm，同时在横向上偏离接近轴 2 cm。
    """

    axial_offset: np.ndarray
    lateral_offset: np.ndarray


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

    def decompose_offset(
        self,
        tcp_position: np.ndarray,
        target_position: np.ndarray,
    ) -> OffsetComponents:
        """Split the TCP-to-target position offset into axial and lateral components."""
        tcp_position = np.array(tcp_position, dtype=float)
        target_position = np.array(target_position, dtype=float)
        position_offset = target_position - tcp_position
        axial_offset = (
            np.dot(position_offset, self.approach_direction) * self.approach_direction
        )
        lateral_offset = position_offset - axial_offset
        return OffsetComponents(axial_offset=axial_offset, lateral_offset=lateral_offset)

    def is_pre_standoff_reached(
        self,
        tcp_position: np.ndarray,
        plan: ApproachPlan,
        *,
        lateral_tolerance: float,
        axial_tolerance: float,
    ) -> bool:
        """Check whether the TCP is close enough to the pre-standoff waypoint."""
        components = self.decompose_offset(tcp_position, plan.pre_standoff_position)
        lateral_offset_norm = float(np.linalg.norm(components.lateral_offset))
        axial_offset_norm = float(np.linalg.norm(components.axial_offset))
        reached = (
            lateral_offset_norm <= lateral_tolerance
            and axial_offset_norm <= axial_tolerance
        )
        if reached:
            logger.info(
                "[approach] pre-standoff reached: lateral_offset=%.4f axial_offset=%.4f",
                lateral_offset_norm,
                axial_offset_norm,
            )
        return reached

    def is_standoff_reached(
        self,
        tcp_position: np.ndarray,
        plan: ApproachPlan,
        *,
        position_tolerance: float,
    ) -> bool:
        """Check whether the TCP has reached the standoff goal."""
        tcp_position = np.array(tcp_position, dtype=float)
        position_offset_norm = float(np.linalg.norm(plan.standoff_position - tcp_position))
        reached = position_offset_norm <= position_tolerance
        if reached:
            logger.info(
                "[approach] standoff reached: position_offset=%.4f",
                position_offset_norm,
            )
        return reached
