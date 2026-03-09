"""
Smooth Cartesian reference generation for model-based following.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    """Position and velocity sampled from a reference segment."""

    position: np.ndarray
    velocity: np.ndarray
    progress: float
    finished: bool


class CartesianTrajectory:
    """表示两个三维点之间的一段 minimum-jerk 笛卡尔局部参考轨迹。

    该类保存一段从 ``start_position`` 到 ``goal_position`` 的单段轨迹，
    其中空间路径是两点之间的直线，时间参数化采用 minimum-jerk
    五次多项式。这样做的结果是：轨迹在起点和终点处速度为零，
    中间的加减速过程更平滑，适合机械臂末端跟随与精细接近任务。

    这条轨迹并不是一次性输出整段离散路径，而是先保存：

    - 起点 ``start_position``
    - 终点 ``goal_position``
    - 开始时间 ``start_time``
    - 持续时间 ``duration``

    然后在运行过程中通过 ``sample(now)`` 按当前时刻进行采样，
    返回该时刻的参考位置 ``position`` 和参考速度 ``velocity``。
    控制循环连续调用 ``sample(now)`` 得到的一串采样点，
    合起来就构成了“从已知起点到目标点的平滑局部参考轨迹”。

    其中 ``duration`` 可以直接指定，也可以通过 ``from_distance()``
    按路径长度和最大允许速度自动估算：

    ``duration = max(min_duration, distance / max_speed)``

    这表示轨迹至少要满足速度上限约束，同时对很短的位移保留一个
    最小执行时间，避免参考轨迹变化过急。
    """

    def __init__(
        self,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        *,
        start_time: float,
        duration: float,
    ):
        if duration <= 0.0:
            raise ValueError("duration must be greater than 0")

        self.start_position = np.array(start_position, dtype=float)
        self.goal_position = np.array(goal_position, dtype=float)
        self.start_time = float(start_time)
        self.duration = float(duration)
        self.delta = self.goal_position - self.start_position

    @classmethod
    def from_distance(
        cls,
        start_position: np.ndarray,
        goal_position: np.ndarray,
        *,
        start_time: float,
        max_speed: float,
        min_duration: float = 0.2,
    ) -> "CartesianTrajectory":
        """根据路径长度与速度上限构造一段轨迹。

        轨迹持续时间按 ``max(min_duration, distance / max_speed)`` 计算，
        其中 ``distance`` 是起点到终点的直线距离。
        """
        if max_speed <= 0.0:
            raise ValueError("max_speed must be greater than 0")

        start_position = np.array(start_position, dtype=float)
        goal_position = np.array(goal_position, dtype=float)
        distance = float(np.linalg.norm(goal_position - start_position))
        duration = max(min_duration, distance / max_speed)
        return cls(
            start_position=start_position,
            goal_position=goal_position,
            start_time=start_time,
            duration=duration,
        )

    def sample(self, now: float) -> TrajectorySample:
        """按给定时刻对 minimum-jerk 笛卡尔轨迹进行采样。

        参数：
            now:
                当前采样时刻，通常使用单调时钟时间。

        返回：
            TrajectorySample:
                包含该时刻对应的参考位置、参考速度、轨迹进度以及
                轨迹是否结束的标记。

        说明：
            该函数先将当前时间归一化到 ``[0, 1]``，再通过五次多项式
            计算平滑插值系数 ``blend`` 及其时间导数 ``blend_rate``，
            从而生成速度和加速度都更平滑的参考轨迹。
        """
        tau = np.clip((float(now) - self.start_time) / self.duration, 0.0, 1.0)
        tau2 = tau * tau
        tau3 = tau2 * tau
        tau4 = tau3 * tau
        tau5 = tau4 * tau

        blend = 10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5
        blend_rate = (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4) / self.duration

        return TrajectorySample(
            position=self.start_position + blend * self.delta,
            velocity=blend_rate * self.delta,
            progress=float(tau),
            finished=bool(tau >= 1.0),
        )
