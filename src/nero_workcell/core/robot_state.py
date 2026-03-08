"""
Robot state containers shared by model-based controllers.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RobotState:
    """Snapshot of the robot state used by the differential IK follower."""

    joint_positions: np.ndarray
    joint_velocities: Optional[np.ndarray] = None
    tcp_pose: Optional[np.ndarray] = None
    timestamp: Optional[float] = None
