"""
AGX robot arm controller wrapper.
"""
import logging
import time
from typing import Optional, List
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyAgxArm import create_agx_arm_config, AgxArmFactory

from .robot_state import RobotState

logger = logging.getLogger(__name__)


class ArmController:
    DEFAULT_ROBOT_TYPE = "nero"
    _POLL_INTERVAL = 0.1

    def __init__(self, channel: str = "can_left", *, robot_type: str = DEFAULT_ROBOT_TYPE):
        """
        Initialize the controller.

        :param channel: CAN channel (for example, "can0", "can")
        """
        self.channel = channel
        self.robot_type = robot_type
        self.config = None
        self.robot = None
        self.end_effector = None
        self._connected = False

    def _clear_connection_state(self):
        self.config = None
        self.robot = None
        self.end_effector = None
        self._connected = False

    def _wait_until_enabled(self, robot, *, timeout: float):
        deadline = time.monotonic() + timeout
        while not robot.enable():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Failed to enable robot: Timeout ({timeout}s) exceeded")
            time.sleep(self._POLL_INTERVAL)

    def _wait_until_end_effector_ready(
        self,
        end_effector,
        *,
        timeout: float,
    ):
        deadline = time.monotonic() + timeout
        while not end_effector.is_ok():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Failed to initialize end effector: Timeout ({timeout}s) exceeded"
                )
            time.sleep(self._POLL_INTERVAL)

        logger.info("AGX_GRIPPER end effector initialized")
        return end_effector

    def connect(self, speed_percent: int = 5, timeout: float = 5.0) -> bool:
        """Connect to the robot arm."""
        if self.is_connected():
            logger.warning(
                "connect() called while already connected to %s robot arm: %s",
                self.robot_type, self.channel,
            )
            return True

        if not 0 <= speed_percent <= 100:
            raise ValueError(f"speed_percent must be in [0, 100], got {speed_percent}")
        if timeout <= 0.0:
            raise ValueError(f"timeout must be greater than 0, got {timeout}")

        cfg = create_agx_arm_config(robot=self.robot_type, comm="can", channel=self.channel)
        assert cfg is not None, "create_agx_arm_config() returned None"
        assert isinstance(cfg, dict), (
            f"create_agx_arm_config() returned invalid config type: {type(cfg).__name__}"
        )
        
        robot = AgxArmFactory.create_arm(cfg)
        # The vendor docs recommend creating the effector driver before connect().
        end_effector = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        if end_effector is None:
            raise RuntimeError(
                "Failed to initialize end effector: init_effector() returned None"
            )
        robot.connect()
        self._wait_until_enabled(robot, timeout=timeout)
        robot.set_speed_percent(speed_percent)
        end_effector = self._wait_until_end_effector_ready(
            end_effector,
            timeout=timeout,
        )

        self.config = cfg
        self.robot = robot
        self.end_effector = end_effector
        self._connected = True
        logger.info("Connected to %s robot arm: %s", self.robot_type, self.channel)
        return True

    def disconnect(self):
        """Disconnect from the robot."""
        self._clear_connection_state()
        logger.info("Disconnected from %s robot arm", self.robot_type)

    def is_connected(self) -> bool:
        return self._connected and self.robot is not None

    @property
    def joint_names(self) -> List[str]:
        assert self.is_connected(), "Robot is not connected"
        assert self.config is not None, "joint_names requires connect() to initialize config"

        joint_names = self.config.get("joint_names")
        assert joint_names, "Config missing joint_names"
        assert not isinstance(joint_names, (str, bytes)), (
            "Config joint_names must be a sequence of joint names"
        )
        return [str(name) for name in joint_names]

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        assert self.config is not None, "joint_limits requires connect() to initialize config"
        limits = self.config.get("joint_limits", {})
        return {
            str(name): (float(bounds[0]), float(bounds[1]))
            for name, bounds in limits.items()
        }

    def get_joint_angles(self) -> Optional[List[float]]:
        """Get robot joint positions as [j1, ..., j7] in radians."""
        assert self.is_connected(), "Robot is not connected"

        joint_angles = self.robot.get_joint_angles()
        if joint_angles is not None:
            return list(joint_angles.msg)
        logger.warning("Failed to read joint angles: robot.get_joint_angles() returned None")
        return None

    def get_robot_state(self) -> Optional[RobotState]:
        """Read the latest joint-space state needed by model-based controllers."""
        assert self.is_connected(), "Robot is not connected"

        joint_msg = self.robot.get_joint_angles()
        if joint_msg is None:
            return None

        joint_angles = list(joint_msg.msg)

        tcp_pose = self.get_tcp_pose()
        timestamp = float(joint_msg.timestamp)

        tcp_pose_arr = None
        if tcp_pose is not None:
            tcp_pose_arr = np.array(tcp_pose, dtype=float)

        return RobotState(
            joint_positions=np.array(joint_angles, dtype=float),
            tcp_pose=tcp_pose_arr,
            timestamp=timestamp,
        )

    def get_current_pose(self) -> Optional[np.ndarray]:
        """Get the homogeneous transform (4x4) of flange relative to base."""
        assert self.is_connected(), "Robot is not connected"
        
        pose = self.robot.get_flange_pose()
        if pose is not None:
            x, y, z, roll, pitch, yaw = pose.msg
        else:
            return None

        matrix = np.eye(4)
        matrix[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
        matrix[:3, 3] = [x, y, z]
        return matrix

    def get_flange_pose(self) -> Optional[List[float]]:
        """
        Get robot flange pose as [x, y, z, roll, pitch, yaw].
        """
        assert self.is_connected(), "Robot is not connected"
        
        pose = self.robot.get_flange_pose()
        if pose is not None:
            return list(pose.msg)
        return None

    def get_tcp_pose(self) -> Optional[List[float]]:
        """
        Get robot TCP pose as [x, y, z, roll, pitch, yaw].
        """
        assert self.is_connected(), "Robot is not connected"
        
        pose = self.robot.get_tcp_pose()
        if pose is not None:
            return list(pose.msg)
        return None

    def set_tcp_offset(self, pose: List[float]) -> None:
        """
        Set TCP offset relative to flange, in flange frame.
        pose: [x, y, z, roll, pitch, yaw] (meters / radians)
        """
        assert self.is_connected(), "Robot is not connected"
        self.robot.set_tcp_offset(pose)

    def get_flange2tcp_pose(self, flange_pose: List[float]) -> Optional[List[float]]:
        """Convert a flange pose to TCP pose using the current TCP offset."""
        assert self.is_connected(), "Robot is not connected"
        return self.robot.get_flange2tcp_pose(flange_pose)

    def get_tcp2flange_pose(self, tcp_pose: List[float]) -> Optional[List[float]]:
        """Convert a target TCP pose to flange pose. Result can be passed to move_p()."""
        assert self.is_connected(), "Robot is not connected"
        return self.robot.get_tcp2flange_pose(tcp_pose)

    def get_arm_status(self):
        """
        Get the robot arm status structure.
        """
        assert self.is_connected(), "Robot is not connected"
        return self.robot.get_arm_status()

    def _wait_motion_done(self, timeout: float = 5.0, poll_interval: float = 0.1) -> bool:
        """
        Internal helper: Wait until motion_status == 0 or timeout occurs.
        """
        assert self.is_connected(), "Robot is not connected"

        time.sleep(0.5)  # Initial settling time
        start_t = time.monotonic()
        while True:
            status = self.get_arm_status()
            if status is not None and getattr(status.msg, "motion_status", None) == 0:
                return True
            if time.monotonic() - start_t > timeout:
                return False
            time.sleep(poll_interval)

    def move_j(self, joints: List[float], blocking: bool = False, timeout: float = 5.0) -> bool:
        """
        Send a joint point-to-point motion command.

        :param joints: [j1, j2, j3, j4, j5, j6, j7] (radians)
        :param blocking: If True, wait until motion is complete.
        :param timeout: Max wait time in seconds if blocking is True.
        :return: True if command sent.
        """
        assert self.is_connected(), "Robot is not connected"
        
        self.robot.move_j(joints)
        if blocking:
            return self._wait_motion_done(timeout=timeout)
        return True

    def move_to_home(self, blocking: bool = True) -> bool:
        """
        Move the robot to a safe 'home' position using joint interpolation.
        """
        if self.robot_type != self.DEFAULT_ROBOT_TYPE:
            raise NotImplementedError(
                f"move_to_home() is not defined for robot_type={self.robot_type!r}"
            )
        scale = 57324.840764
        home_joints = [-88000 / scale, 
                       -97000 / scale, 
                       2719 / scale, 
                       118000 / scale,
                       6882 / scale,
                       -9 / scale, 
                       73000 / scale]
        logger.info("Moving to home position: %s", home_joints)
        return self.move_j(home_joints, blocking=blocking)

    def move_p(self, pose: List[float], blocking: bool = False, timeout: float = 5.0) -> bool:
        """
        Send a Cartesian point-to-point motion command.

        :param pose: [x, y, z, r, p, y]
        :param blocking: If True, wait until motion is complete.
        :param timeout: Max wait time in seconds if blocking is True.
        :return: True if command sent (and completed if blocking), False otherwise.
        """
        assert self.is_connected(), "Robot is not connected"
        
        self.robot.move_p(pose)
        if blocking:
            return self._wait_motion_done(timeout=timeout)
        return True

    def set_motion_mode(self, motion_mode: str = "p") -> None:
        """Set controller motion mode ('p' or 'j')."""
        assert self.is_connected(), "Robot is not connected"
        self.robot.set_motion_mode(motion_mode)

    def set_normal_mode(self) -> None:
        """Switch the arm back to normal single-arm control mode."""
        assert self.is_connected(), "Robot is not connected"
        self.robot.set_normal_mode()

    def move_relative(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, 
                      dr: float = 0.0, dp: float = 0.0, dyaw: float = 0.0,
                      *, blocking: bool = False, timeout: float = 5.0) -> bool:
        """
        Move relative to the current TCP pose.

        :param dx, dy, dz: Position deltas (meters)
        :param dr, dp, dyaw: Orientation deltas (radians)
        :param blocking: If True, wait until motion is complete.
        :param timeout: Max wait time in seconds if blocking is True.
        """
        assert self.is_connected(), "Robot is not connected"

        current_pose = self.get_tcp_pose()
        if current_pose is None:
            return False

        # Compute target pose by linear delta composition.
        # Orientation composition is an approximation for small angles.
        target_pose = list(current_pose)
        deltas = [dx, dy, dz, dr, dp, dyaw]
        for i in range(6):
            target_pose[i] += deltas[i]

        return self.move_p(target_pose, blocking=blocking, timeout=timeout)

    def move_gripper(self, width: float, force: float = 1.0):
        """
        Control the AGX gripper.

        :param width: Jaw width (meters), range [0.0, 0.1]
        :param force: Gripping force (newtons), range [0.0, 3.0]
        """
        assert self.is_connected(), "Robot is not connected"
        assert self.end_effector is not None, "End effector is not initialized"
        self.end_effector.move_gripper(width=width, force=force)
