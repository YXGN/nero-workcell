"""
Nero robot arm controller wrapper.
"""
import logging
import time
from typing import Optional, List
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyAgxArm import create_agx_arm_config, AgxArmFactory

logger = logging.getLogger(__name__)

class NeroController:
    def __init__(self, channel: str = "can0", robot_type: str = "nero"):
        """
        Initialize the controller.

        :param channel: CAN channel (for example, "can0", "can")
        :param robot_type: Robot model (default: "nero")
        """
        self.channel = channel
        self.robot_type = robot_type
        self.robot = None
        self.end_effector = None
        self._connected = False

    def connect(self, speed_percent: int = 20, timeout: float = 5.0) -> bool:
        """Connect to the robot arm."""
        cfg = create_agx_arm_config(robot=self.robot_type, comm="can", channel=self.channel)
        
        self.robot = AgxArmFactory.create_arm(cfg)
        self.robot.connect()
        
        # Enable robot with timeout check
        start_t = time.monotonic()
        while True:
            self.robot.enable()
            time.sleep(0.5)
            if self.robot.get_arm_status() is not None:
                break
            if time.monotonic() - start_t > timeout:
                raise RuntimeError(f"Failed to enable robot: Timeout ({timeout}s) exceeded")

        self.robot.set_speed_percent(speed_percent)
        
        # Initialize end effector (AGX_GRIPPER).
        self.end_effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        # Wait for effector to be ready (up to 1.0s)
        for _ in range(10):
            if self.end_effector.is_ok():
                break
            time.sleep(0.1)

        if self.end_effector.is_ok():
            logger.info("AGX_GRIPPER end effector initialized")
        else:
            logger.warning("Failed to initialize end effector: is_ok() is False")
            self.end_effector = None
        
        self._connected = True
        logger.info(f"Connected to Nero robot arm: {self.channel}")
        return True

    def disconnect(self):
        """Disconnect from the robot."""
        self._connected = False
        self.robot = None
        self.end_effector = None
        logger.info("Disconnected from Nero robot arm")

    def is_connected(self) -> bool:
        return self._connected and self.robot is not None

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

        :param joints: [j1, j2, j3, j4, j5, j6] (radians)
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
        if self.end_effector is None:
            logger.warning("Cannot control gripper: end effector not initialized")
            return

        self.end_effector.move_gripper(width=width, force=force)