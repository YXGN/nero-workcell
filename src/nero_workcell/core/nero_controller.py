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

    def connect(self, speed_percent: int = 20) -> bool:
        """Connect to the robot arm."""
        try:
            cfg = create_agx_arm_config(robot=self.robot_type, comm="can", channel=self.channel)
            
            self.robot = AgxArmFactory.create_arm(cfg)
            self.robot.connect()
            self.robot.enable()
            self.robot.set_speed_percent(speed_percent)
            
            # Initialize end effector (AGX_GRIPPER).
            try:
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
            except Exception as e:
                logger.warning(f"Failed to initialize end effector (possibly not installed or model mismatch): {e}")
            
            self._connected = True
            logger.info(f"Connected to Nero robot arm: {self.channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Nero robot arm: {e}")
            self._connected = False
            return False

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
        if not self.is_connected():
            return None
        try:
            pose = self.robot.get_flange_pose()
            if pose is not None:
                x, y, z, roll, pitch, yaw = pose.msg
            else:
                return None

            matrix = np.eye(4)
            matrix[:3, :3] = R.from_euler('xyz', [roll, pitch, yaw], degrees=False).as_matrix()
            matrix[:3, 3] = [x, y, z]
        
        except Exception as e:
            logger.error(f"Failed to get flange pose: {e}")
            return None
        return matrix

    def get_flange_pose(self) -> Optional[List[float]]:
        """
        Get robot flange pose as [x, y, z, roll, pitch, yaw].
        """
        if not self.is_connected():
            return None
        try:
            pose = self.robot.get_flange_pose()
            if pose is not None:
                return list(pose.msg)
            return None
        except Exception as e:
            logger.error(f"Failed to get flange pose: {e}")
            return None

    def get_tcp_pose(self) -> Optional[List[float]]:
        """
        Get robot TCP pose as [x, y, z, roll, pitch, yaw].
        """
        if not self.is_connected():
            return None
        try:
            pose = self.robot.get_tcp_pose()
            if pose is not None:
                return list(pose.msg)
            return None
        except Exception as e:
            logger.error(f"Failed to get TCP pose: {e}")
            return None

    def move_p(self, pose: List[float]):
        """
        Send a Cartesian point-to-point motion command.

        :param pose: [x, y, z, r, p, y]
        """
        if not self.is_connected():
            return
        try:
            self.robot.move_p(pose)
        except Exception as e:
            logger.error(f"Failed to send motion command: {e}")

    def move_relative(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, 
                      dr: float = 0.0, dp: float = 0.0, dyaw: float = 0.0):
        """
        Move relative to the current TCP pose.

        :param dx, dy, dz: Position deltas (meters)
        :param dr, dp, dyaw: Orientation deltas (radians)
        """
        if not self.is_connected():
            return

        current_pose = self.get_tcp_pose()
        if current_pose is None:
            return

        # Compute target pose by linear delta composition.
        # Orientation composition is an approximation for small angles.
        target_pose = list(current_pose)
        deltas = [dx, dy, dz, dr, dp, dyaw]
        for i in range(6):
            target_pose[i] += deltas[i]

        self.move_p(target_pose)

    def move_gripper(self, width: float, force: float = 1.0):
        """
        Control the AGX gripper.

        :param width: Jaw width (meters), range [0.0, 0.1]
        :param force: Gripping force (newtons), range [0.0, 3.0]
        """
        if not self.is_connected() or self.end_effector is None:
            logger.warning("Cannot control gripper: robot not connected or end effector not initialized")
            return

        try:
            self.end_effector.move_gripper(width=width, force=force)
        except Exception as e:
            logger.error(f"Gripper control failed: {e}")