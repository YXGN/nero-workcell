"""
Nero 机械臂控制器封装
"""
import logging
from typing import Optional, List
import numpy as np
from scipy.spatial.transform import Rotation as R
from pyAgxArm import create_agx_arm_config, AgxArmFactory

logger = logging.getLogger(__name__)

class NeroController:
    def __init__(self, channel: str = "can0", robot_type: str = "nero"):
        """
        初始化控制器
        :param channel: CAN通道 (如 "can0", "can")
        :param robot: 机械臂型号 (默认 "nero")
        """
        self.channel = channel
        self.robot_type = robot_type
        self.robot = None
        self.end_effector = None
        self._connected = False

    def connect(self) -> bool:
        """连接机械臂"""
        try:
            cfg = create_agx_arm_config(robot=self.robot_type, comm="can", channel=self.channel)
            
            self.robot = AgxArmFactory.create_arm(cfg)
            self.robot.connect()
            self.robot.enable()
            
            # 初始化末端执行器 (AGX_GRIPPER)
            try:
                self.end_effector = self.robot.init_effector(self.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
                logger.info("已初始化 AGX_GRIPPER 末端执行器")
            except Exception as e:
                logger.warning(f"初始化末端执行器失败 (可能未安装或型号不匹配): {e}")
            
            self._connected = True
            logger.info(f"已连接到 Nero 机械臂: {self.channel}")
            return True
        except Exception as e:
            logger.error(f"连接 Nero 机械臂失败: {e}")
            self._connected = False
            return False

    def disconnect(self):
        """断开连接"""
        self._connected = False
        self.robot = None
        self.end_effector = None
        logger.info("已断开 Nero 机械臂连接")

    def is_connected(self) -> bool:
        return self._connected and self.robot is not None

    def get_flange_pose(self) -> Optional[np.ndarray]:
        """获取法兰相对于基座的齐次变换矩阵 (4x4)"""
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
            logger.error(f"获取法兰位姿失败: {e}")
            return None
        return matrix

    def get_tcp_pose(self) -> Optional[List[float]]:
        """
        获取机械臂 TCP 位姿 [x, y, z, r, p, y]
        """
        if not self.is_connected():
            return None
        try:
            pose = self.robot.get_tcp_pose()
            if pose is not None:
                return pose.msg
            return None
        except Exception as e:
            logger.error(f"获取 TCP 位姿失败: {e}")
            return None

    def move_p(self, pose: List[float]):
        """
        发送笛卡尔空间点到点运动指令
        :param pose: [x, y, z, r, p, y]
        """
        if not self.is_connected():
            return
        try:
            self.robot.move_p(pose)
        except Exception as e:
            logger.error(f"发送运动指令失败: {e}")

    def move_relative(self, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0, 
                      dr: float = 0.0, dp: float = 0.0, dyaw: float = 0.0):
        """
        相对当前 TCP 位姿进行移动
        :param dx, dy, dz: 位置增量 (米)
        :param dr, dp, dyaw: 姿态增量 (弧度)
        """
        if not self.is_connected():
            return

        current_pose = self.get_tcp_pose()
        if current_pose is None:
            return

        # 计算目标位姿 (简单的线性叠加，姿态叠加在小角度下近似可用)
        target_pose = list(current_pose)
        deltas = [dx, dy, dz, dr, dp, dyaw]
        for i in range(6):
            target_pose[i] += deltas[i]

        self.move_p(target_pose)

    def control_gripper(self, width: float, force: float = 1.0):
        """
        控制 AGX Gripper
        :param width: 开口宽度 (米), 范围 [0.0, 0.1]
        :param force: 夹持力 (牛顿), 范围 [0.0, 3.0]
        """
        if not self.is_connected() or self.end_effector is None:
            logger.warning("无法控制夹爪: 未连接或未初始化")
            return

        try:
            self.end_effector.move_gripper(width=width, force=force)
        except Exception as e:
            logger.error(f"夹爪控制失败: {e}")