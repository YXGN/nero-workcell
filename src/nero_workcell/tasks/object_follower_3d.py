#!/usr/bin/env python3
# coding=utf-8
"""
物体跟随任务 (Visual Servoing)
结合 RealSense 相机和 YOLO 模型，控制机械臂跟随指定物体。

用法:
    python -m nero_workcell.tasks.object_follower --target bottle --conf 0.5
"""

import time
import logging
import argparse
from typing import List

import cv2
import numpy as np
from nero_workcell.core.target_object import TargetObject
from nero_workcell.utils.common import transform_to_base
import pyrealsense2 as rs
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R

from nero_workcell.core import NeroController, RealSenseCamera

logger = logging.getLogger(__name__)


class PIDController:
    """简单的 PID 控制器，用于计算运动速度"""
    def __init__(self, kp: float, ki: float, kd: float, max_out: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_out = max_out
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def compute(self, error: float) -> float:
        current_time = time.time()
        dt = current_time - self.last_time
        if dt <= 0:
            return 0.0

        # 比例项
        p_term = self.kp * error

        # 积分项
        self.integral += error * dt
        i_term = self.ki * self.integral

        # 微分项
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # 计算输出
        output = p_term + i_term + d_term
        
        # 限幅
        output = np.clip(output, -self.max_out, self.max_out)

        self.prev_error = error
        self.last_time = current_time
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()


class ObjectFollower:
    def __init__(self, 
                 target_class: str, 
                 robot_channel: str = "can0",
                 model_path: str = 'yolov8n.pt',
                 conf_threshold: float = 0.5):
        
        self.target_class = target_class
        self.conf_threshold = conf_threshold
        
        # 初始化相机
        self.width = 640
        self.height = 480
        self.camera = None
        
        # 初始化 YOLO
        logger.info(f"加载 YOLO 模型: {model_path}")
        self.model = YOLO(model_path)

        # 获取目标类别的 ID
        self.target_class_id = None
        for idx, name in self.model.names.items():
            if name == self.target_class:
                self.target_class_id = idx
                break
        if self.target_class_id is None:
            logger.warning(f"目标类别 '{self.target_class}' 未在模型中找到。")
        
        # 初始化 PID 控制器 (X轴和Y轴)
        # 参数需要根据实际机械臂响应速度进行调整
        self.pid_x = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05) # 这里的单位是 m/s
        self.pid_y = PIDController(kp=0.0005, ki=0.0, kd=0.0001, max_out=0.05)
        
        # 机械臂实例
        self.robot = NeroController(robot_channel, "nero")
        self.is_running = False

    def setup_camera(self):
        """自动查找并连接第一个 RealSense 相机"""
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未找到 RealSense 相机")
        
        camera_serial = devices[0].get_info(rs.camera_info.serial_number)
        logger.info(f"使用相机: {camera_serial}")
        self.camera = RealSenseCamera(width=self.width, height=self.height, fps=30, serial_number=camera_serial) 
        
        if not self.camera.start():
            raise RuntimeError("相机启动失败")

    def move_robot(self, vx: float, vy: float):
        """
        控制机械臂移动
        vx, vy: PID 输出的控制量，此处作为位移增量处理
        
        注意：这里假设了相机坐标系与机械臂末端坐标系的关系。
        通常 RealSense: +X 右, +Y 下, +Z 前
        机械臂末端: 需要根据实际安装确认。
        这里假设：
        - 图像 X 轴偏差 -> 控制机械臂沿 Y 轴移动 (左右)
        - 图像 Y 轴偏差 -> 控制机械臂沿 X 轴移动 (上下/前后，取决于安装)
        
        *请根据实际情况修改轴映射*
        """
        if not self.robot.is_connected():
            return

        # 简单的死区设置，避免微小抖动
        if abs(vx) < 0.001 and abs(vy) < 0.001:
            return

        try:
            # 坐标系映射 (根据实际安装调整)
            # 假设: 图像 X+ (右) -> 机械臂 Y-
            # 假设: 图像 Y+ (下) -> 机械臂 X-
            
            scale = 0.5
            dx = -vy * scale
            dy = -vx * scale
            
            # 发送相对运动指令
            self.robot.move_relative(dx=dx, dy=dy)
            
        except Exception as e:
            logger.error(f"运动控制失败: {e}")
    def _yolo_detect(self, color: np.ndarray, depth: np.ndarray) -> List[TargetObject]:
        """
        Run object detection and project valid detections into the camera frame.

        This method performs YOLO inference on the color image, filters detections
        by confidence and target classes, reads local depth around each box center,
        and converts the pixel center into a 3D point in camera
        coordinates (`position = [x, y, z]`, `frame="camera"`).

        Args:
            color: RGB/BGR color image used for YOLO inference.
            depth: Depth image aligned with `color`.

        Returns:
            A list of camera-frame detections. Each item follows `TargetObject`
            and includes class info, 2D box/center, confidence, and 3D position.
        """
        results = self.model(color, verbose=False)
        h, w = depth.shape
        target_objects: List[TargetObject] = []

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf < self.detection_confidence:
                    continue
                if cls_id not in [self.bottle_class_id, self.bowl_class_id]:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cu, cv_pt = (x1 + x2) // 2, (y1 + y2) // 2

                # Estimate depth with a local median window for robustness.
                region = depth[max(0, cv_pt - 5):min(h, cv_pt + 5), max(0, cu - 5):min(w, cu + 5)]
                valid = region[region > 0]
                d = np.median(valid) if len(valid) > 0 else 0
                if d <= 0:
                    continue

                p_cam = np.array([(cu - self.cx) * d / self.fx, (cv_pt - self.cy) * d / self.fy, d])
                obj_name = "bottle" if cls_id == self.bottle_class_id else "bowl"
                target_objects.append(
                    TargetObject(
                        name=obj_name,
                        class_id=cls_id,
                        bbox=(x1, y1, x2, y2),
                        center=(cu, cv_pt),
                        position=p_cam,
                        conf=conf,
                        frame="camera",
                    )
                )

        logger.debug("VisionDetector.detect: %d camera objects detected", len(target_objects))
        return target_objects

    def _pick_best_target_by_class(target_objects_base, class_id):
        """从同类别候选中选取置信度最高的目标（基座坐标系）。"""
        candidates = [
            obj for obj in target_objects_base
            if obj.class_id == class_id and obj.frame == "base"
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda obj: float(obj.conf))

    def detect_object(self):
        """
        单次采集并识别目标，返回基座坐标系中的位置。

        Returns:
            dict | None: 包含基座坐标系下的目标位置、检测对象列表等信息；
            当读取帧或获取末端位姿失败时返回 None。
        """
        # 读取图像
        frame = self.camera.read_frame()
        color, depth = frame["color"], frame["depth"]
        if color is None or depth is None:
            return None
        # YOLO检测
        detected_objects_camera = self._yolo_detect(color, depth)

        # 获取机械臂末端位姿
        try:
            # 获取 Nero 机械臂法兰位姿 [x, y, z, roll, pitch, yaw] (单位: m, rad)
            T_gripper2base = self.robot.get_flange_pose()
            if T_gripper2base is None:
                logger.warning("[检测] 获取位姿失败: 无数据")
                return None
        except Exception as e:
            logger.warning(f"[检测] 获取位姿失败: {e}")
            return None

        T_cam2base = T_gripper2base @ self.T_cam2gripper
        detected_objects_base = transform_to_base(
            detected_objects_camera,
            T_cam2base,
        )

        bottle_obj = _pick_best_target_by_class(detected_objects_base, class_id)
        
        positions_base = {
            "bottle": None if bottle_obj is None else np.array(bottle_obj.position, dtype=float),
        }

        return {
            "color": color,
            "depth": depth,
            "T_cam2base": T_cam2base.copy(),
            "target_objects_camera": detected_objects_camera,
            "target_objects_base": detected_objects_base,
            "targets_base": {
                "bottle": bottle_obj,
            },
            "positions_base": positions_base,
            "gripper_pos": T_gripper2base[:3, 3].copy(),
            "timestamp": time.time(),
        }
    def generate_waypoints(target_pos):
        """生成静态抓取路径点序列（基于固定姿态，目标 3D 点位）。"""
        approach_height_offset = min(APPROACH_HEIGHT_OFFSET, STATIC_PICK_APPROACH_HEIGHT_OFFSET)
        waypoints = [
            ("pre_grasp", compute_target_with_offset(target_pos, APPROACH_HEIGHT_OFFSET)),
            ("approach", compute_target_with_offset(target_pos, approach_height_offset)),
            ("grasp", compute_target_with_offset(target_pos, GRASP_HEIGHT_OFFSET)),
            ("retreat", compute_target_with_offset(target_pos, LIFT_HEIGHT_OFFSET)),
        ]
        return waypoints
    
    def execute_waypoints(self, waypoints, tilted=True, blocking=True, max_step=WAYPOINT_MAX_STEP):
        """顺序执行路径点，并做简单步长约束检查。"""
        current_pos = robot.get_current_pose()
        prev_pos = np.array(current_pos, dtype=float)

        for name, target_pos in waypoints:
            target_pos = np.array(target_pos, dtype=float)
            step = np.linalg.norm(target_pos - prev_pos)
            if max_step is not None and step > max_step:
                logger.warning(
                    f"路径点步长过大: {name}, step={step:.3f}m > {max_step:.3f}m"
                )
                return False

            logger.info(
                f"执行路径点 {name}: "
                f"({target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f})"
            )
            if not robot.move_to(target_pos, tilted=tilted, blocking=blocking):
                logger.warning(f"执行路径点失败: {name}")
                return False

            prev_pos = target_pos

        return True

    def follow_object(self, object):
        bottle_pos = np.array(bottle_pos, dtype=float)
        bottle_waypoints = generate_waypoints(np.array(bottle_pos, dtype=float))
        execute_waypoints(self.robot, bottle_waypoints)


    def run(self):
        # 1. 加载手眼标定
        calib_file = "configs" / "eye_in_hand_calibration.json"
        self.T_cam2gripper = load_eye_in_hand_calibration(calib_file)
        # 2. 启动相机，加载相机内参
        self.setup_camera()
        intrinsics = self.camera.get_intrinsics()

        # 3. 连接机械臂
        if not self.robot.connect():
            logger.error("机械臂连接失败，任务终止")
            return
        self.is_running = True
        
        logger.info(f"开始跟随任务，目标: {self.target_class}")
        logger.info("按 'q' 退出")

        center_x, center_y = self.width // 2, self.height // 2

        try:
            while self.is_running:
                
                target_box = None
                max_conf = 0

                object = self.detect_object()
                if object is None:
                    logger.debug("没有检测到目标")
                    continue
                
                cv2.imshow('Detection (Eye-in-Hand)', object["color"])
                bottle_pos = scene["positions_base"]["bottle"]
                logger.info("瓶子和碗已就绪，按's'确认，开始执行任务...")
                self.follow_object(bottle_pos)

                task_ok = execute_grasp_task(
                    robot,
                    camera,
                    vision_detector,
                    T_cam2gripper,
                    bottle_pos,
                    bowl_pos,
                )
                
                

                # 显示中心十字
                cv2.line(display_img, (center_x-20, center_y), (center_x+20, center_y), (255, 0, 0), 1)
                cv2.line(display_img, (center_x, center_y-20), (center_x, center_y+20), (255, 0, 0), 1)

                cv2.imshow("Object Follower", display_img)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.camera.stop()
            cv2.destroyAllWindows()
            logger.info("任务结束")


def main():
    parser = argparse.ArgumentParser(description="Nero Workcell - Object Following Task")
    parser.add_argument("--target", type=str, default="bottle", help="Target object class name (e.g., bottle, cup)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )

    follower = ObjectFollower(
        target_class=args.target,
        robot_channel="can0",
        model_path=args.model,
        conf_threshold=args.conf
    )
    
    follower.run()


if __name__ == "__main__":
    main()
