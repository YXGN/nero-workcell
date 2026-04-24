#!/usr/bin/env python3
# coding=utf-8
"""
手眼标定数据采集程序 V2 - 实时角点检测版本
- 实时显示角点检测结果（不保存到图片）
- 只有检测到角点才能保存
- 显示角点起始位置，确保一致性

按键说明:
- 's': 采集当前位姿和图像数据（仅角点检测成功时有效）
- 'q': 退出程序
- 'g': 切换井字格显示
- 'c': 切换角点检测显示
- '1': 只显示相机1
- '2': 只显示相机2
- '3': 显示双相机
"""

import os
import sys
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs

from nero_workcell.core import ArmController, RealSenseCamera

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
)

logger_ = logging.getLogger(__name__)


def get_connected_cameras():
    """获取所有连接的RealSense相机序列号"""

    ctx = rs.context()
    devices = ctx.query_devices()
    serial_numbers = []
    for dev in devices:
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        serial_numbers.append(serial)
        logger_.info(f"发现相机: {name} (序列号: {serial})")
    return serial_numbers


def create_folder_with_date():
    """创建带日期的数据文件夹（在当前目录下）"""
    today = datetime.now().strftime('%Y%m%d')
    # 直接在当前目录下创建 images 文件夹
    prefix_files = "./images"
    
    if not os.path.exists(prefix_files):
        os.makedirs(prefix_files)
    
    base_folder_name = os.path.join(prefix_files, f"data{today}")
    index = 0
    folder_path = base_folder_name
    
    while os.path.exists(folder_path):
        index += 1
        folder_path = f"{base_folder_name}{str(index).zfill(2)}"
    
    os.makedirs(folder_path)
    logger_.info(f"创建数据文件夹: {folder_path}")
    return folder_path


def draw_grid(image, grid_size=3):
    """在图像上绘制井字格"""
    h, w = image.shape[:2]
    color = (0, 255, 0)
    thickness = 1
    
    for i in range(1, grid_size):
        x = w * i // grid_size
        cv2.line(image, (x, 0), (x, h), color, thickness)
    
    for i in range(1, grid_size):
        y = h * i // grid_size
        cv2.line(image, (0, y), (w, y), color, thickness)
    
    return image


def detect_corners(image, corner_long, corner_short):
    """
    检测棋盘格角点，并确保起始点在左下角
    返回: (success, corners, corners_subpix)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (corner_long, corner_short), None)
    
    if ret:
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
        corners_subpix = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        
        # 确保起始点在左下角
        corners_subpix = normalize_corner_order(corners_subpix)
        
        return True, corners, corners_subpix
    
    return False, None, None


def normalize_corner_order(corners):
    """
    调整角点顺序，确保起始点始终在左下角
    
    逻辑：
    - 比较第一个点和最后一个点的位置
    - 如果第一个点不在左下角（y值较大，x值较小），则翻转整个数组
    """
    first_point = corners[0][0]
    last_point = corners[-1][0]
    
    # 判断哪个点更接近左下角
    # 左下角的特征：x值小，y值大（图像坐标系y向下）
    # 用 (y - x) 作为判断依据，值越大越接近左下角
    first_score = first_point[1] - first_point[0]  # y - x
    last_score = last_point[1] - last_point[0]
    
    # 如果最后一个点更接近左下角，翻转数组
    if last_score > first_score:
        corners = corners[::-1].copy()
    
    return corners


def draw_corners_with_info(image, corners, corner_long, corner_short, detected):
    """
    在图像上绘制角点和起始/终点标记
    """
    display = image.copy()
    
    if detected and corners is not None:
        # 绘制所有角点
        cv2.drawChessboardCorners(display, (corner_long, corner_short), corners, detected)
        
        # 标记起始点（第一个角点）- 用大绿色圆圈
        start_point = tuple(corners[0][0].astype(int))
        cv2.circle(display, start_point, 15, (0, 255, 0), 3)
        cv2.putText(display, "START", (start_point[0] + 20, start_point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 标记终点（最后一个角点）- 用大红色圆圈
        end_point = tuple(corners[-1][0].astype(int))
        cv2.circle(display, end_point, 15, (0, 0, 255), 3)
        cv2.putText(display, "END", (end_point[0] + 20, end_point[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 显示检测状态 - 绿色
        cv2.putText(display, "Corners: DETECTED", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(display, "Press 's' to save", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        # 显示检测状态 - 红色
        cv2.putText(display, "Corners: NOT FOUND", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display, "Cannot save!", (10, 105), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return display


def main():
    """主函数"""
    
    # 从配置文件读取参数（与脚本同目录）
    config_file = Path(__file__).with_name("config.json")
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 相机参数
    WIDTH = config['camera']['width']
    HEIGHT = config['camera']['height']
    FPS = config['camera']['fps']
    
    # 标定板参数
    corner_long = config['checkerboard']['corner_point_long']
    corner_short = config['checkerboard']['corner_point_short']
    
    # 连接 Nero 机械臂
    logger_.info("连接 Nero 机械臂...")
    robot = ArmController()
    
    if not robot.connect():
        logger_.error("无法连接机械臂!")
        return
    
    logger_.info("机械臂连接成功!")
    
    # 获取连接的相机
    logger_.info("扫描连接的相机...")
    serial_numbers = get_connected_cameras()
    
    if len(serial_numbers) < 1:
        logger_.error("未发现相机!")
        robot.disconnect()
        return
    
    use_dual_camera = len(serial_numbers) >= 2
    
    if use_dual_camera:
        logger_.info("检测到双相机模式")
        logger_.info("相机1 (眼在手上): " + serial_numbers[0])
        logger_.info("相机2 (眼在手外): " + serial_numbers[1])
    else:
        logger_.info("单相机模式")
    
    # 初始化相机
    cameras = []
    
    for i, serial in enumerate(serial_numbers[:2] if use_dual_camera else serial_numbers[:1]):
        logger_.info(f"初始化相机 {i+1} (序列号: {serial})...")
        
        try:
            camera = RealSenseCamera(
                width=WIDTH,
                height=HEIGHT,
                fps=FPS,
                serial_number=serial
            )
            camera.start()
        except Exception as e:
            logger_.error(f"相机 {i+1} 创建失败: {e}")
            for cam in cameras:
                cam.stop()
            robot.disconnect()
            return
        
        cameras.append(camera)
        
        # 设置自动曝光 (RealSenseCamera 内部已初始化 pipeline)
        device = camera.profile.get_device()
        color_sensor = device.first_color_sensor()
        color_sensor.set_option(rs.option.enable_auto_exposure, 1)
    
    # 创建数据保存目录
    data_folder = create_folder_with_date()
    poses_file = os.path.join(data_folder, "poses.txt")
    
    logger_.info(f"\n数据将保存到: {data_folder}")
    logger_.info(f"标定板: {corner_long}x{corner_short}")
    logger_.info("\n" + "="*50)
    logger_.info("手眼标定数据采集程序 V2 (实时角点检测)")
    logger_.info("="*50)
    logger_.info("按键说明:")
    logger_.info("  's' - 采集数据 (仅角点检测成功时有效)")
    logger_.info("  'q' - 退出")
    logger_.info("  'g' - 切换井字格")
    logger_.info("  'c' - 切换角点检测显示")
    if use_dual_camera:
        logger_.info("  '1' - 只显示相机1")
        logger_.info("  '2' - 只显示相机2 (眼在手外)")
        logger_.info("  '3' - 双相机显示")
    logger_.info("="*50)
    logger_.info("\n注意: 确保每张图片的角点起始位置(绿色)一致!")
    
    # 创建窗口
    window_name = 'HandEye Calibration V2'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    if use_dual_camera:
        cv2.resizeWindow(window_name, WIDTH * 2, HEIGHT)
    else:
        cv2.resizeWindow(window_name, WIDTH, HEIGHT)
    
    count = 1
    show_grid = True
    show_corners = True
    display_mode = 3 if use_dual_camera else 2
    
    # 记录第一次保存时的起始点位置（用于一致性检查）
    first_start_position = None
    
    try:
        while True:
            # 读取相机图像
            color_images = []
            raw_images = []  # 保存原始图像用于存储
            corner_detected = [False, False]
            corners_list = [None, None]
            
            for i, camera in enumerate(cameras):
                frame_data = camera.read_frame()
                color_image = frame_data['color']
                
                if color_image is None:
                    color_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
                
                raw_images.append(color_image.copy())
                info_image = color_image.copy()
                
                # 实时角点检测
                if show_corners:
                    detected, corners, corners_subpix = detect_corners(
                        color_image, corner_long, corner_short)
                    corner_detected[i] = detected
                    corners_list[i] = corners_subpix if detected else None
                    
                    # 绘制角点（仅用于显示）
                    info_image = draw_corners_with_info(
                        info_image, corners_subpix, corner_long, corner_short, detected)
                
                # 绘制井字格
                if show_grid:
                    draw_grid(info_image)
                
                # 绘制中心十字
                h, w = info_image.shape[:2]
                cx, cy = w // 2, h // 2
                cv2.line(info_image, (cx - 20, cy), (cx + 20, cy), (0, 255, 0), 2)
                cv2.line(info_image, (cx, cy - 20), (cx, cy + 20), (0, 255, 0), 2)
                
                # 显示相机标签
                label = f"Camera {i+1}"
                if use_dual_camera:
                    label += " (Eye-in-Hand)" if i == 0 else " (Eye-to-Hand)"
                cv2.putText(info_image, label, (10, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # 显示采集计数
                cv2.putText(info_image, f"Count: {count}", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                color_images.append(info_image)
            
            # 根据显示模式组合图像
            if use_dual_camera:
                if display_mode == 1:
                    display_image = color_images[0]
                elif display_mode == 2:
                    display_image = color_images[1]
                else:
                    display_image = np.hstack(color_images)
            else:
                display_image = color_images[0]
            
            # 显示提示信息
            cv2.putText(display_image, "Press 's' to capture, 'q' to quit", 
                       (10, display_image.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(window_name, display_image)
            
            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            
            elif key == ord('s'):
                # 确定要保存的相机索引
                save_cam_idx = 1 if use_dual_camera else 0
                
                # 检查角点是否检测成功
                if not corner_detected[save_cam_idx]:
                    logger_.warning("角点未检测到，无法保存！请调整标定板位置。")
                    continue
                
                # 检查起始点一致性
                current_start = corners_list[save_cam_idx][0][0]
                if first_start_position is not None:
                    # 计算与第一次的距离差异
                    dist = np.linalg.norm(current_start - first_start_position)
                    # 如果起始点位置差异过大（比如在图像的另一侧），给出警告
                    if dist > min(WIDTH, HEIGHT) * 0.5:
                        logger_.warning(f"警告: 角点起始位置与第一张图片差异较大 (距离: {dist:.0f}px)")
                        logger_.warning("可能是标定板方向反了，建议检查！")
                

                # 采集数据
                #增加偏移量（执行器坐标）
                robot.set_tcp_offset([0.17500, 0.00000, -0.02350, 0.0, 0.0, 0.0])

                try:
                    pose = robot.get_tcp_pose()
                    success = pose is not None and len(pose) >= 6
                except Exception as e:
                    logger_.error(f"获取位姿异常: {e}")
                    success = False
                
                if success:
                    # 保存位姿
                    pose_str = ','.join([str(p) for p in pose])
                    with open(poses_file, 'a+') as f:
                        f.write(pose_str + '\n')
                    
                    # 保存原始图像（不带角点标记）
                    image_to_save = raw_images[save_cam_idx]
                    image_path = os.path.join(data_folder, f"{count}.jpg")
                    cv2.imwrite(image_path, image_to_save)
                    
                    # 记录第一次的起始点位置
                    if first_start_position is None:
                        first_start_position = current_start.copy()
                        logger_.info(f"记录参考起始点位置: ({first_start_position[0]:.0f}, {first_start_position[1]:.0f})")
                    
                    logger_.info(f"=== 采集第 {count} 次数据 ===")
                    logger_.info(f"  位姿: x={pose[0]:.4f}, y={pose[1]:.4f}, z={pose[2]:.4f}")
                    logger_.info(f"        rx={pose[3]:.4f}, ry={pose[4]:.4f}, rz={pose[5]:.4f}")
                    logger_.info(f"  图像: {image_path}")
                    logger_.info(f"  角点起始: ({current_start[0]:.0f}, {current_start[1]:.0f})")
                    
                    count += 1
                else:
                    logger_.error("获取机械臂位姿失败!")
            
            elif key == ord('g'):
                show_grid = not show_grid
                logger_.info(f"井字格: {'开启' if show_grid else '关闭'}")
            
            elif key == ord('c'):
                show_corners = not show_corners
                logger_.info(f"角点检测: {'开启' if show_corners else '关闭'}")
            
            elif key == ord('1') and use_dual_camera:
                display_mode = 1
                logger_.info("显示: 相机1 (眼在手上)")
            
            elif key == ord('2'):
                display_mode = 2
                logger_.info("显示: 相机2 (眼在手外)")
            
            elif key == ord('3') and use_dual_camera:
                display_mode = 3
                logger_.info("显示: 双相机")
    
    except KeyboardInterrupt:
        logger_.info("用户中断")
    
    finally:
        for camera in cameras:
            camera.stop()
        robot.disconnect()
        cv2.destroyAllWindows()
        
        logger_.info(f"\n采集完成! 共采集 {count - 1} 组数据")
        logger_.info(f"数据保存在: {data_folder}")
        logger_.info("\n下一步: 运行 eyeToHand.py 计算标定结果")


if __name__ == '__main__':
    main()
