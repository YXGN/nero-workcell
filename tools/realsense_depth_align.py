#!/usr/bin/env python3
# coding=utf-8
"""
测试深度对齐 - 对比对齐前后的深度数据
"""

import cv2
import numpy as np
import pyrealsense2 as rs

print("=== 深度对齐测试 ===\n")

# 创建管道
pipeline = rs.pipeline()
config = rs.config()

# 启用深度流和彩色流
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 启动
print("启动相机...")
profile = pipeline.start(config)

# 创建对齐对象
align = rs.align(rs.stream.color)

# 获取深度比例
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print(f"深度比例: {depth_scale}")

print("\n按 'q' 退出")

def colorize_depth(depth_image):
    """将深度图转换为彩色图"""
    valid_mask = depth_image > 0
    if np.any(valid_mask):
        min_depth = np.min(depth_image[valid_mask])
        max_depth = np.percentile(depth_image[valid_mask], 95)
        depth_normalized = np.zeros_like(depth_image, dtype=np.uint8)
        depth_normalized[valid_mask] = np.clip(
            255 * (depth_image[valid_mask] - min_depth) / (max_depth - min_depth + 1),
            0, 255
        ).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_colormap[~valid_mask] = [0, 0, 0]
    else:
        depth_colormap = np.zeros((depth_image.shape[0], depth_image.shape[1], 3), dtype=np.uint8)
    return depth_colormap

try:
    while True:
        # 等待帧
        frames = pipeline.wait_for_frames()
        
        # 获取原始深度帧（未对齐）
        depth_frame_raw = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame_raw or not color_frame:
            continue
        
        # 对齐
        aligned_frames = align.process(frames)
        depth_frame_aligned = aligned_frames.get_depth_frame()
        
        # 转换为numpy
        depth_raw = np.asanyarray(depth_frame_raw.get_data())
        depth_aligned = np.asanyarray(depth_frame_aligned.get_data()) if depth_frame_aligned else np.zeros_like(depth_raw)
        color_image = np.asanyarray(color_frame.get_data())
        
        # 统计
        valid_raw = np.sum(depth_raw > 0)
        valid_aligned = np.sum(depth_aligned > 0)
        
        # 可视化
        depth_raw_vis = colorize_depth(depth_raw)
        depth_aligned_vis = colorize_depth(depth_aligned)
        
        # 添加文字
        cv2.putText(depth_raw_vis, f"RAW: {100*valid_raw/depth_raw.size:.1f}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(depth_aligned_vis, f"ALIGNED: {100*valid_aligned/depth_aligned.size:.1f}%", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # 显示
        cv2.imshow('Color', color_image)
        cv2.imshow('Depth RAW', depth_raw_vis)
        cv2.imshow('Depth ALIGNED', depth_aligned_vis)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("已停止")
