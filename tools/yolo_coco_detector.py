#!/usr/bin/env python3
"""
Test YOLO for detecting all objects.
Used for debugging and checking what the camera can detect.
"""
import argparse
import logging
import cv2
import pyrealsense2 as rs

from nero_workcell.core import RealSenseCamera
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="YOLO COCO Detector")
    parser.add_argument("--model", type=str, default="yolo11x.pt", help="Path to YOLO model")
    parser.add_argument("--serial", type=str, default=None, help="RealSense camera serial number")
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger(__name__)

    # Determine camera serial
    camera_serial = args.serial
    if camera_serial is None:
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            logger.error("No RealSense devices found")
            return
        camera_serial = devices[0].get_info(rs.camera_info.serial_number)
        logger.info(f"Auto-detected camera: {camera_serial}")

    logger.info(f"Initializing camera ({camera_serial})...")
    camera = RealSenseCamera(width=640, height=480, fps=30, serial_number=camera_serial)
    if not camera.start():
        logger.error("Failed to start camera")
        return
    
    logger.info(f"Loading YOLO model: {args.model}")
    model = YOLO(args.model)
    
    logger.info("Starting all-object detection...")
    logger.info("Press 'q' to quit")
    
    while True:
        frame_data = camera.read_frame()
        color_image = frame_data['color']
        
        if color_image is None:
            continue
        
        # Detect all objects
        results = model(color_image, verbose=False)
        
        # Render results
        vis_image = color_image.copy()
        detected_objects = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if conf < 0.3:  # Low-confidence threshold
                    continue
                
                # Get class name
                class_name = model.names[cls_id]
                
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw box
                color = (0, 255, 0)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label text
                text = f"{class_name} ({conf:.2f})"
                cv2.putText(vis_image, text, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detected_objects.append((class_name, conf, cls_id))
        
        # Show detected object list
        if detected_objects:
            objects_str = ", ".join([f"{name}(ID:{oid}, {conf:.2f})" for name, conf, oid in detected_objects])
            logger.info(f"Detected {len(detected_objects)} objects: {objects_str}")
        
        cv2.imshow('YOLO All Objects Detection', vis_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    camera.stop()
    cv2.destroyAllWindows()
    logger.info("Test finished")

if __name__ == '__main__':
    main()
