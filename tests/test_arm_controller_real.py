#!/usr/bin/env python3
# coding=utf-8
"""
ArmController Real Robot Test Cases

Description:
- This script tests the core functionality of ArmController on a real robot arm.
- Includes connection, status reading, and simple motion tests.

WARNING:
- Running test_03_relative_movement will move the robot!
- Ensure there are no obstacles around the robot.
- Keep your hand on the emergency stop button.
"""

import unittest
import time
import logging

from nero_workcell.core.arm_controller import ArmController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TestArmControllerReal")


class TestArmControllerReal(unittest.TestCase):
    """ArmController real robot integration test."""

    def setUp(self):
        """Initialize controller before each test"""
        self.controller = ArmController(channel="can0", robot_type="nero")
        
    def tearDown(self):
        """Disconnect after each test"""
        if self.controller.is_connected():
            self.controller.disconnect()

    def test_01_connection(self):
        """Test: Connect and disconnect robot"""
        logger.info("=== Test 01: Connection Test ===")
        success = self.controller.connect()
        self.assertTrue(success, "Failed to connect to robot, check CAN connection or power")
        self.assertTrue(self.controller.is_connected(), "Connection status should be True")
        
        # Disconnect
        self.controller.disconnect()
        self.assertFalse(self.controller.is_connected(), "Connection status should be False after disconnect")
        logger.info("Connection test passed")

    def test_02_read_poses(self):
        """Test: Read flange and TCP poses"""
        logger.info("=== Test 02: Pose Reading Test ===")
        if not self.controller.connect():
            self.skipTest("Cannot connect to robot, skipping test")
            
        # 1. Get flange pose
        flange_pose = self.controller.get_flange_pose()
        logger.info(f"Flange pose: {flange_pose}")
        self.assertIsNotNone(flange_pose, "Flange pose should not be None")
        self.assertEqual(len(flange_pose), 6, "Pose data length should be 6 [x,y,z,r,p,y]")
        
        # 2. Get TCP pose
        tcp_pose = self.controller.get_tcp_pose()
        logger.info(f"TCP pose: {tcp_pose}")
        self.assertIsNotNone(tcp_pose, "TCP pose should not be None")
        self.assertEqual(len(tcp_pose), 6, "Pose data length should be 6")
        
        logger.info("Pose reading test passed")

    def test_03_relative_movement(self):
        """Test: Relative movement (Z-axis +1cm)"""
        logger.info("=== Test 03: Relative Movement Test (Z-axis +1cm) ===")
        if not self.controller.connect():
            self.skipTest("Cannot connect to robot, skipping test")
            
        # Get initial pose
        start_pose = self.controller.get_tcp_pose()
        self.assertIsNotNone(start_pose)
        initial_z = start_pose[2]
        
        move_dist = 0.05  # 1cm
        logger.info(f"Current Z: {initial_z:.4f}, moving up {move_dist}m")
        
        # Execute move
        self.controller.move_relative(dz=move_dist)
        time.sleep(2.0)  # Wait for motion
        
        # Check new position
        mid_pose = self.controller.get_tcp_pose()
        self.assertIsNotNone(mid_pose)
        mid_z = mid_pose[2]
        logger.info(f"Z after move: {mid_z:.4f}")
        
        # Verify Z change (allow 2mm error)
        self.assertAlmostEqual(mid_z, initial_z + move_dist, delta=0.002, 
                             msg="Z-axis movement distance mismatch")
        
        # Return to original position
        logger.info("Returning to original position...")
        self.controller.move_relative(dz=-move_dist)
        time.sleep(2.0)
        
        end_pose = self.controller.get_tcp_pose()
        end_z = end_pose[2]
        logger.info(f"Z after return: {end_z:.4f}")
        
        self.assertAlmostEqual(end_z, initial_z, delta=0.002,
                             msg="Failed to return to original position accurately")
        logger.info("Relative movement test passed")

    def test_04_gripper_control(self):
        """Test: Gripper control (Open/Close)"""
        logger.info("=== Test 04: Gripper Control Test ===")
        if not self.controller.connect():
            self.skipTest("Cannot connect to robot, skipping test")
            
        if self.controller.end_effector is None:
            logger.warning("Gripper not initialized, skipping this test")
            return

        # 1. Open gripper
        logger.info("Opening gripper (5cm)...")
        self.controller.move_gripper(width=0.05, force=1.0)
        time.sleep(2.0)
        
        # 2. Close gripper
        logger.info("Closing gripper...")
        self.controller.move_gripper(width=0.0, force=1.0)
        time.sleep(2.0)
        logger.info("Gripper test passed")

if __name__ == '__main__':
    unittest.main()
