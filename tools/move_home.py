#!/usr/bin/env python3
# coding=utf-8
"""
Tool to move the robot to the home position.

Usage:
    python -m nero_workcell.tools.move_home
"""

import logging
import argparse
from nero_workcell.core import ArmController

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Nero Workcell - Move to Home")
    parser.add_argument("--channel", type=str, default="can0", help="CAN channel")
    parser.add_argument("--speed", type=int, default=20, help="Movement speed percent")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%H:%M:%S'
    )

    controller = ArmController(channel=args.channel)
    if not controller.connect(speed_percent=args.speed):
        logger.error("Failed to connect to robot")
        return

    try:
        controller.move_to_home(blocking=True)
        logger.info("Robot moved to home position")
    except Exception as e:
        logger.error(f"Failed to move home: {e}")
    finally:
        controller.disconnect()

if __name__ == "__main__":
    main()
