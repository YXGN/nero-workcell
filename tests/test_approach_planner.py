#!/usr/bin/env python3
# coding=utf-8

import importlib.util
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "nero_workcell" / "core" / "approach_planner.py"
)
SPEC = importlib.util.spec_from_file_location("approach_planner", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
approach_planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(approach_planner)

ApproachPlanner = approach_planner.ApproachPlanner
OffsetComponents = approach_planner.OffsetComponents


class TestApproachPlanner(unittest.TestCase):
    def test_decompose_offset_splits_axial_and_lateral_components(self):
        planner = ApproachPlanner(approach_direction=(0.0, 0.0, -1.0))

        components = planner.decompose_offset(
            tcp_position=np.array([1.0, 2.0, 3.0]),
            target_position=np.array([4.0, 8.0, 1.0]),
        )

        np.testing.assert_allclose(components.axial_offset, np.array([0.0, 0.0, -2.0]))
        np.testing.assert_allclose(components.lateral_offset, np.array([3.0, 6.0, 0.0]))

    def test_decompose_offset_returns_offset_components(self):
        planner = ApproachPlanner(approach_direction=(0.0, 0.0, -1.0))

        components = planner.decompose_offset(
            tcp_position=np.array([0.0, 0.0, 0.0]),
            target_position=np.array([1.0, 2.0, -3.0]),
        )

        self.assertIsInstance(components, OffsetComponents)
        np.testing.assert_allclose(components.axial_offset, np.array([0.0, 0.0, -3.0]))
        np.testing.assert_allclose(components.lateral_offset, np.array([1.0, 2.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
