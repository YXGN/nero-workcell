#!/usr/bin/env python3
# coding=utf-8

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

@dataclass(frozen=True)
class TargetObject:
    """Single tracked/observed target with an associated coordinate frame."""

    name: str
    class_id: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    position: np.ndarray
    conf: float
    frame: Literal["camera", "base"]
