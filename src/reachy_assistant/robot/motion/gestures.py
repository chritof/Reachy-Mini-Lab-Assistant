"""
Enkle bevegelser for Reachy Mini.
"""

import numpy as np
from reachy_mini import ReachyMini
from scipy.spatial.transform import Rotation as R


def _head_pose(yaw: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R.from_euler("zy", [yaw, pitch]).as_matrix()
    return pose


def neutral(mini: ReachyMini) -> None:
    mini.goto_target(head=_head_pose(yaw=0.0, pitch=0.0), antennas=[0.0, 0.0], duration=0.4)


def listening(mini: ReachyMini) -> None:
    mini.goto_target(
        head=_head_pose(yaw=0.0, pitch=-0.08),
        antennas=[0.15, -0.15],
        duration=0.35,
    )


def thinking(mini: ReachyMini) -> None:
    mini.goto_target(
        head=_head_pose(yaw=0.18, pitch=0.1),
        antennas=[-0.1, 0.1],
        duration=0.35,
    )


def speaking(mini: ReachyMini) -> None:
    mini.goto_target(
        head=_head_pose(yaw=-0.06, pitch=0.0),
        antennas=[0.25, -0.25],
        duration=0.25,
    )


def happy(mini: ReachyMini) -> None:
    mini.goto_target(antennas=[0.5, -0.5], duration=0.3)
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.3)
    mini.goto_target(antennas=[0.0, 0.0], duration=0.3)


def nod(mini: ReachyMini) -> None:
    mini.goto_target(head=_head_pose(pitch=0.2), duration=0.3)
    mini.goto_target(head=_head_pose(pitch=-0.2), duration=0.3)
    mini.goto_target(head=_head_pose(pitch=0.0), duration=0.3)
