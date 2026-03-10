"""
Enkle bevegelser for Reachy Mini.
"""

from reachy_mini import ReachyMini
import time


def happy(mini: ReachyMini) -> None:
    mini.goto_target(antennas=[0.5, -0.5], duration=0.3)
    mini.goto_target(antennas=[-0.5, 0.5], duration=0.3)
    mini.goto_target(antennas=[0.0, 0.0], duration=0.3)


def nod(mini: ReachyMini) -> None:
    mini.goto_target(head_pitch=0.2, duration=0.3)
    mini.goto_target(head_pitch=-0.2, duration=0.3)
    mini.goto_target(head_pitch=0.0, duration=0.3)