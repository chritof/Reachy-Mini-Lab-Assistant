from reachy_mini import ReachyMini
from reachy_assistant.robot.motion.gestures import happy

with ReachyMini() as mini:
    happy(mini)