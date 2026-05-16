from reachy_mini import ReachyMini

from reachy_assistant.robot.motion.gestures import happy


def main() -> None:
    with ReachyMini() as mini:
        happy(mini)


if __name__ == "__main__":
    main()
