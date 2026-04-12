from reachy_assistant.robot.motion.motion_controller import ReachyMotionController


class FakeMini:
    pass


def test_sleep_state_blocks_late_motion_changes(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        "reachy_assistant.robot.motion.motion_controller.gestures.sleeping",
        lambda mini: calls.append("sleeping"),
    )
    monkeypatch.setattr(
        "reachy_assistant.robot.motion.motion_controller.gestures.neutral",
        lambda mini: calls.append("idle"),
    )
    monkeypatch.setattr(
        "reachy_assistant.robot.motion.motion_controller.gestures.thinking",
        lambda mini: calls.append("thinking"),
    )

    controller = ReachyMotionController(mini=FakeMini())

    controller.sleeping()
    controller.idle()
    controller.thinking()

    assert calls == ["sleeping"]


def test_waking_releases_sleep_lock(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr(
        "reachy_assistant.robot.motion.motion_controller.gestures.sleeping",
        lambda mini: calls.append("sleeping"),
    )
    monkeypatch.setattr(
        "reachy_assistant.robot.motion.motion_controller.gestures.waking",
        lambda mini: calls.append("waking"),
    )
    monkeypatch.setattr(
        "reachy_assistant.robot.motion.motion_controller.gestures.neutral",
        lambda mini: calls.append("idle"),
    )
    monkeypatch.setattr(
        "reachy_assistant.robot.motion.motion_controller.gestures.listening",
        lambda mini: calls.append("listening"),
    )

    controller = ReachyMotionController(mini=FakeMini())

    controller.sleeping()
    controller.waking()
    controller.listening()

    assert calls == ["sleeping", "waking", "listening"]
