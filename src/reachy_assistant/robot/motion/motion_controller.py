"""
Enkel tilstandskontroll for Reachy-bevegelser brukt av samtalepipelinen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from reachy_assistant.robot.motion import gestures


@dataclass
class ReachyMotionController:
    mini: object
    current_state: str = field(init=False, default="")

    def sleeping(self) -> None:
        if self.current_state != "sleeping":
            gestures.sleeping(self.mini)
            self.current_state = "sleeping"

    def waking(self) -> None:
        gestures.waking(self.mini)
        self.current_state = "idle"

    def idle(self) -> None:
        if self.current_state != "idle":
            gestures.neutral(self.mini)
            self.current_state = "idle"

    def listening(self) -> None:
        if self.current_state != "listening":
            gestures.listening(self.mini)
            self.current_state = "listening"

    def thinking(self) -> None:
        if self.current_state != "thinking":
            gestures.thinking(self.mini)
            self.current_state = "thinking"

    def speaking(self) -> None:
        if self.current_state != "speaking":
            gestures.speaking(self.mini)
            self.current_state = "speaking"

    def acknowledge(self) -> None:
        gestures.nod(self.mini)

    def celebrate(self) -> None:
        gestures.happy(self.mini)
