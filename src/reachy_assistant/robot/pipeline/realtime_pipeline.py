"""
Setter sammen Reachy-spesifikke realtime-komponenter til en kjørbar pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.realtime.engine import RealtimeConversationEngine
from reachy_assistant.realtime.openwakeword_source import OpenWakewordAudioSource
from reachy_assistant.realtime.pipeline import RealtimeConversationPipeline
from reachy_assistant.realtime.reachy_audio import ReachyRealtimeAudioSink, ReachyRealtimeAudioSource
from reachy_assistant.robot.motion.motion_controller import ReachyMotionController


@dataclass
class ReachyRealtimePipeline:
    mini: object
    config: RealtimeConfig
    wakeword_model_path: str | None = None
    sleepword_model_path: str | None = None

    def build(self) -> RealtimeConversationPipeline:
        motion = ReachyMotionController(mini=self.mini)
        source = ReachyRealtimeAudioSource(mini=self.mini)
        if self.wakeword_model_path:
            source = OpenWakewordAudioSource(
                base_source=source,
                model_path=str(Path(self.wakeword_model_path)),
                sleep_model_path=(
                    str(Path(self.sleepword_model_path)) if self.sleepword_model_path else None
                ),
                inactivity_timeout_sec=self.config.wake_phrase_timeout_sec,
                on_wake=motion.waking,
                on_sleep=motion.sleeping,
            )
        sink = ReachyRealtimeAudioSink(mini=self.mini)
        engine = RealtimeConversationEngine(
            config=self.config,
            source=source,
            sink=sink,
            motion=motion,
        )
        return RealtimeConversationPipeline(engine=engine)
