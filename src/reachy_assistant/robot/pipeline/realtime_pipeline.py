from __future__ import annotations

from dataclasses import dataclass

from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.realtime.engine import RealtimeConversationEngine
from reachy_assistant.realtime.pipeline import RealtimeConversationPipeline
from reachy_assistant.realtime.reachy_audio import ReachyRealtimeAudioSink, ReachyRealtimeAudioSource
from reachy_assistant.robot.motion.motion_controller import ReachyMotionController


@dataclass
class ReachyRealtimePipeline:
    mini: object
    config: RealtimeConfig

    def build(self) -> RealtimeConversationPipeline:
        source = ReachyRealtimeAudioSource(mini=self.mini)
        sink = ReachyRealtimeAudioSink(mini=self.mini)
        motion = ReachyMotionController(mini=self.mini)
        engine = RealtimeConversationEngine(
            config=self.config,
            source=source,
            sink=sink,
            motion=motion,
        )
        return RealtimeConversationPipeline(engine=engine)
