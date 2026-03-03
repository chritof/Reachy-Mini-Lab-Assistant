"""
State-driven conversation pipeline for Reachy Mini.
"""

from reachy_assistant.robot.pipeline.states import ConversationState

class ConversationPipeline:

    def __init__(self, stt, rag, tts, motion):
        self.stt = stt
        self.rag = rag
        self.tts = tts
        self.motion = motion

        self.state = ConversationState.IDLE

    def run_once(self):
        if self.state == ConversationState.IDLE:
            self._handle_idle()

        elif self.state == ConversationState.LISTENING:
            self._handle_listening()

        elif self.state == ConversationState.THINKING:
            self._handle_thinking()

        elif self.state == ConversationState.SPEAKING:
            self._handle_speaking()

    # ---- state handlers ----

    def _handle_idle(self):
        print("Idle..")
        self.state = ConversationState.LISTENING

    def _handle_listening(self):
        print("Listening..")
        self.state = ConversationState.THINKING

    def _handle_thinking(self):
        print("Thinking..")
        self.state = ConversationState.SPEAKING

    def _handle_speaking(self):
        print("Speaking..")
        # TODO: call TTS + motion
        self.state = ConversationState.IDLE