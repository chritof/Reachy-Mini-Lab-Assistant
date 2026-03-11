"""
State-driven conversation pipeline for Reachy Mini.
"""

from reachy_assistant.robot.pipeline.states import ConversationState
from pathlib import Path

class ConversationPipeline:

    def __init__(self, stt, rag, tts, recorder, player, motion):
        self.stt = stt
        self.rag = rag
        self.tts = tts
        self.recorder = recorder
        self.player = player
        self.motion = motion

        self.state = ConversationState.IDLE

        self.input_wav_path: Path | None = None
        self.user_text: str | None = None
        self.answer_text: str | None = None
        self.last_tts_wav: bytes | None = None


    def run_once(self):
        if self.state == ConversationState.IDLE:
            self._handle_idle()

        elif self.state == ConversationState.LISTENING:
            self._handle_listening()

        elif self.state == ConversationState.THINKING:
            self._handle_thinking()

        elif self.state == ConversationState.SPEAKING:
            self._handle_speaking()

        else:
            raise ValueError(f"Unknown state: {self.state}")



    # ---- state handlers ----

    def _handle_idle(self):
        print("[STATE] IDLE")

        if self.motion and hasattr(self.motion, "idle"):
            self.motion.idle()

        self.state = ConversationState.LISTENING

    def _handle_listening(self) -> None:
        print("[STATE] LISTENING")

        if self.motion and hasattr(self.motion, "listening"):
            self.motion.listening()

        self.input_wav_path = self.recorder.record_to_file()
        result = self.stt.transcribe_wav(self.input_wav_path)

        self.user_text = (result or "").strip()
        print("User:", self.user_text)

        if not self.user_text:
            print("[INFO] No speech detected.")
            self._reset_turn()
            self.state = ConversationState.IDLE
            return

        self.state = ConversationState.THINKING

    def _handle_thinking(self):
        print("[STATE] THINKING")

        if self.motion and hasattr(self.motion, "thinking"):
            self.motion.thinking()

        rag_result = self.rag.ask(self.user_text)

        if hasattr(rag_result, "answer"):
            self.answer_text = rag_result.answer
        else:
            self.answer_text = str(rag_result)

        print("Assistant:", self.answer_text)

        if not self.answer_text or not self.answer_text:
            self.answer_text = "Sorry, I could not find an answer."

        self.state = ConversationState.SPEAKING

    def _handle_speaking(self):
        print("[STATE] SPEAKING")

        if self.motion and hasattr(self.motion, "speaking"):
            self.motion.speaking()

        self.last_tts_wav = self.tts.synthesize(self.answer_text)
        self.player.play_wav(self.last_tts_wav)

        self._reset_turn()
        self.state = ConversationState.IDLE


    def _reset_turn(self):
        self.input_wav_path = None
        self.user_text = None
        self.answer_text = None
        self.last_tts_wav = None