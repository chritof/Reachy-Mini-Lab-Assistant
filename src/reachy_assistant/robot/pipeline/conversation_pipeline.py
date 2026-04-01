"""
State-driven conversation pipeline for Reachy Mini.
"""

from __future__ import annotations

from pathlib import Path
from time import sleep

from reachy_assistant.robot.pipeline.states import ConversationState


class ConversationPipeline:
    def __init__(
        self,
        stt,
        rag,
        tts,
        recorder,
        player,
        motion,
        *,
        intro_text: str | None = "Hei. Jeg er klar. Hva kan jeg hjelpe deg med?",
        no_answer_text: str = "Beklager, jeg fant ikke et godt svar.",
        error_text: str = "Beklager, noe gikk galt. Vi kan prove igjen.",
    ):
        self.stt = stt
        self.rag = rag
        self.tts = tts
        self.recorder = recorder
        self.player = player
        self.motion = motion
        self.intro_text = intro_text
        self.no_answer_text = no_answer_text
        self.error_text = error_text

        self.state = ConversationState.IDLE
        self.has_greeted = False

        self.input_wav_path: Path | None = None
        self.user_text: str | None = None
        self.answer_text: str | None = None
        self.last_tts_wav: bytes | None = None

    def run_once(self) -> None:
        try:
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
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[ERROR] Turn failed: {exc}")
            self._motion_call("idle")
            self._safe_speak(self.error_text)
            self._reset_turn()
            self.state = ConversationState.IDLE

    def run_forever(self, poll_interval: float = 0.1) -> None:
        while True:
            self.run_once()
            sleep(poll_interval)

    def _handle_idle(self) -> None:
        print("[STATE] IDLE")
        self._motion_call("idle")

        if not self.has_greeted and self.intro_text:
            self._safe_speak(self.intro_text)
            self.has_greeted = True

        self.state = ConversationState.LISTENING

    def _handle_listening(self) -> None:
        print("[STATE] LISTENING")
        self._motion_call("listening")

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

    def _handle_thinking(self) -> None:
        print("[STATE] THINKING")
        self._motion_call("thinking")

        rag_result = self.rag.ask(self.user_text)

        if hasattr(rag_result, "answer"):
            self.answer_text = (rag_result.answer or "").strip()
        else:
            self.answer_text = str(rag_result).strip()

        print("Assistant:", self.answer_text)

        if not self.answer_text:
            self.answer_text = self.no_answer_text

        self.state = ConversationState.SPEAKING

    def _handle_speaking(self) -> None:
        print("[STATE] SPEAKING")
        self._motion_call("speaking")

        self.last_tts_wav = self.tts.synthesize(self.answer_text)
        if self.last_tts_wav:
            self.player.play_wav(self.last_tts_wav)

        self._reset_turn()
        self.state = ConversationState.IDLE

    def _reset_turn(self) -> None:
        self.input_wav_path = None
        self.user_text = None
        self.answer_text = None
        self.last_tts_wav = None

    def _motion_call(self, name: str) -> None:
        if self.motion is None or not hasattr(self.motion, name):
            return
        getattr(self.motion, name)()

    def _safe_speak(self, text: str | None) -> None:
        text = (text or "").strip()
        if not text:
            return
        try:
            self._motion_call("speaking")
            wav_bytes = self.tts.synthesize(text)
            if wav_bytes:
                self.player.play_wav(wav_bytes)
        finally:
            self._motion_call("idle")
