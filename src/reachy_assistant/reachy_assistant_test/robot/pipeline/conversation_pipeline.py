from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import traceback
import time

from reachy_assistant_test.audio.preprocessing import load_wav_as_float32_mono_16k
from reachy_assistant_test.audio.vad import get_vad_segments, extract_voiced_audio
from reachy_assistant_test.robot.pipeline.states import ConversationState

TARGET_SR = 16000


@dataclass
class PipelineContext:
    wav_path: Optional[Path] = None
    transcribed_text: str = ""
    answer_text: str = ""
    synthesized_wav: bytes = b""
    last_error: Optional[str] = None


class ConversationPipeline:
    def __init__(self, recorder, stt, rag, tts, motion, listen_duration_sec: float = 6.0):
        self.recorder = recorder
        self.stt = stt
        self.rag = rag
        self.tts = tts
        self.motion = motion

        self.listen_duration_sec = listen_duration_sec
        self.state = ConversationState.IDLE
        self.ctx = PipelineContext()

    def run_once(self):
        try:
            if self.state == ConversationState.IDLE:
                self._handle_idle()
            elif self.state == ConversationState.LISTENING:
                self._handle_listening()
            elif self.state == ConversationState.THINKING:
                self._handle_thinking()
            elif self.state == ConversationState.SPEAKING:
                self._handle_speaking()
            elif self.state == ConversationState.ERROR:
                self._handle_error()
        except Exception as e:
            self.ctx.last_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            self.state = ConversationState.ERROR

    def reset_cycle(self):
        self.ctx.wav_path = None
        self.ctx.transcribed_text = ""
        self.ctx.answer_text = ""
        self.ctx.synthesized_wav = b""
        self.ctx.last_error = None

    # ---------------- states ----------------

    def _handle_idle(self):
        print("[STATE] IDLE")
        self.motion.idle()
        time.sleep(0.3)
        self.state = ConversationState.LISTENING

    def _handle_listening(self):
        print("[STATE] LISTENING")
        self.motion.listening_pose()

        wav_path = Path("data/audio/recordings/reachy_input.wav")
        wav_path, orig_sr = self.recorder.record_audio_wav(wav_path, self.listen_duration_sec)
        self.ctx.wav_path = wav_path

        audio_16k = load_wav_as_float32_mono_16k(wav_path, orig_sr)
        segments = get_vad_segments(audio_16k)
        voiced = extract_voiced_audio(audio_16k, segments)

        if voiced.size < TARGET_SR * 0.2:
            print("[LISTENING] Ingen tale funnet.")
            self.ctx.transcribed_text = ""
            self.state = ConversationState.IDLE
            return

        text = self.stt.transcribe_audio_array(voiced).strip()
        self.ctx.transcribed_text = text
        print(f"[USER] {text}")

        if not text:
            self.state = ConversationState.IDLE
            return

        self.state = ConversationState.THINKING

    def _handle_thinking(self):
        print("[STATE] THINKING")
        self.motion.thinking_pose()

        question = self.ctx.transcribed_text.strip()
        if not question:
            self.state = ConversationState.IDLE
            return

        rag_answer = self.rag.ask(question)
        answer = rag_answer.answer.strip()
        self.ctx.answer_text = answer

        print(f"[ASSISTANT] {answer}")
        self.ctx.synthesized_wav = self.tts.synthesize(answer)
        self.state = ConversationState.SPEAKING

    def _handle_speaking(self):
        print("[STATE] SPEAKING")
        self.motion.speaking_pose()

        if self.ctx.synthesized_wav:
            self.motion.play_wav_bytes_blocking(self.ctx.synthesized_wav)
        else:
            print("[SPEAKING] Ingen lyd generert.")

        self.reset_cycle()
        self.state = ConversationState.IDLE

    def _handle_error(self):
        print("[STATE] ERROR")
        print(f"[ERROR] {self.ctx.last_error}")
        self.reset_cycle()
        self.state = ConversationState.IDLE