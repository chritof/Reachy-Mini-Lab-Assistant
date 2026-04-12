"""
Kjernelogikk for realtime-samtalen: lyd inn/ut, events, tool-calls og tilstand.
"""

from __future__ import annotations

import asyncio
import json
import time
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from websockets.exceptions import ConnectionClosedError

from reachy_assistant.realtime.audio import REALTIME_SAMPLE_RATE, from_base64_pcm16, to_base64_pcm16
from reachy_assistant.realtime.config import RealtimeConfig
from reachy_assistant.realtime.io_base import (
    RealtimeAudioSink,
    RealtimeAudioSource,
    RealtimeSessionRestart,
)
from reachy_assistant.realtime.rag_tool import OpenAIRagRealtimeTool

if TYPE_CHECKING:
    from reachy_assistant.realtime.client import OpenAIRealtimeClient


@dataclass
class RealtimeConversationEngine:
    config: RealtimeConfig
    source: RealtimeAudioSource
    sink: RealtimeAudioSink
    client: "OpenAIRealtimeClient | None" = None
    motion: object | None = None
    rag_tool: OpenAIRagRealtimeTool | None = None
    last_user_transcript: str = ""
    last_assistant_transcript: str = ""
    _stop_event: asyncio.Event = field(init=False, default_factory=asyncio.Event)
    _active_response_id: str | None = field(init=False, default=None)
    _active_item_id: str | None = field(init=False, default=None)
    _active_content_index: int | None = field(init=False, default=None)
    _active_audio_end_ms: int = field(init=False, default=0)
    _interrupting_response_id: str | None = field(init=False, default=None)
    _response_create_pending: bool = field(init=False, default=False)
    _tool_calls: dict[str, str] = field(init=False, default_factory=dict)
    _suppress_input_until: float = field(init=False, default=0.0)
    _wake_active_until: float = field(init=False, default=0.0)
    _wake_waiting_logged: bool = field(init=False, default=False)
    _sleep_after_response: bool = field(init=False, default=False)
    _skip_next_response_create: bool = field(init=False, default=False)
    reconnect_delay_sec: float = 2.0

    def __post_init__(self) -> None:
        if self.client is None:
            from reachy_assistant.realtime.client import OpenAIRealtimeClient

            self.client = OpenAIRealtimeClient(self.config)
        if self.rag_tool is None:
            self.rag_tool = OpenAIRagRealtimeTool.from_env(self.config.api_key)
        if not self.config.wake_phrase_enabled:
            self._wake_active_until = float("inf")

    async def run(self) -> None:
        print("[realtime] Opening audio devices...")
        await self.source.open()
        await self.sink.open(self.config.output_sample_rate)
        starts_asleep = bool(getattr(self.source, "starts_asleep", False))
        self._motion_call("sleeping" if starts_asleep or self.config.wake_phrase_enabled else "idle")

        try:
            while not self._stop_event.is_set():
                try:
                    self._reset_conversation_state()
                    await self._run_session()
                    break
                except RealtimeSessionRestart as exc:
                    if self._stop_event.is_set():
                        break
                    print(f"[realtime] {exc}")
                    self._reset_conversation_state()
                    self._reset_source_state()
                    starts_asleep = bool(getattr(self.source, "starts_asleep", False))
                    self._motion_call("sleeping" if starts_asleep or self.config.wake_phrase_enabled else "idle")
                    continue
                except (ConnectionClosedError, TimeoutError) as exc:
                    if self._stop_event.is_set():
                        break
                    print(
                        "[realtime] Connection lost. "
                        f"Retrying in {self.reconnect_delay_sec:.1f}s: {exc}"
                    )
                    self._clear_active_response()
                    self._reset_source_state()
                    starts_asleep = bool(getattr(self.source, "starts_asleep", False))
                    self._motion_call("sleeping" if starts_asleep or self.config.wake_phrase_enabled else "idle")
                    await asyncio.sleep(self.reconnect_delay_sec)
        finally:
            self._stop_event.set()
            await self.source.close()
            await self.sink.close()

    def stop(self) -> None:
        self._stop_event.set()

    async def _send_audio(self, connection) -> None:
        async for chunk in self.source.chunks():
            if self._stop_event.is_set():
                break
            if self._should_suppress_input():
                continue
            payload = to_base64_pcm16(chunk, self.source.sample_rate, REALTIME_SAMPLE_RATE)
            await connection.input_audio_buffer.append(audio=payload)

    async def _run_session(self) -> None:
        print("[realtime] Connecting to OpenAI Realtime API...")
        async with self.client.connect() as connection:
            await connection.session.update(
                session={
                    "modalities": ["text", "audio"],
                    "instructions": (
                        f"{self.config.instructions} "
                        "When the conversation is clearly over, finish briefly and call the go_to_sleep tool."
                    ),
                    "voice": self.config.voice,
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {
                        "model": self.config.transcription_model,
                        "language": self.config.language,
                    },
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": self.config.vad_threshold,
                        "silence_duration_ms": self.config.vad_silence_ms,
                        "prefix_padding_ms": self.config.vad_prefix_padding_ms,
                        "create_response": (
                            self.config.vad_create_response
                            if self.config.allow_interruptions
                            else False
                        ),
                        "interrupt_response": self.config.allow_interruptions,
                    },
                    "temperature": self.config.temperature,
                    "max_response_output_tokens": self.config.max_output_tokens,
                    "tools": self._tool_definitions(),
                    "tool_choice": "auto",
                }
            )
            print("[realtime] Session ready. Speak into the microphone. Press Ctrl+C to stop.")

            send_task = asyncio.create_task(self._send_audio(connection))
            receive_task = asyncio.create_task(self._receive_events(connection))
            wake_task = asyncio.create_task(self._monitor_wake_timeout())

            done, pending = await asyncio.wait(
                {send_task, receive_task, wake_task},
                return_when=asyncio.FIRST_EXCEPTION,
            )

            for task in pending:
                task.cancel()
            for task in done:
                task.result()

    async def _receive_events(self, connection) -> None:
        async for event in connection:
            event_type = getattr(event, "type", "")

            if event_type == "response.audio.delta":
                if not self._wake_is_active():
                    await self._interrupt_assistant_response(connection)
                    continue
                self._motion_call("speaking")
                self._push_input_suppression()
                self._active_response_id = getattr(event, "response_id", None)
                self._active_item_id = getattr(event, "item_id", None)
                self._active_content_index = getattr(event, "content_index", None)
                audio, sample_rate = from_base64_pcm16(event.delta, REALTIME_SAMPLE_RATE)
                await self.sink.write(audio, sample_rate)
                self._active_audio_end_ms += int(round(len(audio) * 1000 / sample_rate))

            elif event_type == "response.audio.done":
                self._push_input_suppression()
                self._clear_active_response()

            elif event_type == "response.created":
                response = getattr(event, "response", None)
                self._active_response_id = getattr(response, "id", None)
                self._response_create_pending = False
                if not self._wake_is_active():
                    await self._interrupt_assistant_response(connection)
                    continue

            elif event_type == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", "") == "function_call":
                    call_id = getattr(item, "call_id", None)
                    name = getattr(item, "name", None)
                    if call_id and name:
                        self._tool_calls[call_id] = name

            elif event_type == "response.audio_transcript.delta":
                self.last_assistant_transcript += event.delta

            elif event_type == "response.audio_transcript.done":
                transcript = getattr(event, "transcript", "").strip()
                if transcript:
                    self.last_assistant_transcript = transcript
                    print(f"Assistant: {transcript}")
                self._push_input_suppression()
                if not self._should_hold_sleep_pose():
                    self._motion_call("idle")

            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = getattr(event, "transcript", "").strip()
                if transcript:
                    self.last_user_transcript = transcript
                    print(f"User: {transcript}")
                    wake_matched, stripped = self._handle_wake_phrase(transcript)
                    if wake_matched:
                        self._motion_call("acknowledge")
                        await self._request_response_if_idle(connection)
                    elif self._wake_is_active() and self._is_sleep_phrase(transcript):
                        print("[wakeword] Sleep phrase detected from transcript. Returning to sleep mode.")
                        self._skip_next_response_create = True
                        await self._enter_sleep_mode(connection)
                    elif self._wake_is_active():
                        self._extend_wake_window()
                        self._motion_call("acknowledge")

            elif event_type == "input_audio_buffer.speech_started":
                print("[realtime] Speech detected.")
                if self.config.allow_interruptions:
                    await self._interrupt_assistant_response(connection)
                if self._wake_is_active():
                    self._motion_call("listening")
                elif not self._wake_waiting_logged:
                    print("[realtime] Waiting for wake phrase.")
                    self._wake_waiting_logged = True

            elif event_type == "input_audio_buffer.speech_stopped":
                print("[realtime] Processing speech...")
                skip_response = self._skip_next_response_create
                if skip_response:
                    self._skip_next_response_create = False
                elif not self.config.allow_interruptions and self._wake_is_active():
                    await self._request_response_if_idle(connection)
                elif not self._wake_is_active() and not self._wake_waiting_logged:
                    print("[realtime] Waiting for wake phrase.")
                    self._wake_waiting_logged = True
                if self._wake_is_active() and not skip_response:
                    self._motion_call("thinking")

            elif event_type == "response.done":
                self._push_input_suppression()
                self._clear_active_response()
                if self._sleep_after_response:
                    self._sleep_after_response = False
                    await self._enter_sleep_mode(connection)
                elif not self._should_hold_sleep_pose():
                    self._motion_call("idle")

            elif event_type == "response.function_call_arguments.done":
                await self._handle_function_call(connection, event)

            elif event_type == "error":
                error = getattr(event, "error", None)
                error_code = getattr(error, "code", None)
                if error_code == "response_cancel_not_active":
                    print("[realtime] Interrupt race: response was already stopped.")
                    self._clear_active_response()
                    continue
                if error_code == "conversation_already_has_active_response":
                    print("[realtime] Response race: server already has an active response.")
                    self._response_create_pending = False
                    continue
                raise RuntimeError(f"Realtime API error: {error}")

    def _motion_call(self, name: str) -> None:
        if self.motion is None or not hasattr(self.motion, name):
            return
        try:
            getattr(self.motion, name)()
        except Exception as exc:
            print(f"[realtime] Motion '{name}' failed: {exc}")

    async def _interrupt_assistant_response(self, connection) -> None:
        if self._active_response_id is None:
            if hasattr(self.sink, "clear"):
                await self.sink.clear()
            return

        if self._interrupting_response_id == self._active_response_id:
            return

        print("[realtime] Interrupting assistant response.")
        response_id = self._active_response_id
        self._interrupting_response_id = response_id
        await connection.response.cancel(response_id=response_id)

        if self._active_item_id is not None and self._active_content_index is not None:
            await connection.conversation.item.truncate(
                item_id=self._active_item_id,
                content_index=self._active_content_index,
                audio_end_ms=max(0, self._active_audio_end_ms),
            )

        if hasattr(self.sink, "clear"):
            await self.sink.clear()

        self._clear_active_response()

    def _clear_active_response(self) -> None:
        self._active_response_id = None
        self._active_item_id = None
        self._active_content_index = None
        self._active_audio_end_ms = 0
        self._interrupting_response_id = None
        self._response_create_pending = False

    def _reset_conversation_state(self) -> None:
        self.last_user_transcript = ""
        self.last_assistant_transcript = ""
        self._tool_calls.clear()
        self._wake_waiting_logged = False
        self._sleep_after_response = False
        self._skip_next_response_create = False
        self._clear_active_response()

    def _reset_source_state(self) -> None:
        if hasattr(self.source, "reset_session_state"):
            self.source.reset_session_state()

    def _should_suppress_input(self) -> bool:
        if not self.config.suppress_input_while_speaking:
            return False
        return time.monotonic() < self._suppress_input_until

    def _push_input_suppression(self) -> None:
        if not self.config.suppress_input_while_speaking:
            return
        delay_sec = max(0.0, self.config.input_resume_delay_ms / 1000.0)
        self._suppress_input_until = max(
            self._suppress_input_until,
            time.monotonic() + delay_sec,
        )

    async def _monitor_wake_timeout(self) -> None:
        while not self._stop_event.is_set():
            self._keep_source_awake_if_busy()
            self._wake_is_active()
            await asyncio.sleep(0.25)

    def _wake_is_active(self) -> bool:
        if bool(getattr(self.source, "starts_asleep", False)) and hasattr(self.source, "is_awake"):
            return bool(self.source.is_awake())

        if not self.config.wake_phrase_enabled:
            return True

        now = time.monotonic()
        if now < self._wake_active_until:
            return True

        if self._wake_active_until != 0.0:
            self._wake_active_until = 0.0
            print("[wakeword] Inactivity timeout reached. Returning to sleep mode.")
            self._motion_call("sleeping")
        return False

    def _extend_wake_window(self) -> None:
        if not self.config.wake_phrase_enabled:
            return
        self._wake_active_until = time.monotonic() + self.config.wake_phrase_timeout_sec
        self._wake_waiting_logged = False

    def _handle_wake_phrase(self, transcript: str) -> tuple[bool, str]:
        if not self.config.wake_phrase_enabled:
            return False, transcript

        text = transcript.strip().lower()
        if self._wake_is_active():
            self._extend_wake_window()
            return False, transcript

        matched_phrase, stripped = self._match_wake_phrase(text)
        if matched_phrase:
            self._extend_wake_window()
            print(f"[wakeword] Wake phrase detected ({matched_phrase}). Reachy is awake.")
            self._motion_call("waking")
            if not stripped:
                print("[wakeword] Awaiting your question.")
            return True, stripped

        return False, transcript

    def _match_wake_phrase(self, text: str) -> tuple[str | None, str]:
        normalized = re.sub(r"[^\w\s]", " ", text)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return None, text

        greeting_only = {"hei", "hey", "hallo", "hello"}
        if normalized in greeting_only:
            return normalized, ""

        for phrase in self.config.wake_phrases:
            if phrase and phrase in normalized:
                stripped = re.sub(re.escape(phrase), "", normalized, count=1).strip()
                return phrase, stripped

        tokens = normalized.split()
        if len(tokens) < 2:
            return None, text

        greetings = {"hei", "hey", "hello", "hallo"}
        name_aliases = {"reachy", "richie", "ritchie", "reachie", "richy", "ritchi"}

        for index, token in enumerate(tokens[:-1]):
            if token not in greetings:
                continue
            candidate = tokens[index + 1]
            if candidate in name_aliases or any(
                SequenceMatcher(None, candidate, alias).ratio() >= 0.72
                for alias in name_aliases
            ):
                stripped = " ".join(tokens[:index] + tokens[index + 2 :]).strip()
                matched = f"{token} {candidate}"
                return matched, stripped

        return None, text

    def _is_sleep_phrase(self, transcript: str) -> bool:
        normalized = re.sub(r"[^\w\s]", " ", transcript.lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if not normalized:
            return False

        if normalized in self.config.sleep_phrases:
            return True

        return any(
            phrase and phrase in normalized
            for phrase in self.config.sleep_phrases
        )

    async def _request_response_if_idle(self, connection) -> None:
        if self._active_response_id is not None or self._response_create_pending:
            return
        self._response_create_pending = True
        await connection.response.create()

    def _tool_definitions(self) -> list[dict[str, object]]:
        tools: list[dict[str, object]] = [self._sleep_tool_definition()]
        if self.rag_tool is not None:
            tools.append(self.rag_tool.definition())
        return tools

    @staticmethod
    def _sleep_tool_definition() -> dict[str, object]:
        return {
            "type": "function",
            "name": "go_to_sleep",
            "description": (
                "Use this when the conversation is clearly over and Reachy should go back to sleep "
                "after a short closing reply, for example after goodbye, good night, or when the "
                "user says they do not need more help."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Short reason for ending the conversation.",
                    }
                },
                "required": [],
                "additionalProperties": False,
            },
        }

    async def _handle_function_call(self, connection, event) -> None:
        call_id = getattr(event, "call_id", None)
        arguments = getattr(event, "arguments", "") or "{}"

        if not call_id:
            return

        name = self._tool_calls.get(call_id, "")

        self._clear_active_response()

        if name == "go_to_sleep":
            self._sleep_after_response = True
            output = json.dumps(
                {"status": "ok", "message": "Reachy will go to sleep after this reply."},
                ensure_ascii=False,
            )
        elif name != "openai_rag_search" or self.rag_tool is None:
            output = json.dumps({"error": f"Unknown tool: {name}"}, ensure_ascii=False)
        else:
            print("[realtime] Running tool: openai_rag_search")
            debug = await asyncio.to_thread(self.rag_tool.debug_summary, arguments)
            print(
                "[realtime][rag] "
                f"query={debug.get('query')!r} "
                f"category={debug.get('category')!r} "
                f"limit={debug.get('limit')} "
                f"results={debug.get('results_count')} "
                f"sources={debug.get('sources')}"
            )
            answer_preview = (debug.get("answer") or "")[:240]
            if answer_preview:
                print(f"[realtime][rag] answer={answer_preview}")
            if debug.get("error"):
                print(f"[realtime][rag] error={debug['error']}")
            output = await asyncio.to_thread(self.rag_tool.execute, arguments)
        self._tool_calls.pop(call_id, None)

        await connection.conversation.item.create(
            item={
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            }
        )
        await self._request_response_if_idle(connection)

    async def _enter_sleep_mode(self, connection=None) -> None:
        self._sleep_after_response = False
        if connection is not None and self._active_response_id is not None:
            await self._interrupt_assistant_response(connection)
        elif hasattr(self.sink, "clear"):
            await self.sink.clear()

        self._clear_active_response()
        self._reset_source_state()
        self._motion_call("sleeping")
        if bool(getattr(self.source, "starts_asleep", False)):
            raise RealtimeSessionRestart("Sleep requested; starting a fresh wakeword session.")

    def _keep_source_awake_if_busy(self) -> None:
        if not hasattr(self.source, "touch_activity"):
            return
        if not bool(getattr(self.source, "starts_asleep", False)):
            return
        if (
            self._response_create_pending
            or self._active_response_id is not None
        ):
            self.source.touch_activity()

    def _should_hold_sleep_pose(self) -> bool:
        if self._sleep_after_response:
            return True
        if bool(getattr(self.source, "starts_asleep", False)) and hasattr(self.source, "is_awake"):
            return not bool(self.source.is_awake())
        return False
