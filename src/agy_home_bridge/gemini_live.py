"""Full-Duplex Gemini Live bidirectional audio & tool bridge for Antigravity + Google Home."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
from typing import Any
from unittest.mock import NonCallableMock

from fastapi import WebSocket, WebSocketDisconnect

from agy_home_bridge.antigravity_agent import AntigravityAgentEngine
from agy_home_bridge.cast_manager import GoogleHomeCastManager, normalize_room_key
from agy_home_bridge.config import BridgeSettings
from agy_home_bridge.event_hub import EventHub, safe_json_dumps
from agy_home_bridge.project_manager import ProjectManager

logger = logging.getLogger(__name__)


class _BoundedEventIdSet:
    """FIFO-bounded set of event IDs preventing unbounded growth on slow/evicted sockets."""

    def __init__(self, maxlen: int = 256) -> None:
        self._maxlen = max(1, int(maxlen))
        self._items: dict[str, None] = {}

    def add(self, item: str) -> None:
        if item in self._items:
            return
        if len(self._items) >= self._maxlen:
            oldest = next(iter(self._items))
            del self._items[oldest]
        self._items[item] = None

    def discard(self, item: str) -> None:
        self._items.pop(item, None)

    def __contains__(self, item: object) -> bool:
        return item in self._items

    def __len__(self) -> int:
        return len(self._items)


class FullDuplexLiveBridge:
    """Bridges a browser/mobile/satellite WebSocket to Gemini Live with Antigravity & Google Home tools.

    Protocol (JSON messages over WebSocket `/ws/live`):
    Client -> Server:
      - `{"type": "audio_chunk", "pcm_b64": "<base64 16kHz 16-bit mono PCM>"}`
      - `{"type": "text", "text": "Follow me to the kitchen", "cast": true}`
      - `{"type": "barge_in"}` (Immediately interrupts playback and clears audio buffer)
      - `{"type": "set_room", "room": "Kitchen"}`
      - `{"type": "set_routing", "mode": "both" | "direct" | "cast"}`

    Server -> Client:
      - `{"type": "status", "state": "connected" | "listening" | "thinking" | "speaking", ...}`
      - `{"type": "audio_chunk", "pcm_b64": "<base64 24kHz 16-bit mono PCM>", "sample_rate": 24000}`
      - `{"type": "transcript", "role": "user" | "agent", "text": "..."}`
      - `{"type": "tool_call", "name": "...", "args": {...}, "result": "..."}`
      - `{"type": "interrupted"}`
    """

    def __init__(
        self,
        config: BridgeSettings,
        agent_engine: AntigravityAgentEngine,
        cast_manager: GoogleHomeCastManager,
        event_hub: EventHub | None = None,
        project_manager: ProjectManager | None = None,
    ) -> None:
        self.config = config
        self.agent_engine = agent_engine
        self.cast_manager = cast_manager
        self.event_hub = (
            event_hub
            if event_hub is not None
            else getattr(agent_engine, "event_hub", None)
        )
        self.project_manager = (
            project_manager
            if project_manager is not None
            else getattr(agent_engine, "project_manager", None)
        )
        self._voice_status: str = "idle"
        self._voice_muted: bool = False

    @property
    def voice_status(self) -> str:
        """Current live voice status ('idle', 'listening', 'thinking', 'speaking', 'muted')."""
        return "muted" if self._voice_muted else self._voice_status

    @property
    def voice_muted(self) -> bool:
        """True when global voice input is muted."""
        return self._voice_muted

    async def set_voice_muted(self, muted: bool) -> bool:
        """Set global voice mute state and broadcast to EventHub (/ws/hud and /ws/live)."""
        self._voice_muted = bool(muted)
        if self._voice_muted:
            self._on_live_interrupted()
        effective = "muted" if self._voice_muted else (self._voice_status or "listening")
        if self.event_hub is not None:
            target = self._voice_status_target()
            await self.event_hub.publish(
                "voice_status",
                project_id=target.get("project_id"),
                project_name=target.get("project_name"),
                short_name=target.get("short_name"),
                agent_id=target.get("agent_id"),
                project_agent=target.get("project_agent"),
                conversation_id=target.get("conversation_id"),
                status=effective,
                voice_status=effective,
                voice_muted=self._voice_muted,
            )
        return self._voice_muted

    def _voice_status_target(self) -> dict[str, Any]:
        if self.project_manager is not None:
            try:
                return self.project_manager.get_target()
            except Exception:
                return {}
        return {}

    def _set_voice_status_sync(self, state: str, **extra: Any) -> None:
        """Synchronously update voice_status and publish to EventHub without awaiting."""
        self._voice_status = state
        if self.event_hub is None:
            return
        target = self._voice_status_target()
        try:
            self.event_hub.publish_threadsafe(
                "voice_status",
                project_id=target.get("project_id"),
                project_name=target.get("project_name"),
                short_name=target.get("short_name"),
                agent_id=target.get("agent_id"),
                project_agent=target.get("project_agent"),
                conversation_id=target.get("conversation_id"),
                status=state,
                voice_status=state,
                active_room=self.cast_manager.active_room,
                **extra,
            )
        except Exception as exc:
            logger.debug("Failed to publish voice_status '%s': %s", state, exc)

    async def _publish_transcript(
        self,
        role: str,
        text: str,
        partial: bool = False,
        local_emitted_event_ids: set[str] | _BoundedEventIdSet | None = None,
    ) -> None:
        """Broadcast user/agent speech transcript to EventHub (/ws/hud & other PWAs)."""
        if self.event_hub is None or not text:
            return
        target = self._voice_status_target()
        ev = await self.event_hub.publish(
            "transcript",
            project_id=target.get("project_id"),
            project_name=target.get("project_name"),
            short_name=target.get("short_name"),
            agent_id=target.get("agent_id"),
            project_agent=target.get("project_agent"),
            conversation_id=target.get("conversation_id"),
            role=role,
            text=text,
            partial=partial,
        )
        if local_emitted_event_ids is not None and isinstance(ev, dict):
            ev_id = ev.get("event_id")
            if isinstance(ev_id, str) and ev_id:
                local_emitted_event_ids.add(ev_id)

    async def _set_voice_status(self, state: str, **extra: Any) -> None:
        self._voice_status = state
        if self.event_hub is None:
            return
        target = self._voice_status_target()
        try:
            await self.event_hub.publish(
                "voice_status",
                project_id=target.get("project_id"),
                project_name=target.get("project_name"),
                short_name=target.get("short_name"),
                agent_id=target.get("agent_id"),
                project_agent=target.get("project_agent"),
                conversation_id=target.get("conversation_id"),
                status=state,
                voice_status=state,
                active_room=self.cast_manager.active_room,
                **extra,
            )
        except Exception as exc:
            logger.debug("Failed to publish voice_status '%s': %s", state, exc)

    async def _forward_event_hub_to_ws(
        self,
        websocket: WebSocket,
        queue: asyncio.Queue[dict[str, Any]],
        local_emitted_event_ids: set[str] | _BoundedEventIdSet | None = None,
    ) -> None:
        """Forward EventHub events to `/ws/live` without disrupting `/ws/live` status ordering."""
        try:
            while True:
                event = await queue.get()
                if event.get("type") == "voice_status":
                    continue
                ev_id = event.get("event_id")
                if (
                    local_emitted_event_ids is not None
                    and ev_id is not None
                    and ev_id in local_emitted_event_ids
                ):
                    local_emitted_event_ids.discard(ev_id)
                    continue
                try:
                    payload_text = safe_json_dumps(event)
                except (TypeError, ValueError) as exc:
                    logger.warning("Skipping unserializable live event: %s", exc)
                    continue
                await websocket.send_text(payload_text)
        except (asyncio.CancelledError, WebSocketDisconnect):
            return
        except Exception:
            return

    async def handle_websocket(self, websocket: WebSocket) -> None:
        """Main full-duplex WebSocket loop for a connected voice client or room mic satellite."""
        await websocket.accept()
        await self._send_json(
            websocket,
            {
                "type": "status",
                "state": "connected",
                "active_room": self.cast_manager.active_room,
                "active_presence_room": self.cast_manager.active_presence_room,
                "active_tv": self.cast_manager.active_tv,
                "routing_mode": self.cast_manager.routing_mode,
                "speakers": self.cast_manager.list_speakers(),
                "live_model": self.config.gemini_live_model,
            },
        )

        hub_queue: asyncio.Queue[dict[str, Any]] | None = None
        forwarder_task: asyncio.Task[Any] | None = None
        local_emitted_event_ids = _BoundedEventIdSet(maxlen=256)
        if self.event_hub is not None:
            hub_queue = self.event_hub.subscribe()
            forwarder_task = asyncio.create_task(
                self._forward_event_hub_to_ws(
                    websocket,
                    hub_queue,
                    local_emitted_event_ids=local_emitted_event_ids,
                )
            )

        try:
            api_key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
            if api_key:
                await self._run_gemini_live_duplex_loop(
                    websocket,
                    api_key,
                    local_emitted_event_ids=local_emitted_event_ids,
                )
            else:
                await self._run_text_and_vad_duplex_loop(
                    websocket,
                    local_emitted_event_ids=local_emitted_event_ids,
                )
        finally:
            if self.event_hub is not None and hub_queue is not None:
                self.event_hub.unsubscribe(hub_queue)
            if forwarder_task is not None and not forwarder_task.done():
                forwarder_task.cancel()
            self._set_voice_status_sync("idle")
            if forwarder_task is not None:
                try:
                    await asyncio.gather(forwarder_task, return_exceptions=True)
                except asyncio.CancelledError:
                    pass

    async def _run_gemini_live_duplex_loop(
        self,
        websocket: WebSocket,
        api_key: str,
        local_emitted_event_ids: set[str] | _BoundedEventIdSet | None = None,
    ) -> None:
        """Connect to the native Gemini Live bidirectional streaming API (`client.aio.live.connect`)."""
        try:
            from google import genai
            from google.genai import types as genai_types

            use_vertex = self.config.gemini_use_vertex
            client = genai.Client(
                api_key=api_key,
                vertexai=use_vertex,
                http_options=None if use_vertex else {"api_version": "v1beta"},
            )

            tool_declarations = [
                genai_types.FunctionDeclaration(
                    name="delegate_to_antigravity_agent",
                    description=(
                        "Non-blocking background delegation: start a coding, shell command, file editing, "
                        "Proxmox homelab check, or workspace task on the persistent Google Antigravity SDK "
                        "agent. Returns immediately with task_id so you can briefly acknowledge the user "
                        "while the agent works in the background."
                    ),
                    parameters=genai_types.Schema(
                        type="OBJECT",
                        properties={
                            "task": genai_types.Schema(
                                type="STRING",
                                description="Detailed task or question for the Antigravity workspace agent.",
                            ),
                            "project": genai_types.Schema(
                                type="STRING",
                                description="Optional target project name, id, or verbal short name override.",
                            ),
                            "project_agent": genai_types.Schema(
                                type="STRING",
                                description="Optional combined verbal target (e.g. 'project Clark Home Server, agent 102').",
                            ),
                            "agent": genai_types.Schema(
                                type="STRING",
                                description="Optional agent slot id override (e.g. '101', '102').",
                            ),
                        },
                        required=["task"],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="switch_project",
                    description=(
                        "Switch the active workspace project and/or project_agent slot for subsequent "
                        "voice commands (e.g. 'Switch project to Clark Home Server' or 'project proxmox monitor, agent 102')."
                    ),
                    parameters=genai_types.Schema(
                        type="OBJECT",
                        properties={
                            "project": genai_types.Schema(
                                type="STRING",
                                description="Target project id, directory name, or verbal short name.",
                            ),
                            "project_agent": genai_types.Schema(
                                type="STRING",
                                description="Combined verbal project-agent phrase (e.g. 'project Clark Home Server, agent 101').",
                            ),
                            "agent": genai_types.Schema(
                                type="STRING",
                                description="Optional agent slot id (e.g. '101', '102').",
                            ),
                        },
                        required=[],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="cast_hud_to_tv",
                    description=(
                        "Strictly on-demand: cast the read-only JARVIS /hud dashboard to a room TV "
                        "only when the user explicitly asks to show or cast the HUD on a TV."
                    ),
                    parameters=genai_types.Schema(
                        type="OBJECT",
                        properties={
                            "room": genai_types.Schema(
                                type="STRING",
                                description="Optional room name (e.g. 'office', 'game_room', '98_qled', 'living_room').",
                            ),
                            "tv": genai_types.Schema(
                                type="STRING",
                                description="Optional explicit TV device name (e.g. 'Game Room TV', 'Office TV', '98\" QLED').",
                            ),
                        },
                        required=[],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="switch_google_home_room",
                    description=(
                        "Switch which Google Home room speaker or speaker group plays Antigravity's voice "
                        "(e.g. when the user says 'Follow me to the kitchen' or 'Switch to the office speaker')."
                    ),
                    parameters=genai_types.Schema(
                        type="OBJECT",
                        properties={
                            "room": genai_types.Schema(
                                type="STRING",
                                description="Name of the Google Home room or speaker group.",
                            )
                        },
                        required=["room"],
                    ),
                ),
                genai_types.FunctionDeclaration(
                    name="broadcast_to_google_home",
                    description="Broadcast a spoken message out loud to a specific Google Home speaker or the whole house.",
                    parameters=genai_types.Schema(
                        type="OBJECT",
                        properties={
                            "message": genai_types.Schema(
                                type="STRING",
                                description="The message to announce.",
                            ),
                            "room": genai_types.Schema(
                                type="STRING",
                                description="Target room or 'Whole Home'.",
                            ),
                        },
                        required=["message"],
                    ),
                ),
            ]

            live_instructions = (
                f"{self.agent_engine.system_instructions}\n"
                "- When delegating workspace tasks via `delegate_to_antigravity_agent`, it runs asynchronously "
                "in the background: briefly acknowledge the started task to the user right away.\n"
                "- Use `switch_project` when the user asks to switch projects or `project_agent` slots.\n"
                "- Use `cast_hud_to_tv` ONLY when the user explicitly requests casting or showing the HUD on a TV."
            )

            live_config = genai_types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                system_instruction=genai_types.Content(
                    parts=[
                        genai_types.Part.from_text(
                            text=live_instructions
                        )
                    ]
                ),
                tools=[genai_types.Tool(function_declarations=tool_declarations)],
                speech_config=genai_types.SpeechConfig(
                    voice_config=genai_types.VoiceConfig(
                        prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                            voice_name=self.config.gemini_live_voice
                        )
                    )
                ),
                # Respond quickly once the user stops talking (default waits longer).
                realtime_input_config=genai_types.RealtimeInputConfig(
                    automatic_activity_detection=genai_types.AutomaticActivityDetection(
                        end_of_speech_sensitivity=genai_types.EndSensitivity.END_SENSITIVITY_HIGH,
                        silence_duration_ms=self.config.gemini_live_silence_ms,
                        prefix_padding_ms=100,
                    )
                ),
                # Server-side transcripts replace the phone's Web Speech recognition.
                input_audio_transcription=genai_types.AudioTranscriptionConfig(),
                output_audio_transcription=genai_types.AudioTranscriptionConfig(),
            )

            async with client.aio.live.connect(
                model=self.config.gemini_live_model,
                config=live_config,
            ) as session:
                if self.cast_manager.routing_mode in ("cast", "both"):
                    asyncio.create_task(self.cast_manager.warm_room())

                await self._set_voice_status("listening", mode="gemini-live-native-audio")
                await self._send_json(
                    websocket,
                    {
                        "type": "status",
                        "state": "listening",
                        "mode": "gemini-live-native-audio",
                        "active_room": self.cast_manager.active_room,
                    },
                )

                client_to_gemini = asyncio.create_task(
                    self._forward_client_to_gemini(
                        websocket,
                        session,
                        local_emitted_event_ids=local_emitted_event_ids,
                    )
                )
                gemini_to_client = asyncio.create_task(
                    self._forward_gemini_to_client(
                        websocket, session, local_emitted_event_ids
                    )
                )

                done: set[asyncio.Task[Any]] = set()
                pending: set[asyncio.Task[Any]] = {client_to_gemini, gemini_to_client}
                try:
                    done, pending = await asyncio.wait(
                        [client_to_gemini, gemini_to_client],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                finally:
                    for task in pending:
                        if not task.done():
                            task.cancel()
                    if pending:
                        try:
                            await asyncio.gather(*pending, return_exceptions=True)
                        except asyncio.CancelledError:
                            pass
                for task in done:
                    exc = task.exception()
                    if exc and not isinstance(exc, WebSocketDisconnect):
                        # e.g. Gemini closing an idle session (1008 client inactivity).
                        # Close cleanly so the PWA auto-reconnects with a fresh session.
                        logger.info("Gemini Live session ended: %s", exc)
                        try:
                            await websocket.close(code=1011, reason="live session ended")
                        except Exception:
                            pass

        except WebSocketDisconnect:
            logger.info("Live WebSocket client disconnected.")
        except Exception as exc:
            logger.warning(
                "Native Gemini Live session fell back to turn-based bridge (%s)", exc
            )
            await self._run_text_and_vad_duplex_loop(
                websocket,
                local_emitted_event_ids=local_emitted_event_ids,
            )

    @staticmethod
    async def _receive_ws_frame_text(
        websocket: WebSocket,
    ) -> tuple[str, str | None] | None:
        """Receive one WebSocket text frame safely across Starlette WebSocket and test mocks.

        Returns ``None`` on disconnect, ``("", error)`` on unsupported (binary) frames,
        or ``(text, None)`` for a text frame.
        """
        is_mock = isinstance(websocket, NonCallableMock)
        has_real_receive = (
            "receive" in getattr(websocket, "__dict__", {})
            if is_mock
            else callable(getattr(websocket, "receive", None))
        )
        if not has_real_receive:
            raw = await websocket.receive_text()
            return (raw, None)

        frame = await websocket.receive()
        if not isinstance(frame, dict):
            return ("", "Unsupported WebSocket frame.")
        if frame.get("type") == "websocket.disconnect":
            return None
        text_val = frame.get("text")
        bytes_val = frame.get("bytes")
        if text_val is None and bytes_val is not None:
            return (
                "",
                "Binary WebSocket frames are not supported; send JSON text frames.",
            )
        if text_val is None:
            return None
        return (text_val, None)

    async def _forward_client_to_gemini(
        self,
        websocket: WebSocket,
        session: Any,
        local_emitted_event_ids: set[str] | _BoundedEventIdSet | None = None,
    ) -> None:
        is_hardware_mic = False
        while True:
            received = await self._receive_ws_frame_text(websocket)
            if received is None:
                break
            raw, frame_err = received
            if frame_err is not None:
                await self._send_json(
                    websocket,
                    {"type": "error", "error": frame_err},
                )
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError as exc:
                await self._send_json(
                    websocket,
                    {
                        "type": "error",
                        "error": f"Invalid JSON payload: {exc.msg}",
                    },
                )
                continue
            if not isinstance(msg, dict):
                await self._send_json(
                    websocket,
                    {
                        "type": "error",
                        "error": "Invalid message format: expected JSON object",
                    },
                )
                continue
            msg_type = msg.get("type")

            if msg_type == "client_hello":
                if msg.get("client_type") == "hardware_mic":
                    is_hardware_mic = True
                continue

            if msg_type == "set_mute":
                muted = bool(msg.get("muted", False))
                await self.set_voice_muted(muted)
                await self._send_json(
                    websocket,
                    {
                        "type": "status",
                        "state": self.voice_status,
                        "voice_muted": self._voice_muted,
                    },
                )
                continue

            if msg_type == "audio_chunk":
                raw_b64 = msg.get("pcm_b64")
                if not isinstance(raw_b64, str):
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": "Invalid audio_chunk: 'pcm_b64' must be a base64 string",
                        },
                    )
                    continue
                if not raw_b64.strip():
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": "Empty audio_chunk pcm_b64",
                        },
                    )
                    continue
                try:
                    pcm_bytes = base64.b64decode(raw_b64, validate=True)
                except (binascii.Error, ValueError):
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": "Invalid audio_chunk: malformed base64 in 'pcm_b64'",
                        },
                    )
                    continue
                if len(pcm_bytes) == 0:
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": "Empty audio_chunk pcm_b64",
                        },
                    )
                    continue

                from google.genai import types as genai_types

                # Prevent echo loop on wearable/USB mics without hardware AEC by muting
                # mic audio to digital silence while Google Home is speaking or globally muted.
                forward_pcm = (
                    bytes(len(pcm_bytes))
                    if self._voice_muted
                    or (is_hardware_mic and self.cast_manager.is_casting_audio)
                    else pcm_bytes
                )
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=forward_pcm, mime_type="audio/pcm;rate=16000"
                    ),
                )
            elif msg_type == "text":
                raw_text = msg.get("text", "")
                if not isinstance(raw_text, str):
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": "Invalid text message: 'text' must be a string",
                        },
                    )
                    continue
                text = raw_text.strip()
                if text:
                    from google.genai import types as genai_types

                    await self._send_json(
                        websocket,
                        {"type": "transcript", "role": "user", "text": text},
                    )
                    await self._publish_transcript(
                        "user",
                        text,
                        partial=False,
                        local_emitted_event_ids=local_emitted_event_ids,
                    )
                    await session.send_client_content(
                        turns=genai_types.Content(
                            role="user", parts=[genai_types.Part(text=text)]
                        ),
                        turn_complete=True,
                    )
            elif msg_type == "barge_in":
                self._on_live_interrupted()
                await self._send_json(websocket, {"type": "interrupted"})
            elif msg_type == "set_room":
                raw_room = msg.get("room", "")
                if not isinstance(raw_room, str):
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": "Invalid set_room message: 'room' must be a string",
                        },
                    )
                    continue
                clean_room = raw_room.strip()
                if not clean_room:
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": "room must be a non-empty string",
                        },
                    )
                    continue
                if not normalize_room_key(clean_room):
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": "room must contain at least one alphanumeric character",
                        },
                    )
                    continue
                room = self.cast_manager.set_active_room(clean_room)
                if self.event_hub is not None:
                    ev = await self.event_hub.publish(
                        "room_changed",
                        active_room=room,
                        active_speaker=room,
                        active_presence_room=self.cast_manager.active_presence_room,
                        presence_room=self.cast_manager.active_presence_room,
                        active_tv=self.cast_manager.active_tv,
                        routing_mode=self.cast_manager.routing_mode,
                        source="ws_live",
                    )
                    if local_emitted_event_ids is not None:
                        local_emitted_event_ids.add(ev["event_id"])
                if self.cast_manager.routing_mode in ("cast", "both"):
                    asyncio.create_task(self.cast_manager.warm_room(room))
                await self._send_json(
                    websocket,
                    {
                        "type": "status",
                        "state": "listening",
                        "active_room": room,
                        "speakers": self.cast_manager.list_speakers(),
                    },
                )
            elif msg_type == "set_routing":
                raw_mode = msg.get("mode", "both")
                if not isinstance(raw_mode, str):
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": "Invalid set_routing message: 'mode' must be a string",
                        },
                    )
                    continue
                try:
                    mode = self.cast_manager.set_routing_mode(raw_mode)
                except ValueError as exc:
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": str(exc),
                        },
                    )
                    continue
                await self._send_json(
                    websocket,
                    {
                        "type": "status",
                        "state": "listening",
                        "routing_mode": mode,
                    },
                )

    def _on_live_interrupted(self) -> None:
        """Stop active Google Home speaker playback immediately on barge-in."""
        if self.cast_manager.routing_mode in ("cast", "both"):
            try:
                asyncio.create_task(self.cast_manager.stop_room())
            except RuntimeError:
                pass

    async def _forward_gemini_to_client(
        self,
        websocket: WebSocket,
        session: Any,
        local_emitted_event_ids: set[str] | _BoundedEventIdSet | None = None,
    ) -> None:
        turn_pcm_buffer = bytearray()
        turn_text_chunks: list[str] = []
        phrase_idx = 0
        last_agent_tr = ""
        cast_tasks: list[asyncio.Task[Any]] = []

        def _track_cast_task(coro: Any) -> asyncio.Task[Any]:
            cast_tasks[:] = [t for t in cast_tasks if not t.done()]
            task = asyncio.create_task(coro)
            cast_tasks.append(task)

            def _cleanup(done_task: asyncio.Task[Any]) -> None:
                try:
                    cast_tasks.remove(done_task)
                except ValueError:
                    pass

            task.add_done_callback(_cleanup)
            return task

        async def _flush_cast_phrase() -> None:
            nonlocal phrase_idx
            if (
                self.cast_manager.routing_mode not in ("cast", "both")
                or not turn_pcm_buffer
            ):
                return
            pcm_copy = bytes(turn_pcm_buffer)
            turn_pcm_buffer.clear()
            enqueue = phrase_idx > 0 or self.cast_manager.is_casting_audio
            phrase_idx += 1
            # Ensure sequential ordering if a previous phrase task is still uploading
            cast_tasks[:] = [t for t in cast_tasks if not t.done()]
            prev = cast_tasks[-1] if cast_tasks else None

            async def _run_cast() -> None:
                if prev is not None and not prev.done():
                    try:
                        await prev
                    except Exception:
                        pass
                await self.cast_manager.cast_pcm_audio(
                    pcm_copy,
                    sample_rate=24000,
                    room_name=self.cast_manager.active_room,
                    enqueue=enqueue,
                )

            _track_cast_task(_run_cast())

        async def _receive_across_turns() -> Any:
            # google-genai's session.receive() ends after each turn (including tool-call
            # turns), so keep re-entering it for the life of the session.
            while True:
                got_any = False
                async for item in session.receive():
                    got_any = True
                    yield item
                if not got_any:
                    return

        try:
            async for response in _receive_across_turns():
                server_content = getattr(response, "server_content", None)
                if server_content is not None:
                    if getattr(server_content, "interrupted", False):
                        turn_pcm_buffer.clear()
                        turn_text_chunks.clear()
                        phrase_idx = 0
                        last_agent_tr = ""
                        self._on_live_interrupted()
                        await self._set_voice_status("listening")
                        await self._send_json(websocket, {"type": "interrupted"})
                        continue

                    for attr, role in (
                        ("input_transcription", "user"),
                        ("output_transcription", "agent"),
                    ):
                        tr = getattr(server_content, attr, None)
                        tr_text = getattr(tr, "text", None) if tr is not None else None
                        if isinstance(tr_text, str) and tr_text:
                            if role == "agent":
                                last_agent_tr += tr_text
                            await self._send_json(
                                websocket,
                                {
                                    "type": "transcript",
                                    "role": role,
                                    "text": tr_text,
                                    "partial": True,
                                },
                            )
                            await self._publish_transcript(
                                role,
                                tr_text,
                                partial=True,
                                local_emitted_event_ids=local_emitted_event_ids,
                            )

                    model_turn = getattr(server_content, "model_turn", None)
                    if model_turn is not None:
                        for part in (getattr(model_turn, "parts", None) or []):
                            inline_data = getattr(part, "inline_data", None)
                            if inline_data and getattr(inline_data, "data", None):
                                if self._voice_status != "speaking":
                                    await self._set_voice_status("speaking")
                                audio_bytes = inline_data.data
                                turn_pcm_buffer.extend(audio_bytes)
                                if self.cast_manager.routing_mode in ("direct", "both"):
                                    await self._send_json(
                                        websocket,
                                        {
                                            "type": "audio_chunk",
                                            "pcm_b64": base64.b64encode(audio_bytes).decode(
                                                "ascii"
                                            ),
                                            "sample_rate": 24000,
                                        },
                                    )
                            part_text = getattr(part, "text", None)
                            if part_text:
                                if self._voice_status != "speaking":
                                    await self._set_voice_status("speaking")
                                turn_text_chunks.append(part_text)
                                await self._send_json(
                                    websocket,
                                    {
                                        "type": "transcript",
                                        "role": "agent",
                                        "text": part_text,
                                    },
                                )
                                await self._publish_transcript(
                                    "agent",
                                    part_text,
                                    partial=False,
                                    local_emitted_event_ids=local_emitted_event_ids,
                                )

                    # Flush mid-turn only for very long (>8s) monologues so normal replies
                    # play as a single seamless WAV with zero Cast queue cutoff at the end.
                    if self.cast_manager.routing_mode in ("cast", "both") and turn_pcm_buffer:
                        if len(turn_pcm_buffer) >= 384000:  # 8.0s @ 24kHz 16-bit mono
                            await _flush_cast_phrase()
                            last_agent_tr = ""

                    if getattr(server_content, "turn_complete", False):
                        if self.cast_manager.routing_mode in ("cast", "both"):
                            if turn_pcm_buffer:
                                await _flush_cast_phrase()
                            elif turn_text_chunks and phrase_idx == 0:
                                full_text = "".join(turn_text_chunks).strip()
                                if full_text:
                                    _track_cast_task(
                                        self.cast_manager.speak_on_room(
                                            full_text,
                                            room_name=self.cast_manager.active_room,
                                        )
                                    )
                        turn_pcm_buffer.clear()
                        turn_text_chunks.clear()
                        phrase_idx = 0
                        last_agent_tr = ""
                        await self._set_voice_status("listening")
                        await self._send_json(
                            websocket,
                            {"type": "status", "state": "listening"},
                        )

                # Handle live function/tool calls from Gemini Live -> Antigravity Agent / Google Home
                tool_call = getattr(response, "tool_call", None)
                if tool_call is not None:
                    await self._handle_live_tool_calls(websocket, session, tool_call)
        finally:
            pending_cast = [t for t in cast_tasks if not t.done()]
            for task in pending_cast:
                task.cancel()
            if pending_cast:
                try:
                    await asyncio.gather(*pending_cast, return_exceptions=True)
                except asyncio.CancelledError:
                    pass
            cast_tasks.clear()

    async def _handle_live_tool_calls(
        self, websocket: WebSocket, session: Any, tool_call: Any
    ) -> None:
        from google.genai import types as genai_types

        function_responses = []
        for fc in (getattr(tool_call, "function_calls", None) or []):
            name = fc.name
            args = dict(fc.args or {})
            await self._set_voice_status("thinking", tool=name)
            await self._send_json(
                websocket,
                {"type": "status", "state": "thinking", "tool": name},
            )

            if name == "delegate_to_antigravity_agent":
                task_prompt = args.get("task", "")
                try:
                    ack = self.agent_engine.start_background_task(
                        task_prompt,
                        project=args.get("project") or None,
                        agent=args.get("agent") or None,
                        project_agent=args.get("project_agent") or None,
                    )
                    response_payload: dict[str, Any] = {
                        **ack,
                        "status": ack.get("status", "started"),
                        "instruction": (
                            "Briefly acknowledge to the user that the agent is working on "
                            "this task in the background and will announce the result when done."
                        ),
                    }
                except Exception as exc:
                    response_payload = {"status": "error", "error": str(exc)}
                output = json.dumps(response_payload)
            elif name == "switch_project":
                if self.project_manager is None:
                    response_payload = {
                        "status": "error",
                        "error": "ProjectManager is not configured.",
                    }
                else:
                    try:
                        selection = self.project_manager.select_project(
                            project=args.get("project") or None,
                            agent=args.get("agent") or None,
                            project_agent=args.get("project_agent") or None,
                        )
                        if self.event_hub is not None:
                            await self.event_hub.publish(
                                "project_changed",
                                project_id=selection["project_id"],
                                project_name=selection["project_name"],
                                short_name=selection["short_name"],
                                agent_id=selection["agent_id"],
                                project_agent=selection["project_agent"],
                                conversation_id=selection.get("conversation_id"),
                                path=selection.get("path"),
                                git_branch=selection.get("git_branch"),
                                uncommitted_changes=selection.get(
                                    "uncommitted_changes"
                                ),
                                source="voice_tool",
                            )
                        response_payload = {"status": "switched", **selection}
                    except Exception as exc:
                        response_payload = {"status": "error", "error": str(exc)}
                output = json.dumps(response_payload)
            elif name == "cast_hud_to_tv":
                try:
                    cast_res = await self.cast_manager.cast_hud_to_tv(
                        room_name=args.get("room") or None,
                        tv_name=args.get("tv") or None,
                    )
                    response_payload = dict(cast_res)
                except Exception as exc:
                    response_payload = {"status": "error", "error": str(exc)}
                output = json.dumps(response_payload)
            elif name == "switch_google_home_room":
                try:
                    raw_room = str(args.get("room") or "").strip()
                    if not raw_room or not normalize_room_key(raw_room):
                        raise ValueError("room must contain at least one alphanumeric character")
                    new_room = self.cast_manager.set_active_room(raw_room)
                    if self.event_hub is not None:
                        await self.event_hub.publish(
                            "room_changed",
                            active_room=new_room,
                            active_speaker=new_room,
                            active_presence_room=self.cast_manager.active_presence_room,
                            presence_room=self.cast_manager.active_presence_room,
                            active_tv=self.cast_manager.active_tv,
                            source="voice_tool",
                        )
                    output = f"Active Google Home speaker switched to {new_room}"
                    response_payload = {"result": output}
                except Exception as exc:
                    response_payload = {"status": "error", "error": str(exc)}
                    output = json.dumps(response_payload)
            elif name == "broadcast_to_google_home":
                try:
                    cast_res = await self.cast_manager.speak_on_room(
                        args.get("message", ""),
                        room_name=args.get("room") or None,
                    )
                    output = json.dumps(cast_res)
                    response_payload = {"result": output}
                except Exception as exc:
                    response_payload = {"status": "error", "error": str(exc)}
                    output = json.dumps(response_payload)
            else:
                output = f"Unknown tool: {name}"
                response_payload = {"result": output}

            await self._send_json(
                websocket,
                {
                    "type": "tool_call",
                    "name": name,
                    "args": args,
                    "result": output,
                    "active_room": self.cast_manager.active_room,
                },
            )
            function_responses.append(
                genai_types.FunctionResponse(
                    id=getattr(fc, "id", None),
                    name=name,
                    response=response_payload,
                )
            )

        if function_responses:
            await session.send_tool_response(function_responses=function_responses)

    async def _run_text_and_vad_duplex_loop(
        self,
        websocket: WebSocket,
        local_emitted_event_ids: set[str] | _BoundedEventIdSet | None = None,
    ) -> None:
        """Fallback full-duplex WebSocket loop supporting browser Web Speech / text / barge-in + Google Home Cast."""
        active_task: asyncio.Task[Any] | None = None
        await self._set_voice_status("listening", mode="text-vad-fallback")

        try:
            while True:
                received = await self._receive_ws_frame_text(websocket)
                if received is None:
                    break
                raw, frame_err = received
                if frame_err is not None:
                    await self._send_json(
                        websocket,
                        {"type": "error", "error": frame_err},
                    )
                    continue
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError as exc:
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": f"Invalid JSON payload: {exc.msg}",
                        },
                    )
                    continue
                if not isinstance(msg, dict):
                    await self._send_json(
                        websocket,
                        {
                            "type": "error",
                            "error": "Invalid message format: expected JSON object",
                        },
                    )
                    continue
                msg_type = msg.get("type")

                if msg_type == "barge_in":
                    if active_task is not None:
                        if not active_task.done():
                            active_task.cancel()
                        await asyncio.gather(active_task, return_exceptions=True)
                        active_task = None
                    await self._set_voice_status("listening")
                    await self._send_json(websocket, {"type": "interrupted"})
                    await self._send_json(
                        websocket, {"type": "status", "state": "listening"}
                    )

                elif msg_type == "set_room":
                    raw_room = msg.get("room", "")
                    if not isinstance(raw_room, str):
                        await self._send_json(
                            websocket,
                            {
                                "type": "error",
                                "error": "Invalid set_room message: 'room' must be a string",
                            },
                        )
                        continue
                    clean_room = raw_room.strip()
                    if not clean_room:
                        await self._send_json(
                            websocket,
                            {
                                "type": "error",
                                "error": "room must be a non-empty string",
                            },
                        )
                        continue
                    if not normalize_room_key(clean_room):
                        await self._send_json(
                            websocket,
                            {
                                "type": "error",
                                "error": "room must contain at least one alphanumeric character",
                            },
                        )
                        continue
                    room = self.cast_manager.set_active_room(clean_room)
                    if self.event_hub is not None:
                        ev = await self.event_hub.publish(
                            "room_changed",
                            active_room=room,
                            active_speaker=room,
                            active_presence_room=self.cast_manager.active_presence_room,
                            presence_room=self.cast_manager.active_presence_room,
                            active_tv=self.cast_manager.active_tv,
                            routing_mode=self.cast_manager.routing_mode,
                            source="ws_live",
                        )
                        if local_emitted_event_ids is not None:
                            local_emitted_event_ids.add(ev["event_id"])
                    await self._send_json(
                        websocket,
                        {
                            "type": "status",
                            "state": "listening",
                            "active_room": room,
                            "speakers": self.cast_manager.list_speakers(),
                        },
                    )

                elif msg_type == "set_routing":
                    raw_mode = msg.get("mode", "both")
                    if not isinstance(raw_mode, str):
                        await self._send_json(
                            websocket,
                            {
                                "type": "error",
                                "error": "Invalid set_routing message: 'mode' must be a string",
                            },
                        )
                        continue
                    try:
                        mode = self.cast_manager.set_routing_mode(raw_mode)
                    except ValueError as exc:
                        await self._send_json(
                            websocket,
                            {
                                "type": "error",
                                "error": str(exc),
                            },
                        )
                        continue
                    await self._send_json(
                        websocket,
                        {
                            "type": "status",
                            "state": "listening",
                            "routing_mode": mode,
                        },
                    )

                elif msg_type == "text":
                    raw_text = msg.get("text", "")
                    if not isinstance(raw_text, str):
                        await self._send_json(
                            websocket,
                            {
                                "type": "error",
                                "error": "Invalid text message: 'text' must be a string",
                            },
                        )
                        continue
                    user_text = raw_text.strip()
                    if not user_text:
                        continue

                    if active_task is not None:
                        if not active_task.done():
                            active_task.cancel()
                        await asyncio.gather(active_task, return_exceptions=True)
                        active_task = None

                    await self._set_voice_status("thinking")
                    await self._send_json(
                        websocket,
                        {"type": "transcript", "role": "user", "text": user_text},
                    )
                    await self._send_json(
                        websocket, {"type": "status", "state": "thinking"}
                    )

                    async def _process_turn(prompt_text: str) -> None:
                        try:
                            result = await self.agent_engine.ask(
                                prompt_text,
                                cast_response=(
                                    self.cast_manager.routing_mode in ("cast", "both")
                                ),
                                emit_events=False,
                            )
                            for tc in result.tool_calls:
                                await self._send_json(
                                    websocket,
                                    {
                                        "type": "tool_call",
                                        "name": tc["name"],
                                        "args": tc.get("args", {}),
                                        "result": "executed",
                                        "active_room": self.cast_manager.active_room,
                                    },
                                )
                            await self._set_voice_status("speaking")
                            await self._send_json(
                                websocket,
                                {
                                    "type": "transcript",
                                    "role": "agent",
                                    "text": result.text,
                                    "backend": result.backend,
                                    "speak_direct": (
                                        self.cast_manager.routing_mode
                                        in ("direct", "both")
                                    ),
                                    "active_room": self.cast_manager.active_room,
                                },
                            )
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            logger.warning("Text/VAD turn processing failed: %s", exc)
                            await self._send_json(
                                websocket,
                                {
                                    "type": "error",
                                    "error": str(exc),
                                },
                            )
                        await self._set_voice_status("listening")
                        await self._send_json(
                            websocket,
                            {
                                "type": "status",
                                "state": "listening",
                                "active_room": self.cast_manager.active_room,
                            },
                        )

                    active_task = asyncio.create_task(_process_turn(user_text))

        except WebSocketDisconnect:
            pass
        finally:
            if active_task and not active_task.done():
                active_task.cancel()
            if active_task is not None:
                try:
                    await asyncio.gather(active_task, return_exceptions=True)
                except asyncio.CancelledError:
                    pass

    @staticmethod
    async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
        try:
            await websocket.send_text(safe_json_dumps(payload))
        except Exception:
            pass
