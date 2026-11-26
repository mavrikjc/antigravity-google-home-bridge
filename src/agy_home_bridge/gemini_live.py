"""Full-Duplex Gemini Live bidirectional audio & tool bridge for Antigravity + Google Home."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from agy_home_bridge.antigravity_agent import AntigravityAgentEngine
from agy_home_bridge.cast_manager import GoogleHomeCastManager
from agy_home_bridge.config import BridgeSettings

logger = logging.getLogger(__name__)


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
    ) -> None:
        self.config = config
        self.agent_engine = agent_engine
        self.cast_manager = cast_manager

    async def handle_websocket(self, websocket: WebSocket) -> None:
        """Main full-duplex WebSocket loop for a connected voice client or room mic satellite."""
        await websocket.accept()
        await self._send_json(
            websocket,
            {
                "type": "status",
                "state": "connected",
                "active_room": self.cast_manager.active_room,
                "routing_mode": self.cast_manager.routing_mode,
                "speakers": self.cast_manager.list_speakers(),
                "live_model": self.config.gemini_live_model,
            },
        )

        api_key = self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            await self._run_gemini_live_duplex_loop(websocket, api_key)
        else:
            await self._run_text_and_vad_duplex_loop(websocket)

    async def _run_gemini_live_duplex_loop(
        self, websocket: WebSocket, api_key: str
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
                        "Delegate any coding, shell command, file editing, Proxmox homelab check, "
                        "or workspace task to the persistent 24/7 Google Antigravity SDK agent."
                    ),
                    parameters=genai_types.Schema(
                        type="OBJECT",
                        properties={
                            "task": genai_types.Schema(
                                type="STRING",
                                description="Detailed task or question for the Antigravity workspace agent.",
                            )
                        },
                        required=["task"],
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

            live_config = genai_types.LiveConnectConfig(
                response_modalities=["AUDIO"],
                system_instruction=genai_types.Content(
                    parts=[
                        genai_types.Part.from_text(
                            text=self.agent_engine.system_instructions
                        )
                    ]
                ),
                tools=[genai_types.Tool(function_declarations=tool_declarations)],
            )

            async with client.aio.live.connect(
                model=self.config.gemini_live_model,
                config=live_config,
            ) as session:
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
                    self._forward_client_to_gemini(websocket, session)
                )
                gemini_to_client = asyncio.create_task(
                    self._forward_gemini_to_client(websocket, session)
                )

                done, pending = await asyncio.wait(
                    [client_to_gemini, gemini_to_client],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()

        except WebSocketDisconnect:
            logger.info("Live WebSocket client disconnected.")
        except Exception as exc:
            logger.warning(
                "Native Gemini Live session fell back to turn-based bridge (%s)", exc
            )
            await self._run_text_and_vad_duplex_loop(websocket)

    async def _forward_client_to_gemini(
        self, websocket: WebSocket, session: Any
    ) -> None:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            msg_type = msg.get("type")

            if msg_type == "audio_chunk":
                from google.genai import types as genai_types

                pcm_bytes = base64.b64decode(msg["pcm_b64"])
                await session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=pcm_bytes, mime_type="audio/pcm;rate=16000"
                    ),
                )
            elif msg_type == "text":
                from google.genai import types as genai_types

                text = msg.get("text", "").strip()
                if text:
                    await self._send_json(
                        websocket,
                        {"type": "transcript", "role": "user", "text": text},
                    )
                    await session.send_client_content(
                        turns=genai_types.Content(
                            role="user", parts=[genai_types.Part(text=text)]
                        ),
                        turn_complete=True,
                    )
            elif msg_type == "barge_in":
                await self._send_json(websocket, {"type": "interrupted"})
            elif msg_type == "set_room":
                room = self.cast_manager.set_active_room(msg.get("room", ""))
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
                mode = self.cast_manager.set_routing_mode(msg.get("mode", "both"))
                await self._send_json(
                    websocket,
                    {
                        "type": "status",
                        "state": "listening",
                        "routing_mode": mode,
                    },
                )

    async def _forward_gemini_to_client(
        self, websocket: WebSocket, session: Any
    ) -> None:
        turn_pcm_buffer = bytearray()
        turn_text_chunks: list[str] = []

        async for response in session.receive():
            server_content = getattr(response, "server_content", None)
            if server_content is not None:
                if getattr(server_content, "interrupted", False):
                    turn_pcm_buffer.clear()
                    turn_text_chunks.clear()
                    await self._send_json(websocket, {"type": "interrupted"})
                    continue

                model_turn = getattr(server_content, "model_turn", None)
                if model_turn is not None:
                    for part in getattr(model_turn, "parts", []):
                        inline_data = getattr(part, "inline_data", None)
                        if inline_data and getattr(inline_data, "data", None):
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
                            turn_text_chunks.append(part_text)
                            await self._send_json(
                                websocket,
                                {
                                    "type": "transcript",
                                    "role": "agent",
                                    "text": part_text,
                                },
                            )

                if getattr(server_content, "turn_complete", False):
                    # Cast the completed turn to the active Google Home room speaker if cast routing is enabled
                    if self.cast_manager.routing_mode in ("cast", "both"):
                        if turn_pcm_buffer:
                            pcm_copy = bytes(turn_pcm_buffer)
                            asyncio.create_task(
                                self.cast_manager.cast_pcm_audio(
                                    pcm_copy,
                                    sample_rate=24000,
                                    room_name=self.cast_manager.active_room,
                                )
                            )
                        elif turn_text_chunks:
                            full_text = "".join(turn_text_chunks).strip()
                            if full_text:
                                asyncio.create_task(
                                    self.cast_manager.speak_on_room(
                                        full_text,
                                        room_name=self.cast_manager.active_room,
                                    )
                                )
                    turn_pcm_buffer.clear()
                    turn_text_chunks.clear()
                    await self._send_json(
                        websocket,
                        {"type": "status", "state": "listening"},
                    )

            # Handle live function/tool calls from Gemini Live -> Antigravity Agent / Google Home
            tool_call = getattr(response, "tool_call", None)
            if tool_call is not None:
                await self._handle_live_tool_calls(websocket, session, tool_call)

    async def _handle_live_tool_calls(
        self, websocket: WebSocket, session: Any, tool_call: Any
    ) -> None:
        from google.genai import types as genai_types

        function_responses = []
        for fc in getattr(tool_call, "function_calls", []):
            name = fc.name
            args = dict(fc.args or {})
            await self._send_json(
                websocket,
                {"type": "status", "state": "thinking", "tool": name},
            )

            if name == "delegate_to_antigravity_agent":
                task_prompt = args.get("task", "")
                res = await self.agent_engine.ask(task_prompt, cast_response=False)
                output = res.text
            elif name == "switch_google_home_room":
                new_room = self.cast_manager.set_active_room(args.get("room", ""))
                output = f"Active Google Home speaker switched to {new_room}"
            elif name == "broadcast_to_google_home":
                cast_res = await self.cast_manager.speak_on_room(
                    args.get("message", ""),
                    room_name=args.get("room") or None,
                )
                output = json.dumps(cast_res)
            else:
                output = f"Unknown tool: {name}"

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
                    response={"result": output},
                )
            )

        if function_responses:
            await session.send_tool_response(function_responses=function_responses)

    async def _run_text_and_vad_duplex_loop(self, websocket: WebSocket) -> None:
        """Fallback full-duplex WebSocket loop supporting browser Web Speech / text / barge-in + Google Home Cast."""
        active_task: asyncio.Task[Any] | None = None

        try:
            while True:
                raw = await websocket.receive_text()
                msg = json.loads(raw)
                msg_type = msg.get("type")

                if msg_type == "barge_in":
                    if active_task and not active_task.done():
                        active_task.cancel()
                    await self._send_json(websocket, {"type": "interrupted"})
                    await self._send_json(
                        websocket, {"type": "status", "state": "listening"}
                    )

                elif msg_type == "set_room":
                    room = self.cast_manager.set_active_room(msg.get("room", ""))
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
                    mode = self.cast_manager.set_routing_mode(msg.get("mode", "both"))
                    await self._send_json(
                        websocket,
                        {
                            "type": "status",
                            "state": "listening",
                            "routing_mode": mode,
                        },
                    )

                elif msg_type == "text":
                    user_text = msg.get("text", "").strip()
                    if not user_text:
                        continue

                    if active_task and not active_task.done():
                        active_task.cancel()

                    await self._send_json(
                        websocket,
                        {"type": "transcript", "role": "user", "text": user_text},
                    )
                    await self._send_json(
                        websocket, {"type": "status", "state": "thinking"}
                    )

                    async def _process_turn(prompt_text: str) -> None:
                        result = await self.agent_engine.ask(
                            prompt_text,
                            cast_response=(
                                self.cast_manager.routing_mode in ("cast", "both")
                            ),
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
            if active_task and not active_task.done():
                active_task.cancel()

    @staticmethod
    async def _send_json(websocket: WebSocket, payload: dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(payload))
        except Exception:
            pass
