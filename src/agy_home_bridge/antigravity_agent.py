"""24/7 Google Antigravity SDK Agent & Local ConversationService Engine for Proxmox."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from agy_home_bridge.cast_manager import GoogleHomeCastManager
from agy_home_bridge.config import BridgeSettings

logger = logging.getLogger(__name__)


@dataclass
class AgentTurnResult:
    """Result from executing a prompt against the Antigravity Agent Engine."""

    text: str
    conversation_id: str | None = None
    thoughts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    backend: str = "antigravity-sdk"


class AntigravityAgentEngine:
    """Manages an always-on Google Antigravity SDK Agent with Google Home room-casting tools.

    Supports three automatic execution tiers:
    1. Official `google-antigravity` Python SDK (`Agent`, `LocalAgentConfig`, `CapabilitiesConfig`)
       with persistent `conversation_id` across sessions and built-in workspace tools.
    2. Local Antigravity Language Server (`ConversationService` over Connect HTTP+JSON)
       when `ANTIGRAVITY_LS_ADDRESS` and `ANTIGRAVITY_CSRF_TOKEN` are present.
    3. `google-genai` tool-enabled Gemini fallback so the voice bridge works out-of-the-box
       even before `google-antigravity` wheel installation completes.
    """

    def __init__(
        self,
        config: BridgeSettings,
        cast_manager: GoogleHomeCastManager,
    ) -> None:
        self.config = config
        self.cast_manager = cast_manager
        self.conversation_id: str | None = None
        self._state_file = Path(self.config.agy_save_dir) / "active_session.json"
        self._load_persisted_conversation_id()

    def _load_persisted_conversation_id(self) -> None:
        try:
            if self._state_file.exists():
                data = json.loads(self._state_file.read_text(encoding="utf-8"))
                self.conversation_id = data.get("conversation_id")
        except Exception as exc:
            logger.debug("Could not load persisted conversation_id: %s", exc)

    def _save_persisted_conversation_id(self, conv_id: str | None) -> None:
        if not conv_id:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(
                json.dumps({"conversation_id": conv_id}, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Could not save persisted conversation_id: %s", exc)

    def build_custom_tools(self) -> list[Any]:
        """Build custom Python tools injected into the Antigravity SDK Agent."""
        cast_mgr = self.cast_manager
        workspace_dir = self.config.agy_workspace_dir

        def speak_on_google_home(
            message: str,
            room: str = "",
        ) -> str:
            """Speak a message out loud on a Google Home speaker or speaker group in the house.

            Args:
                message: The spoken text to announce on the Google Home speaker.
                room: Optional room or speaker group name (e.g., "Kitchen", "Office", "Living Room", "Whole Home"). Leave empty to use the active room.
            """
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(
                    cast_mgr.speak_on_room(message, room_name=room or None),
                    loop,
                )
                result = future.result(timeout=15)
            except RuntimeError:
                result = asyncio.run(
                    cast_mgr.speak_on_room(message, room_name=room or None)
                )
            return json.dumps(result)

        def switch_active_google_home_room(room: str) -> str:
            """Switch which Google Home room speaker Antigravity Voice Live follows you into.

             Use this when the user says "Follow me to the kitchen", "Switch audio to the office",
            or "Talk to me on Whole Home".

            Args:
                room: Target Google Home room or speaker group name (e.g., "Kitchen", "Office", "Whole Home").
            """
            new_room = cast_mgr.set_active_room(room)
            return f"Switched active Google Home room speaker to '{new_room}'."

        def list_google_home_speakers() -> str:
            """List all discovered Google Home and Chromecast speakers and speaker groups on the LAN."""
            speakers = cast_mgr.list_speakers()
            return json.dumps(
                {
                    "active_room": cast_mgr.active_room,
                    "routing_mode": cast_mgr.routing_mode,
                    "speakers": speakers,
                }
            )

        def check_homelab_status() -> str:
            """Check system uptime, CPU/memory load, and disk usage on this Proxmox / Home Server host."""
            try:
                uptime = subprocess.check_output(
                    ["uptime"], text=True, timeout=5
                ).strip()
                df = subprocess.check_output(
                    ["df", "-h", workspace_dir], text=True, timeout=5
                ).strip()
                return f"Host Uptime: {uptime}\nWorkspace Disk:\n{df}"
            except Exception as exc:
                return f"Error checking host status: {exc}"

        return [
            speak_on_google_home,
            switch_active_google_home_room,
            list_google_home_speakers,
            check_homelab_status,
        ]

    @property
    def system_instructions(self) -> str:
        return (
            "You are Antigravity Live, an always-on full-duplex voice agent running 24/7 on the user's "
            "Proxmox home server and connected to their Google Home speakers across the house.\n"
            f"- Current active Google Home speaker room: '{self.cast_manager.active_room}'\n"
            f"- Active workspace directory: '{self.config.agy_workspace_dir}'\n"
            "- Keep your spoken responses natural, concise, and conversational (1-3 sentences unless asked for detail).\n"
            "- When the user moves rooms (e.g. 'Follow me to the kitchen' or 'Switch to the office speaker'), "
            "call `switch_active_google_home_room` immediately.\n"
            "- When the user asks you to broadcast or announce something in another room, call `speak_on_google_home`.\n"
            "- You have full access to your Antigravity tools to inspect files, run commands, manage the Proxmox homelab, "
            "and write code hands-free while talking to the user."
        )

    async def ask(
        self,
        prompt: str,
        cast_response: bool | None = None,
        target_room: str | None = None,
    ) -> AgentTurnResult:
        """Send a prompt to the Antigravity Agent, stream/collect the response, and optionally cast to Google Home."""
        result = await self._execute_agent_turn(prompt)

        should_cast = (
            cast_response
            if cast_response is not None
            else self.cast_manager.routing_mode in ("cast", "both")
        )
        if should_cast and result.text.strip():
            await self.cast_manager.speak_on_room(
                result.text,
                room_name=target_room or self.cast_manager.active_room,
            )

        return result

    async def _execute_agent_turn(self, prompt: str) -> AgentTurnResult:
        # Tier 1: Official google-antigravity Python SDK
        try:
            from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig  # type: ignore

            save_dir = str(Path(self.config.agy_save_dir).resolve())
            app_data_dir = str(Path(self.config.agy_app_data_dir).resolve())
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            Path(app_data_dir).mkdir(parents=True, exist_ok=True)

            kwargs: dict[str, Any] = {
                "system_instructions": self.system_instructions,
                "tools": self.build_custom_tools(),
                "save_dir": save_dir,
                "app_data_dir": app_data_dir,
            }
            if self.config.gemini_api_key:
                kwargs["api_key"] = self.config.gemini_api_key
            if self.conversation_id:
                kwargs["conversation_id"] = self.conversation_id
            if self.config.agy_enable_write_tools:
                kwargs["capabilities"] = CapabilitiesConfig()

            ag_config = LocalAgentConfig(**kwargs)
            chunks: list[str] = []
            thoughts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            async with Agent(ag_config) as agent:
                response = await agent.chat(prompt)
                async for token in response:
                    chunks.append(token)
                self.conversation_id = getattr(agent, "conversation_id", None)
                self._save_persisted_conversation_id(self.conversation_id)

            return AgentTurnResult(
                text="".join(chunks).strip(),
                conversation_id=self.conversation_id,
                thoughts=thoughts,
                tool_calls=tool_calls,
                backend="google-antigravity-sdk",
            )
        except ImportError:
            logger.debug(
                "google-antigravity SDK package not installed in current venv; checking fallback backends."
            )
        except Exception as exc:
            logger.warning("google-antigravity SDK execution error: %s", exc)

        # Tier 2: google-genai with our custom Google Home & Homelab tools
        if self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY"):
            try:
                from google import genai
                from google.genai import types as genai_types

                api_key = self.config.gemini_api_key or os.environ.get(
                    "GEMINI_API_KEY", ""
                )
                client = genai.Client(api_key=api_key)
                tools = self.build_custom_tools()

                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=self.system_instructions,
                        tools=tools,
                    ),
                )
                return AgentTurnResult(
                    text=(response.text or "").strip(),
                    conversation_id=self.conversation_id or "genai-live-session",
                    backend="google-genai",
                )
            except Exception as exc:
                logger.warning("google-genai execution error: %s", exc)

        # Tier 3: Local Antigravity Language Server ConversationService
        if (
            self.config.antigravity_ls_address
            and self.config.antigravity_csrf_token
            and not any(k in prompt.lower() for k in ("follow me", "switch"))
        ):
            try:
                conv_data = await self.create_ls_conversation(prompt)
                conv_id = conv_data.get("conversation", {}).get("id", "ls-conv")
                return AgentTurnResult(
                    text=f"Started Antigravity conversation {conv_id} for: {prompt}",
                    conversation_id=conv_id,
                    backend="antigravity-ls-conversation-service",
                )
            except Exception as exc:
                logger.warning("Antigravity LS ConversationService error: %s", exc)

        # Offline / No-Key Diagnostic Response (still executes local voice room-switching keywords)
        lower = prompt.lower()
        for tool_fn in self.build_custom_tools():
            if tool_fn.__name__ == "switch_active_google_home_room" and (
                "switch" in lower or "follow me" in lower
            ):
                for candidate in ("kitchen", "office", "living room", "bedroom", "whole home"):
                    if candidate in lower:
                        msg = tool_fn(candidate.title())
                        return AgentTurnResult(
                            text=msg,
                            conversation_id="local-offline",
                            tool_calls=[{"name": "switch_active_google_home_room", "args": {"room": candidate.title()}}],
                            backend="local-fallback",
                        )

        return AgentTurnResult(
            text=(
                f"Antigravity Home Bridge is online (Active Google Home speaker: "
                f"{self.cast_manager.active_room}). Set GEMINI_API_KEY in .env to enable "
                f"live Gemini and Google Antigravity SDK reasoning. You said: {prompt}"
            ),
            conversation_id="local-offline",
            backend="local-fallback",
        )

    async def create_ls_conversation(
        self, prompt_text: str, title: str = "Google Home Voice Request"
    ) -> dict[str, Any]:
        """Call local Antigravity Language Server `ConversationService/CreateConversation`."""
        url = (
            f"http://{self.config.antigravity_ls_address}"
            "/product.v1.ConversationService/CreateConversation"
        )
        headers = {
            "Content-Type": "application/json",
            "Connect-Protocol-Version": "1",
            "x-antigravity-csrf-token": self.config.antigravity_csrf_token,
        }
        payload: dict[str, Any] = {
            "title": title,
            "model": "flash",
            "tags": ["google-home", "voice-live"],
            "prompt": [{"text": prompt_text}],
        }
        project_id = os.environ.get("ANTIGRAVITY_PROJECT_ID")
        if project_id:
            payload["projectId"] = project_id

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()
