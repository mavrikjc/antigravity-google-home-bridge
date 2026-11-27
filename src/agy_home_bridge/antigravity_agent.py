"""24/7 Google Antigravity SDK Agent & Local ConversationService Engine for Proxmox."""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from agy_home_bridge.cast_manager import GoogleHomeCastManager
from agy_home_bridge.config import BridgeSettings
from agy_home_bridge.event_hub import EventHub, safe_json_value
from agy_home_bridge.project_manager import ProjectManager

logger = logging.getLogger(__name__)


def summarize_for_speech(text: str, max_chars: int = 450) -> str:
    """Strip markdown/code fences and condense agent output for spoken Google Home TTS."""
    if not text:
        return ""
    cleaned = re.sub(r"```[\s\S]*?```", " ", text)
    cleaned = cleaned.replace("```", " ")
    cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"(?m)^\s*[-*+]\s+", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*>\s*", "", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "").replace("*", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    if len(cleaned) <= max_chars:
        return cleaned
    cutoff = max(1, max_chars - 3)
    truncated = cleaned[:cutoff].rsplit(" ", 1)[0].rstrip(" ,;:-")
    if not truncated:
        truncated = cleaned[:cutoff].rstrip(" ,;:-")
    return f"{truncated}..."


_safe_json_value = safe_json_value


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
        project_manager: ProjectManager | None = None,
        event_hub: EventHub | None = None,
    ) -> None:
        self.config = config
        self.cast_manager = cast_manager
        self.project_manager = project_manager
        self.event_hub = event_hub
        self.conversation_id: str | None = None
        self._state_file = Path(self.config.agy_save_dir) / "active_session.json"
        self._background_tasks: dict[str, asyncio.Task[Any]] = {}
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

    def _resolve_turn_target(
        self,
        *,
        project: str | None = None,
        agent: str | None = None,
        project_agent: str | None = None,
    ) -> dict[str, Any]:
        """Return an immutable snapshot of the target `(project, agent)` for this turn."""
        if self.project_manager is not None:
            return self.project_manager.get_target(
                project=project,
                agent=agent,
                project_agent=project_agent,
            )

        ws_path = str(Path(self.config.agy_workspace_dir).resolve())
        ws_name = Path(ws_path).name or "workspace"
        default_agent = getattr(self.config, "default_agent_id", "101") or "101"
        return {
            "project_id": "default",
            "project_name": ws_name,
            "short_name": ws_name,
            "path": ws_path,
            "agent_id": default_agent,
            "project_agent": f"project {ws_name}, agent {default_agent}",
            "conversation_id": self.conversation_id,
            "git_branch": None,
            "uncommitted_changes": 0,
        }

    @staticmethod
    def _event_scope(
        target: dict[str, Any],
        task_id: str,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        return {
            "project_id": target.get("project_id"),
            "project_name": target.get("project_name"),
            "short_name": target.get("short_name"),
            "agent_id": target.get("agent_id"),
            "project_agent": target.get("project_agent"),
            "conversation_id": conversation_id
            if conversation_id is not None
            else target.get("conversation_id"),
            "task_id": task_id,
        }

    async def _emit_event(
        self,
        event_type: str,
        *,
        target: dict[str, Any],
        task_id: str,
        conversation_id: str | None = None,
        **payload: Any,
    ) -> None:
        if self.event_hub is None:
            return
        scope = self._event_scope(target, task_id, conversation_id=conversation_id)
        await self.event_hub.publish(event_type, **scope, **payload)

    def _emit_event_threadsafe(
        self,
        event_type: str,
        *,
        target: dict[str, Any],
        task_id: str,
        conversation_id: str | None = None,
        **payload: Any,
    ) -> None:
        if self.event_hub is None:
            return
        scope = self._event_scope(target, task_id, conversation_id=conversation_id)
        self.event_hub.publish_threadsafe(event_type, **scope, **payload)

    def build_custom_tools(
        self,
        workspace_dir: str | None = None,
        *,
        emit_events: bool = True,
    ) -> list[Any]:
        """Build custom Python tools injected into the Antigravity SDK Agent."""
        cast_mgr = self.cast_manager
        event_hub = self.event_hub
        resolved_workspace = workspace_dir or self.config.agy_workspace_dir

        def speak_on_google_home(
            message: str,
            room: str = "",
        ) -> str:
            """Speak a message out loud on a Google Home speaker or speaker group in the house.

            Args:
                message: The spoken text to announce on the Google Home speaker.
                room: Optional room or speaker group name (e.g., "Kitchen", "Office", "Living Room", "Whole Home"). Leave empty to use the active room.
            """
            def _run_in_fresh_loop() -> dict[str, Any]:
                return asyncio.run(
                    cast_mgr.speak_on_room(message, room_name=room or None)
                )

            try:
                asyncio.get_running_loop()
            except RuntimeError:
                result = _run_in_fresh_loop()
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    result = pool.submit(_run_in_fresh_loop).result(timeout=15)
            return json.dumps(result)

        def switch_active_google_home_room(room: str) -> str:
            """Switch which Google Home room speaker Antigravity Voice Live follows you into.

             Use this when the user says "Follow me to the kitchen", "Switch audio to the office",
            or "Talk to me on Whole Home".

            Args:
                room: Target Google Home room or speaker group name (e.g., "Kitchen", "Office", "Whole Home").
            """
            new_room = cast_mgr.set_active_room(room)
            if emit_events and event_hub is not None:
                event_hub.publish_threadsafe(
                    "room_changed",
                    active_room=new_room,
                    active_speaker=new_room,
                    active_presence_room=cast_mgr.active_presence_room,
                    presence_room=cast_mgr.active_presence_room,
                    active_tv=cast_mgr.active_tv,
                    source="agent_tool",
                )
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
                    ["df", "-h", resolved_workspace], text=True, timeout=5
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

    def _build_system_instructions(self, target: dict[str, Any] | None = None) -> str:
        project_block = f"- Active workspace directory: '{self.config.agy_workspace_dir}'\n"
        if self.project_manager is not None:
            try:
                active_target = target or self.project_manager.get_target()
                summary = self.project_manager.get_state_summary()
                proj_lines: list[str] = []
                for p in summary.get("projects", []):
                    agent_labels = ", ".join(
                        a.get("project_agent") or a.get("short_name", "")
                        for a in p.get("agents", [])
                    )
                    proj_lines.append(
                        f"  * {p.get('short_name')} (id='{p.get('id')}', dir='{p.get('name')}', "
                        f"branch={p.get('git_branch') or 'n/a'}, agents=[{agent_labels}])"
                    )
                project_block = (
                    f"- Active project: '{active_target.get('short_name')}' "
                    f"(id='{active_target.get('project_id')}', path='{active_target.get('path')}')\n"
                    f"- Active project_agent: '{active_target.get('project_agent')}'\n"
                    "- Available projects & verbal short names (use these when switching or delegating):\n"
                    + ("\n".join(proj_lines) if proj_lines else "  * (none discovered)")
                    + "\n"
                )
            except Exception as exc:
                logger.debug("Could not build project instructions block: %s", exc)

        return (
            "You are Antigravity Live, an always-on full-duplex voice agent running 24/7 on the user's "
            "Proxmox home server and connected to their Google Home speakers across the house.\n"
            f"- Current active Google Home speaker room: '{self.cast_manager.active_room}'\n"
            f"- Available Google Home speakers/groups (use these exact names): "
            f"{', '.join(repr(s['name']) for s in self.cast_manager.list_speakers()) or 'none discovered yet'}\n"
            "- Only call broadcast/speak tools when the user explicitly asks to announce something in a room; "
            "otherwise just answer by voice.\n"
            f"{project_block}"
            "- Keep your spoken responses natural, concise, and conversational (1-3 sentences unless asked for detail).\n"
            "- When the user moves rooms (e.g. 'Follow me to the kitchen' or 'Switch to the office speaker'), "
            "call `switch_active_google_home_room` immediately.\n"
            "- When the user asks you to broadcast or announce something in another room, call `speak_on_google_home`.\n"
            "- You have full access to your Antigravity tools to inspect files, run commands, manage the Proxmox homelab, "
            "and write code hands-free while talking to the user."
        )

    @property
    def system_instructions(self) -> str:
        return self._build_system_instructions()

    async def ask(
        self,
        prompt: str,
        cast_response: bool | None = None,
        target_room: str | None = None,
        *,
        project: str | None = None,
        agent: str | None = None,
        project_agent: str | None = None,
        emit_events: bool = True,
    ) -> AgentTurnResult:
        """Send a prompt to the Antigravity Agent, stream/collect the response, and optionally cast to Google Home."""
        target = self._resolve_turn_target(
            project=project,
            agent=agent,
            project_agent=project_agent,
        )
        task_id = uuid.uuid4().hex[:12]

        if self.project_manager is not None:
            self.project_manager.update_agent_status(
                target["project_id"],
                target["agent_id"],
                status="running",
                current_task=prompt.strip()[:200],
                task_id=task_id,
            )

        try:
            if emit_events:
                await self._emit_event(
                    "task_started",
                    target=target,
                    task_id=task_id,
                    prompt=prompt,
                )
            result, swarm_nodes = await self._execute_agent_turn(
                prompt,
                target=target,
                task_id=task_id,
                emit_events=emit_events,
                raise_on_sdk_error=False,
            )
        except asyncio.CancelledError:
            if self.project_manager is not None:
                self.project_manager.update_agent_status(
                    target["project_id"],
                    target["agent_id"],
                    status="idle",
                    current_task=None,
                    task_id=task_id,
                )
            if emit_events:
                try:
                    await self._emit_event(
                        "task_failed",
                        target=target,
                        task_id=task_id,
                        error="Task cancelled",
                    )
                except Exception:
                    pass
            raise
        except Exception as exc:
            if self.project_manager is not None:
                self.project_manager.update_agent_status(
                    target["project_id"],
                    target["agent_id"],
                    status="error",
                    current_task=None,
                    last_summary=f"Error: {exc}",
                    task_id=task_id,
                )
            if emit_events:
                await self._emit_event(
                    "task_failed",
                    target=target,
                    task_id=task_id,
                    error=str(exc),
                )
            raise

        summary = summarize_for_speech(result.text)
        if self.project_manager is not None:
            self.project_manager.update_agent_status(
                target["project_id"],
                target["agent_id"],
                status="idle",
                current_task=None,
                last_summary=summary or None,
                swarm_nodes=swarm_nodes,
                task_id=task_id,
            )
        if emit_events:
            await self._emit_event(
                "task_completed",
                target=target,
                task_id=task_id,
                conversation_id=result.conversation_id or target.get("conversation_id"),
                summary=summary,
                text=result.text,
                backend=result.backend,
            )

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

    def start_background_task(
        self,
        prompt: str,
        *,
        project: str | None = None,
        agent: str | None = None,
        project_agent: str | None = None,
    ) -> dict[str, Any]:
        """Launch a non-blocking Antigravity turn in the background and return immediate ack metadata."""
        target = self._resolve_turn_target(
            project=project,
            agent=agent,
            project_agent=project_agent,
        )
        loop = asyncio.get_running_loop()
        task_id = uuid.uuid4().hex[:12]

        if self.project_manager is not None:
            self.project_manager.update_agent_status(
                target["project_id"],
                target["agent_id"],
                status="running",
                current_task=prompt.strip()[:200],
                task_id=task_id,
            )
        self._emit_event_threadsafe(
            "task_started",
            target=target,
            task_id=task_id,
            prompt=prompt,
        )

        task = loop.create_task(
            self._run_background_task(
                prompt=prompt,
                target=target,
                task_id=task_id,
            )
        )
        self._background_tasks[task_id] = task
        task.add_done_callback(lambda _t, tid=task_id: self._background_tasks.pop(tid, None))

        return {
            "status": "started",
            "task_id": task_id,
            "project_id": target["project_id"],
            "project_name": target["project_name"],
            "short_name": target["short_name"],
            "agent_id": target["agent_id"],
            "project_agent": target["project_agent"],
        }

    async def _run_background_task(
        self,
        *,
        prompt: str,
        target: dict[str, Any],
        task_id: str,
    ) -> None:
        try:
            result, swarm_nodes = await self._execute_agent_turn(
                prompt,
                target=target,
                task_id=task_id,
                emit_events=True,
                raise_on_sdk_error=True,
            )
            summary = summarize_for_speech(result.text, max_chars=450)
            final_conv_id = result.conversation_id or target.get("conversation_id")

            if self.project_manager is not None:
                self.project_manager.update_agent_status(
                    target["project_id"],
                    target["agent_id"],
                    status="idle",
                    current_task=None,
                    last_summary=summary or None,
                    swarm_nodes=swarm_nodes,
                    task_id=task_id,
                )

            await self._emit_event(
                "task_completed",
                target=target,
                task_id=task_id,
                conversation_id=final_conv_id,
                summary=summary,
                text=result.text,
                backend=result.backend,
            )

            if summary:
                try:
                    # Evaluated at completion time so room changes mid-task are honored.
                    active_room = self.cast_manager.active_room
                    await self.cast_manager.speak_on_room(
                        summary,
                        room_name=active_room,
                    )
                except Exception as speak_exc:
                    logger.warning(
                        "Background completion speech broadcast failed: %s",
                        speak_exc,
                    )
        except asyncio.CancelledError:
            if self.project_manager is not None:
                self.project_manager.update_agent_status(
                    target["project_id"],
                    target["agent_id"],
                    status="idle",
                    current_task=None,
                    task_id=task_id,
                )
            try:
                await self._emit_event(
                    "task_failed",
                    target=target,
                    task_id=task_id,
                    error="Task cancelled",
                )
            except Exception:
                pass
            raise
        except Exception as exc:
            logger.warning("Background agent task %s failed: %s", task_id, exc)
            if self.project_manager is not None:
                self.project_manager.update_agent_status(
                    target["project_id"],
                    target["agent_id"],
                    status="error",
                    current_task=None,
                    last_summary=f"Error: {exc}",
                    task_id=task_id,
                )
            await self._emit_event(
                "task_failed",
                target=target,
                task_id=task_id,
                error=str(exc),
            )

    async def shutdown(self) -> None:
        """Cancel and await all running background agent tasks."""
        tasks = list(self._background_tasks.values())
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._background_tasks.clear()

    @staticmethod
    def _record_swarm_node(
        swarm_map: dict[str, dict[str, Any]],
        node_id: str,
        *,
        parent_id: str | None = None,
        depth: int = 1,
        status: str = "completed",
        label: str | None = None,
    ) -> None:
        if not node_id:
            return
        existing = swarm_map.get(node_id) or {}
        existing_status = existing.get("status")
        next_status = status or existing_status or "completed"
        if existing_status in {"completed", "error"} and next_status == "running":
            next_status = existing_status
        swarm_map[node_id] = {
            "node_id": node_id,
            "parent_id": parent_id or existing.get("parent_id"),
            "depth": max(int(depth), int(existing.get("depth", 0))),
            "status": next_status,
            "label": label or existing.get("label") or node_id,
        }

    def _inspect_swarm_metadata(
        self,
        obj: Any,
        swarm_map: dict[str, dict[str, Any]],
        *,
        default_status: str = "running",
    ) -> None:
        if obj is None:
            return
        traj_id = getattr(obj, "trajectory_id", None)
        parent_traj_id = getattr(obj, "parent_trajectory_id", None)
        depth = getattr(obj, "depth", None)
        if traj_id and (parent_traj_id or (isinstance(depth, int) and depth > 0)):
            self._record_swarm_node(
                swarm_map,
                str(traj_id),
                parent_id=str(parent_traj_id) if parent_traj_id else None,
                depth=int(depth) if isinstance(depth, int) and depth > 0 else 1,
                status=str(getattr(obj, "status", default_status) or default_status),
            )

    async def _execute_agent_turn(
        self,
        prompt: str,
        *,
        target: dict[str, Any] | None = None,
        task_id: str | None = None,
        emit_events: bool = True,
        raise_on_sdk_error: bool = False,
    ) -> tuple[AgentTurnResult, list[dict[str, Any]] | None]:
        active_target = target or self._resolve_turn_target()
        active_task_id = task_id or uuid.uuid4().hex[:12]
        target_conv_id = active_target.get("conversation_id")
        target_ws_path = str(Path(active_target["path"]).resolve())
        sys_instructions = self._build_system_instructions(active_target)
        custom_tools = self.build_custom_tools(
            workspace_dir=target_ws_path,
            emit_events=emit_events,
        )

        # Tier 1: Official google-antigravity Python SDK
        try:
            from google.antigravity import Agent, CapabilitiesConfig, LocalAgentConfig  # type: ignore

            save_dir = str(Path(self.config.agy_save_dir).resolve())
            app_data_dir = str(Path(self.config.agy_app_data_dir).resolve())
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            Path(app_data_dir).mkdir(parents=True, exist_ok=True)

            kwargs: dict[str, Any] = {
                "system_instructions": sys_instructions,
                "tools": custom_tools,
                "save_dir": save_dir,
                # Without this the SDK only allows the process cwd, so the agent
                # could not reach the projects workspace.
                "workspaces": [target_ws_path],
                "app_data_dir": app_data_dir,
            }
            if self.config.gemini_api_key:
                kwargs["api_key"] = self.config.gemini_api_key
                if self.config.gemini_use_vertex:
                    kwargs["vertex"] = True
            if target_conv_id:
                kwargs["conversation_id"] = target_conv_id
            if self.config.agy_enable_write_tools:
                if self.config.agy_enable_sandbox:
                    try:
                        from google.antigravity import types as agy_types  # type: ignore

                        run_cfg_cls = getattr(agy_types, "RunCommandConfig", None)
                        if callable(run_cfg_cls):
                            kwargs["capabilities"] = CapabilitiesConfig(
                                run_command_config=run_cfg_cls(enable_sandbox=True)
                            )
                        else:
                            kwargs["capabilities"] = CapabilitiesConfig()
                    except Exception:
                        kwargs["capabilities"] = CapabilitiesConfig()
                else:
                    kwargs["capabilities"] = CapabilitiesConfig()

            emitted_tool_result_ids: set[str] = set()
            unnamed_call_steps: set[str] = set()
            named_calls_by_step: dict[str, set[str]] = {}
            emitted_calls_by_step: dict[str, set[str]] = {}
            swarm_map: dict[str, dict[str, Any]] = {}
            pending_unnamed_subagents: list[str] = []
            pending_subagents_by_step: dict[str, list[str]] = {}
            named_subagents_by_step: dict[str, list[str]] = {}
            unnamed_subagent_seq = 0
            live_conv_id: str | None = target_conv_id

            def _claim_tool_result_emit(call_id: Any, step_id: Any) -> bool:
                c_key = f"call:{call_id}" if call_id not in (None, "") else None
                step_str = str(step_id) if step_id not in (None, "") else None
                s_key = f"step:{step_str}" if step_str is not None else None
                has_unnamed_call = step_str is not None and step_str in unnamed_call_steps
                known_named_calls = (
                    named_calls_by_step.get(step_str, ()) if step_str is not None else ()
                )
                emitted_named_calls = (
                    emitted_calls_by_step.get(step_str, ()) if step_str is not None else ()
                )
                if c_key is not None:
                    if c_key in emitted_tool_result_ids:
                        return False
                    if (
                        s_key is not None
                        and s_key in emitted_tool_result_ids
                        and (
                            has_unnamed_call
                            or (len(known_named_calls) <= 1 and not emitted_named_calls)
                        )
                    ):
                        return False
                    emitted_tool_result_ids.add(c_key)
                    if step_str is not None:
                        emitted_calls_by_step.setdefault(step_str, set()).add(
                            str(call_id)
                        )
                        if s_key is not None and has_unnamed_call:
                            emitted_tool_result_ids.add(s_key)
                    return True
                if s_key is not None:
                    if s_key in emitted_tool_result_ids:
                        return False
                    if (
                        not has_unnamed_call
                        and len(known_named_calls) <= 1
                        and len(emitted_named_calls) == 1
                    ):
                        emitted_tool_result_ids.add(s_key)
                        return False
                    emitted_tool_result_ids.add(s_key)
                    return True
                return True

            def _remove_subagent_tracking(node_id: str) -> None:
                if node_id in pending_unnamed_subagents:
                    pending_unnamed_subagents.remove(node_id)
                for mapping in (pending_subagents_by_step, named_subagents_by_step):
                    for k, q_list in list(mapping.items()):
                        if node_id in q_list:
                            q_list.remove(node_id)
                            if not q_list:
                                del mapping[k]

            def _next_subagent_node_id(
                call_id: Any,
                step_id: Any = None,
                *,
                is_start: bool,
                create_if_missing: bool = True,
            ) -> str | None:
                nonlocal unnamed_subagent_seq
                step_str = str(step_id) if step_id not in (None, "") else None
                if call_id not in (None, ""):
                    call_str = str(call_id)
                    if is_start:
                        if step_str is not None and call_str not in swarm_map:
                            named_subagents_by_step.setdefault(step_str, []).append(
                                call_str
                            )
                        return call_str
                    if (
                        call_str not in swarm_map
                        and step_str is not None
                        and pending_subagents_by_step.get(step_str)
                    ):
                        node_id = pending_subagents_by_step[step_str].pop(0)
                        if not pending_subagents_by_step[step_str]:
                            del pending_subagents_by_step[step_str]
                        if node_id in pending_unnamed_subagents:
                            pending_unnamed_subagents.remove(node_id)
                        return node_id
                    return call_str
                if is_start:
                    unnamed_subagent_seq += 1
                    node_id = f"subagent-{unnamed_subagent_seq}"
                    pending_unnamed_subagents.append(node_id)
                    if step_str is not None:
                        pending_subagents_by_step.setdefault(step_str, []).append(
                            node_id
                        )
                    return node_id
                if step_str is not None and pending_subagents_by_step.get(step_str):
                    node_id = pending_subagents_by_step[step_str].pop(0)
                    if not pending_subagents_by_step[step_str]:
                        del pending_subagents_by_step[step_str]
                    if node_id in pending_unnamed_subagents:
                        pending_unnamed_subagents.remove(node_id)
                    return node_id
                if step_str is not None and named_subagents_by_step.get(step_str):
                    return named_subagents_by_step[step_str].pop(0)
                if pending_unnamed_subagents:
                    node_id = pending_unnamed_subagents.pop(0)
                    _remove_subagent_tracking(node_id)
                    return node_id
                if not create_if_missing:
                    return None
                unnamed_subagent_seq += 1
                return f"subagent-{unnamed_subagent_seq}"

            try:
                import google.antigravity.hooks as hooks_pkg  # type: ignore

                policy_mod = getattr(hooks_pkg, "policy", None)
                if policy_mod is not None:
                    if self.config.agy_allow_all_commands and callable(
                        getattr(policy_mod, "allow_all", None)
                    ):
                        kwargs["policies"] = [policy_mod.allow_all()]
                    elif self.config.agy_enable_write_tools and callable(
                        getattr(policy_mod, "confirm_run_command", None)
                    ):
                        confirm_pol = policy_mod.confirm_run_command()
                        kwargs["policies"] = (
                            list(confirm_pol)
                            if isinstance(confirm_pol, (list, tuple))
                            else [confirm_pol]
                        )

                sdk_hooks = getattr(hooks_pkg, "hooks", None)
                if sdk_hooks is not None and callable(
                    getattr(sdk_hooks, "post_tool_call", None)
                ):
                    registered_hooks: list[Any] = []

                    @sdk_hooks.post_tool_call
                    async def _on_post_tool_call(data: Any) -> None:
                        tool_name = str(getattr(data, "name", "") or "tool")
                        call_id = getattr(data, "id", None)
                        step_id = getattr(data, "step_id", None)
                        if not _claim_tool_result_emit(call_id, step_id):
                            return
                        result_val = getattr(data, "result", None)
                        err_val = getattr(data, "error", None) or getattr(
                            data, "exception", None
                        )
                        if tool_name == "start_subagent":
                            sub_node_id = _next_subagent_node_id(
                                call_id, step_id, is_start=False
                            )
                            if sub_node_id is not None:
                                self._record_swarm_node(
                                    swarm_map,
                                    sub_node_id,
                                    status="error" if err_val else "completed",
                                    label=None if sub_node_id in swarm_map else "subagent",
                                )
                        if emit_events:
                            await self._emit_event(
                                "tool_result",
                                target=active_target,
                                task_id=active_task_id,
                                conversation_id=live_conv_id,
                                name=tool_name,
                                tool_call_id=call_id,
                                step_id=step_id,
                                result=_safe_json_value(result_val),
                                error=_safe_json_value(err_val)
                                if err_val is not None
                                else None,
                            )

                    registered_hooks.append(_on_post_tool_call)

                    if callable(getattr(sdk_hooks, "on_tool_error", None)):
                        @sdk_hooks.on_tool_error
                        async def _on_tool_error(data: Any) -> None:
                            tool_name = str(getattr(data, "name", "") or "tool")
                            call_id = getattr(data, "id", None)
                            step_id = getattr(data, "step_id", None)
                            if not _claim_tool_result_emit(call_id, step_id):
                                return None
                            err_val = (
                                getattr(data, "error", None)
                                or getattr(data, "exception", None)
                                or data
                            )
                            if tool_name == "start_subagent":
                                sub_node_id = _next_subagent_node_id(
                                    call_id, step_id, is_start=False
                                )
                                if sub_node_id is not None:
                                    self._record_swarm_node(
                                        swarm_map,
                                        sub_node_id,
                                        status="error",
                                        label=None if sub_node_id in swarm_map else "subagent",
                                    )
                            if emit_events:
                                await self._emit_event(
                                    "tool_result",
                                    target=active_target,
                                    task_id=active_task_id,
                                    conversation_id=live_conv_id,
                                    name=tool_name,
                                    tool_call_id=call_id,
                                    step_id=step_id,
                                    result=None,
                                    error=_safe_json_value(err_val),
                                )
                            return None

                        registered_hooks.append(_on_tool_error)

                    kwargs["hooks"] = registered_hooks
            except ImportError:
                pass

            ag_config = LocalAgentConfig(**kwargs)
            chunks: list[str] = []
            thoughts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            async with Agent(ag_config) as agent:
                response = await agent.chat(prompt)
                live_conv_id = getattr(agent, "conversation_id", None) or live_conv_id
                chunks_attr = getattr(response, "chunks", None)
                stream = (
                    chunks_attr
                    if (chunks_attr is not None and hasattr(chunks_attr, "__aiter__"))
                    else response
                )
                async for chunk in stream:
                    self._inspect_swarm_metadata(chunk, swarm_map)
                    chunk_type = type(chunk).__name__

                    if isinstance(chunk, str):
                        chunks.append(chunk)
                        if emit_events and chunk:
                            await self._emit_event(
                                "token",
                                target=active_target,
                                task_id=active_task_id,
                                conversation_id=live_conv_id,
                                text=chunk,
                            )
                    elif chunk_type == "Thought":
                        thought_text = str(
                            getattr(chunk, "text", None)
                            or getattr(chunk, "content", None)
                            or ""
                        )
                        if thought_text:
                            thoughts.append(thought_text)
                            if emit_events:
                                await self._emit_event(
                                    "thought",
                                    target=active_target,
                                    task_id=active_task_id,
                                    conversation_id=live_conv_id,
                                    text=thought_text,
                                )
                    elif chunk_type == "ToolCall" or (
                        hasattr(chunk, "name")
                        and hasattr(chunk, "args")
                        and chunk_type != "ToolResult"
                    ):
                        tool_name = str(getattr(chunk, "name", "tool"))
                        tool_args = _safe_json_value(getattr(chunk, "args", {}) or {})
                        call_id = getattr(chunk, "id", None)
                        step_id = getattr(chunk, "step_id", None)
                        if step_id not in (None, ""):
                            if call_id not in (None, ""):
                                named_calls_by_step.setdefault(
                                    str(step_id), set()
                                ).add(str(call_id))
                            else:
                                unnamed_call_steps.add(str(step_id))
                        tool_calls.append(
                            {
                                "id": call_id,
                                "name": tool_name,
                                "args": tool_args,
                            }
                        )
                        sub_node_id: str | None = None
                        if tool_name == "start_subagent":
                            sub_node_id = _next_subagent_node_id(
                                call_id, step_id, is_start=True
                            )
                            assert sub_node_id is not None
                            self._record_swarm_node(
                                swarm_map,
                                sub_node_id,
                                status="running",
                                label=str(
                                    tool_args.get("agent_name")
                                    or tool_args.get("name")
                                    or "subagent"
                                )
                                if isinstance(tool_args, dict)
                                else "subagent",
                            )
                        if emit_events:
                            await self._emit_event(
                                "tool_call",
                                target=active_target,
                                task_id=active_task_id,
                                conversation_id=live_conv_id,
                                name=tool_name,
                                args=tool_args,
                                tool_call_id=call_id,
                                step_id=step_id,
                            )
                        inline_result = getattr(chunk, "result", None)
                        inline_err = getattr(chunk, "error", None)
                        if inline_result is not None or inline_err is not None:
                            if tool_name == "start_subagent" and sub_node_id is not None:
                                _remove_subagent_tracking(sub_node_id)
                                self._record_swarm_node(
                                    swarm_map,
                                    sub_node_id,
                                    status="error" if inline_err else "completed",
                                    label=None if sub_node_id in swarm_map else "subagent",
                                )
                            should_emit = _claim_tool_result_emit(call_id, step_id)
                            if should_emit and emit_events:
                                await self._emit_event(
                                    "tool_result",
                                    target=active_target,
                                    task_id=active_task_id,
                                    conversation_id=live_conv_id,
                                    name=tool_name,
                                    tool_call_id=call_id,
                                    step_id=step_id,
                                    result=_safe_json_value(inline_result),
                                    error=_safe_json_value(inline_err)
                                    if inline_err is not None
                                    else None,
                                )
                    elif chunk_type == "ToolResult":
                        tool_name = str(getattr(chunk, "name", "tool"))
                        call_id = getattr(chunk, "id", None)
                        step_id = getattr(chunk, "step_id", None)
                        err_val = getattr(chunk, "error", None) or getattr(
                            chunk, "exception", None
                        )
                        should_emit = _claim_tool_result_emit(call_id, step_id)
                        if should_emit and tool_name == "start_subagent":
                            sub_node_id = _next_subagent_node_id(
                                call_id,
                                step_id,
                                is_start=False,
                                create_if_missing=False,
                            )
                            if sub_node_id is not None:
                                self._record_swarm_node(
                                    swarm_map,
                                    sub_node_id,
                                    status="error" if err_val else "completed",
                                    label=None if sub_node_id in swarm_map else "subagent",
                                )
                        if should_emit and emit_events:
                            await self._emit_event(
                                "tool_result",
                                target=active_target,
                                task_id=active_task_id,
                                conversation_id=live_conv_id,
                                name=tool_name,
                                tool_call_id=call_id,
                                step_id=step_id,
                                result=_safe_json_value(
                                    getattr(chunk, "result", None)
                                ),
                                error=_safe_json_value(err_val)
                                if err_val is not None
                                else None,
                            )
                    else:
                        tok_text = getattr(chunk, "text", None)
                        if isinstance(tok_text, str) and tok_text:
                            chunks.append(tok_text)
                            if emit_events:
                                await self._emit_event(
                                    "token",
                                    target=active_target,
                                    task_id=active_task_id,
                                    conversation_id=live_conv_id,
                                    text=tok_text,
                                )

                # Inspect already-collected steps without double-consuming receive_steps().
                conv_obj = getattr(agent, "conversation", None)
                recorded_steps = (
                    getattr(conv_obj, "steps", None)
                    or getattr(conv_obj, "_steps", None)
                    or []
                )
                if isinstance(recorded_steps, list):
                    for step in recorded_steps:
                        self._inspect_swarm_metadata(
                            step, swarm_map, default_status="completed"
                        )

                final_conv_id = getattr(agent, "conversation_id", None) or live_conv_id
                if self.project_manager is not None:
                    if final_conv_id:
                        self.project_manager.set_conversation_id(
                            active_target["project_id"],
                            active_target["agent_id"],
                            final_conv_id,
                        )
                else:
                    self.conversation_id = final_conv_id
                    self._save_persisted_conversation_id(self.conversation_id)

            swarm_nodes = list(swarm_map.values()) if swarm_map else None
            return (
                AgentTurnResult(
                    text="".join(chunks).strip(),
                    conversation_id=final_conv_id,
                    thoughts=thoughts,
                    tool_calls=tool_calls,
                    backend="google-antigravity-sdk",
                ),
                swarm_nodes,
            )
        except ImportError:
            logger.debug(
                "google-antigravity SDK package not installed in current venv; checking fallback backends."
            )
        except Exception as exc:
            logger.warning("google-antigravity SDK execution error: %s", exc)
            if raise_on_sdk_error:
                raise

        # Tier 2: google-genai with our custom Google Home & Homelab tools
        if self.config.gemini_api_key or os.environ.get("GEMINI_API_KEY"):
            try:
                from google import genai
                from google.genai import types as genai_types

                api_key = self.config.gemini_api_key or os.environ.get(
                    "GEMINI_API_KEY", ""
                )
                client = genai.Client(
                    api_key=api_key,
                    vertexai=self.config.gemini_use_vertex,
                )

                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        system_instruction=sys_instructions,
                        tools=custom_tools,
                    ),
                )
                out_text = (response.text or "").strip()
                if emit_events and out_text:
                    await self._emit_event(
                        "token",
                        target=active_target,
                        task_id=active_task_id,
                        conversation_id=target_conv_id,
                        text=out_text,
                    )
                return (
                    AgentTurnResult(
                        text=out_text,
                        conversation_id=target_conv_id or "genai-live-session",
                        backend="google-genai",
                    ),
                    None,
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
                out_text = f"Started Antigravity conversation {conv_id} for: {prompt}"
                if emit_events:
                    await self._emit_event(
                        "token",
                        target=active_target,
                        task_id=active_task_id,
                        conversation_id=conv_id,
                        text=out_text,
                    )
                return (
                    AgentTurnResult(
                        text=out_text,
                        conversation_id=conv_id,
                        backend="antigravity-ls-conversation-service",
                    ),
                    None,
                )
            except Exception as exc:
                logger.warning("Antigravity LS ConversationService error: %s", exc)

        # Offline / No-Key Diagnostic Response (still executes local voice room-switching keywords)
        lower = prompt.lower()
        for tool_fn in custom_tools:
            if tool_fn.__name__ == "switch_active_google_home_room" and (
                "switch" in lower or "follow me" in lower
            ):
                for candidate in ("kitchen", "office", "living room", "bedroom", "whole home"):
                    if candidate in lower:
                        if emit_events:
                            await self._emit_event(
                                "tool_call",
                                target=active_target,
                                task_id=active_task_id,
                                conversation_id=target_conv_id,
                                name="switch_active_google_home_room",
                                args={"room": candidate.title()},
                            )
                        msg = tool_fn(candidate.title())
                        if emit_events:
                            await self._emit_event(
                                "tool_result",
                                target=active_target,
                                task_id=active_task_id,
                                conversation_id=target_conv_id,
                                name="switch_active_google_home_room",
                                result=msg,
                            )
                            await self._emit_event(
                                "token",
                                target=active_target,
                                task_id=active_task_id,
                                conversation_id=target_conv_id,
                                text=msg,
                            )
                        return (
                            AgentTurnResult(
                                text=msg,
                                conversation_id=target_conv_id or "local-offline",
                                tool_calls=[
                                    {
                                        "name": "switch_active_google_home_room",
                                        "args": {"room": candidate.title()},
                                    }
                                ],
                                backend="local-fallback",
                            ),
                            None,
                        )

        fallback_text = (
            f"Antigravity Home Bridge is online (Active Google Home speaker: "
            f"{self.cast_manager.active_room}). Set GEMINI_API_KEY in .env to enable "
            f"live Gemini and Google Antigravity SDK reasoning. You said: {prompt}"
        )
        if emit_events:
            await self._emit_event(
                "token",
                target=active_target,
                task_id=active_task_id,
                conversation_id=target_conv_id,
                text=fallback_text,
            )
        return (
            AgentTurnResult(
                text=fallback_text,
                conversation_id=target_conv_id or "local-offline",
                backend="local-fallback",
            ),
            None,
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
