"""Thread- and asyncio-safe live event broadcast hub and snapshot builder for JARVIS mode."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any


def safe_json_value(
    value: Any,
    max_str_len: int = 2000,
    *,
    max_depth: int = 8,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    """Convert arbitrary SDK objects/values into bounded JSON-serializable primitives."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        if len(value) > max_str_len:
            return f"{value[:max_str_len]}..."
        return value
    if _depth >= max_depth:
        return "<max depth exceeded>"

    if isinstance(value, (list, tuple, set, frozenset)):
        seen = _seen if _seen is not None else set()
        obj_id = id(value)
        if obj_id in seen:
            return "<circular reference>"
        seen.add(obj_id)
        try:
            items = list(value)[:50]
            return [
                safe_json_value(
                    item,
                    max_str_len=max_str_len,
                    max_depth=max_depth,
                    _seen=seen,
                    _depth=_depth + 1,
                )
                for item in items
            ]
        finally:
            seen.discard(obj_id)

    if isinstance(value, dict):
        seen = _seen if _seen is not None else set()
        obj_id = id(value)
        if obj_id in seen:
            return "<circular reference>"
        seen.add(obj_id)
        try:
            out: dict[str, Any] = {}
            for k, v in list(value.items())[:50]:
                try:
                    key_str = k if isinstance(k, str) else str(k)
                except Exception:
                    key_str = f"<{type(k).__name__}>"
                if len(key_str) > max_str_len:
                    key_str = f"{key_str[:max_str_len]}..."
                out[key_str] = safe_json_value(
                    v,
                    max_str_len=max_str_len,
                    max_depth=max_depth,
                    _seen=seen,
                    _depth=_depth + 1,
                )
            return out
        finally:
            seen.discard(obj_id)

    try:
        text = str(value)
    except Exception:
        text = f"<{type(value).__name__}>"
    if len(text) > max_str_len:
        return f"{text[:max_str_len]}..."
    return text


def _sanitize_for_json(
    value: Any,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
    max_depth: int = 32,
) -> Any:
    """Recursively convert circular refs, NaN/Inf, non-string dict keys, and custom objects for JSON."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, str):
        return value
    if _depth >= max_depth:
        return "<max depth exceeded>"

    if isinstance(value, (list, tuple, set, frozenset)):
        seen = _seen if _seen is not None else set()
        obj_id = id(value)
        if obj_id in seen:
            return "<circular reference>"
        seen.add(obj_id)
        try:
            return [
                _sanitize_for_json(
                    item, _seen=seen, _depth=_depth + 1, max_depth=max_depth
                )
                for item in value
            ]
        finally:
            seen.discard(obj_id)

    if isinstance(value, dict):
        seen = _seen if _seen is not None else set()
        obj_id = id(value)
        if obj_id in seen:
            return "<circular reference>"
        seen.add(obj_id)
        try:
            out: dict[str, Any] = {}
            for k, v in value.items():
                try:
                    key_str = k if isinstance(k, str) else str(k)
                except Exception:
                    key_str = f"<{type(k).__name__}>"
                out[key_str] = _sanitize_for_json(
                    v, _seen=seen, _depth=_depth + 1, max_depth=max_depth
                )
            return out
        finally:
            seen.discard(obj_id)

    try:
        return str(value)
    except Exception:
        return f"<{type(value).__name__}>"


def safe_json_dumps(obj: Any) -> str:
    """Serialize `obj` to strict RFC-8259 JSON, safe against circular refs, NaN/Inf, and non-str keys."""
    try:
        return json.dumps(obj, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        sanitized = _sanitize_for_json(obj)
        return json.dumps(sanitized, allow_nan=False)

KNOWN_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "thought",
        "tool_call",
        "tool_result",
        "token",
        "transcript",
        "task_started",
        "task_completed",
        "task_failed",
        "project_changed",
        "room_changed",
        "voice_status",
    }
)


class EventHub:
    """Bounded async/threadsafe pub-sub hub for JARVIS HUD and live WebSocket streams."""

    def __init__(self, maxlen: int = 200, subscriber_queue_size: int = 256) -> None:
        self._maxlen = max(1, int(maxlen))
        self._subscriber_queue_size = max(1, int(subscriber_queue_size))
        self._history: deque[dict[str, Any]] = deque(maxlen=self._maxlen)
        self._subscribers: dict[
            asyncio.Queue[dict[str, Any]], asyncio.AbstractEventLoop | None
        ] = {}
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a subscriber queue bound to the current event loop (if running)."""
        try:
            loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(
            maxsize=self._subscriber_queue_size
        )
        with self._lock:
            self._subscribers[queue] = loop
            if loop is not None:
                self._loop = loop
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove a subscriber queue (idempotent)."""
        with self._lock:
            self._subscribers.pop(queue, None)

    def _build_event(
        self,
        event_type: str,
        *,
        project_id: str | None = None,
        project_name: str | None = None,
        short_name: str | None = None,
        agent_id: str | None = None,
        project_agent: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        normalized_type = str(event_type).strip()
        if not normalized_type:
            raise ValueError("event_type must be a non-empty string")

        return {
            **payload,
            "type": normalized_type,
            "event_id": uuid.uuid4().hex,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_id": project_id,
            "project_name": project_name,
            "short_name": short_name,
            "agent_id": agent_id,
            "project_agent": project_agent,
            "conversation_id": conversation_id,
            "task_id": task_id,
        }

    def _deliver_to_queue(
        self, queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]
    ) -> None:
        try:
            queue.put_nowait(dict(event))
        except Exception:
            # Immediate eviction on QueueFull or closed/broken queue so slow consumers never block.
            with self._lock:
                self._subscribers.pop(queue, None)

    async def publish(
        self,
        event_type: str,
        *,
        project_id: str | None = None,
        project_name: str | None = None,
        short_name: str | None = None,
        agent_id: str | None = None,
        project_agent: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        """Append event to bounded history and broadcast non-blockingly to all subscribers."""
        running_loop = asyncio.get_running_loop()
        event = self._build_event(
            event_type,
            project_id=project_id,
            project_name=project_name,
            short_name=short_name,
            agent_id=agent_id,
            project_agent=project_agent,
            conversation_id=conversation_id,
            task_id=task_id,
            **payload,
        )

        with self._lock:
            self._loop = running_loop
            self._history.append(event)
            targets = list(self._subscribers.items())

        for queue, sub_loop in targets:
            if sub_loop is None or sub_loop is running_loop:
                self._deliver_to_queue(queue, event)
            elif not sub_loop.is_closed():
                try:
                    sub_loop.call_soon_threadsafe(self._deliver_to_queue, queue, event)
                except RuntimeError:
                    with self._lock:
                        self._subscribers.pop(queue, None)
            else:
                with self._lock:
                    self._subscribers.pop(queue, None)

        return dict(event)

    def publish_threadsafe(
        self,
        event_type: str,
        *,
        project_id: str | None = None,
        project_name: str | None = None,
        short_name: str | None = None,
        agent_id: str | None = None,
        project_agent: str | None = None,
        conversation_id: str | None = None,
        task_id: str | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        """Publish an event safely from synchronous callbacks or worker threads."""
        event = self._build_event(
            event_type,
            project_id=project_id,
            project_name=project_name,
            short_name=short_name,
            agent_id=agent_id,
            project_agent=project_agent,
            conversation_id=conversation_id,
            task_id=task_id,
            **payload,
        )

        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        with self._lock:
            self._history.append(event)
            fallback_loop = self._loop
            targets = list(self._subscribers.items())

        for queue, sub_loop in targets:
            target_loop = sub_loop or fallback_loop
            if target_loop is None or target_loop is current_loop:
                self._deliver_to_queue(queue, event)
            elif not target_loop.is_closed():
                try:
                    target_loop.call_soon_threadsafe(self._deliver_to_queue, queue, event)
                except RuntimeError:
                    with self._lock:
                        self._subscribers.pop(queue, None)
            else:
                with self._lock:
                    self._subscribers.pop(queue, None)

        return dict(event)

    def recent_events(
        self,
        limit: int = 100,
        *,
        project_id: str | None = None,
        agent_id: str | None = None,
        event_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent chronological events, optionally filtered by project/agent/type."""
        if limit <= 0:
            return []
        with self._lock:
            items = list(self._history)

        if project_id is not None:
            items = [e for e in items if e.get("project_id") == project_id]
        if agent_id is not None:
            items = [e for e in items if e.get("agent_id") == agent_id]
        if event_type is not None:
            items = [e for e in items if e.get("type") == event_type]

        return [dict(e) for e in items[-limit:]]

    def clear_history(self) -> None:
        """Clear the bounded event history buffer."""
        with self._lock:
            self._history.clear()

    def build_snapshot(
        self,
        project_manager: Any = None,
        cast_manager: Any = None,
        voice_status: str = "idle",
        voice_muted: bool = False,
        *,
        event_limit: int = 100,
    ) -> dict[str, Any]:
        """Build an initial `hud_snapshot` payload resilient to missing or partial managers."""
        if cast_manager is not None:
            active_room = getattr(cast_manager, "active_room", None)
            active_presence_room = getattr(
                cast_manager, "active_presence_room", active_room
            )
            active_speaker = getattr(cast_manager, "active_speaker", None) or active_room
            active_tv = getattr(cast_manager, "active_tv", None)
            routing_mode = getattr(cast_manager, "routing_mode", None)
        else:
            active_room = None
            active_presence_room = None
            active_speaker = None
            active_tv = None
            routing_mode = None

        summary: dict[str, Any] = {}
        if project_manager is not None and hasattr(project_manager, "get_state_summary"):
            summary = project_manager.get_state_summary() or {}

        return {
            "type": "hud_snapshot",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_room": active_room,
            "active_presence_room": active_presence_room,
            "active_speaker": active_speaker,
            "active_tv": active_tv,
            "routing_mode": routing_mode,
            "voice_status": "muted" if voice_muted else voice_status,
            "voice_muted": bool(voice_muted),
            "projects": list(summary.get("projects") or []),
            "active_project_id": summary.get("active_project_id"),
            "active_agent_id": summary.get("active_agent_id"),
            "active_project_agent": summary.get("active_project_agent"),
            "recent_events": self.recent_events(limit=event_limit),
        }
