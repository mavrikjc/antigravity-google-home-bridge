"""Opaque-box End-to-End (E2E) test suite for JARVIS Mode on antigravity-google-home-bridge.

Structured across four rigor tiers:
- Tier 1 (45 tests): Primary feature coverage for F1-F9.
- Tier 2 (45 tests): Boundary, corner-case, error-path, and adversarial input tests for F1-F9.
- Tier 3 (12 tests): Cross-feature pairwise integration tests (T3-01 .. T3-12).
- Tier 4 (6 tests): Real-world multi-step JARVIS application scenarios (T4-01 .. T4-06).

All tests are strictly opaque-box against HTTP REST endpoints, ASGI WebSockets
(`/ws/hud`, `/ws/live`), and external SDK/device boundaries (`google.antigravity`,
`google.genai.Client`, `pychromecast`, `DashCastController`, `gTTS`).
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import json
import re
import subprocess
import sys
import time
import types
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from weakref import WeakKeyDictionary

import httpx
import pytest
from fastapi import WebSocketDisconnect
from fastapi.testclient import TestClient

from agy_home_bridge.config import BridgeSettings
from agy_home_bridge.server import create_app


# ============================================================================
# Shared Helpers (Opaque-Box WS, Git, HTML, EventHub)
# ============================================================================
def _init_git_repo(
    repo_dir: Path,
    *,
    branch: str = "main",
    dirty_modified: int = 0,
    dirty_untracked: int = 0,
    dirty_staged: int = 0,
) -> Path:
    """Create a real isolated git repository in tmp_path with controlled state."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "JARVIS E2E"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "jarvis-e2e@example.local"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "checkout", "-b", branch],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )
    readme = repo_dir / "README.md"
    readme.write_text(f"# {repo_dir.name}\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repo_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial commit"],
        cwd=repo_dir,
        check=True,
        capture_output=True,
    )

    for idx in range(dirty_modified):
        target = repo_dir / f"tracked_{idx}.txt"
        target.write_text(f"base {idx}\n", encoding="utf-8")
        subprocess.run(["git", "add", target.name], cwd=repo_dir, check=True, capture_output=True)
    if dirty_modified:
        subprocess.run(
            ["git", "commit", "-m", "add tracked files"],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        )
        for idx in range(dirty_modified):
            (repo_dir / f"tracked_{idx}.txt").write_text(
                f"modified {idx}\n", encoding="utf-8"
            )

    for idx in range(dirty_untracked):
        (repo_dir / f"untracked_{idx}.txt").write_text(
            f"untracked {idx}\n", encoding="utf-8"
        )

    for idx in range(dirty_staged):
        staged = repo_dir / f"staged_{idx}.txt"
        staged.write_text(f"staged {idx}\n", encoding="utf-8")
        subprocess.run(["git", "add", staged.name], cwd=repo_dir, check=True, capture_output=True)

    return repo_dir


def recv_json_with_timeout(ws: Any, timeout: float = 3.0) -> dict[str, Any]:
    """Receive one JSON frame from a Starlette WebSocketTestSession with a hard timeout."""
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(ws.receive_json)
    try:
        return fut.result(timeout=timeout)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def recv_until_type(
    ws: Any,
    target_type: str | set[str],
    *,
    timeout: float = 4.0,
    max_frames: int = 40,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read WebSocket frames until a matching `type` arrives; returns (matched, collected)."""
    targets = {target_type} if isinstance(target_type, str) else set(target_type)
    collected: list[dict[str, Any]] = []
    for _ in range(max_frames):
        frame = recv_json_with_timeout(ws, timeout=timeout)
        collected.append(frame)
        if frame.get("type") in targets:
            return frame, collected
    raise AssertionError(
        f"Did not receive event type(s) {sorted(targets)} within {max_frames} frames; "
        f"got {[f.get('type') for f in collected]}"
    )


def collect_until_types(
    ws: Any,
    required_types: set[str],
    *,
    timeout: float = 4.0,
    max_frames: int = 60,
) -> list[dict[str, Any]]:
    """Collect frames until every type in `required_types` has been observed at least once."""
    seen: set[str] = set()
    frames: list[dict[str, Any]] = []
    for _ in range(max_frames):
        frame = recv_json_with_timeout(ws, timeout=timeout)
        frames.append(frame)
        ftype = frame.get("type")
        if isinstance(ftype, str):
            seen.add(ftype)
        if required_types.issubset(seen):
            return frames
    raise AssertionError(
        f"Missing expected event types {sorted(required_types - seen)}; "
        f"received {[f.get('type') for f in frames]}"
    )


_HUB_UNCONSUMED: WeakKeyDictionary[asyncio.Queue[dict[str, Any]], list[dict[str, Any]]] = WeakKeyDictionary()
_HUB_SEEN: WeakKeyDictionary[asyncio.Queue[dict[str, Any]], list[dict[str, Any]]] = WeakKeyDictionary()


async def wait_for_hub_event(
    queue: asyncio.Queue[dict[str, Any]],
    target_type: str,
    *,
    predicate: Any | None = None,
    timeout: float = 4.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Wait on an EventHub subscriber queue for a matching event without dropping out-of-order events."""
    unconsumed = _HUB_UNCONSUMED.setdefault(queue, [])
    all_seen = _HUB_SEEN.setdefault(queue, [])

    # Check previously buffered out-of-order events first
    for idx, ev in enumerate(unconsumed):
        if ev.get("type") == target_type and (predicate is None or predicate(ev)):
            unconsumed.pop(idx)
            return ev, list(all_seen)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise AssertionError(
                f"Timed out waiting for hub event '{target_type}'; "
                f"observed {[e.get('type') for e in all_seen]}"
            )
        try:
            ev = await asyncio.wait_for(queue.get(), timeout=remaining)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise AssertionError(
                f"Timed out waiting for hub event '{target_type}'; "
                f"observed {[e.get('type') for e in all_seen]}"
            ) from exc
        all_seen.append(ev)
        if ev.get("type") == target_type and (predicate is None or predicate(ev)):
            return ev, list(all_seen)
        unconsumed.append(ev)


class FakeWebSocket:
    """Minimal ASGI WebSocket double for single-loop async FullDuplexLiveBridge tests."""

    def __init__(self) -> None:
        self.accepted = False
        self.closed = False
        self.close_code: int | None = None
        self.sent: list[dict[str, Any]] = []
        self._consumed_indices: set[int] = set()
        self._cond: asyncio.Condition | None = None
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()

    def _cond_obj(self) -> asyncio.Condition:
        if self._cond is None:
            self._cond = asyncio.Condition()
        return self._cond

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, data: str) -> None:
        payload = json.loads(data)
        self.sent.append(payload)
        cond = self._cond_obj()
        async with cond:
            cond.notify_all()

    async def send_json(self, payload: dict[str, Any]) -> None:
        self.sent.append(payload)
        cond = self._cond_obj()
        async with cond:
            cond.notify_all()

    async def receive_text(self) -> str:
        item = await self.incoming.get()
        if item is None:
            raise WebSocketDisconnect(code=1000)
        return item

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = True
        self.close_code = code
        await self.incoming.put(None)

    def push_client_json(self, msg: dict[str, Any]) -> None:
        self.incoming.put_nowait(json.dumps(msg))

    def disconnect(self) -> None:
        self.incoming.put_nowait(None)

    async def wait_for_sent(
        self,
        target_type: str,
        *,
        predicate: Any | None = None,
        timeout: float = 4.0,
    ) -> dict[str, Any]:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        cond = self._cond_obj()
        async with cond:
            while True:
                for idx, msg in enumerate(self.sent):
                    if idx in self._consumed_indices:
                        continue
                    if msg.get("type") == target_type and (predicate is None or predicate(msg)):
                        self._consumed_indices.add(idx)
                        return msg
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise AssertionError(
                        f"Timed out waiting for WS frame '{target_type}'; "
                        f"sent={[m.get('type') for m in self.sent]}"
                    )
                try:
                    await asyncio.wait_for(cond.wait(), timeout=remaining)
                except (TimeoutError, asyncio.TimeoutError) as exc:
                    raise AssertionError(
                        f"Timed out waiting for WS frame '{target_type}'; "
                        f"sent={[m.get('type') for m in self.sent]}"
                    ) from exc


class _HTMLInspector(HTMLParser):
    """Collect tags, attributes, visibility, and element text for opaque-box HTML verification."""

    _VOID_TAGS = frozenset(
        {
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "link",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
        }
    )

    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self._stack: list[tuple[str, dict[str, str | None], list[str], bool]] = []
        self.elements: list[tuple[str, dict[str, str | None], str, bool]] = []

    @staticmethod
    def _is_hidden_attrs(attrs_dict: dict[str, str | None]) -> bool:
        if "hidden" in attrs_dict:
            return True
        if (attrs_dict.get("aria-hidden") or "").strip().lower() == "true":
            return True
        style = (attrs_dict.get("style") or "").replace(" ", "").lower()
        return "display:none" in style or "visibility:hidden" in style

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        attrs_dict = {k.lower(): v for k, v in attrs}
        self.tags.append((t, attrs_dict))
        if t in self._VOID_TAGS:
            return
        parent_hidden = any(item[3] for item in self._stack)
        is_hidden = parent_hidden or self._is_hidden_attrs(attrs_dict)
        self._stack.append((t, attrs_dict, [], is_hidden))

    def handle_data(self, data: str) -> None:
        for item in self._stack:
            item[2].append(data)

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        for idx in range(len(self._stack) - 1, -1, -1):
            if self._stack[idx][0] == t:
                name, attrs_dict, chunks, is_hidden = self._stack.pop(idx)
                self.elements.append((name, attrs_dict, "".join(chunks), is_hidden))
                break

    def close(self) -> None:
        super().close()
        while self._stack:
            name, attrs_dict, chunks, is_hidden = self._stack.pop()
            self.elements.append((name, attrs_dict, "".join(chunks), is_hidden))

    def has_visible_button_with_text(self, needle: str) -> bool:
        target = needle.lower()
        for tag, attrs, text, is_hidden in self.elements:
            if tag == "button" and not is_hidden and "disabled" not in attrs:
                if target in " ".join(text.split()).lower():
                    return True
        return False

    def has_visible_node_container(self) -> bool:
        pattern = re.compile(r"(project|node|agent|swarm|jarvis|workspace)", re.IGNORECASE)
        for tag, attrs, _text, is_hidden in self.elements:
            if is_hidden or tag not in {"section", "div", "ul", "ol", "nav", "aside", "article"}:
                continue
            haystack = " ".join(
                str(attrs.get(k) or "") for k in ("id", "class", "aria-label", "data-role")
            )
            if pattern.search(haystack):
                return True
        return False


def _inspect_html(html: str) -> _HTMLInspector:
    parser = _HTMLInspector()
    parser.feed(html)
    parser.close()
    return parser


def _strip_js_comments(js_source: str) -> str:
    """Strip JS // and /* */ comments while preserving strings, template literals, and regexes."""
    out: list[str] = []
    i = 0
    n = len(js_source)
    in_quote: str | None = None
    last_sig = ""
    word_buf: list[str] = []
    regex_prev = set("=(,[!&|?:;{}~^+-*%<>")
    regex_words = {"return", "case", "typeof", "void", "delete", "throw", "in", "of", "else"}

    while i < n:
        ch = js_source[i]
        if in_quote is not None:
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(js_source[i + 1])
                i += 2
                continue
            if ch == in_quote:
                in_quote = None
                last_sig = ch
            i += 1
            continue

        if ch in ("'", '"', "`"):
            in_quote = ch
            word_buf.clear()
            out.append(ch)
            i += 1
        elif ch == "/" and i + 1 < n and js_source[i + 1] == "/":
            i += 2
            while i < n and js_source[i] != "\n":
                i += 1
        elif ch == "/" and i + 1 < n and js_source[i + 1] == "*":
            i += 2
            while i + 1 < n and not (js_source[i] == "*" and js_source[i + 1] == "/"):
                if js_source[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
        elif ch == "/" and (
            not last_sig or last_sig in regex_prev or "".join(word_buf) in regex_words
        ):
            word_buf.clear()
            out.append(ch)
            i += 1
            in_class = False
            while i < n:
                rc = js_source[i]
                out.append(rc)
                if rc == "\\" and i + 1 < n:
                    out.append(js_source[i + 1])
                    i += 2
                    continue
                if rc == "[":
                    in_class = True
                elif rc == "]":
                    in_class = False
                elif rc == "/" and not in_class:
                    i += 1
                    break
                elif rc == "\n":
                    i += 1
                    break
                i += 1
            last_sig = "/"
        else:
            out.append(ch)
            if ch.isalnum() or ch in ("_", "$"):
                word_buf.append(ch)
                last_sig = ch
            elif not ch.isspace():
                word_buf.clear()
                last_sig = ch
            i += 1
    return "".join(out)


def _wait_until(
    predicate: Any,
    *,
    timeout: float = 2.0,
    interval: float = 0.02,
    message: str = "Condition not met before timeout",
) -> None:
    """Poll a synchronous predicate until truthy or raise AssertionError."""
    deadline = time.monotonic() + timeout
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise AssertionError(message)
        time.sleep(interval)


async def _async_wait_until(
    predicate: Any,
    *,
    timeout: float = 2.0,
    interval: float = 0.02,
    message: str = "Async condition not met before timeout",
) -> None:
    """Poll a predicate on the active event loop until truthy or raise AssertionError."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        if predicate():
            return
        if loop.time() >= deadline:
            raise AssertionError(message)
        await asyncio.sleep(interval)


def _sync_discovery(
    client: TestClient,
    spy: "CastEnvironmentSpy",
    timeout: float = 2.0,
) -> None:
    """Wait for lifespan background discover_speakers() when mock devices are configured."""
    expected_speakers = sum(
        1 for d in spy.devices.values() if d.cast_type in ("audio", "group")
    )
    expected_cast_tvs = {
        d.name for d in spy.devices.values() if d.cast_type == "cast"
    }
    if expected_speakers == 0 and not expected_cast_tvs:
        return

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sp_resp = client.get("/api/speakers").json()
        speakers_list = sp_resp.get("speakers") or []
        tvs_list = sp_resp.get("tvs") or []
        discovered_names = {d.get("name") for d in speakers_list + tvs_list}
        if len(speakers_list) >= expected_speakers and expected_cast_tvs.issubset(discovered_names):
            return
        time.sleep(0.02)

    client.get("/api/speakers?refresh=true")


def _assert_iso8601_utc(ts: str) -> None:
    assert isinstance(ts, str) and ts
    normalized = ts[:-1] + "+00:00" if ts.endswith("Z") else ts
    parsed = datetime.fromisoformat(normalized)
    assert parsed is not None


# ============================================================================
# Fixtures: Isolated Workspaces, Settings, Cast/TV Spy, AGY SDK Spy, Live Spy
# ============================================================================
@pytest.fixture()
def isolated_workspaces(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, Path]:
    """Scrub ambient env vars and provision isolated temp git workspaces."""
    for env_name in (
        "GEMINI_API_KEY",
        "GEMINI_LIVE_MODEL",
        "GEMINI_USE_VERTEX",
        "ANTIGRAVITY_LS_ADDRESS",
        "ANTIGRAVITY_CSRF_TOKEN",
        "ANTIGRAVITY_PROJECT_ID",
        "AGY_WORKSPACE_DIR",
        "AGY_SAVE_DIR",
        "AGY_APP_DATA_DIR",
        "KNOWN_CAST_HOSTS",
        "HOST_LAN_IP",
        "BRIDGE_PORT",
        "DEFAULT_CAST_ROOM",
        "DEFAULT_AUDIO_ROUTING",
        "ENABLE_HTTPS",
        "PROJECTS_STATE_FILE",
        "DEFAULT_PRESENCE_ROOM",
        "ROOM_DEVICE_MAP",
        "DEFAULT_AGENT_ID",
        "EVENT_HISTORY_SIZE",
        "HUD_CAST_FORCE",
        "CAST_DISCOVERY_TIMEOUT",
        "CAST_TV_TIMEOUT",
        "HW_MIC_ENABLED",
        "SMART_HOME_WEBHOOK_TOKEN",
    ):
        monkeypatch.delenv(env_name, raising=False)

    workspace_dir = tmp_path / "workspace"
    save_dir = tmp_path / "save"
    app_data_dir = tmp_path / "app_data"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    app_data_dir.mkdir(parents=True, exist_ok=True)

    clark = _init_git_repo(workspace_dir / "clark-home-server", branch="main")
    proxmox = _init_git_repo(
        workspace_dir / "proxmox-monitor",
        branch="feat/alerts",
        dirty_modified=1,
        dirty_untracked=1,
    )
    smart_ui = _init_git_repo(
        workspace_dir / "smart-home-ui",
        branch="main",
        dirty_staged=1,
    )

    # Noise directories that ProjectManager.scan_projects() must skip
    for noise in (".git", ".venv", "__pycache__", "node_modules", ".hidden-dir"):
        noise_dir = workspace_dir / noise
        noise_dir.mkdir(parents=True, exist_ok=True)
        (noise_dir / "marker.txt").write_text("ignore", encoding="utf-8")

    return {
        "root": tmp_path,
        "workspace_dir": workspace_dir,
        "save_dir": save_dir,
        "app_data_dir": app_data_dir,
        "clark": clark,
        "proxmox": proxmox,
        "smart_ui": smart_ui,
    }


@pytest.fixture()
def e2e_settings(isolated_workspaces: dict[str, Path]) -> BridgeSettings:
    """Isolated BridgeSettings ignoring any local .env file."""
    return BridgeSettings(
        _env_file=None,
        GEMINI_API_KEY="",
        ANTIGRAVITY_LS_ADDRESS="",
        ANTIGRAVITY_CSRF_TOKEN="",
        AGY_WORKSPACE_DIR=str(isolated_workspaces["workspace_dir"]),
        AGY_SAVE_DIR=str(isolated_workspaces["save_dir"]),
        AGY_APP_DATA_DIR=str(isolated_workspaces["app_data_dir"]),
        HOST_LAN_IP="192.168.1.50",
        BRIDGE_PORT=8765,
        DEFAULT_CAST_ROOM="All Speakers",
        DEFAULT_AUDIO_ROUTING="cast",
        ENABLE_HTTPS=False,
    )


@dataclass
class _FakeDeviceSpec:
    name: str
    host: str
    cast_type: str
    model_name: str
    port: int = 8009


class CastEnvironmentSpy:
    """Tracks pychromecast discovery, media playback, DashCast TV casting, and gTTS calls."""

    def __init__(self) -> None:
        self.devices: dict[str, _FakeDeviceSpec] = {
            "Office": _FakeDeviceSpec("Office", "192.168.1.101", "audio", "Google Nest Mini"),
            "Living Room speaker": _FakeDeviceSpec(
                "Living Room speaker", "192.168.1.102", "audio", "Google Nest Audio"
            ),
            "All Speakers": _FakeDeviceSpec(
                "All Speakers", "192.168.1.103", "group", "Google Cast Group"
            ),
            '98" QLED': _FakeDeviceSpec(
                '98" QLED', "192.168.1.110", "cast", "Google TV / Chromecast"
            ),
            "Game Room TV": _FakeDeviceSpec(
                "Game Room TV", "192.168.1.111", "cast", "Chromecast Ultra"
            ),
            "Office TV": _FakeDeviceSpec(
                "Office TV", "192.168.1.112", "cast", "Vizio SmartCast TV"
            ),
        }
        self.wait_errors: dict[str, Exception] = {}
        self.dashcast_error: Exception | None = None
        self.gtts_error: Exception | None = None
        self.play_media_calls: list[dict[str, Any]] = []
        self.stop_calls: list[str] = []
        self.volume_calls: list[tuple[str, float]] = []
        self.registered_handlers: list[tuple[str, Any]] = []
        self.dashcast_instances: list[Any] = []
        self.dashcast_calls: list[dict[str, Any]] = []
        self.tts_texts: list[str] = []

    def remove_device(self, name: str) -> None:
        self.devices.pop(name, None)

    def clear_devices(self) -> None:
        self.devices.clear()

    def set_device_wait_error(self, name: str, exc: Exception) -> None:
        self.wait_errors[name] = exc

    def set_dashcast_error(self, exc: Exception | None) -> None:
        self.dashcast_error = exc

    def set_gtts_error(self, exc: Exception | None) -> None:
        self.gtts_error = exc

    def build_chromecasts(self) -> list[Any]:
        spy = self
        result = []
        for spec in self.devices.values():
            cc = MagicMock()
            cc.cast_info.friendly_name = spec.name
            cc.cast_info.uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{spec.name}-{spec.host}"))
            cc.cast_info.model_name = spec.model_name
            cc.cast_info.cast_type = spec.cast_type
            cc.cast_info.host = spec.host
            cc.cast_info.port = spec.port
            cc.app_id = None

            def _make_wait(dev_name: str):
                def _wait(timeout: float = 5.0) -> None:
                    err = spy.wait_errors.get(dev_name)
                    if err is not None:
                        raise err

                return _wait

            def _make_set_vol(dev_name: str):
                def _set_vol(vol: float) -> None:
                    spy.volume_calls.append((dev_name, vol))

                return _set_vol

            def _make_play_media(dev_name: str):
                def _play_media(
                    url: str,
                    content_type: str,
                    title: str = "",
                    stream_type: str = "",
                    enqueue: bool = False,
                ) -> None:
                    spy.play_media_calls.append(
                        {
                            "speaker": dev_name,
                            "url": url,
                            "content_type": content_type,
                            "title": title,
                            "stream_type": stream_type,
                            "enqueue": enqueue,
                        }
                    )

                return _play_media

            def _make_stop(dev_name: str):
                def _stop() -> None:
                    spy.stop_calls.append(dev_name)

                return _stop

            def _make_register(dev_name: str):
                def _register(handler: Any) -> None:
                    spy.registered_handlers.append((dev_name, handler))

                return _register

            cc.wait.side_effect = _make_wait(spec.name)
            cc.set_volume.side_effect = _make_set_vol(spec.name)
            cc.register_handler.side_effect = _make_register(spec.name)
            cc.media_controller.play_media.side_effect = _make_play_media(spec.name)
            cc.media_controller.block_until_active.return_value = None
            cc.media_controller.stop.side_effect = _make_stop(spec.name)
            result.append(cc)
        return result

    def target_device_for_controller(self, controller: Any) -> str | None:
        for dev_name, handler in reversed(self.registered_handlers):
            if handler is controller:
                return dev_name
        return None


@pytest.fixture()
def mock_chromecast_env(monkeypatch: pytest.MonkeyPatch) -> CastEnvironmentSpy:
    """Patch pychromecast.get_chromecasts, DashCastController, and gTTS before app lifespan."""
    spy = CastEnvironmentSpy()

    def _fake_get_chromecasts(timeout: float = 4.0, known_hosts: Any = None):
        browser = MagicMock()
        return spy.build_chromecasts(), browser

    class _FakeDashCastController:
        def __init__(
            self,
            appNamespace: str = "urn:x-cast:com.madmod.dashcast",
            appId: str = "84912283",
        ) -> None:
            self.appNamespace = appNamespace
            self.appId = appId
            spy.dashcast_instances.append(self)

        def load_url(
            self,
            url: str,
            *,
            force: bool = False,
            reload_seconds: float = 0,
            callback_function: Any = None,
        ) -> None:
            if spy.dashcast_error is not None:
                raise spy.dashcast_error
            target_dev = spy.target_device_for_controller(self)
            spy.dashcast_calls.append(
                {
                    "url": url,
                    "force": force,
                    "reload_seconds": reload_seconds,
                    "target_device": target_dev,
                    "controller": self,
                }
            )

    class _FakeGTTS:
        def __init__(self, text: str, lang: str = "en", slow: bool = False) -> None:
            self.text = text
            self.lang = lang
            self.slow = slow

        def save(self, filepath: str) -> None:
            if spy.gtts_error is not None:
                raise spy.gtts_error
            spy.tts_texts.append(self.text)
            Path(filepath).write_bytes(b"ID3-FAKE-MP3-BYTES")

    import pychromecast
    import pychromecast.controllers.dashcast as dashcast_mod

    async def _no_gemini_tts(self: Any, text: str) -> None:
        return None

    monkeypatch.setattr(pychromecast, "get_chromecasts", _fake_get_chromecasts)
    monkeypatch.setattr(dashcast_mod, "DashCastController", _FakeDashCastController)
    monkeypatch.setattr("agy_home_bridge.cast_manager.gTTS", _FakeGTTS)
    monkeypatch.setattr(
        "agy_home_bridge.cast_manager.GoogleHomeCastManager.synthesize_gemini_pcm",
        _no_gemini_tts,
    )
    return spy


class Thought:
    """Fake google.antigravity Thought chunk."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.content = text


class ToolCall:
    """Fake google.antigravity ToolCall chunk."""

    def __init__(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        result: Any = None,
    ) -> None:
        self.name = name
        self.args = args or {}
        self.result = result


class Text:
    """Fake google.antigravity Text chunk."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.content = text

    def __str__(self) -> str:
        return self.text


@dataclass
class ScriptedTurn:
    chunks: list[Any]
    conversation_id: str | None = None
    hold_event: asyncio.Event | None = None
    error: Exception | None = None


class AntigravitySDKSpy:
    """Tracks LocalAgentConfig and drives deterministic scripted Agent.chat() streams."""

    def __init__(self) -> None:
        self.configs: list[dict[str, Any]] = []
        self.prompts: list[str] = []
        self.scripted_by_workspace: dict[str, list[ScriptedTurn]] = {}
        self.default_turns: list[ScriptedTurn] = []
        self._seq = 0

    def queue_turn(
        self,
        chunks: list[Any],
        *,
        workspace: str | Path | None = None,
        conversation_id: str | None = None,
        hold_event: asyncio.Event | None = None,
        error: Exception | None = None,
    ) -> None:
        turn = ScriptedTurn(
            chunks=chunks,
            conversation_id=conversation_id,
            hold_event=hold_event,
            error=error,
        )
        if workspace is not None:
            key = str(Path(workspace).resolve())
            self.scripted_by_workspace.setdefault(key, []).append(turn)
        else:
            self.default_turns.append(turn)

    def next_turn_for(self, cfg: dict[str, Any], prompt: str) -> ScriptedTurn:
        workspaces = cfg.get("workspaces") or []
        if workspaces:
            ws_key = str(Path(workspaces[0]).resolve())
            q = self.scripted_by_workspace.get(ws_key)
            if q:
                return q.pop(0)
        if self.default_turns:
            return self.default_turns.pop(0)
        self._seq += 1
        ws_name = Path(workspaces[0]).name if workspaces else "default"
        return ScriptedTurn(
            chunks=[
                Thought(f"Analyzing {ws_name}"),
                ToolCall("inspect_repo", {"workspace": ws_name}, result="ok"),
                Text(f"Completed '{prompt}' in {ws_name}."),
            ],
            conversation_id=cfg.get("conversation_id") or f"conv-{ws_name}-{self._seq}",
        )


@pytest.fixture()
def mock_antigravity_sdk(monkeypatch: pytest.MonkeyPatch) -> AntigravitySDKSpy:
    """Safely inject fake google.antigravity + google.antigravity.hooks without touching google.genai."""
    import google  # namespace package must remain intact for google.genai

    spy = AntigravitySDKSpy()

    class FakeResponseStream:
        def __init__(self, turn: ScriptedTurn, cfg: dict[str, Any]) -> None:
            self._turn = turn
            self._cfg = cfg
            self.chunks = list(turn.chunks)

        def __aiter__(self):
            turn = self._turn
            cfg = self._cfg

            async def _gen():
                if turn.error is not None:
                    raise turn.error
                # Yield Thought chunks before hold_event so live thought streaming
                # can be observed mid-task while the turn is paused.
                held = False
                for item in turn.chunks:
                    if (
                        turn.hold_event is not None
                        and not held
                        and not isinstance(item, Thought)
                    ):
                        held = True
                        await turn.hold_event.wait()
                    if isinstance(item, ToolCall):
                        for hook in cfg.get("hooks") or []:
                            post_fn = getattr(hook, "post_tool_call", None)
                            if callable(post_fn):
                                maybe_coro = post_fn(item)
                                if asyncio.iscoroutine(maybe_coro):
                                    await maybe_coro
                    yield item
                if turn.hold_event is not None and not held:
                    await turn.hold_event.wait()

            return _gen()

    class FakeAgent:
        def __init__(self, cfg: dict[str, Any]) -> None:
            self.cfg = cfg
            self.conversation_id = cfg.get("conversation_id")

        async def __aenter__(self) -> FakeAgent:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def chat(self, prompt: str) -> FakeResponseStream:
            spy.prompts.append(prompt)
            turn = spy.next_turn_for(self.cfg, prompt)
            if turn.conversation_id:
                self.conversation_id = turn.conversation_id
            elif not self.conversation_id:
                spy._seq += 1
                self.conversation_id = f"conv-auto-{spy._seq}"
            return FakeResponseStream(turn, self.cfg)

    def _local_agent_config(**kwargs: Any) -> dict[str, Any]:
        cfg = dict(kwargs)
        spy.configs.append(cfg)
        return cfg

    def _capabilities_config(**kwargs: Any) -> dict[str, Any]:
        return dict(kwargs)

    fake_agy = types.ModuleType("google.antigravity")
    fake_agy.Agent = FakeAgent  # type: ignore[attr-defined]
    fake_agy.LocalAgentConfig = _local_agent_config  # type: ignore[attr-defined]
    fake_agy.CapabilitiesConfig = _capabilities_config  # type: ignore[attr-defined]
    fake_agy.Thought = Thought  # type: ignore[attr-defined]
    fake_agy.ToolCall = ToolCall  # type: ignore[attr-defined]
    fake_agy.Text = Text  # type: ignore[attr-defined]

    fake_hooks = types.ModuleType("google.antigravity.hooks")
    fake_policy = MagicMock()
    fake_policy.allow_all.return_value = {"policy": "allow_all"}
    fake_hooks.policy = fake_policy  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google.antigravity", fake_agy)
    monkeypatch.setitem(sys.modules, "google.antigravity.hooks", fake_hooks)
    monkeypatch.setattr(google, "antigravity", fake_agy, raising=False)
    return spy


class GeminiLiveSpy:
    """Scriptable mock for google.genai.Client.aio.live.connect."""

    def __init__(self) -> None:
        self.connect_calls: list[dict[str, Any]] = []
        self.tool_responses: list[Any] = []
        self.realtime_inputs: list[Any] = []
        self.client_contents: list[Any] = []
        self._pending_turns: list[list[Any] | None] = []
        self._session_queue: asyncio.Queue[list[Any] | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self.tool_response_event: asyncio.Event | None = None

    @property
    def declared_tool_names(self) -> set[str]:
        names: set[str] = set()
        for call in self.connect_calls:
            cfg = call.get("config")
            for tool in getattr(cfg, "tools", None) or []:
                for fd in getattr(tool, "function_declarations", None) or []:
                    if getattr(fd, "name", None):
                        names.add(fd.name)
        return names

    def enqueue_turn(self, responses: list[Any]) -> None:
        if self._session_queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._session_queue.put_nowait, responses)
        else:
            self._pending_turns.append(responses)

    def end_session(self) -> None:
        if self._session_queue is not None and self._loop is not None:
            self._loop.call_soon_threadsafe(self._session_queue.put_nowait, None)
        else:
            self._pending_turns.append(None)

    def make_tool_call_turn(
        self,
        calls: list[tuple[str, dict[str, Any], str]],
    ) -> Any:
        fc_objs = []
        for call_id, args, name in calls:
            fc = MagicMock()
            fc.id = call_id
            fc.name = name
            fc.args = args
            fc_objs.append(fc)
        resp = MagicMock()
        resp.server_content = None
        resp.tool_call = MagicMock()
        resp.tool_call.function_calls = fc_objs
        return resp

    def make_audio_text_turn(
        self,
        *,
        pcm_bytes: bytes = b"\x02\x00" * 240,
        output_text: str = "On it.",
        input_text: str | None = None,
        turn_complete: bool = True,
    ) -> Any:
        part = MagicMock()
        part.inline_data = MagicMock()
        part.inline_data.data = pcm_bytes
        part.text = output_text

        sc = MagicMock()
        sc.interrupted = False
        sc.turn_complete = turn_complete
        sc.model_turn = MagicMock()
        sc.model_turn.parts = [part]
        if input_text is not None:
            sc.input_transcription = MagicMock()
            sc.input_transcription.text = input_text
        else:
            sc.input_transcription = None
        sc.output_transcription = MagicMock()
        sc.output_transcription.text = output_text

        resp = MagicMock()
        resp.server_content = sc
        resp.tool_call = None
        return resp


@pytest.fixture()
def mock_genai_live(monkeypatch: pytest.MonkeyPatch) -> GeminiLiveSpy:
    """Patch google.genai.Client with an async-capable Gemini Live double."""
    spy = GeminiLiveSpy()

    class _FakeLiveSession:
        async def send_tool_response(self, function_responses: Any = None, **kwargs: Any) -> None:
            payload = function_responses if function_responses is not None else kwargs
            spy.tool_responses.append(payload)
            if spy.tool_response_event is not None:
                spy.tool_response_event.set()

        async def send_realtime_input(self, **kwargs: Any) -> None:
            spy.realtime_inputs.append(kwargs)

        async def send_client_content(self, **kwargs: Any) -> None:
            spy.client_contents.append(kwargs)

        def receive(self):
            assert spy._session_queue is not None

            async def _gen():
                turn_items = await spy._session_queue.get()
                if turn_items is None:
                    return
                for item in turn_items:
                    yield item

            return _gen()

    class _FakeLiveConnectCM:
        def __init__(self, model: str, config: Any) -> None:
            self.model = model
            self.config = config

        async def __aenter__(self) -> _FakeLiveSession:
            spy._loop = asyncio.get_running_loop()
            spy._session_queue = asyncio.Queue()
            spy.tool_response_event = asyncio.Event()
            for item in spy._pending_turns:
                spy._session_queue.put_nowait(item)
            spy._pending_turns.clear()
            spy.connect_calls.append({"model": self.model, "config": self.config})
            return _FakeLiveSession()

        async def __aexit__(self, *args: Any) -> None:
            return None

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.aio = MagicMock()
            self.aio.live.connect = lambda model, config=None: _FakeLiveConnectCM(model, config)

    monkeypatch.setattr("google.genai.Client", _FakeClient)
    return spy


@pytest.fixture()
def jarvis_client(
    e2e_settings: BridgeSettings,
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> Any:
    """Context-managed TestClient running FastAPI lifespan with mocked LAN/SDK."""
    app = create_app(e2e_settings)
    with TestClient(app) as client:
        _sync_discovery(client, mock_chromecast_env)
        yield client


# ============================================================================
# TIER 1 — Primary Feature Coverage (45 Tests: F1-F9)
# ============================================================================
def test_t1_f1_presence_office_maps_speaker_tv_and_warms(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    resp = jarvis_client.post(
        "/api/presence/room",
        json={"room": "office", "source": "frigate_ha"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert (data.get("active_presence_room") or data.get("presence_room")) == "office"
    assert data["active_room"] == "Office"
    assert data["active_tv"] in ("Office TV", "Office - Samsung Q60CD 65")

    health = jarvis_client.get("/api/health").json()
    assert health["active_presence_room"] == "office"
    assert health["active_room"] == "Office"
    _wait_until(
        lambda: any(
            c["speaker"] == "Office" and "warmup_silence.wav" in c["url"]
            for c in mock_chromecast_env.play_media_calls
        ),
        message="Expected background warm_room() play_media call for Office",
    )


def test_t1_f1_presence_living_room_maps_speaker_and_tv(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    resp = jarvis_client.post(
        "/api/presence/room",
        json={"room": "living_room", "source": "frigate_ha"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert (data.get("active_presence_room") or data.get("presence_room")) == "living_room"
    assert data["active_room"] == "Living Room speaker"
    assert data["active_tv"] in ("Living Room TV", '98" QLED')
    _wait_until(
        lambda: any(
            c["speaker"] == "Living Room speaker" and "warmup_silence.wav" in c["url"]
            for c in mock_chromecast_env.play_media_calls
        ),
        message="Expected background warm_room() play_media call for Living Room speaker",
    )


def test_t1_f1_presence_kitchen_maps_speaker_and_kitchen_tv(
    jarvis_client: TestClient,
) -> None:
    resp = jarvis_client.post(
        "/api/presence/room",
        json={"room": "kitchen", "source": "frigate_ha"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert (data.get("active_presence_room") or data.get("presence_room")) == "kitchen"
    assert data["active_room"] == "Living Room speaker"
    assert data["active_tv"] == "Kitchen TV"


def test_t1_f1_presence_game_room_and_98_qled_mappings(
    jarvis_client: TestClient,
) -> None:
    r1 = jarvis_client.post("/api/presence/room", json={"room": "game_room"}).json()
    assert r1["active_room"] == "All Speakers"
    assert r1["active_tv"] in ("Game Room TV", '98" QLED')

    r2 = jarvis_client.post("/api/presence/room", json={"room": '98" QLED'}).json()
    assert r2["active_room"] == "All Speakers"
    assert r2["active_tv"] == '98" QLED'

    r3 = jarvis_client.post("/api/presence/room", json={"room": "theater"}).json()
    assert r3["active_room"] == "All Speakers"
    assert r3["active_tv"] == '98" QLED'


def test_t1_f1_presence_master_bedroom_and_whole_home(
    jarvis_client: TestClient,
) -> None:
    mb = jarvis_client.post("/api/presence/room", json={"room": "master_bedroom"}).json()
    assert mb["active_room"] == "All Speakers"
    assert mb["active_tv"] in ("The Frame", "Master Bedroom Frame", "Master Bedroom TV")

    wh = jarvis_client.post("/api/presence/room", json={"room": "whole_home"}).json()
    assert wh["active_room"] == "All Speakers"
    assert wh["active_tv"] in ('98" QLED', "Living Room TV", "Office TV", "Game Room TV")


def test_t1_f1_presence_emits_room_changed_and_never_auto_casts_tv(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    with jarvis_client.websocket_connect("/ws/hud") as ws:
        snap = recv_json_with_timeout(ws)
        assert snap["type"] == "hud_snapshot"

        resp = jarvis_client.post(
            "/api/presence/room",
            json={"room": "office", "source": "frigate_ha"},
        )
        assert resp.status_code == 200
        ev, _ = recv_until_type(ws, "room_changed")
        assert ev["type"] == "room_changed"
        assert ev.get("active_room") == "Office"
    assert mock_chromecast_env.dashcast_calls == []


def test_t1_f2_scan_projects_lists_all_workspace_git_repos(
    jarvis_client: TestClient,
) -> None:
    resp = jarvis_client.get("/api/projects")
    assert resp.status_code == 200
    payload = resp.json()
    names = {p["name"] for p in payload["projects"]}
    assert {"clark-home-server", "proxmox-monitor", "smart-home-ui"}.issubset(names)
    for proj in payload["projects"]:
        for key in (
            "id",
            "name",
            "short_name",
            "path",
            "git_branch",
            "uncommitted_changes",
            "active",
            "active_agent_id",
            "agents",
        ):
            assert key in proj


def test_t1_f2_git_branch_and_dirty_counts_accurate(
    jarvis_client: TestClient,
) -> None:
    projects = {p["name"]: p for p in jarvis_client.get("/api/projects").json()["projects"]}
    assert projects["clark-home-server"]["git_branch"] == "main"
    assert projects["clark-home-server"]["uncommitted_changes"] == 0
    assert projects["proxmox-monitor"]["git_branch"] == "feat/alerts"
    assert projects["proxmox-monitor"]["uncommitted_changes"] == 2
    assert projects["smart-home-ui"]["git_branch"] == "main"
    assert projects["smart-home-ui"]["uncommitted_changes"] == 1


def test_t1_f2_verbal_short_names_and_project_agent_labels(
    jarvis_client: TestClient,
) -> None:
    projects = jarvis_client.get("/api/projects").json()["projects"]
    short_names = [p["short_name"] for p in projects]
    assert all(isinstance(s, str) and s.strip() for s in short_names)
    assert len(set(s.lower() for s in short_names)) == len(short_names)
    for proj in projects:
        agents = proj["agents"]
        assert len(agents) >= 1
        agent_101 = next((a for a in agents if a["agent_id"] == "101"), agents[0])
        assert agent_101["short_name"]
        assert "101" in agent_101["project_agent"]
        assert proj["short_name"].lower() in agent_101["project_agent"].lower()


def test_t1_f2_ignores_non_project_and_hidden_directories(
    jarvis_client: TestClient,
) -> None:
    names = {p["name"] for p in jarvis_client.get("/api/projects").json()["projects"]}
    for ignored in (".git", ".venv", "__pycache__", "node_modules", ".hidden-dir"):
        assert ignored not in names


def test_t1_f2_concurrent_agents_and_swarm_nodes_tracked(
    jarvis_client: TestClient,
) -> None:
    sel = jarvis_client.post(
        "/api/projects/select",
        json={"project": "clark-home-server", "agent": "102"},
    )
    assert sel.status_code == 200
    clark_id = sel.json()["project_id"]

    pm = jarvis_client.app.state.project_manager
    swarm = [
        {"node_id": "sub-lint", "role": "linter", "status": "running"},
        {"node_id": "sub-test", "role": "tester", "status": "completed"},
    ]
    pm.update_agent_status(
        clark_id,
        "102",
        status="running",
        current_task="Verify swarm",
        swarm_nodes=swarm,
    )

    projects = {p["id"]: p for p in jarvis_client.get("/api/projects").json()["projects"]}
    agents = {a["agent_id"]: a for a in projects[clark_id]["agents"]}
    assert "101" in agents and "102" in agents
    assert len(agents["102"]["swarm_nodes"]) == 2
    assert agents["102"]["status"] == "running"


def test_t1_f3_select_project_by_name_and_short_name(
    jarvis_client: TestClient,
) -> None:
    r1 = jarvis_client.post("/api/projects/select", json={"project": "proxmox-monitor"})
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["project_name"] == "proxmox-monitor"
    short_target = d1["short_name"]

    r2 = jarvis_client.post("/api/projects/select", json={"project": "clark-home-server"})
    assert r2.status_code == 200

    r3 = jarvis_client.post("/api/projects/select", json={"project": short_target})
    assert r3.status_code == 200
    assert r3.json()["project_name"] == "proxmox-monitor"

    health = jarvis_client.get("/api/health").json()
    assert "proxmox" in health["active_project"].lower()
    assert health["active_project_agent"]
    projs = {p["name"]: p for p in jarvis_client.get("/api/projects").json()["projects"]}
    assert projs["proxmox-monitor"]["active"] is True
    assert projs["clark-home-server"]["active"] is False


def test_t1_f3_select_combined_project_agent_phrase(
    jarvis_client: TestClient,
) -> None:
    with jarvis_client.websocket_connect("/ws/hud") as ws:
        _ = recv_json_with_timeout(ws)
        resp = jarvis_client.post(
            "/api/projects/select",
            json={"project_agent": "project clark home server, agent 102"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["project_name"] == "clark-home-server"
        assert data["agent_id"] == "102"

        ev, _ = recv_until_type(ws, "project_changed")
        assert ev["agent_id"] == "102"
        assert ev["project_name"] == "clark-home-server"


def test_t1_f3_switch_updates_local_agent_config_workspaces(
    jarvis_client: TestClient,
    isolated_workspaces: dict[str, Path],
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    jarvis_client.post("/api/projects/select", json={"project": "clark-home-server"})
    r1 = jarvis_client.post(
        "/api/agent/ask",
        json={"prompt": "Check repo one", "cast_response": False},
    )
    assert r1.status_code == 200

    jarvis_client.post("/api/projects/select", json={"project": "proxmox-monitor"})
    r2 = jarvis_client.post(
        "/api/agent/ask",
        json={"prompt": "Check repo two", "cast_response": False},
    )
    assert r2.status_code == 200

    assert len(mock_antigravity_sdk.configs) >= 2
    assert mock_antigravity_sdk.configs[0]["workspaces"] == [
        str(isolated_workspaces["clark"].resolve())
    ]
    assert mock_antigravity_sdk.configs[1]["workspaces"] == [
        str(isolated_workspaces["proxmox"].resolve())
    ]


def test_t1_f3_per_project_agent_conversation_id_isolation_and_persistence(
    jarvis_client: TestClient,
    isolated_workspaces: dict[str, Path],
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    mock_antigravity_sdk.queue_turn(
        [Text("ok clark 101")],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-clark-101",
    )
    mock_antigravity_sdk.queue_turn(
        [Text("ok proxmox 101")],
        workspace=isolated_workspaces["proxmox"],
        conversation_id="conv-proxmox-101",
    )
    mock_antigravity_sdk.queue_turn(
        [Text("ok clark 102")],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-clark-102",
    )
    mock_antigravity_sdk.queue_turn(
        [Text("resumed clark 101")],
        workspace=isolated_workspaces["clark"],
    )

    jarvis_client.post("/api/projects/select", json={"project": "clark-home-server", "agent": "101"})
    jarvis_client.post("/api/agent/ask", json={"prompt": "t1", "cast_response": False})

    jarvis_client.post("/api/projects/select", json={"project": "proxmox-monitor", "agent": "101"})
    jarvis_client.post("/api/agent/ask", json={"prompt": "t2", "cast_response": False})

    jarvis_client.post("/api/projects/select", json={"project": "clark-home-server", "agent": "102"})
    jarvis_client.post("/api/agent/ask", json={"prompt": "t3", "cast_response": False})

    jarvis_client.post("/api/projects/select", json={"project": "clark-home-server", "agent": "101"})
    jarvis_client.post("/api/agent/ask", json={"prompt": "t4", "cast_response": False})

    assert mock_antigravity_sdk.configs[3].get("conversation_id") == "conv-clark-101"
    state_file = isolated_workspaces["save_dir"] / "projects_state.json"
    assert state_file.exists()
    raw = state_file.read_text(encoding="utf-8")
    assert "conv-clark-101" in raw
    assert "conv-proxmox-101" in raw
    assert "conv-clark-102" in raw


def test_t1_f3_persisted_state_survives_app_restart(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    mock_antigravity_sdk.queue_turn(
        [Text("saved turn")],
        workspace=isolated_workspaces["proxmox"],
        conversation_id="conv-proxmox-102-persist",
    )

    app1 = create_app(e2e_settings)
    with TestClient(app1) as c1:
        c1.post("/api/projects/select", json={"project": "proxmox-monitor", "agent": "102"})
        c1.post("/api/agent/ask", json={"prompt": "persist me", "cast_response": False})

    app2 = create_app(e2e_settings)
    with TestClient(app2) as c2:
        summary = c2.get("/api/projects").json()
        assert summary["active_agent_id"] == "102"
        proxmox = next(p for p in summary["projects"] if p["name"] == "proxmox-monitor")
        assert proxmox["active"] is True
        a102 = next(a for a in proxmox["agents"] if a["agent_id"] == "102")
        assert a102["conversation_id"] == "conv-proxmox-102-persist"


def test_t1_f3_gemini_live_switch_project_voice_tool(
    e2e_settings: BridgeSettings,
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    e2e_settings.gemini_api_key = "fake-live-key"
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [("fc-sw-1", {"project_agent": "proxmox monitor agent 103"}, "switch_project")]
            )
        ]
    )

    app = create_app(e2e_settings)
    with TestClient(app) as client:
        _sync_discovery(client, mock_chromecast_env)
        with client.websocket_connect("/ws/hud") as hud_ws, client.websocket_connect("/ws/live") as live_ws:
            _ = recv_json_with_timeout(hud_ws)
            _ = recv_until_type(live_ws, "status")
            ev, _ = recv_until_type(hud_ws, "project_changed", timeout=4.0)
            assert ev["agent_id"] == "103"
            assert ev["project_name"] == "proxmox-monitor"
            _wait_until(
                lambda: len(mock_genai_live.tool_responses) >= 1,
                timeout=2.0,
                message="Expected switch_project tool_response before ending session",
            )
            mock_genai_live.end_session()

    assert "switch_project" in mock_genai_live.declared_tool_names
    assert len(mock_genai_live.tool_responses) >= 1


def test_t1_f4_ws_hud_initial_snapshot_structure(
    jarvis_client: TestClient,
) -> None:
    with jarvis_client.websocket_connect("/ws/hud") as ws:
        snap = recv_json_with_timeout(ws)
        assert snap["type"] == "hud_snapshot"
        for key in (
            "timestamp",
            "active_room",
            "active_speaker",
            "active_tv",
            "voice_status",
            "projects",
            "active_project_id",
            "active_agent_id",
            "active_project_agent",
            "recent_events",
        ):
            assert key in snap
        _assert_iso8601_utc(snap["timestamp"])
        assert isinstance(snap["projects"], list) and len(snap["projects"]) >= 3


async def test_t1_f4_streams_thought_tool_call_tool_result_token_task_events(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    mock_antigravity_sdk.queue_turn(
        [
            Thought("Checking systemd units"),
            ToolCall("run_cmd", {"cmd": "systemctl status"}, result="active (running)"),
            Text("All services healthy."),
        ],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-stream-1",
    )
    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    q = app.state.event_hub.subscribe()
    try:
        ack = app.state.agent_engine.start_background_task(
            "Check systemd", project="clark-home-server", agent="101"
        )
        assert ack["status"] == "started"
        done_ev, seen = await wait_for_hub_event(q, "task_completed")
        types_seen = [e["type"] for e in seen]
        for expected in ("task_started", "thought", "tool_call", "tool_result", "token", "task_completed"):
            assert expected in types_seen, f"Missing {expected} in {types_seen}"
        for key in ("project_id", "agent_id", "project_agent", "conversation_id", "task_id", "timestamp", "event_id"):
            assert done_ev.get(key)
    finally:
        app.state.event_hub.unsubscribe(q)


async def test_t1_f4_concurrent_project_agent_event_scoping(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    gate_a = asyncio.Event()
    gate_b = asyncio.Event()
    mock_antigravity_sdk.queue_turn(
        [Thought("Alpha thought"), Text("Alpha done")],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-c1",
        hold_event=gate_a,
    )
    mock_antigravity_sdk.queue_turn(
        [Thought("Beta thought"), Text("Beta done")],
        workspace=isolated_workspaces["proxmox"],
        conversation_id="conv-p2",
        hold_event=gate_b,
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    q = app.state.event_hub.subscribe()
    try:
        ack_a = app.state.agent_engine.start_background_task(
            "Task A", project="clark-home-server", agent="101"
        )
        ack_b = app.state.agent_engine.start_background_task(
            "Task B", project="proxmox-monitor", agent="102"
        )
        assert ack_a["task_id"] != ack_b["task_id"]
        assert ack_a["project_id"] != ack_b["project_id"]

        gate_b.set()
        gate_a.set()

        completed: list[dict[str, Any]] = []
        while len(completed) < 2:
            ev, _ = await wait_for_hub_event(q, "task_completed")
            completed.append(ev)

        by_task = {e["task_id"]: e for e in completed}
        assert by_task[ack_a["task_id"]]["agent_id"] == "101"
        assert by_task[ack_b["task_id"]]["agent_id"] == "102"
        assert by_task[ack_a["task_id"]]["project_id"] != by_task[ack_b["task_id"]]["project_id"]
    finally:
        app.state.event_hub.unsubscribe(q)


def test_t1_f4_events_forwarded_to_both_ws_hud_and_ws_live(
    jarvis_client: TestClient,
) -> None:
    with jarvis_client.websocket_connect("/ws/hud") as hud_ws, jarvis_client.websocket_connect("/ws/live") as live_ws:
        _ = recv_json_with_timeout(hud_ws)
        _ = recv_until_type(live_ws, "status")

        resp = jarvis_client.post(
            "/api/presence/room",
            json={"room": "game_room", "source": "frigate_ha"},
        )
        assert resp.status_code == 200

        hud_ev, _ = recv_until_type(hud_ws, "room_changed")
        live_ev, _ = recv_until_type(live_ws, "room_changed")
        assert hud_ev["active_tv"] in ("Game Room TV", '98" QLED')
        assert live_ev["active_tv"] in ("Game Room TV", '98" QLED')


def test_t1_f4_recent_events_ring_buffer_populates_late_hud_subscriber(
    jarvis_client: TestClient,
) -> None:
    jarvis_client.post("/api/presence/room", json={"room": "office", "source": "frigate_ha"})
    jarvis_client.post("/api/projects/select", json={"project": "proxmox-monitor", "agent": "102"})

    with jarvis_client.websocket_connect("/ws/hud") as ws:
        snap = recv_json_with_timeout(ws)
        recent_types = [e.get("type") for e in snap["recent_events"]]
        assert "room_changed" in recent_types
        assert "project_changed" in recent_types
        idx_room = recent_types.index("room_changed")
        idx_proj = recent_types.index("project_changed")
        assert idx_room < idx_proj


async def test_t1_f5_delegate_to_antigravity_agent_is_non_blocking(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    e2e_settings.gemini_api_key = "fake-live-key"
    hold = asyncio.Event()
    mock_antigravity_sdk.queue_turn(
        [Thought("Working"), Text("Finished long task")],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-nb-1",
        hold_event=hold,
    )
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [("fc-nb-1", {"task": "Run heavy build"}, "delegate_to_antigravity_agent")]
            )
        ]
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    fake_ws = FakeWebSocket()
    live_task = asyncio.create_task(app.state.live_bridge.handle_websocket(fake_ws))
    q = app.state.event_hub.subscribe()
    try:
        assert mock_genai_live.tool_response_event is None or not mock_genai_live.tool_response_event.is_set()
        # Wait until live_bridge enters session and sends tool_response BEFORE releasing hold
        for _ in range(40):
            if mock_genai_live.tool_response_event and mock_genai_live.tool_response_event.is_set():
                break
            await asyncio.sleep(0.02)
        assert mock_genai_live.tool_response_event is not None
        await asyncio.wait_for(mock_genai_live.tool_response_event.wait(), timeout=2.0)
        assert not hold.is_set()
        payload_str = json.dumps(mock_genai_live.tool_responses[0], default=str)
        assert "started" in payload_str.lower()

        hold.set()
        ev, _ = await wait_for_hub_event(q, "task_completed", timeout=3.0)
        assert ev["conversation_id"] == "conv-nb-1"
    finally:
        app.state.event_hub.unsubscribe(q)
        mock_genai_live.end_session()
        fake_ws.disconnect()
        await asyncio.wait_for(live_task, timeout=2.0)


async def test_t1_f5_delegate_respects_optional_project_agent_override(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    e2e_settings.gemini_api_key = "fake-live-key"
    mock_antigravity_sdk.queue_turn(
        [Text("Override target done")],
        workspace=isolated_workspaces["proxmox"],
        conversation_id="conv-ov-105",
    )
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [
                    (
                        "fc-ov-1",
                        {
                            "task": "Check CPU",
                            "project": "proxmox-monitor",
                            "project_agent": "proxmox monitor agent 105",
                        },
                        "delegate_to_antigravity_agent",
                    )
                ]
            )
        ]
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    app.state.project_manager.select_project(project="clark-home-server", agent="101")
    fake_ws = FakeWebSocket()
    live_task = asyncio.create_task(app.state.live_bridge.handle_websocket(fake_ws))
    q = app.state.event_hub.subscribe()
    try:
        ev, _ = await wait_for_hub_event(q, "task_completed", timeout=3.0)
        assert ev["agent_id"] == "105"
        assert ev["project_name"] == "proxmox-monitor"
        assert mock_antigravity_sdk.configs[-1]["workspaces"] == [
            str(isolated_workspaces["proxmox"].resolve())
        ]
    finally:
        app.state.event_hub.unsubscribe(q)
        mock_genai_live.end_session()
        fake_ws.disconnect()
        await asyncio.wait_for(live_task, timeout=2.0)


async def test_t1_f5_completion_broadcasts_concise_summary_to_active_room(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    mock_antigravity_sdk.queue_turn(
        [Text("Deployment to staging completed cleanly.")],
        workspace=isolated_workspaces["clark"],
    )
    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        await ac.post("/api/presence/room", json={"room": "office", "source": "frigate_ha"})
        mock_chromecast_env.play_media_calls.clear()
        mock_chromecast_env.tts_texts.clear()

        q = app.state.event_hub.subscribe()
        try:
            app.state.agent_engine.start_background_task("Deploy staging")
            await wait_for_hub_event(q, "task_completed")
            await _async_wait_until(
                lambda: any(
                    c["speaker"] == "Office" and c["content_type"] == "audio/mp3"
                    for c in mock_chromecast_env.play_media_calls
                ),
                timeout=2.0,
                message="Expected completion TTS playback on Office speaker",
            )
        finally:
            app.state.event_hub.unsubscribe(q)

    assert any("Deployment" in t for t in mock_chromecast_env.tts_texts)
    assert any(
        c["speaker"] == "Office" and c["content_type"] == "audio/mp3"
        for c in mock_chromecast_env.play_media_calls
    )
    assert not any(
        c["speaker"] == "Living Room speaker" and c["content_type"] == "audio/mp3"
        for c in mock_chromecast_env.play_media_calls
    )


async def test_t1_f5_completion_follows_room_change_mid_task(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    hold = asyncio.Event()
    mock_antigravity_sdk.queue_turn(
        [Thought("Running suite"), Text("Suite passed across all targets.")],
        workspace=isolated_workspaces["clark"],
        hold_event=hold,
    )
    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        await ac.post("/api/presence/room", json={"room": "office", "source": "frigate_ha"})
        q = app.state.event_hub.subscribe()
        try:
            app.state.agent_engine.start_background_task("Run suite")
            await wait_for_hub_event(q, "task_started")
            mock_chromecast_env.play_media_calls.clear()

            # Walk to living room while task is still blocked
            await ac.post("/api/presence/room", json={"room": "living_room", "source": "frigate_ha"})
            hold.set()
            await wait_for_hub_event(q, "task_completed")
            await _async_wait_until(
                lambda: any(
                    c["content_type"] == "audio/mp3" for c in mock_chromecast_env.play_media_calls
                ),
                timeout=2.0,
                message="Expected completion TTS playback on Living Room speaker",
            )
        finally:
            app.state.event_hub.unsubscribe(q)

    mp3_calls = [c for c in mock_chromecast_env.play_media_calls if c["content_type"] == "audio/mp3"]
    assert mp3_calls and mp3_calls[-1]["speaker"] == "Living Room speaker"
    assert not any(c["speaker"] == "Office" for c in mp3_calls)


async def test_t1_f5_background_task_failure_emits_task_failed_without_crashing_bridge(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    mock_antigravity_sdk.queue_turn(
        [],
        workspace=isolated_workspaces["clark"],
        error=RuntimeError("Simulated SDK crash"),
    )
    mock_antigravity_sdk.queue_turn(
        [Text("Recovered successfully")],
        workspace=isolated_workspaces["clark"],
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    q = app.state.event_hub.subscribe()
    try:
        app.state.agent_engine.start_background_task("Trigger failure")
        fail_ev, _ = await wait_for_hub_event(q, "task_failed", timeout=3.0)
        assert fail_ev["type"] == "task_failed"

        # Subsequent task still succeeds cleanly
        app.state.agent_engine.start_background_task("Recover")
        ok_ev, _ = await wait_for_hub_event(q, "task_completed", timeout=3.0)
        assert "Recovered" in json.dumps(ok_ev)
    finally:
        app.state.event_hub.unsubscribe(q)


def test_t1_f6_pwa_index_retains_required_markers_and_serves_static_assets(
    jarvis_client: TestClient,
) -> None:
    html_resp = jarvis_client.get("/")
    assert html_resp.status_code == 200
    assert "Antigravity Live" in html_resp.text
    assert 'closedby="any"' in html_resp.text
    assert 'name="color-scheme" content="light dark"' in html_resp.text

    assert jarvis_client.get("/static/app.js").status_code == 200
    assert jarvis_client.get("/static/styles.css").status_code == 200


def test_t1_f6_pwa_contains_node_view_and_verbal_short_names_ui(
    jarvis_client: TestClient,
) -> None:
    html = jarvis_client.get("/").text
    app_js = jarvis_client.get("/static/app.js").text
    inspector = _inspect_html(html)
    tag_names = {t for t, _ in inspector.tags}
    assert {"header", "main", "section", "dialog"}.issubset(tag_names)
    assert inspector.has_visible_node_container()

    clean_js = _strip_js_comments(app_js)
    assert re.search(r"\bfetch\s*\(", clean_js)
    assert re.search(r"""['"`]/api/projects['"`]""", clean_js)
    assert any(
        dom_op in clean_js
        for dom_op in ("createElement", "innerHTML", "textContent", "replaceChildren", "appendChild")
    )
    for marker in (
        "short_name",
        "project_agent",
        "git_branch",
        "uncommitted_changes",
        "conversation_id",
        "agents",
    ):
        assert marker in clean_js
    assert "swarm_nodes" in clean_js or "swarm" in clean_js


def test_t1_f6_pwa_supports_interactive_project_agent_selection(
    jarvis_client: TestClient,
) -> None:
    app_js = jarvis_client.get("/static/app.js").text
    clean_js = _strip_js_comments(app_js)
    assert re.search(r"\bfetch\s*\(", clean_js)
    assert re.search(r"""['"`]/api/projects/select['"`]""", clean_js)
    assert re.search(r"""method\s*:\s*['"]POST['"]""", clean_js, re.IGNORECASE)
    assert re.search(
        r"""addEventListener\s*\(\s*['"](?:click|pointerdown|touchend)['"]|\.onclick\s*=""",
        clean_js,
    )

    sel = jarvis_client.post(
        "/api/projects/select",
        json={"project": "smart-home-ui", "agent": "102"},
    )
    assert sel.status_code == 200
    assert sel.json()["agent_id"] == "102"
    assert jarvis_client.get("/api/health").json()["active_project_agent"]


def test_t1_f6_pwa_has_cast_hud_to_room_tv_button(
    jarvis_client: TestClient,
) -> None:
    html = jarvis_client.get("/").text
    app_js = jarvis_client.get("/static/app.js").text
    inspector = _inspect_html(html)
    assert inspector.has_visible_button_with_text("Cast HUD to Room TV")

    clean_js = _strip_js_comments(app_js)
    assert re.search(r"\bfetch\s*\(", clean_js)
    assert re.search(r"""['"`]/api/cast/hud['"`]""", clean_js)
    assert re.search(r"""method\s*:\s*['"]POST['"]""", clean_js, re.IGNORECASE)
    assert re.search(
        r"""addEventListener\s*\(\s*['"](?:click|pointerdown|touchend)['"]|\.onclick\s*=""",
        clean_js,
    )


def test_t1_f6_pwa_subscribes_to_live_events_and_displays_active_room(
    jarvis_client: TestClient,
) -> None:
    app_js = jarvis_client.get("/static/app.js").text
    clean_js = _strip_js_comments(app_js)
    assert re.search(r"\bnew\s+WebSocket\s*\(", clean_js)
    assert "/ws/live" in clean_js or "/ws/hud" in clean_js
    assert "JSON.parse" in clean_js
    assert any(
        dom_op in clean_js
        for dom_op in ("textContent", "innerHTML", "createElement", "replaceChildren")
    )
    assert "active_room" in clean_js and (
        "active_tv" in clean_js or "voice_status" in clean_js or "active_speaker" in clean_js
    )
    if "/ws/hud" in clean_js:
        assert "hud_snapshot" in clean_js
    else:
        assert "status" in clean_js
    for expected in (
        "project_changed",
        "room_changed",
        "thought",
        "tool_call",
        "task_started",
        "task_completed",
    ):
        assert expected in clean_js


def test_t1_f7_hud_route_serves_html_css_js(
    jarvis_client: TestClient,
) -> None:
    hud = jarvis_client.get("/hud")
    assert hud.status_code == 200
    assert "text/html" in hud.headers.get("content-type", "")
    assert "color-scheme" in hud.text
    assert "/static/hud.js" in hud.text
    assert "/static/hud.css" in hud.text
    assert jarvis_client.get("/static/hud.js").status_code == 200
    assert jarvis_client.get("/static/hud.css").status_code == 200


def test_t1_f7_hud_displays_node_view_active_target_room_and_voice_status(
    jarvis_client: TestClient,
) -> None:
    hud_html = jarvis_client.get("/hud").text
    hud_js = jarvis_client.get("/static/hud.js").text
    inspector = _inspect_html(hud_html)
    assert inspector.has_visible_node_container()

    clean_hud_js = _strip_js_comments(hud_js)
    assert re.search(r"\bnew\s+WebSocket\s*\(", clean_hud_js)
    assert "/ws/hud" in clean_hud_js
    assert "JSON.parse" in clean_hud_js
    assert any(
        dom_op in clean_hud_js
        for dom_op in ("createElement", "innerHTML", "textContent", "replaceChildren", "appendChild")
    )
    for marker in ("active_room", "active_tv", "voice_status", "project_agent", "short_name", "agents"):
        assert marker in clean_hud_js
    assert "swarm_nodes" in clean_hud_js or "recent_events" in clean_hud_js


def test_t1_f7_hud_is_strictly_read_only_no_interactive_inputs(
    jarvis_client: TestClient,
) -> None:
    hud_html = jarvis_client.get("/hud").text
    hud_js = jarvis_client.get("/static/hud.js").text
    inspector = _inspect_html(hud_html)
    tag_names = {t for t, _ in inspector.tags}
    assert not tag_names.intersection({"input", "textarea", "select", "form", "button"})
    assert "contenteditable" not in hud_html.lower()
    clean_hud_js = _strip_js_comments(hud_js)
    assert "/api/projects/select" not in clean_hud_js
    assert "/api/cast/hud" not in clean_hud_js
    assert "/api/agent/ask" not in clean_hud_js
    assert "getUserMedia" not in clean_hud_js
    assert "AudioWorklet" not in clean_hud_js


def test_t1_f7_hud_connects_to_ws_hud_and_handles_all_event_types(
    jarvis_client: TestClient,
) -> None:
    hud_js = jarvis_client.get("/static/hud.js").text
    clean_hud_js = _strip_js_comments(hud_js)
    assert re.search(r"\bnew\s+WebSocket\s*\(", clean_hud_js)
    assert "/ws/hud" in clean_hud_js
    assert "JSON.parse" in clean_hud_js
    assert any(
        dom_op in clean_hud_js
        for dom_op in ("createElement", "innerHTML", "textContent", "replaceChildren")
    )
    for event_name in (
        "hud_snapshot",
        "thought",
        "tool_call",
        "tool_result",
        "token",
        "task_started",
        "task_completed",
        "task_failed",
        "project_changed",
        "room_changed",
        "voice_status",
    ):
        assert event_name in clean_hud_js


def test_t1_f7_hud_tv_styling_10_foot_fullscreen_layout(
    jarvis_client: TestClient,
) -> None:
    hud_css = jarvis_client.get("/static/hud.css").text
    assert any(unit in hud_css for unit in ("100vh", "100dvh", "100vw"))
    assert "clamp(" in hud_css or re.search(r"font-size\s*:\s*(?:[2-9]\dpx|[2-9](?:\.\d+)?rem)", hud_css)


def test_t1_f8_cast_hud_to_cast_tv_via_rest_endpoint(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    resp = jarvis_client.post(
        "/api/cast/hud",
        json={"room": "game_room", "tv": "Game Room TV"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["hud_url"] == "http://192.168.1.50:8765/hud"
    assert data["tv"] == "Game Room TV"
    assert len(mock_chromecast_env.dashcast_calls) == 1
    call = mock_chromecast_env.dashcast_calls[0]
    assert call["url"] == "http://192.168.1.50:8765/hud"
    assert call["force"] is True
    assert call["target_device"] == "Game Room TV"


def test_t1_f8_cast_hud_to_98_qled_and_office_cast_tv(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    r1 = jarvis_client.post("/api/cast/hud", json={"tv": '98" QLED'})
    r2 = jarvis_client.post("/api/cast/hud", json={"tv": "Office TV"})
    assert r1.status_code == 200 and r2.status_code == 200
    targets = [c["target_device"] for c in mock_chromecast_env.dashcast_calls]
    assert targets == ['98" QLED', "Office TV"]
    assert all(c["force"] is True for c in mock_chromecast_env.dashcast_calls)


def test_t1_f8_cast_hud_non_cast_samsung_tizen_tvs_non_fatal_fallback(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    for tv_name in (
        "Office - Samsung Q60CD 65",
        "Living Room TV",
        "Kitchen TV",
        "The Frame",
    ):
        resp = jarvis_client.post("/api/cast/hud", json={"tv": tv_name})
        assert resp.status_code == 200
        data = resp.json()
        for key in ("status", "room", "tv", "hud_url", "protocol", "message"):
            assert key in data
        assert data["tv"] == tv_name
        assert data["hud_url"].endswith("/hud")
        assert isinstance(data["status"], str) and data["status"]
        assert isinstance(data["message"], str) and data["message"]
    assert mock_chromecast_env.dashcast_calls == []


def test_t1_f8_cast_hud_defaults_to_active_presence_room(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    jarvis_client.post("/api/presence/room", json={"room": "game_room"})
    resp = jarvis_client.post("/api/cast/hud", json={})
    assert resp.status_code == 200
    assert len(mock_chromecast_env.dashcast_calls) == 1
    assert mock_chromecast_env.dashcast_calls[0]["target_device"] in ("Game Room TV", '98" QLED')


def test_t1_f8_gemini_live_cast_hud_to_tv_voice_tool(
    e2e_settings: BridgeSettings,
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    e2e_settings.gemini_api_key = "fake-live-key"
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [("fc-cast-1", {"room": "office", "tv": "Office TV"}, "cast_hud_to_tv")]
            )
        ]
    )

    app = create_app(e2e_settings)
    with TestClient(app) as client:
        _sync_discovery(client, mock_chromecast_env)
        with client.websocket_connect("/ws/live") as live_ws:
            _ = recv_until_type(live_ws, "status")
            ev, _ = recv_until_type(live_ws, "tool_call", timeout=4.0)
            assert ev["name"] == "cast_hud_to_tv"
            _wait_until(
                lambda: (
                    len(mock_genai_live.tool_responses) >= 1
                    and len(mock_chromecast_env.dashcast_calls) >= 1
                ),
                timeout=2.0,
                message="Expected cast_hud_to_tv tool_response and dashcast call before closing session",
            )
            mock_genai_live.end_session()

    assert "cast_hud_to_tv" in mock_genai_live.declared_tool_names
    assert len(mock_chromecast_env.dashcast_calls) == 1
    assert mock_chromecast_env.dashcast_calls[0]["target_device"] == "Office TV"
    assert len(mock_genai_live.tool_responses) >= 1


def test_t1_f9_health_reports_jarvis_fields(
    jarvis_client: TestClient,
) -> None:
    jarvis_client.post("/api/presence/room", json={"room": "office"})
    jarvis_client.post("/api/projects/select", json={"project": "clark-home-server", "agent": "101"})
    resp = jarvis_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["lan_url"] == "http://192.168.1.50:8765"
    assert data["speakers_count"] >= 3
    for key in ("active_room", "active_presence_room", "active_tv", "active_project", "active_project_agent"):
        assert key in data


def test_t1_f9_speakers_endpoint_distinguishes_speakers_and_tvs(
    jarvis_client: TestClient,
) -> None:
    resp = jarvis_client.get("/api/speakers")
    assert resp.status_code == 200
    data = resp.json()
    assert "speakers" in data
    by_name = {d["name"]: d for d in data["speakers"]}
    assert by_name["Office"]["cast_type"] == "audio"
    assert by_name["All Speakers"]["cast_type"] == "group"
    if "tvs" in data:
        tvs_by_name = {d["name"]: d for d in data["tvs"]}
        assert tvs_by_name["Game Room TV"]["cast_type"] == "cast"
    elif "Game Room TV" in by_name:
        assert by_name["Game Room TV"]["cast_type"] == "cast"


def test_t1_f9_backward_compat_agent_ask_and_warmup_wav(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    jarvis_client.post("/api/presence/room", json={"room": "office"})
    _wait_until(
        lambda: any("warmup_silence.wav" in c["url"] for c in mock_chromecast_env.play_media_calls),
        timeout=2.0,
        message="Expected background warm_room() to generate and play warmup_silence.wav",
    )
    wav_resp = jarvis_client.get("/api/tts/warmup_silence.wav")
    assert wav_resp.status_code == 200
    assert wav_resp.content[:4] == b"RIFF"

    ask_resp = jarvis_client.post(
        "/api/agent/ask",
        json={"prompt": "Report status", "cast_response": True},
    )
    assert ask_resp.status_code == 200
    body = ask_resp.json()
    assert body["text"]
    assert body["active_room"] == "Office"
    assert any(
        c["speaker"] == "Office" and c["content_type"] == "audio/mp3"
        for c in mock_chromecast_env.play_media_calls
    )


# ============================================================================
# TIER 2 — Boundary & Corner Cases (45 Tests: F1-F9)
# ============================================================================
def test_t2_f1_room_name_normalization_variants(
    jarvis_client: TestClient,
) -> None:
    cases = [
        ("Office", "Office"),
        ("LIVING ROOM", "Living Room speaker"),
        ("living-room", "Living Room speaker"),
        ('98" QLED', "All Speakers"),
        ("98_qled", "All Speakers"),
        ("qled", "All Speakers"),
        ("Master Bedroom", "All Speakers"),
    ]
    for raw_room, expected_speaker in cases:
        resp = jarvis_client.post("/api/presence/room", json={"room": raw_room})
        assert resp.status_code == 200, f"Failed for room={raw_room!r}: {resp.text}"
        assert resp.json()["active_room"] == expected_speaker


def test_t2_f1_speaker_and_tv_overrides_honored(
    jarvis_client: TestClient,
) -> None:
    resp = jarvis_client.post(
        "/api/presence/room",
        json={"room": "office", "speaker": "All Speakers", "tv": '98" QLED'},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["active_room"] == "All Speakers"
    assert data["active_tv"] == '98" QLED'


def test_t2_f1_unknown_room_and_empty_whitespace_handling(
    jarvis_client: TestClient,
) -> None:
    assert jarvis_client.post("/api/presence/room", json={"room": ""}).status_code in (400, 422)
    assert jarvis_client.post("/api/presence/room", json={}).status_code == 422

    unk = jarvis_client.post("/api/presence/room", json={"room": "garage"})
    assert unk.status_code < 500
    assert jarvis_client.get("/api/health").status_code == 200


def test_t2_f1_kitchen_falls_back_to_all_speakers_when_living_room_offline(
    e2e_settings: BridgeSettings,
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    mock_chromecast_env.remove_device("Living Room speaker")
    app = create_app(e2e_settings)
    with TestClient(app) as client:
        _sync_discovery(client, mock_chromecast_env)
        resp = client.post("/api/presence/room", json={"room": "kitchen"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_room"] == "All Speakers"
        assert data["active_tv"] == "Kitchen TV"


def test_t2_f1_warm_room_failure_does_not_fail_presence_webhook(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    mock_chromecast_env.set_device_wait_error("Office", RuntimeError("Cast socket timeout"))
    with jarvis_client.websocket_connect("/ws/hud") as ws:
        _ = recv_json_with_timeout(ws)
        resp = jarvis_client.post("/api/presence/room", json={"room": "office"})
        assert resp.status_code == 200
        assert resp.json()["active_room"] == "Office"
        ev, _ = recv_until_type(ws, "room_changed")
        assert ev["active_room"] == "Office"


def test_t2_f2_single_repo_workspace_fallback(
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    single_dir = isolated_workspaces["root"] / "single_repo"
    _init_git_repo(single_dir, branch="develop", dirty_untracked=1)
    (single_dir / "src").mkdir()
    (single_dir / "src" / "main.py").write_text("print('ok')\n", encoding="utf-8")

    cfg = BridgeSettings(
        _env_file=None,
        AGY_WORKSPACE_DIR=str(single_dir),
        AGY_SAVE_DIR=str(isolated_workspaces["root"] / "single_save"),
        AGY_APP_DATA_DIR=str(isolated_workspaces["root"] / "single_app"),
        HOST_LAN_IP="192.168.1.50",
        ENABLE_HTTPS=False,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        projects = client.get("/api/projects").json()["projects"]
        assert len(projects) >= 1
        assert any(Path(p["path"]).resolve() == single_dir.resolve() for p in projects)
        fallback = next(p for p in projects if Path(p["path"]).resolve() == single_dir.resolve())
        assert fallback["git_branch"] == "develop"


def test_t2_f2_empty_workspace_fallback(
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    empty_dir = isolated_workspaces["root"] / "empty_ws"
    empty_dir.mkdir(parents=True, exist_ok=True)
    (empty_dir / "README.md").write_text("Just docs\n", encoding="utf-8")

    cfg = BridgeSettings(
        _env_file=None,
        AGY_WORKSPACE_DIR=str(empty_dir),
        AGY_SAVE_DIR=str(isolated_workspaces["root"] / "empty_save"),
        AGY_APP_DATA_DIR=str(isolated_workspaces["root"] / "empty_app"),
        HOST_LAN_IP="192.168.1.50",
        ENABLE_HTTPS=False,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        projects = client.get("/api/projects").json()["projects"]
        assert len(projects) == 1
        p = projects[0]
        assert p["git_branch"] is None
        assert p["uncommitted_changes"] == 0
        assert p["active_agent_id"] == "101"


def test_t2_f2_non_git_subdirectory_handled_gracefully(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    plain = isolated_workspaces["workspace_dir"] / "notes-plain"
    plain.mkdir()
    (plain / "todo.md").write_text("- item\n", encoding="utf-8")

    app = create_app(e2e_settings)
    with TestClient(app) as client:
        projects = {p["name"]: p for p in client.get("/api/projects").json()["projects"]}
        assert "notes-plain" in projects
        assert projects["notes-plain"]["git_branch"] is None
        assert projects["notes-plain"]["uncommitted_changes"] == 0


def test_t2_f2_short_name_collision_disambiguation(
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    collide_ws = isolated_workspaces["root"] / "collide_ws"
    _init_git_repo(collide_ws / "home-app", branch="main")
    _init_git_repo(collide_ws / "home_app", branch="main")

    cfg = BridgeSettings(
        _env_file=None,
        AGY_WORKSPACE_DIR=str(collide_ws),
        AGY_SAVE_DIR=str(isolated_workspaces["root"] / "collide_save"),
        AGY_APP_DATA_DIR=str(isolated_workspaces["root"] / "collide_app"),
        HOST_LAN_IP="192.168.1.50",
        ENABLE_HTTPS=False,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        projects = client.get("/api/projects").json()["projects"]
        assert len(projects) == 2
        ids = {p["id"] for p in projects}
        short_names = {p["short_name"].lower() for p in projects}
        assert len(ids) == 2
        assert len(short_names) == 2


def test_t2_f2_corrupted_projects_state_json_recovery(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    state_file = isolated_workspaces["save_dir"] / "projects_state.json"
    state_file.write_text("{corrupt", encoding="utf-8")

    app = create_app(e2e_settings)
    with TestClient(app) as client:
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active_project_id"] is not None
        assert data["active_agent_id"] == "101"
        assert len(data["projects"]) >= 3


def test_t2_f3_unknown_project_selection_returns_clean_error(
    jarvis_client: TestClient,
) -> None:
    before = jarvis_client.post(
        "/api/projects/select",
        json={"project": "clark-home-server", "agent": "101"},
    ).json()
    bad = jarvis_client.post(
        "/api/projects/select",
        json={"project": "nonexistent-repo-xyz"},
    )
    assert bad.status_code in (400, 404)
    after = jarvis_client.get("/api/projects").json()
    assert after["active_project_id"] == before["project_id"]


def test_t2_f3_select_case_insensitive_and_substring_fuzzy_match(
    jarvis_client: TestClient,
) -> None:
    r1 = jarvis_client.post("/api/projects/select", json={"project": "PROXMOX"})
    assert r1.status_code == 200
    assert r1.json()["project_name"] == "proxmox-monitor"

    r2 = jarvis_client.post("/api/projects/select", json={"project": "clark"})
    assert r2.status_code == 200
    assert r2.json()["project_name"] == "clark-home-server"


def test_t2_f3_select_agent_without_project_keeps_current_project(
    jarvis_client: TestClient,
) -> None:
    base = jarvis_client.post(
        "/api/projects/select", json={"project": "proxmox-monitor", "agent": "101"}
    ).json()
    switched = jarvis_client.post("/api/projects/select", json={"agent": "104"})
    assert switched.status_code == 200
    data = switched.json()
    assert data["project_id"] == base["project_id"]
    assert data["agent_id"] == "104"


def test_t2_f3_natural_voice_phrasing_parsing(
    jarvis_client: TestClient,
) -> None:
    phrases = [
        ("project Clark Home Server, agent 202", "clark-home-server", "202"),
        ("proxmox-monitor agent 303", "proxmox-monitor", "303"),
        ("smart home ui, agent 101", "smart-home-ui", "101"),
    ]
    for phrase, expected_name, expected_agent in phrases:
        resp = jarvis_client.post("/api/projects/select", json={"project_agent": phrase})
        assert resp.status_code == 200, f"Failed phrase {phrase!r}: {resp.text}"
        body = resp.json()
        assert body["project_name"] == expected_name
        assert body["agent_id"] == expected_agent


async def test_t2_f3_non_mutating_get_target_does_not_overwrite_active_selection(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    clark = app.state.project_manager.select_project(project="clark-home-server", agent="101")

    q = app.state.event_hub.subscribe()
    try:
        app.state.agent_engine.start_background_task(
            "One-off override", project="proxmox-monitor", agent="109"
        )
        await wait_for_hub_event(q, "task_completed")
    finally:
        app.state.event_hub.unsubscribe(q)

    summary = app.state.project_manager.get_state_summary()
    assert summary["active_project_id"] == clark["project_id"]
    assert summary["active_agent_id"] == "101"


async def test_t2_f4_dead_ws_subscriber_does_not_block_healthy_subscribers(
    e2e_settings: BridgeSettings,
) -> None:
    app = create_app(e2e_settings)
    hub = app.state.event_hub
    dead_q = hub.subscribe()
    healthy_q = hub.subscribe()
    dead_q.put_nowait = MagicMock(side_effect=RuntimeError("closed socket"))
    try:
        await hub.publish("thought", project_id="p1", agent_id="101", text="Testing isolation")
        ev = await asyncio.wait_for(healthy_q.get(), timeout=1.0)
        assert ev["type"] == "thought"
        assert ev["text"] == "Testing isolation"
    finally:
        hub.unsubscribe(dead_q)
        hub.unsubscribe(healthy_q)


async def test_t2_f4_ring_buffer_bounded_at_200_events(
    e2e_settings: BridgeSettings,
) -> None:
    app = create_app(e2e_settings)
    hub = app.state.event_hub
    for idx in range(250):
        await hub.publish("token", seq=idx, text=f"tok-{idx}")
    recent = hub.recent_events(limit=300)
    assert len(recent) <= 200
    assert recent[-1]["seq"] == 249
    assert recent[0]["seq"] >= 50


async def test_t2_f4_event_payload_metadata_completeness(
    e2e_settings: BridgeSettings,
) -> None:
    app = create_app(e2e_settings)
    hub = app.state.event_hub
    e1 = await hub.publish("voice_status", status="speaking")
    e2 = await hub.publish("voice_status", status="listening")
    _assert_iso8601_utc(e1["timestamp"])
    _assert_iso8601_utc(e2["timestamp"])
    assert e1["event_id"] and e2["event_id"]
    assert e1["event_id"] != e2["event_id"]


def test_t2_f4_rapid_hud_connect_disconnect_reconnect_cycle(
    jarvis_client: TestClient,
) -> None:
    for idx in range(5):
        with jarvis_client.websocket_connect("/ws/hud") as ws:
            snap = recv_json_with_timeout(ws)
            assert snap["type"] == "hud_snapshot"
        jarvis_client.post(
            "/api/presence/room",
            json={"room": "office" if idx % 2 == 0 else "game_room"},
        )

    with jarvis_client.websocket_connect("/ws/hud") as ws_final:
        snap = recv_json_with_timeout(ws_final)
        assert snap["type"] == "hud_snapshot"
        jarvis_client.post("/api/presence/room", json={"room": "kitchen"})
        ev, _ = recv_until_type(ws_final, "room_changed")
        assert ev["active_tv"] == "Kitchen TV"


def test_t2_f4_ws_live_preserves_control_protocol_with_event_hub(
    jarvis_client: TestClient,
) -> None:
    with jarvis_client.websocket_connect("/ws/live") as ws:
        init = recv_json_with_timeout(ws)
        assert init["type"] == "status" and init["state"] == "connected"

        ws.send_json({"type": "set_room", "room": "Office"})
        st_room, _ = recv_until_type(ws, "status")
        assert st_room["active_room"] == "Office"

        ws.send_json({"type": "set_routing", "mode": "direct"})
        st_route, _ = recv_until_type(ws, "status")
        assert st_route["routing_mode"] == "direct"

        ws.send_json({"type": "barge_in"})
        intr, _ = recv_until_type(ws, "interrupted")
        assert intr["type"] == "interrupted"

        ws.send_json({"type": "text", "text": "Hello bridge"})
        agent_msg, _ = recv_until_type(
            ws, "transcript", timeout=4.0
        )
        # First transcript may be user; collect until agent transcript arrives
        if agent_msg.get("role") != "agent":
            for _ in range(15):
                nxt = recv_json_with_timeout(ws)
                if nxt.get("type") == "transcript" and nxt.get("role") == "agent":
                    agent_msg = nxt
                    break
        assert agent_msg.get("role") == "agent"


async def test_t2_f5_two_concurrent_background_tasks_same_project_different_agents(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    gate_101 = asyncio.Event()
    gate_102 = asyncio.Event()
    mock_antigravity_sdk.queue_turn(
        [Text("Agent 101 finished")],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-same-101",
        hold_event=gate_101,
    )
    mock_antigravity_sdk.queue_turn(
        [Text("Agent 102 finished")],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-same-102",
        hold_event=gate_102,
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    q = app.state.event_hub.subscribe()
    try:
        a1 = app.state.agent_engine.start_background_task(
            "Task 101", project="clark-home-server", agent="101"
        )
        a2 = app.state.agent_engine.start_background_task(
            "Task 102", project="clark-home-server", agent="102"
        )
        assert a1["task_id"] != a2["task_id"]
        # Out-of-order completion: 102 finishes before 101
        gate_102.set()
        ev_first, _ = await wait_for_hub_event(q, "task_completed")
        assert ev_first["agent_id"] == "102"
        gate_101.set()
        ev_second, _ = await wait_for_hub_event(q, "task_completed")
        assert ev_second["agent_id"] == "101"
    finally:
        app.state.event_hub.unsubscribe(q)

    clark = next(
        p for p in app.state.project_manager.get_state_summary()["projects"]
        if p["name"] == "clark-home-server"
    )
    agents = {a["agent_id"]: a for a in clark["agents"]}
    assert agents["101"]["conversation_id"] == "conv-same-101"
    assert agents["102"]["conversation_id"] == "conv-same-102"
    assert "101" in (agents["101"]["last_summary"] or "")
    assert "102" in (agents["102"]["last_summary"] or "")


async def test_t2_f5_empty_task_prompt_or_empty_sdk_text_handled_gracefully(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    mock_antigravity_sdk.queue_turn(
        [Text("   ")],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-empty-text",
    )
    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    q = app.state.event_hub.subscribe()
    try:
        app.state.agent_engine.start_background_task("Silent turn", project="clark-home-server")
        ev, _ = await wait_for_hub_event(q, "task_completed", timeout=3.0)
        assert ev["type"] == "task_completed"
    finally:
        app.state.event_hub.unsubscribe(q)
    assert mock_chromecast_env.tts_texts == []


async def test_t2_f5_long_agent_output_summarized_for_spoken_broadcast(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    long_text = (
        "## Diagnostic Report\n\n"
        + ("Proxmox node 1 CPU and memory telemetry nominal. " * 20)
        + "\n```bash\nsystemctl status agy-home-bridge\n```"
    )
    assert len(long_text) > 600
    mock_antigravity_sdk.queue_turn(
        [Text(long_text)],
        workspace=isolated_workspaces["clark"],
    )
    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    q = app.state.event_hub.subscribe()
    try:
        app.state.agent_engine.start_background_task("Full diagnostics")
        ev, _ = await wait_for_hub_event(q, "task_completed")
        await _async_wait_until(
            lambda: bool(mock_chromecast_env.tts_texts),
            timeout=2.0,
            message="Expected completion TTS summary for long agent output",
        )
    finally:
        app.state.event_hub.unsubscribe(q)

    assert mock_chromecast_env.tts_texts
    spoken = mock_chromecast_env.tts_texts[-1]
    assert len(spoken) <= 450
    assert len(spoken) < len(long_text)
    assert "```" not in spoken
    assert "##" not in spoken
    assert ev.get("summary") or ev.get("text")


async def test_t2_f5_tts_broadcast_error_does_not_lose_task_completion(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_antigravity_sdk.queue_turn(
        [Text("Build succeeded cleanly.")],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-tts-err-1",
    )
    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    monkeypatch.setattr(
        app.state.cast_manager,
        "speak_on_room",
        AsyncMock(side_effect=RuntimeError("Speaker unreachable")),
    )
    q = app.state.event_hub.subscribe()
    try:
        app.state.agent_engine.start_background_task("Run build", project="clark-home-server", agent="101")
        ev, _ = await wait_for_hub_event(q, "task_completed", timeout=3.0)
        assert ev["conversation_id"] == "conv-tts-err-1"
    finally:
        app.state.event_hub.unsubscribe(q)

    clark = next(
        p for p in app.state.project_manager.get_state_summary()["projects"]
        if p["name"] == "clark-home-server"
    )
    a101 = next(a for a in clark["agents"] if a["agent_id"] == "101")
    assert a101["conversation_id"] == "conv-tts-err-1"


async def test_t2_f5_fallback_when_google_antigravity_not_installed(
    e2e_settings: BridgeSettings,
    mock_chromecast_env: CastEnvironmentSpy,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import google

    monkeypatch.setitem(sys.modules, "google.antigravity", None)  # forces ImportError
    monkeypatch.delattr(google, "antigravity", raising=False)

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    q = app.state.event_hub.subscribe()
    try:
        app.state.agent_engine.start_background_task("Check offline fallback")
        ev, seen = await wait_for_hub_event(q, "task_completed", timeout=3.0)
        assert any(e["type"] == "task_started" for e in seen)
        assert ev["type"] == "task_completed"
    finally:
        app.state.event_hub.unsubscribe(q)


def test_t2_f6_pwa_renders_with_empty_or_single_project_workspace(
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    empty_dir = isolated_workspaces["root"] / "pwa_empty"
    empty_dir.mkdir()
    (empty_dir / "README.md").write_text("empty\n", encoding="utf-8")
    cfg = BridgeSettings(
        _env_file=None,
        AGY_WORKSPACE_DIR=str(empty_dir),
        AGY_SAVE_DIR=str(isolated_workspaces["root"] / "pwa_save"),
        AGY_APP_DATA_DIR=str(isolated_workspaces["root"] / "pwa_app"),
        HOST_LAN_IP="192.168.1.50",
        ENABLE_HTTPS=False,
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        projs = client.get("/api/projects").json()["projects"]
        assert len(projs) >= 1


def test_t2_f6_pwa_cast_hud_action_handles_non_cast_samsung_tv_gracefully(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    jarvis_client.post("/api/presence/room", json={"room": "kitchen"})
    resp = jarvis_client.post("/api/cast/hud", json={})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tv"] == "Kitchen TV"
    assert body["hud_url"].endswith("/hud")
    assert body["message"]
    assert mock_chromecast_env.dashcast_calls == []


def test_t2_f6_pwa_retains_accessibility_and_dark_mode_meta(
    jarvis_client: TestClient,
) -> None:
    html = jarvis_client.get("/").text
    assert 'name="color-scheme" content="light dark"' in html
    assert 'closedby="any"' in html
    assert "aria-live" in html
    assert "Antigravity Live" in html


def test_t2_f6_pwa_api_projects_select_empty_payload_returns_4xx_or_current(
    jarvis_client: TestClient,
) -> None:
    resp = jarvis_client.post("/api/projects/select", json={})
    assert resp.status_code in (200, 400, 422)
    after = jarvis_client.get("/api/projects").json()
    assert isinstance(after.get("active_project_id"), str) and after["active_project_id"]
    assert isinstance(after.get("active_agent_id"), str) and after["active_agent_id"]
    if resp.status_code == 200:
        body = resp.json()
        assert body.get("project_id") == after["active_project_id"]
        assert body.get("agent_id") == after["active_agent_id"]
    assert jarvis_client.get("/api/health").status_code == 200


def test_t2_f6_pwa_rapid_node_switches_converge_to_last_selection(
    jarvis_client: TestClient, isolated_workspaces: dict[str, Path]
) -> None:
    sequence = [
        {"project": "clark-home-server", "agent": "101"},
        {"project": "proxmox-monitor", "agent": "102"},
        {"project": "smart-home-ui", "agent": "103"},
        {"project": "clark-home-server", "agent": "104"},
        {"project": "proxmox-monitor", "agent": "105"},
    ]
    for item in sequence:
        r = jarvis_client.post("/api/projects/select", json=item)
        assert r.status_code == 200

    summary = jarvis_client.get("/api/projects").json()
    assert summary["active_agent_id"] == "105"
    active_proj = next(p for p in summary["projects"] if p["active"])
    assert active_proj["name"] == "proxmox-monitor"
    persisted = json.loads(
        (isolated_workspaces["save_dir"] / "projects_state.json").read_text(encoding="utf-8")
    )
    assert persisted.get("active_agent_id") == "105"


def test_t2_f7_hud_serves_correctly_when_no_speakers_or_tvs_discovered(
    e2e_settings: BridgeSettings,
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    mock_chromecast_env.clear_devices()
    app = create_app(e2e_settings)
    with TestClient(app) as client:
        assert client.get("/hud").status_code == 200
        with client.websocket_connect("/ws/hud") as ws:
            snap = recv_json_with_timeout(ws)
            assert snap["type"] == "hud_snapshot"
            assert isinstance(snap["projects"], list)


def test_t2_f7_hud_html_escapes_or_avoids_inline_script_xss_sinks(
    jarvis_client: TestClient,
) -> None:
    hud_html = jarvis_client.get("/hud").text
    hud_js = jarvis_client.get("/static/hud.js").text
    assert "<script>" not in hud_html.lower()  # scripts loaded from /static/hud.js
    clean_hud_js = _strip_js_comments(hud_js)
    assert "textContent" in clean_hud_js or "escapeHtml" in clean_hud_js


def test_t2_f7_hud_head_and_static_cache_headers(
    jarvis_client: TestClient,
) -> None:
    hud_r = jarvis_client.get("/hud")
    js_r = jarvis_client.get("/static/hud.js")
    css_r = jarvis_client.get("/static/hud.css")
    assert hud_r.status_code == 200 and "text/html" in hud_r.headers.get("content-type", "")
    assert js_r.status_code == 200 and "javascript" in js_r.headers.get("content-type", "")
    assert css_r.status_code == 200 and "text/css" in css_r.headers.get("content-type", "")
    assert hud_r.text.strip() and js_r.text.strip() and css_r.text.strip()


def test_t2_f7_hud_snapshot_includes_active_swarm_and_recent_events_after_activity(
    jarvis_client: TestClient,
) -> None:
    jarvis_client.post("/api/presence/room", json={"room": "office"})
    sel = jarvis_client.post(
        "/api/projects/select", json={"project": "proxmox-monitor", "agent": "102"}
    ).json()
    jarvis_client.app.state.project_manager.update_agent_status(
        sel["project_id"],
        "102",
        status="running",
        current_task="Inspect alerts",
        swarm_nodes=[{"node_id": "worker-1", "status": "running"}],
    )
    jarvis_client.post("/api/agent/ask", json={"prompt": "Check node", "cast_response": False})

    with jarvis_client.websocket_connect("/ws/hud") as ws:
        snap = recv_json_with_timeout(ws)
        assert snap["active_presence_room"] == "office" or snap["active_room"] == "Office"
        assert snap["active_agent_id"] == "102"
        proxmox = next(p for p in snap["projects"] if p["id"] == sel["project_id"])
        a102 = next(a for a in proxmox["agents"] if a["agent_id"] == "102")
        assert len(a102["swarm_nodes"]) == 1
        assert len(snap["recent_events"]) >= 2


def test_t2_f7_hud_url_matches_cast_hud_target_url(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    cast_res = jarvis_client.post("/api/cast/hud", json={"tv": "Office TV"}).json()
    hud_url = cast_res["hud_url"]
    assert hud_url.endswith("/hud")
    assert jarvis_client.get("/hud").status_code == 200


def test_t2_f8_explicit_tv_override_beats_room_mapping(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    jarvis_client.post("/api/presence/room", json={"room": "office"})
    resp = jarvis_client.post("/api/cast/hud", json={"tv": "Game Room TV"})
    assert resp.status_code == 200
    assert len(mock_chromecast_env.dashcast_calls) == 1
    assert mock_chromecast_env.dashcast_calls[0]["target_device"] == "Game Room TV"


def test_t2_f8_offline_cast_tv_returns_structured_non_fatal_result(
    e2e_settings: BridgeSettings,
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    mock_chromecast_env.remove_device("Game Room TV")
    app = create_app(e2e_settings)
    with TestClient(app) as client:
        _sync_discovery(client, mock_chromecast_env)
        resp = client.post("/api/cast/hud", json={"tv": "Game Room TV"})
        assert resp.status_code == 200
        body = resp.json()
        for key in ("status", "tv", "hud_url", "message"):
            assert key in body
        assert body["tv"] == "Game Room TV"
        assert mock_chromecast_env.dashcast_calls == []


def test_t2_f8_dashcast_exception_returns_structured_error_without_500(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    mock_chromecast_env.set_dashcast_error(RuntimeError("DashCast launch failed"))
    resp = jarvis_client.post("/api/cast/hud", json={"tv": "Office TV"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["tv"] == "Office TV"
    assert data["hud_url"].endswith("/hud")
    assert data["message"]
    assert jarvis_client.get("/api/health").status_code == 200


def test_t2_f8_cast_never_triggered_by_agent_ask_or_speak_or_room_select(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    jarvis_client.post("/api/presence/room", json={"room": "game_room"})
    jarvis_client.post("/api/cast/select", json={"room": "Office", "routing_mode": "cast"})
    jarvis_client.post("/api/cast/speak", json={"text": "Test announcement", "room": "Office"})
    jarvis_client.post("/api/agent/ask", json={"prompt": "Say hi", "cast_response": True})
    jarvis_client.post("/api/projects/select", json={"project": "proxmox-monitor"})
    assert mock_chromecast_env.dashcast_calls == []


def test_t2_f8_case_insensitive_and_quoted_tv_name_matching(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    r1 = jarvis_client.post("/api/cast/hud", json={"tv": "game room tv"})
    r2 = jarvis_client.post("/api/cast/hud", json={"tv": '98" qled'})
    assert r1.status_code == 200 and r2.status_code == 200
    assert [c["target_device"] for c in mock_chromecast_env.dashcast_calls] == [
        "Game Room TV",
        '98" QLED',
    ]


def test_t2_f9_cast_select_unknown_speaker_and_invalid_routing_mode(
    jarvis_client: TestClient,
) -> None:
    bad = jarvis_client.post(
        "/api/cast/select",
        json={"room": "Office", "routing_mode": "invalid_routing_xyz"},
    )
    assert bad.status_code == 400

    unk = jarvis_client.post("/api/cast/select", json={"room": "Unknown Room 99"})
    assert unk.status_code == 200
    assert jarvis_client.get("/api/health").status_code == 200


def test_t2_f9_cast_speak_blank_text_returns_422(
    jarvis_client: TestClient,
) -> None:
    empty = jarvis_client.post("/api/cast/speak", json={"text": ""})
    assert empty.status_code == 422


def test_t2_f9_agent_ask_empty_prompt_returns_422(
    jarvis_client: TestClient,
) -> None:
    resp = jarvis_client.post("/api/agent/ask", json={"prompt": ""})
    assert resp.status_code == 422


def test_t2_f9_tts_missing_file_returns_404_and_path_traversal_safe(
    jarvis_client: TestClient,
) -> None:
    assert jarvis_client.get("/api/tts/nonexistent_audio_file.mp3").status_code == 404
    assert jarvis_client.get("/api/tts/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_t2_f9_health_without_active_speakers_still_returns_200(
    e2e_settings: BridgeSettings,
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    mock_chromecast_env.clear_devices()
    app = create_app(e2e_settings)
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["speakers_count"] == 0
        assert data["active_project"]


# ============================================================================
# TIER 3 — Cross-Feature Pairwise Integration Tests (12 Tests: T3-01 .. T3-12)
# ============================================================================
async def test_t3_01_presence_move_mid_task_routes_completion_to_new_room(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    """T3-01 (F1 x F5): Presence move mid-task routes completion TTS to new room speaker."""
    e2e_settings.gemini_api_key = "fake-live-key"
    hold = asyncio.Event()
    mock_antigravity_sdk.queue_turn(
        [Thought("Auditing"), Text("Kitchen audit complete.")],
        workspace=isolated_workspaces["clark"],
        hold_event=hold,
    )
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [("fc-t3-1", {"task": "Audit bridge"}, "delegate_to_antigravity_agent")]
            )
        ]
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    fake_ws = FakeWebSocket()
    live_task = asyncio.create_task(app.state.live_bridge.handle_websocket(fake_ws))
    q = app.state.event_hub.subscribe()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            await ac.post("/api/presence/room", json={"room": "office", "source": "frigate_ha"})
            for _ in range(50):
                if mock_genai_live.tool_response_event and mock_genai_live.tool_response_event.is_set():
                    break
                await asyncio.sleep(0.02)
            assert mock_genai_live.tool_response_event is not None
            await asyncio.wait_for(mock_genai_live.tool_response_event.wait(), timeout=2.0)
            assert not hold.is_set()
            assert "started" in json.dumps(mock_genai_live.tool_responses[0], default=str).lower()

            mock_chromecast_env.play_media_calls.clear()
            await ac.post("/api/presence/room", json={"room": "kitchen", "source": "frigate_ha"})
            health = (await ac.get("/api/health")).json()
            assert health["active_room"] == "Living Room speaker"

            hold.set()
            await wait_for_hub_event(q, "task_completed", timeout=3.0)
            await _async_wait_until(
                lambda: any(
                    c["content_type"] == "audio/mp3" for c in mock_chromecast_env.play_media_calls
                ),
                timeout=2.0,
                message="Expected completion TTS playback on Kitchen -> Living Room speaker",
            )
    finally:
        app.state.event_hub.unsubscribe(q)
        mock_genai_live.end_session()
        fake_ws.disconnect()
        await asyncio.wait_for(live_task, timeout=2.0)

    mp3_calls = [c for c in mock_chromecast_env.play_media_calls if c["content_type"] == "audio/mp3"]
    assert mp3_calls and mp3_calls[-1]["speaker"] == "Living Room speaker"
    assert not any(c["speaker"] == "Office" for c in mp3_calls)


def test_t3_02_project_manager_switcher_and_event_hub_sync(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    """T3-02 (F2 x F3 x F4): REST select + turn + voice switch_project -> HUD snapshot sync."""
    e2e_settings.gemini_api_key = "fake-live-key"
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [("fc-t3-2", {"project": "clark-home-server", "agent": "101"}, "switch_project")]
            )
        ]
    )

    app = create_app(e2e_settings)
    with TestClient(app) as client:
        _sync_discovery(client, mock_chromecast_env)
        with client.websocket_connect("/ws/hud") as hud_ws:
            _ = recv_json_with_timeout(hud_ws)
            sel = client.post(
                "/api/projects/select",
                json={"project_agent": "project proxmox monitor, agent 102"},
            )
            assert sel.status_code == 200
            ev1, _ = recv_until_type(hud_ws, "project_changed")
            assert ev1["project_name"] == "proxmox-monitor"
            assert ev1["agent_id"] == "102"

            client.post("/api/agent/ask", json={"prompt": "Inspect proxmox", "cast_response": False})

            with client.websocket_connect("/ws/live") as live_ws:
                _ = recv_until_type(live_ws, "status")
                ev2, _ = recv_until_type(hud_ws, "project_changed", timeout=4.0)
                assert ev2["project_name"] == "clark-home-server"
                assert ev2["agent_id"] == "101"
                _wait_until(
                    lambda: len(mock_genai_live.tool_responses) >= 1,
                    timeout=2.0,
                    message="Expected switch_project tool_response before closing live_ws",
                )
                mock_genai_live.end_session()

        client.post("/api/agent/ask", json={"prompt": "Inspect clark", "cast_response": False})

        with client.websocket_connect("/ws/hud") as hud_ws2:
            snap2 = recv_json_with_timeout(hud_ws2)
            assert "clark" in snap2["active_project_agent"].lower()
            assert "101" in snap2["active_project_agent"]

    assert mock_antigravity_sdk.configs[0]["workspaces"] == [
        str(isolated_workspaces["proxmox"].resolve())
    ]
    assert mock_antigravity_sdk.configs[1]["workspaces"] == [
        str(isolated_workspaces["clark"].resolve())
    ]


async def test_t3_03_voice_switch_delegate_and_cast_hud_tool_chain(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    """T3-03 (F3 x F5 x F8): Voice tool chain switch_project -> delegate -> cast_hud_to_tv."""
    e2e_settings.gemini_api_key = "fake-live-key"
    hold = asyncio.Event()
    mock_antigravity_sdk.queue_turn(
        [Thought("Refactoring"), Text("Auth middleware refactored.")],
        workspace=isolated_workspaces["clark"],
        hold_event=hold,
    )
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [
                    ("fc-1", {"project_agent": "project clark home server, agent 101"}, "switch_project"),
                    ("fc-2", {"task": "Refactor auth"}, "delegate_to_antigravity_agent"),
                    ("fc-3", {"room": "game_room"}, "cast_hud_to_tv"),
                ]
            )
        ]
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    fake_ws = FakeWebSocket()
    live_task = asyncio.create_task(app.state.live_bridge.handle_websocket(fake_ws))
    q = app.state.event_hub.subscribe()
    try:
        for _ in range(50):
            if mock_genai_live.tool_response_event and mock_genai_live.tool_response_event.is_set():
                break
            await asyncio.sleep(0.02)
        assert mock_genai_live.tool_response_event is not None
        await asyncio.wait_for(mock_genai_live.tool_response_event.wait(), timeout=2.0)
        assert not hold.is_set()
        assert len(mock_chromecast_env.dashcast_calls) == 1
        assert mock_chromecast_env.dashcast_calls[0]["force"] is True
        assert mock_chromecast_env.dashcast_calls[0]["url"].endswith("/hud")

        hold.set()
        ev, _ = await wait_for_hub_event(q, "task_completed", timeout=3.0)
        assert ev["project_name"] == "clark-home-server"
    finally:
        app.state.event_hub.unsubscribe(q)
        mock_genai_live.end_session()
        fake_ws.disconnect()
        await asyncio.wait_for(live_task, timeout=2.0)


async def test_t3_04_concurrent_multi_project_agent_scoped_streams_no_bleed(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    """T3-04 (F2 x F4 x F5): Concurrent overlapping tasks emit isolated scoped streams."""
    gate_a = asyncio.Event()
    gate_b = asyncio.Event()
    mock_antigravity_sdk.queue_turn(
        [
            Thought("Clark thinking"),
            ToolCall("read_file", {"path": "a.py"}, result="a-ok"),
            Text("Clark result"),
        ],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-alpha-101",
        hold_event=gate_a,
    )
    mock_antigravity_sdk.queue_turn(
        [
            Thought("Proxmox thinking"),
            ToolCall("read_file", {"path": "b.py"}, result="b-ok"),
            Text("Proxmox result"),
        ],
        workspace=isolated_workspaces["proxmox"],
        conversation_id="conv-beta-102",
        hold_event=gate_b,
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    q_hud = app.state.event_hub.subscribe()
    q_live = app.state.event_hub.subscribe()
    try:
        ack1 = app.state.agent_engine.start_background_task(
            "Task alpha", project="clark-home-server", agent="101"
        )
        ack2 = app.state.agent_engine.start_background_task(
            "Task beta", project="proxmox-monitor", agent="102"
        )
        await wait_for_hub_event(q_hud, "thought", predicate=lambda e: e["task_id"] == ack1["task_id"])
        await wait_for_hub_event(q_hud, "thought", predicate=lambda e: e["task_id"] == ack2["task_id"])

        gate_b.set()
        gate_a.set()
        _, seen_hud_1 = await wait_for_hub_event(
            q_hud, "task_completed", predicate=lambda e: e["task_id"] == ack1["task_id"]
        )
        _, seen_hud_2 = await wait_for_hub_event(
            q_hud, "task_completed", predicate=lambda e: e["task_id"] == ack2["task_id"]
        )
        all_events = seen_hud_1 + seen_hud_2
        alpha_events = [e for e in all_events if e.get("task_id") == ack1["task_id"]]
        beta_events = [e for e in all_events if e.get("task_id") == ack2["task_id"]]
        assert alpha_events and all(e["agent_id"] == "101" for e in alpha_events)
        assert beta_events and all(e["agent_id"] == "102" for e in beta_events)
        assert alpha_events[-1]["conversation_id"] == "conv-alpha-101"
        assert beta_events[-1]["conversation_id"] == "conv-beta-102"
    finally:
        app.state.event_hub.unsubscribe(q_hud)
        app.state.event_hub.unsubscribe(q_live)

    assert len(mock_antigravity_sdk.configs) == 2
    ws_sets = {tuple(c["workspaces"]) for c in mock_antigravity_sdk.configs}
    assert ws_sets == {
        (str(isolated_workspaces["clark"].resolve()),),
        (str(isolated_workspaces["proxmox"].resolve()),),
    }


def test_t3_05_presence_room_samsung_non_cast_vs_cast_tv_behavior(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    """T3-05 (F1 x F8): Non-Cast Samsung Tizen fallback vs Cast 98" QLED DashCast."""
    jarvis_client.post(
        "/api/presence/room",
        json={"room": "office", "tv": "Office - Samsung Q60CD 65"},
    )
    samsung_cast = jarvis_client.post("/api/cast/hud", json={})
    assert samsung_cast.status_code == 200
    s_data = samsung_cast.json()
    assert s_data["tv"] == "Office - Samsung Q60CD 65"
    assert s_data["hud_url"].endswith("/hud")
    assert s_data["message"]
    assert mock_chromecast_env.dashcast_calls == []

    jarvis_client.post("/api/presence/room", json={"room": "98_qled"})
    qled_cast = jarvis_client.post("/api/cast/hud", json={})
    assert qled_cast.status_code == 200
    assert len(mock_chromecast_env.dashcast_calls) == 1
    assert mock_chromecast_env.dashcast_calls[0]["target_device"] == '98" QLED'
    assert mock_chromecast_env.dashcast_calls[0]["force"] is True


def test_t3_06_pwa_and_hud_assets_match_project_git_and_health_state(
    jarvis_client: TestClient,
) -> None:
    """T3-06 (F6 x F7 x F2): PWA + /hud assets align with /api/projects & /api/health."""
    for path in (
        "/",
        "/hud",
        "/static/app.js",
        "/static/styles.css",
        "/static/hud.js",
        "/static/hud.css",
        "/static/manifest.json",
    ):
        assert jarvis_client.get(path).status_code == 200

    pwa_html = jarvis_client.get("/").text
    assert "Antigravity Live" in pwa_html
    assert 'closedby="any"' in pwa_html
    assert 'name="color-scheme" content="light dark"' in pwa_html
    pwa_inspector = _inspect_html(pwa_html)
    assert pwa_inspector.has_visible_button_with_text("Cast HUD to Room TV")
    assert pwa_inspector.has_visible_node_container()

    clean_app_js = _strip_js_comments(jarvis_client.get("/static/app.js").text)
    assert re.search(r"\bfetch\s*\(", clean_app_js)
    assert "/api/projects" in clean_app_js
    assert "/api/projects/select" in clean_app_js
    assert "/api/cast/hud" in clean_app_js
    assert re.search(r"\bnew\s+WebSocket\s*\(", clean_app_js)

    hud_html = jarvis_client.get("/hud").text
    inspector = _inspect_html(hud_html)
    assert not {t for t, _ in inspector.tags}.intersection(
        {"form", "input", "textarea", "select", "button"}
    )
    assert "contenteditable" not in hud_html.lower()
    clean_hud_js = _strip_js_comments(jarvis_client.get("/static/hud.js").text)
    assert re.search(r"\bnew\s+WebSocket\s*\(", clean_hud_js)
    assert "/ws/hud" in clean_hud_js
    assert "JSON.parse" in clean_hud_js
    assert "/api/cast/hud" not in clean_hud_js
    assert "/api/projects/select" not in clean_hud_js

    projs = {p["name"]: p for p in jarvis_client.get("/api/projects").json()["projects"]}
    assert projs["proxmox-monitor"]["uncommitted_changes"] == 2
    assert projs["clark-home-server"]["uncommitted_changes"] == 0
    health = jarvis_client.get("/api/health").json()
    assert health["active_project"]


def test_t3_07_presence_transitions_never_auto_cast_tv(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    """T3-07 (F1 x F8): Six consecutive room transitions never auto-cast TV."""
    for room in ("office", "living_room", "kitchen", "game_room", "master_bedroom", "whole_home"):
        resp = jarvis_client.post("/api/presence/room", json={"room": room, "source": "frigate_ha"})
        assert resp.status_code == 200
    _wait_until(
        lambda: len(mock_chromecast_env.play_media_calls) >= 1,
        timeout=2.0,
        message="Expected background warm_room() play_media call during room transitions",
    )
    assert mock_chromecast_env.dashcast_instances == []
    assert mock_chromecast_env.dashcast_calls == []


def test_t3_08_presence_webhook_emits_room_changed_and_updates_hud_snapshot(
    jarvis_client: TestClient,
) -> None:
    """T3-08 (F1 x F4 x F7): HA presence pushes room_changed live & updates late /ws/hud snapshot."""
    with jarvis_client.websocket_connect("/ws/hud") as ws_a:
        _ = recv_json_with_timeout(ws_a)
        resp = jarvis_client.post(
            "/api/presence/room",
            json={"room": "game_room", "source": "frigate_ha"},
        )
        assert resp.status_code == 200
        ev, _ = recv_until_type(ws_a, "room_changed")
        assert ev["active_room"] == "All Speakers"
        assert ev["active_tv"] in ("Game Room TV", '98" QLED')
        assert ev.get("source") == "frigate_ha"

    with jarvis_client.websocket_connect("/ws/hud") as ws_b:
        snap = recv_json_with_timeout(ws_b)
        assert snap["active_room"] == "All Speakers"
        assert snap["active_tv"] in ("Game Room TV", '98" QLED')
        assert any(e.get("type") == "room_changed" for e in snap["recent_events"])


async def test_t3_09_pwa_project_agent_selection_routes_default_voice_delegation(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    """T3-09 (F3 x F6): PWA REST selection routes default voice delegation without args."""
    e2e_settings.gemini_api_key = "fake-live-key"
    mock_antigravity_sdk.queue_turn(
        [Text("Checked tests on agent 102")],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-pwa-102",
    )
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [("fc-def-1", {"task": "Check tests"}, "delegate_to_antigravity_agent")]
            )
        ]
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        sel = await ac.post(
            "/api/projects/select", json={"project": "clark-home-server", "agent": "102"}
        )
        assert sel.status_code == 200

    fake_ws = FakeWebSocket()
    live_task = asyncio.create_task(app.state.live_bridge.handle_websocket(fake_ws))
    q = app.state.event_hub.subscribe()
    try:
        ev, _ = await wait_for_hub_event(q, "task_completed", timeout=3.0)
        assert ev["agent_id"] == "102"
        assert ev["project_name"] == "clark-home-server"
        assert ev["conversation_id"] == "conv-pwa-102"
    finally:
        app.state.event_hub.unsubscribe(q)
        mock_genai_live.end_session()
        fake_ws.disconnect()
        await asyncio.wait_for(live_task, timeout=2.0)

    clark = next(
        p for p in app.state.project_manager.get_state_summary()["projects"]
        if p["name"] == "clark-home-server"
    )
    agents = {a["agent_id"]: a for a in clark["agents"]}
    assert agents["102"]["conversation_id"] == "conv-pwa-102"
    assert agents["101"]["conversation_id"] is None


def test_t3_10_event_hub_multi_subscriber_isolation_on_abrupt_disconnect(
    jarvis_client: TestClient,
) -> None:
    """T3-10 (F4 x F5): Abruptly disconnecting /ws/hud #1 does not drop events for #2 or /ws/live."""
    with jarvis_client.websocket_connect("/ws/hud") as hud2, jarvis_client.websocket_connect("/ws/live") as live_ws:
        _ = recv_json_with_timeout(hud2)
        _ = recv_until_type(live_ws, "status")

        # Open and abruptly close hud1
        with jarvis_client.websocket_connect("/ws/hud") as hud1:
            _ = recv_json_with_timeout(hud1)

        jarvis_client.post("/api/presence/room", json={"room": "office"})
        jarvis_client.post("/api/projects/select", json={"project": "proxmox-monitor", "agent": "102"})

        ev_hud2, _ = recv_until_type(hud2, "project_changed")
        ev_live, _ = recv_until_type(live_ws, "project_changed")
        assert ev_hud2["project_name"] == "proxmox-monitor"
        assert ev_live["project_name"] == "proxmox-monitor"


def test_t3_11_presence_explicit_tv_override_used_by_default_cast_hud(
    jarvis_client: TestClient, mock_chromecast_env: CastEnvironmentSpy
) -> None:
    """T3-11 (F1 x F3 x F8): Explicit speaker+tv override on presence is used by default /api/cast/hud."""
    pres = jarvis_client.post(
        "/api/presence/room",
        json={"room": "office", "speaker": "All Speakers", "tv": "Office TV"},
    )
    assert pres.status_code == 200
    assert pres.json()["active_room"] == "All Speakers"
    assert pres.json()["active_tv"] == "Office TV"

    cast = jarvis_client.post("/api/cast/hud", json={})
    assert cast.status_code == 200
    assert len(mock_chromecast_env.dashcast_calls) == 1
    assert mock_chromecast_env.dashcast_calls[0]["target_device"] == "Office TV"
    assert mock_chromecast_env.dashcast_calls[0]["force"] is True


def test_t3_12_ws_live_voice_status_lifecycle_syncs_to_ws_hud(
    jarvis_client: TestClient,
) -> None:
    """T3-12 (F4 x F7): Live voice activity emits voice_status events synced to /ws/hud."""
    with jarvis_client.websocket_connect("/ws/hud") as hud_ws, jarvis_client.websocket_connect("/ws/live") as live_ws:
        _ = recv_json_with_timeout(hud_ws)
        _ = recv_until_type(live_ws, "status")

        live_ws.send_json({"type": "text", "text": "Summarize home status"})
        frames = collect_until_types(hud_ws, {"voice_status"})
        statuses = [f.get("voice_status") or f.get("status") for f in frames if f.get("type") == "voice_status"]
        assert any(s in ("thinking", "listening", "speaking", "idle") for s in statuses)

    with jarvis_client.websocket_connect("/ws/hud") as late_hud:
        snap = recv_json_with_timeout(late_hud)
        assert "voice_status" in snap and isinstance(snap["voice_status"], str)


# ============================================================================
# TIER 4 — Real-World JARVIS Application Scenarios (6 Full E2E Scenarios)
# ============================================================================
async def test_t4_01_walk_and_talk_follow_me_voice_switch_async_refactor_mid_task_move(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    """Scenario 1: Office -> Living Room -> switch_project -> async refactor -> move to Kitchen -> TTS."""
    e2e_settings.gemini_api_key = "fake-live-key"
    hold = asyncio.Event()
    mock_antigravity_sdk.queue_turn(
        [
            Thought("Inspecting auth middleware"),
            ToolCall("edit_file", {"path": "auth.py"}, result="updated"),
            Text("Auth middleware refactored and verified."),
        ],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-scen1-clark",
        hold_event=hold,
    )
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [("sc1-sw", {"project_agent": "project clark home server, agent 101"}, "switch_project")]
            )
        ]
    )
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [("sc1-del", {"task": "Refactor auth middleware"}, "delegate_to_antigravity_agent")]
            )
        ]
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    fake_ws = FakeWebSocket()
    live_task = asyncio.create_task(app.state.live_bridge.handle_websocket(fake_ws))
    q = app.state.event_hub.subscribe()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as ac:
            r_off = await ac.post("/api/presence/room", json={"room": "office", "source": "frigate_ha"})
            assert r_off.json()["active_room"] == "Office"
            await wait_for_hub_event(q, "room_changed")

            r_lr = await ac.post("/api/presence/room", json={"room": "living_room", "source": "frigate_ha"})
            assert r_lr.json()["active_room"] == "Living Room speaker"

            await wait_for_hub_event(q, "task_started", timeout=4.0)
            await wait_for_hub_event(q, "thought", timeout=3.0)
            assert not hold.is_set()
            await _async_wait_until(
                lambda: len(mock_genai_live.tool_responses) >= 2,
                timeout=2.0,
                message="Expected switch_project and delegate tool responses before kitchen move",
            )

            # Mid-task walk to Kitchen -> maps to Living Room speaker
            mock_chromecast_env.play_media_calls.clear()
            r_kit = await ac.post("/api/presence/room", json={"room": "kitchen", "source": "frigate_ha"})
            assert r_kit.json()["active_room"] == "Living Room speaker"
            assert r_kit.json()["active_tv"] == "Kitchen TV"

            hold.set()
            done_ev, seen = await wait_for_hub_event(q, "task_completed", timeout=4.0)
            assert done_ev["project_name"] == "clark-home-server"
            assert done_ev["agent_id"] == "101"
            assert any(e["type"] == "tool_call" for e in seen)
            await _async_wait_until(
                lambda: any(
                    c["content_type"] == "audio/mp3" for c in mock_chromecast_env.play_media_calls
                ),
                timeout=2.0,
                message="Expected completion TTS playback on Living Room speaker in Scenario 1",
            )
    finally:
        app.state.event_hub.unsubscribe(q)
        mock_genai_live.end_session()
        fake_ws.disconnect()
        await asyncio.wait_for(live_task, timeout=2.0)

    assert mock_chromecast_env.dashcast_calls == []
    mp3_calls = [c for c in mock_chromecast_env.play_media_calls if c["content_type"] == "audio/mp3"]
    assert mp3_calls and mp3_calls[-1]["speaker"] == "Living Room speaker"


async def test_t4_02_multi_project_concurrent_swarm_dual_agents_no_bleed(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    """Scenario 2: Concurrent tasks on (clark, 101) & (proxmox, 102) completing out-of-order."""
    gate_a = asyncio.Event()
    gate_b = asyncio.Event()
    mock_antigravity_sdk.queue_turn(
        [Thought("Swarm alpha running"), Text("Alpha swarm finished.")],
        workspace=isolated_workspaces["clark"],
        conversation_id="conv-alpha-101",
        hold_event=gate_a,
    )
    mock_antigravity_sdk.queue_turn(
        [Thought("Swarm beta running"), Text("Beta swarm finished.")],
        workspace=isolated_workspaces["proxmox"],
        conversation_id="conv-beta-102",
        hold_event=gate_b,
    )

    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()
    q = app.state.event_hub.subscribe()
    try:
        ack_a = app.state.agent_engine.start_background_task(
            "Swarm A", project="clark-home-server", agent="101"
        )
        ack_b = app.state.agent_engine.start_background_task(
            "Swarm B", project="proxmox-monitor", agent="102"
        )
        app.state.project_manager.update_agent_status(
            ack_a["project_id"],
            "101",
            status="running",
            swarm_nodes=[{"node_id": "a-sub-1", "status": "running"}],
        )

        # Out-of-order release: B finishes first, then A
        gate_b.set()
        first_done, _ = await wait_for_hub_event(q, "task_completed")
        assert first_done["task_id"] == ack_b["task_id"]

        gate_a.set()
        second_done, _ = await wait_for_hub_event(q, "task_completed")
        assert second_done["task_id"] == ack_a["task_id"]
    finally:
        app.state.event_hub.unsubscribe(q)

    # Restart app and confirm persisted conversations survive without overwriting
    app2 = create_app(e2e_settings)
    with TestClient(app2) as c2:
        projs = {p["name"]: p for p in c2.get("/api/projects").json()["projects"]}
        c_agents = {a["agent_id"]: a for a in projs["clark-home-server"]["agents"]}
        p_agents = {a["agent_id"]: a for a in projs["proxmox-monitor"]["agents"]}
        assert c_agents["101"]["conversation_id"] == "conv-alpha-101"
        assert p_agents["102"]["conversation_id"] == "conv-beta-102"


def test_t4_03_ten_foot_hud_tv_dashboard_full_session(
    jarvis_client: TestClient,
) -> None:
    """Scenario 3: Full /hud read-only 10-foot TV session receiving live room/project/task events."""
    hud = jarvis_client.get("/hud")
    hud_js = jarvis_client.get("/static/hud.js")
    hud_css = jarvis_client.get("/static/hud.css")
    assert hud.status_code == 200 and "text/html" in hud.headers.get("content-type", "")
    assert hud_js.status_code == 200 and hud_css.status_code == 200
    assert "color-scheme" in hud.text
    assert "/static/hud.js" in hud.text and "/static/hud.css" in hud.text
    inspector = _inspect_html(hud.text)
    assert not {t for t, _ in inspector.tags}.intersection(
        {"form", "input", "textarea", "select", "button"}
    )
    assert "contenteditable" not in hud.text.lower()
    assert inspector.has_visible_node_container()
    clean_hud_js = _strip_js_comments(hud_js.text)
    assert re.search(r"\bnew\s+WebSocket\s*\(", clean_hud_js)
    assert "/ws/hud" in clean_hud_js
    assert "JSON.parse" in clean_hud_js
    assert "/api/cast/hud" not in clean_hud_js
    assert "/api/projects/select" not in clean_hud_js
    assert "getUserMedia" not in clean_hud_js and "AudioWorklet" not in clean_hud_js

    with jarvis_client.websocket_connect("/ws/hud") as hud_ws:
        snap = recv_json_with_timeout(hud_ws)
        assert snap["type"] == "hud_snapshot"
        for k in (
            "active_room",
            "active_speaker",
            "active_tv",
            "voice_status",
            "projects",
            "active_project_id",
            "active_project_agent",
            "recent_events",
        ):
            assert k in snap

        jarvis_client.post("/api/presence/room", json={"room": "game_room", "source": "frigate_ha"})
        jarvis_client.post("/api/projects/select", json={"project": "proxmox-monitor", "agent": "102"})
        jarvis_client.post("/api/agent/ask", json={"prompt": "Check proxmox", "cast_response": False})

        frames = collect_until_types(hud_ws, {"room_changed", "project_changed"})
        assert any(f.get("active_tv") in ("Game Room TV", '98" QLED') for f in frames)


def test_t4_04_on_demand_tv_casting_cast_vs_samsung_tizen_workflow(
    e2e_settings: BridgeSettings,
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    """Scenario 4: On-demand /api/cast/hud + cast_hud_to_tv across Cast TVs & Samsung Tizen TVs."""
    e2e_settings.gemini_api_key = "fake-live-key"
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [("sc4-qled", {"room": "98_qled"}, "cast_hud_to_tv")]
            )
        ]
    )

    app = create_app(e2e_settings)
    with TestClient(app) as client:
        _sync_discovery(client, mock_chromecast_env)
        # 1. Game room on-demand Cast HUD
        client.post("/api/presence/room", json={"room": "game_room"})
        r_gr = client.post("/api/cast/hud", json={})
        assert r_gr.status_code == 200
        assert mock_chromecast_env.dashcast_calls[-1]["target_device"] in ("Game Room TV", '98" QLED')
        assert mock_chromecast_env.dashcast_calls[-1]["force"] is True

        # 2. Voice tool cast_hud_to_tv(room="98_qled")
        with client.websocket_connect("/ws/live") as live_ws:
            _ = recv_until_type(live_ws, "status")
            _ = recv_until_type(live_ws, "tool_call", timeout=4.0)
            _wait_until(
                lambda: (
                    len(mock_genai_live.tool_responses) >= 1
                    and any(c["target_device"] == '98" QLED' for c in mock_chromecast_env.dashcast_calls)
                ),
                timeout=2.0,
                message="Expected cast_hud_to_tv tool_response and 98\" QLED dashcast call",
            )
            mock_genai_live.end_session()
        assert any(c["target_device"] == '98" QLED' for c in mock_chromecast_env.dashcast_calls)

        # 3. Non-Cast Samsung Tizen TVs never crash and never call DashCastController
        before_count = len(mock_chromecast_env.dashcast_calls)
        for samsung_tv in (
            "Office - Samsung Q60CD 65",
            "Living Room TV",
            "Kitchen TV",
            "The Frame",
        ):
            res = client.post("/api/cast/hud", json={"tv": samsung_tv})
            assert res.status_code == 200
            body = res.json()
            assert body["tv"] == samsung_tv
            assert body["hud_url"].endswith("/hud")
            assert body["message"]
        assert len(mock_chromecast_env.dashcast_calls) == before_count

        # 4. Explicit override to Cast-capable Office TV succeeds
        r_off = client.post("/api/cast/hud", json={"room": "office", "tv": "Office TV"})
        assert r_off.status_code == 200
        assert mock_chromecast_env.dashcast_calls[-1]["target_device"] == "Office TV"


def test_t4_05_persistent_conversation_resume_across_bridge_restarts(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
) -> None:
    """Scenario 5: Persist (clark,101), (clark,102), (proxmox,101) across full app restart."""
    mock_antigravity_sdk.queue_turn(
        [Text("clark 101")], workspace=isolated_workspaces["clark"], conversation_id="conv-alpha-101"
    )
    mock_antigravity_sdk.queue_turn(
        [Text("clark 102")], workspace=isolated_workspaces["clark"], conversation_id="conv-alpha-102"
    )
    mock_antigravity_sdk.queue_turn(
        [Text("proxmox 101")], workspace=isolated_workspaces["proxmox"], conversation_id="conv-beta-101"
    )

    app1 = create_app(e2e_settings)
    with TestClient(app1) as c1:
        c1.post("/api/projects/select", json={"project": "clark-home-server", "agent": "101"})
        c1.post("/api/agent/ask", json={"prompt": "turn 1", "cast_response": False})

        c1.post("/api/projects/select", json={"project": "clark-home-server", "agent": "102"})
        c1.post("/api/agent/ask", json={"prompt": "turn 2", "cast_response": False})

        c1.post("/api/projects/select", json={"project": "proxmox-monitor", "agent": "101"})
        c1.post("/api/agent/ask", json={"prompt": "turn 3", "cast_response": False})

    assert (isolated_workspaces["save_dir"] / "projects_state.json").exists()

    mock_antigravity_sdk.configs.clear()
    mock_antigravity_sdk.queue_turn(
        [Text("resumed proxmox 101")], workspace=isolated_workspaces["proxmox"]
    )
    mock_antigravity_sdk.queue_turn(
        [Text("resumed clark 102")], workspace=isolated_workspaces["clark"]
    )

    app2 = create_app(e2e_settings)
    with TestClient(app2) as c2:
        health = c2.get("/api/health").json()
        assert "proxmox" in health["active_project"].lower()
        c2.post("/api/agent/ask", json={"prompt": "resume proxmox", "cast_response": False})
        c2.post("/api/projects/select", json={"project": "clark-home-server", "agent": "102"})
        c2.post("/api/agent/ask", json={"prompt": "resume clark", "cast_response": False})

    assert mock_antigravity_sdk.configs[0].get("conversation_id") == "conv-beta-101"
    assert mock_antigravity_sdk.configs[0]["workspaces"] == [
        str(isolated_workspaces["proxmox"].resolve())
    ]
    assert mock_antigravity_sdk.configs[1].get("conversation_id") == "conv-alpha-102"
    assert mock_antigravity_sdk.configs[1]["workspaces"] == [
        str(isolated_workspaces["clark"].resolve())
    ]


async def test_t4_06_voice_driven_full_cycle_ws_live_and_pwa_contract(
    e2e_settings: BridgeSettings,
    isolated_workspaces: dict[str, Path],
    mock_chromecast_env: CastEnvironmentSpy,
    mock_antigravity_sdk: AntigravitySDKSpy,
    mock_genai_live: GeminiLiveSpy,
) -> None:
    """Scenario 6: Full PWA baseline/upgrade verification + Gemini Live audio/tools/barge-in."""
    e2e_settings.gemini_api_key = "fake-live-key"
    app = create_app(e2e_settings)
    await app.state.cast_manager.discover_speakers()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as ac:
        pwa = await ac.get("/")
        assert pwa.status_code == 200
        assert "Antigravity Live" in pwa.text
        assert 'closedby="any"' in pwa.text
        assert 'name="color-scheme" content="light dark"' in pwa.text
        pwa_inspector = _inspect_html(pwa.text)
        assert pwa_inspector.has_visible_button_with_text("Cast HUD to Room TV")
        assert pwa_inspector.has_visible_node_container()
        app_js = (await ac.get("/static/app.js")).text
        clean_app_js = _strip_js_comments(app_js)
        assert re.search(r"\bfetch\s*\(", clean_app_js)
        assert re.search(r"""method\s*:\s*['"]POST['"]""", clean_app_js, re.IGNORECASE)
        assert "/api/projects/select" in clean_app_js and "/api/cast/hud" in clean_app_js
        assert re.search(
            r"""addEventListener\s*\(\s*['"](?:click|pointerdown|touchend)['"]|\.onclick\s*=""",
            clean_app_js,
        )
        assert re.search(r"\bnew\s+WebSocket\s*\(", clean_app_js)
        assert (await ac.get("/static/styles.css")).status_code == 200
        assert (await ac.get("/static/manifest.json")).status_code == 200

    mock_antigravity_sdk.queue_turn(
        [Thought("Checking containers"), Text("All containers healthy.")],
        workspace=isolated_workspaces["clark"],
    )
    mock_genai_live.enqueue_turn(
        [
            mock_genai_live.make_tool_call_turn(
                [
                    ("sc6-1", {"project_agent": "project clark home server, agent 101"}, "switch_project"),
                    ("sc6-2", {"task": "Check containers"}, "delegate_to_antigravity_agent"),
                    ("sc6-3", {"room": "game_room"}, "cast_hud_to_tv"),
                ]
            ),
            mock_genai_live.make_audio_text_turn(
                pcm_bytes=b"\x03\x00" * 320,
                output_text="Started your container check and cast the HUD.",
                turn_complete=True,
            ),
        ]
    )

    fake_ws = FakeWebSocket()
    live_task = asyncio.create_task(app.state.live_bridge.handle_websocket(fake_ws))
    try:
        conn_st = await fake_ws.wait_for_sent("status", predicate=lambda m: m.get("state") == "connected")
        assert conn_st["state"] == "connected"
        list_st = await fake_ws.wait_for_sent("status", predicate=lambda m: m.get("state") == "listening")
        assert list_st.get("mode") == "gemini-live-native-audio"

        pcm_b64 = base64.b64encode(b"\x01\x00" * 160).decode("ascii")
        fake_ws.push_client_json({"type": "audio_chunk", "pcm_b64": pcm_b64})
        await fake_ws.wait_for_sent("tool_call", predicate=lambda m: m.get("name") == "cast_hud_to_tv")
        await fake_ws.wait_for_sent("transcript", timeout=4.0)
        await fake_ws.wait_for_sent("task_completed", timeout=4.0)

        fake_ws.push_client_json({"type": "barge_in"})
        intr = await fake_ws.wait_for_sent("interrupted", timeout=3.0)
        assert intr["type"] == "interrupted"
        await _async_wait_until(
            lambda: len(mock_chromecast_env.stop_calls) >= 1,
            timeout=2.0,
            message="Expected background cast_manager.stop_room() call on barge_in",
        )
    finally:
        mock_genai_live.end_session()
        fake_ws.disconnect()
        await asyncio.wait_for(live_task, timeout=2.0)

    assert len(mock_genai_live.realtime_inputs) >= 1
    assert len(mock_chromecast_env.dashcast_calls) == 1
    assert mock_chromecast_env.stop_calls
