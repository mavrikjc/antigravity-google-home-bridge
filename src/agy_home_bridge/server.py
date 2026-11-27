"""FastAPI Server for Antigravity Google Home Bridge."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import Body, FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agy_home_bridge.antigravity_agent import AntigravityAgentEngine
from agy_home_bridge.cast_manager import GoogleHomeCastManager, normalize_room_key
from agy_home_bridge.config import BridgeSettings, settings
from agy_home_bridge.event_hub import EventHub, safe_json_dumps
from agy_home_bridge.gemini_live import FullDuplexLiveBridge
from agy_home_bridge.hardware_mic import HardwareMicDaemon
from agy_home_bridge.mcp_server import BridgeMCPServer
from agy_home_bridge.project_manager import ProjectManager
from agy_home_bridge.smart_home import SmartHomeFulfillment
from agy_home_bridge.tls import ensure_lan_tls_certificates

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_HUD_FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="color-scheme" content="light dark" />
  <title>Antigravity JARVIS HUD</title>
</head>
<body>
  <header>
    <h1>Antigravity JARVIS HUD</h1>
  </header>
  <main>
    <section class="hud-projects-nodes" id="hud-projects-nodes" aria-live="polite"></section>
  </main>
</body>
</html>
"""


class SpeakRequest(BaseModel):
    """Payload for `/api/cast/speak`."""

    text: str = Field(..., min_length=1, description="Text to speak on Google Home.")
    room: str | None = Field(
        default=None, description="Target Google Home room or speaker group."
    )
    volume: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Optional volume (0.0 to 1.0)."
    )


class SelectRoomRequest(BaseModel):
    """Payload for `/api/cast/select`."""

    room: str = Field(..., min_length=1, description="Room or speaker group name.")
    routing_mode: str | None = Field(
        default=None, description="Optional routing mode: 'direct', 'cast', or 'both'."
    )


class AgentPromptRequest(BaseModel):
    """Payload for `/api/agent/ask`."""

    prompt: str = Field(..., min_length=1, description="Prompt for Antigravity Agent.")
    cast_response: bool | None = Field(
        default=None, description="Whether to speak the response on Google Home."
    )
    room: str | None = Field(
        default=None, description="Optional target Google Home room override."
    )
    project: str | None = Field(
        default=None, description="Optional project id, name, or short name override."
    )
    agent: str | None = Field(
        default=None, description="Optional agent slot identifier override."
    )
    project_agent: str | None = Field(
        default=None, description="Optional combined verbal project-agent override."
    )


class RoomPresenceRequest(BaseModel):
    """Payload for `/api/presence/room`."""

    room: str = Field(..., min_length=1, description="Presence room name.")
    source: str | None = Field(
        default=None, description="Presence update source (e.g., 'frigate_ha')."
    )
    speaker: str | None = Field(
        default=None, description="Optional speaker override."
    )
    tv: str | None = Field(default=None, description="Optional TV override.")


class SelectProjectRequest(BaseModel):
    """Payload for `/api/projects/select`."""

    project: str | None = Field(
        default=None, description="Project id, name, or verbal short name."
    )
    agent: str | None = Field(
        default=None, description="Agent slot identifier (e.g., '101', '102')."
    )
    project_agent: str | None = Field(
        default=None,
        description="Combined verbal project-agent phrase (e.g., 'project Clark Home Server, agent 101').",
    )


class CastHudRequest(BaseModel):
    """Payload for `/api/cast/hud`."""

    room: str | None = Field(default=None, description="Optional room override.")
    tv: str | None = Field(default=None, description="Optional TV name override.")


class VoiceMuteRequest(BaseModel):
    """Payload for `/api/voice/mute`."""

    muted: bool = Field(..., description="Whether global voice input should be muted.")


def _strict_json_loads(raw_bytes: bytes) -> Any:
    """Decode UTF-8 JSON bytes while rejecting non-standard NaN/Infinity constants."""
    def _reject_constant(val: str) -> Any:
        raise ValueError(f"Non-standard JSON constant {val!r} is not allowed")

    return json.loads(raw_bytes.decode("utf-8"), parse_constant=_reject_constant)


def create_app(custom_settings: BridgeSettings | None = None) -> FastAPI:
    """Create and wire the FastAPI application instance."""
    cfg = custom_settings or settings
    event_hub = EventHub(maxlen=cfg.event_history_size)
    project_manager = ProjectManager(
        cfg.agy_workspace_dir,
        cfg.agy_save_dir,
        state_file=cfg.resolved_projects_state_file,
        default_agent_id=cfg.default_agent_id,
    )
    cast_manager = GoogleHomeCastManager(cfg)
    agent_engine = AntigravityAgentEngine(
        cfg,
        cast_manager,
        project_manager=project_manager,
        event_hub=event_hub,
    )
    live_bridge = FullDuplexLiveBridge(
        cfg,
        agent_engine,
        cast_manager,
        event_hub=event_hub,
        project_manager=project_manager,
    )
    smart_home = SmartHomeFulfillment(agent_engine, cast_manager)
    mcp_server = BridgeMCPServer(cast_manager, agent_engine)
    hw_mic = HardwareMicDaemon(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Run non-blocking Google Home speaker discovery on startup (never auto-casts TVs)
        discovery_task = asyncio.create_task(cast_manager.discover_speakers())
        cast_manager.schedule_discovery_resilience()
        hw_task: asyncio.Task[Any] | None = None
        if cfg.hw_mic_enabled:
            hw_task = asyncio.create_task(hw_mic.run())
        try:
            yield
        finally:
            cast_manager.cancel_discovery_resilience()
            if not discovery_task.done():
                discovery_task.cancel()
                await asyncio.gather(discovery_task, return_exceptions=True)
            if hw_task:
                hw_mic.stop()
                if not hw_task.done():
                    hw_task.cancel()
                    await asyncio.gather(hw_task, return_exceptions=True)
            await agent_engine.shutdown()

    app = FastAPI(
        title="Antigravity Google Home Bridge",
        description=(
            "Always-On Full-Duplex Google Antigravity Live Voice & Google Home Bridge "
            "for Proxmox and Home Servers"
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Expose components on app.state for tests and extensions
    app.state.settings = cfg
    app.state.event_hub = event_hub
    app.state.project_manager = project_manager
    app.state.cast_manager = cast_manager
    app.state.agent_engine = agent_engine
    app.state.live_bridge = live_bridge
    app.state.smart_home = smart_home
    app.state.mcp_server = mcp_server

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> HTMLResponse:
        index_file = STATIC_DIR / "index.html"
        return HTMLResponse(index_file.read_text(encoding="utf-8"))

    @app.get("/hud", response_class=HTMLResponse)
    async def hud() -> HTMLResponse:
        hud_file = STATIC_DIR / "hud.html"
        if hud_file.exists():
            return HTMLResponse(hud_file.read_text(encoding="utf-8"))
        return HTMLResponse(_HUD_FALLBACK_HTML)

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        proj_summary = project_manager.get_state_summary()
        active_proj_id = proj_summary.get("active_project_id")
        active_proj = next(
            (p for p in proj_summary.get("projects", []) if p.get("id") == active_proj_id),
            None,
        )
        return {
            "status": "ok",
            "version": "0.1.0",
            "active_room": cast_manager.active_room,
            "active_presence_room": cast_manager.active_presence_room,
            "active_tv": cast_manager.active_tv,
            "routing_mode": cast_manager.routing_mode,
            "lan_url": cfg.base_lan_url,
            "https_lan_url": cfg.https_lan_url,
            "https_port": cfg.bridge_https_port,
            "gemini_configured": bool(cfg.gemini_api_key),
            "speakers_count": len(cast_manager.list_speakers()),
            "hardware_mics": hw_mic.list_audio_input_devices(),
            "active_project": active_proj["name"] if active_proj else active_proj_id,
            "active_project_id": active_proj_id,
            "active_agent_id": proj_summary.get("active_agent_id"),
            "active_project_agent": proj_summary.get("active_project_agent"),
            "voice_status": live_bridge.voice_status,
            "voice_muted": live_bridge.voice_muted,
        }

    @app.get("/api/tls/ca.pem")
    async def download_root_ca() -> FileResponse:
        ca_cert, _, _ = ensure_lan_tls_certificates(cfg.certs_dir, cfg.resolved_lan_ip)
        return FileResponse(
            str(ca_cert),
            media_type="application/x-x509-ca-cert",
            filename="Antigravity-Home-Root-CA.crt",
        )

    @app.get("/api/speakers")
    async def get_speakers(refresh: bool = False) -> dict[str, Any]:
        if refresh:
            await cast_manager.discover_speakers()
        return {
            "active_room": cast_manager.active_room,
            "active_presence_room": cast_manager.active_presence_room,
            "active_tv": cast_manager.active_tv,
            "routing_mode": cast_manager.routing_mode,
            "speakers": cast_manager.list_speakers(),
            "tvs": cast_manager.list_tvs(),
        }

    @app.post("/api/cast/select")
    async def select_room(req: SelectRoomRequest) -> dict[str, Any]:
        clean_room = req.room.strip()
        if not clean_room:
            raise HTTPException(status_code=400, detail="room must not be empty")
        if not normalize_room_key(clean_room):
            raise HTTPException(
                status_code=400,
                detail="room must contain at least one alphanumeric character",
            )
        if req.routing_mode is not None and req.routing_mode not in ("direct", "cast", "both"):
            raise HTTPException(
                status_code=400,
                detail="routing_mode must be 'direct', 'cast', or 'both'",
            )
        room = cast_manager.set_active_room(clean_room)
        if req.routing_mode is not None:
            cast_manager.set_routing_mode(req.routing_mode)
        await event_hub.publish(
            "room_changed",
            active_room=room,
            active_speaker=room,
            active_presence_room=cast_manager.active_presence_room,
            presence_room=cast_manager.active_presence_room,
            active_tv=cast_manager.active_tv,
            routing_mode=cast_manager.routing_mode,
            source="api",
        )
        return {
            "active_room": room,
            "routing_mode": cast_manager.routing_mode,
            "speakers": cast_manager.list_speakers(),
        }

    @app.post("/api/presence/room")
    async def set_presence_room(req: RoomPresenceRequest) -> dict[str, Any]:
        if not req.room.strip() or not normalize_room_key(req.room):
            raise HTTPException(
                status_code=400,
                detail="room must contain at least one alphanumeric character",
            )
        try:
            res = cast_manager.set_presence_room(
                req.room,
                source=req.source,
                speaker_override=req.speaker,
                tv_override=req.tv,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        async def _bg_warm_room(speaker_name: str) -> None:
            try:
                await cast_manager.warm_room(speaker_name)
            except Exception as warm_exc:
                logger.debug(
                    "Background warm_room skipped for '%s': %s",
                    speaker_name,
                    warm_exc,
                )

        asyncio.create_task(_bg_warm_room(res["active_speaker"]))
        await event_hub.publish(
            "room_changed",
            active_room=res["active_room"],
            active_speaker=res["active_speaker"],
            active_presence_room=res["presence_room"],
            presence_room=res["presence_room"],
            active_tv=res["active_tv"],
            source=req.source,
        )
        return {
            "status": "ok",
            "active_presence_room": res["presence_room"],
            **res,
        }

    @app.get("/api/projects")
    async def get_projects() -> dict[str, Any]:
        return await asyncio.to_thread(project_manager.get_state_summary, rescan=True)

    @app.post("/api/projects/select")
    async def select_project(req: SelectProjectRequest) -> dict[str, Any]:
        try:
            selection = project_manager.select_project(
                project=req.project,
                agent=req.agent,
                project_agent=req.project_agent,
            )
        except ValueError as exc:
            msg = str(exc)
            status_code = 404 if ("Unknown project" in msg or "No projects" in msg) else 400
            raise HTTPException(status_code=status_code, detail=msg) from exc

        await event_hub.publish(
            "project_changed",
            project_id=selection["project_id"],
            project_name=selection["project_name"],
            short_name=selection["short_name"],
            agent_id=selection["agent_id"],
            project_agent=selection["project_agent"],
            conversation_id=selection.get("conversation_id"),
            path=selection.get("path"),
            git_branch=selection.get("git_branch"),
            uncommitted_changes=selection.get("uncommitted_changes"),
            source="api",
        )
        return {"status": "ok", **selection}

    @app.get("/api/voice/mute")
    async def get_voice_mute() -> dict[str, Any]:
        return {
            "voice_muted": live_bridge.voice_muted,
            "voice_status": live_bridge.voice_status,
        }

    @app.post("/api/voice/mute")
    async def set_voice_mute(req: VoiceMuteRequest) -> dict[str, Any]:
        muted = await live_bridge.set_voice_muted(req.muted)
        return {
            "status": "ok",
            "voice_muted": muted,
            "voice_status": live_bridge.voice_status,
        }

    @app.post("/api/cast/hud")
    async def cast_hud(
        req: CastHudRequest = Body(default_factory=CastHudRequest),
    ) -> dict[str, Any]:
        return await cast_manager.cast_hud_to_tv(
            room_name=req.room,
            tv_name=req.tv,
        )

    @app.post("/api/cast/speak")
    async def cast_speak(req: SpeakRequest) -> dict[str, Any]:
        return await cast_manager.speak_on_room(
            text=req.text,
            room_name=req.room,
            volume=req.volume,
        )

    @app.get("/api/tts/{filename}")
    async def serve_tts_file(filename: str) -> FileResponse:
        safe_name = Path(filename).name
        if not safe_name or safe_name in (".", ".."):
            raise HTTPException(status_code=404, detail="Audio file not found")
        file_path = cfg.tts_cache_dir / safe_name
        if not file_path.is_file():
            raise HTTPException(status_code=404, detail="Audio file not found")
        media_type = "audio/wav" if safe_name.endswith(".wav") else "audio/mpeg"
        return FileResponse(str(file_path), media_type=media_type)

    @app.post("/api/agent/ask")
    async def agent_ask(req: AgentPromptRequest) -> dict[str, Any]:
        try:
            res = await agent_engine.ask(
                prompt=req.prompt,
                cast_response=req.cast_response,
                target_room=req.room,
                project=req.project,
                agent=req.agent,
                project_agent=req.project_agent,
            )
        except ValueError as exc:
            msg = str(exc)
            status_code = 404 if ("Unknown project" in msg or "No projects" in msg) else 400
            raise HTTPException(status_code=status_code, detail=msg) from exc
        return {
            "text": res.text,
            "conversation_id": res.conversation_id,
            "backend": res.backend,
            "active_room": cast_manager.active_room,
            "tool_calls": res.tool_calls,
        }

    @app.post("/api/google-home/fulfillment")
    async def google_home_fulfillment(
        request: Request,
        authorization: str | None = Header(default=None),
    ) -> JSONResponse:
        # If SMART_HOME_WEBHOOK_TOKEN is configured and non-default, verify Bearer token
        expected = cfg.smart_home_webhook_token
        if expected and expected != "change-me-to-a-secure-token":
            token = (authorization or "").replace("Bearer ", "").strip()
            if token != expected:
                raise HTTPException(status_code=401, detail="Invalid bearer token")

        try:
            body = _strict_json_loads(await request.body())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid JSON payload") from exc
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="JSON body must be an object")
        try:
            response_payload = await smart_home.handle_intent(body)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(response_payload)

    @app.post("/mcp")
    async def mcp_http_endpoint(request: Request) -> JSONResponse:
        try:
            body = _strict_json_loads(await request.body())
        except ValueError:
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": "Parse error"},
                },
                status_code=400,
            )
        if not isinstance(body, dict):
            return JSONResponse(
                {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid Request"},
                },
                status_code=400,
            )
        response_payload = await mcp_server.handle_jsonrpc(body)
        err = response_payload.get("error")
        status_code = (
            400
            if isinstance(err, dict) and err.get("code") in (-32700, -32600, -32602)
            else 200
        )
        return JSONResponse(response_payload, status_code=status_code)

    @app.websocket("/ws/hud")
    async def websocket_hud_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = event_hub.subscribe()
        pump_task: asyncio.Task[Any] | None = None
        try:
            snapshot = event_hub.build_snapshot(
                project_manager,
                cast_manager,
                voice_status=live_bridge.voice_status,
                voice_muted=live_bridge.voice_muted,
            )
            await websocket.send_text(safe_json_dumps(snapshot))

            async def _pump_events() -> None:
                try:
                    while True:
                        event = await queue.get()
                        try:
                            payload_text = safe_json_dumps(event)
                        except (TypeError, ValueError) as exc:
                            logger.warning("Skipping unserializable HUD event: %s", exc)
                            continue
                        await websocket.send_text(payload_text)
                except (asyncio.CancelledError, WebSocketDisconnect):
                    return
                except Exception:
                    return

            pump_task = asyncio.create_task(_pump_events())
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
        except WebSocketDisconnect:
            pass
        finally:
            event_hub.unsubscribe(queue)
            if pump_task is not None and not pump_task.done():
                pump_task.cancel()
            if pump_task is not None:
                try:
                    await asyncio.gather(pump_task, return_exceptions=True)
                except asyncio.CancelledError:
                    pass

    @app.websocket("/ws/live")
    async def websocket_live_endpoint(websocket: WebSocket) -> None:
        await live_bridge.handle_websocket(websocket)

    return app


app = create_app()
