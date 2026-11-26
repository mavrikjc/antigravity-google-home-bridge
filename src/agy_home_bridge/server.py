"""FastAPI Server for Antigravity Google Home Bridge."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agy_home_bridge.antigravity_agent import AntigravityAgentEngine
from agy_home_bridge.cast_manager import GoogleHomeCastManager
from agy_home_bridge.config import BridgeSettings, settings
from agy_home_bridge.gemini_live import FullDuplexLiveBridge
from agy_home_bridge.hardware_mic import HardwareMicDaemon
from agy_home_bridge.mcp_server import BridgeMCPServer
from agy_home_bridge.smart_home import SmartHomeFulfillment

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


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


def create_app(custom_settings: BridgeSettings | None = None) -> FastAPI:
    """Create and wire the FastAPI application instance."""
    cfg = custom_settings or settings
    cast_manager = GoogleHomeCastManager(cfg)
    agent_engine = AntigravityAgentEngine(cfg, cast_manager)
    live_bridge = FullDuplexLiveBridge(cfg, agent_engine, cast_manager)
    smart_home = SmartHomeFulfillment(agent_engine, cast_manager)
    mcp_server = BridgeMCPServer(cast_manager, agent_engine)
    hw_mic = HardwareMicDaemon(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Run non-blocking Google Home speaker discovery on startup
        asyncio.create_task(cast_manager.discover_speakers())
        hw_task: asyncio.Task[Any] | None = None
        if cfg.hw_mic_enabled:
            hw_task = asyncio.create_task(hw_mic.run())
        yield
        if hw_task:
            hw_mic.stop()
            hw_task.cancel()

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

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": "0.1.0",
            "active_room": cast_manager.active_room,
            "routing_mode": cast_manager.routing_mode,
            "lan_url": cfg.base_lan_url,
            "gemini_configured": bool(cfg.gemini_api_key),
            "speakers_count": len(cast_manager.list_speakers()),
            "hardware_mics": hw_mic.list_audio_input_devices(),
        }

    @app.get("/api/speakers")
    async def get_speakers(refresh: bool = False) -> dict[str, Any]:
        if refresh:
            await cast_manager.discover_speakers()
        return {
            "active_room": cast_manager.active_room,
            "routing_mode": cast_manager.routing_mode,
            "speakers": cast_manager.list_speakers(),
        }

    @app.post("/api/cast/select")
    async def select_room(req: SelectRoomRequest) -> dict[str, Any]:
        room = cast_manager.set_active_room(req.room)
        if req.routing_mode:
            try:
                cast_manager.set_routing_mode(req.routing_mode)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "active_room": room,
            "routing_mode": cast_manager.routing_mode,
            "speakers": cast_manager.list_speakers(),
        }

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
        file_path = cfg.tts_cache_dir / safe_name
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Audio file not found")
        media_type = "audio/wav" if safe_name.endswith(".wav") else "audio/mpeg"
        return FileResponse(str(file_path), media_type=media_type)

    @app.post("/api/agent/ask")
    async def agent_ask(req: AgentPromptRequest) -> dict[str, Any]:
        res = await agent_engine.ask(
            prompt=req.prompt,
            cast_response=req.cast_response,
            target_room=req.room,
        )
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

        body = await request.json()
        response_payload = await smart_home.handle_intent(body)
        return JSONResponse(response_payload)

    @app.post("/mcp")
    async def mcp_http_endpoint(request: Request) -> JSONResponse:
        body = await request.json()
        response_payload = await mcp_server.handle_jsonrpc(body)
        return JSONResponse(response_payload)

    @app.websocket("/ws/live")
    async def websocket_live_endpoint(websocket: WebSocket) -> None:
        await live_bridge.handle_websocket(websocket)

    return app


app = create_app()
