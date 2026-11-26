"""Command-line interface for Antigravity Google Home Bridge."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

import uvicorn

from agy_home_bridge.antigravity_agent import AntigravityAgentEngine
from agy_home_bridge.cast_manager import GoogleHomeCastManager
from agy_home_bridge.config import settings
from agy_home_bridge.hardware_mic import HardwareMicDaemon
from agy_home_bridge.mcp_server import BridgeMCPServer, run_stdio_mcp_loop


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agy-home-bridge",
        description="Always-On Full-Duplex Google Antigravity Live Voice & Google Home Bridge",
    )
    subparsers = parser.add_subparsers(dest="command")

    # `serve` subcommand (default)
    serve_parser = subparsers.add_parser(
        "serve", help="Start the FastAPI + WebSocket Full-Duplex Live Voice Server"
    )
    serve_parser.add_argument("--host", default=settings.bridge_host)
    serve_parser.add_argument("--port", type=int, default=settings.bridge_port)
    serve_parser.add_argument("--reload", action="store_true")

    # `discover` subcommand
    subparsers.add_parser(
        "discover", help="Scan the LAN for Google Home speakers and Cast groups"
    )

    # `speak` subcommand
    speak_parser = subparsers.add_parser(
        "speak", help="Speak a message out loud on a Google Home speaker"
    )
    speak_parser.add_argument("message", help="Text message to speak")
    speak_parser.add_argument("--room", default=None, help="Target Google Home room")

    # `hw-mic` subcommand
    hw_parser = subparsers.add_parser(
        "hw-mic",
        help="Run the Proxmox Hardware Wireless Mic Daemon (or list USB audio receivers)",
    )
    hw_parser.add_argument(
        "--list-devices", action="store_true", help="List detected audio input devices"
    )
    hw_parser.add_argument(
        "--ws-url",
        default=None,
        help="Target WebSocket URL (default: ws://127.0.0.1:8765/ws/live)",
    )

    # `mcp-stdio` subcommand
    subparsers.add_parser(
        "mcp-stdio",
        help="Run as an MCP (Model Context Protocol) server over stdio for Antigravity IDE/CLI",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cmd = args.command or "serve"

    if cmd == "serve":
        host = getattr(args, "host", settings.bridge_host)
        port = getattr(args, "port", settings.bridge_port)
        reload_flag = getattr(args, "reload", False)
        uvicorn.run(
            "agy_home_bridge.server:app",
            host=host,
            port=port,
            reload=reload_flag,
        )
        return 0

    if cmd == "discover":
        cast_mgr = GoogleHomeCastManager(settings)
        speakers = asyncio.run(cast_mgr.discover_speakers())
        print(json.dumps(speakers, indent=2))
        return 0

    if cmd == "speak":
        cast_mgr = GoogleHomeCastManager(settings)
        asyncio.run(cast_mgr.discover_speakers(timeout=2.5))
        res = asyncio.run(cast_mgr.speak_on_room(args.message, room_name=args.room))
        print(json.dumps(res, indent=2))
        return 0

    if cmd == "hw-mic":
        daemon = HardwareMicDaemon(settings)
        if args.list_devices:
            print(json.dumps(daemon.list_audio_input_devices(), indent=2))
            return 0
        asyncio.run(daemon.run(ws_url=args.ws_url))
        return 0

    if cmd == "mcp-stdio":
        cast_mgr = GoogleHomeCastManager(settings)
        agent_eng = AntigravityAgentEngine(settings, cast_mgr)
        mcp = BridgeMCPServer(cast_mgr, agent_eng)
        asyncio.run(run_stdio_mcp_loop(mcp))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
