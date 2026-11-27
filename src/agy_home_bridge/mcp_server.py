"""MCP (Model Context Protocol) JSON-RPC Server for Antigravity IDE / CLI integration.

Allows any local or remote Antigravity session to discover Google Home speakers,
switch active rooms, and speak out loud across the house.
"""

from __future__ import annotations

import asyncio
import json
import math
import sys
from typing import Any

from agy_home_bridge.antigravity_agent import AntigravityAgentEngine
from agy_home_bridge.cast_manager import GoogleHomeCastManager, normalize_room_key


class BridgeMCPServer:
    """Handles MCP JSON-RPC 2.0 requests over HTTP (`/mcp`) or stdio (`agy-home-bridge mcp-stdio`)."""

    def __init__(
        self,
        cast_manager: GoogleHomeCastManager,
        agent_engine: AntigravityAgentEngine,
    ) -> None:
        self.cast_manager = cast_manager
        self.agent_engine = agent_engine

    def get_tools_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "speak_on_google_home",
                "description": (
                    "Speak a message out loud on any Google Home speaker or speaker group on the LAN "
                    "(e.g., 'Kitchen', 'Office', 'Living Room', 'Whole Home')."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The spoken message to announce on Google Home.",
                        },
                        "room": {
                            "type": "string",
                            "description": "Optional target Google Home room or speaker group name.",
                        },
                        "volume": {
                            "type": "number",
                            "description": "Optional speaker volume between 0.0 and 1.0.",
                        },
                    },
                    "required": ["message"],
                },
            },
            {
                "name": "switch_active_google_home_room",
                "description": (
                    "Switch the active Google Home room speaker that Antigravity Live follows the user into."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "room": {
                            "type": "string",
                            "description": "Target Google Home speaker or speaker group name.",
                        }
                    },
                    "required": ["room"],
                },
            },
            {
                "name": "list_google_home_speakers",
                "description": "List all discovered Google Home speakers and speaker groups on the home network.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "refresh": {
                            "type": "boolean",
                            "description": "Run a fresh mDNS LAN scan before returning speakers.",
                        }
                    },
                },
            },
        ]

    async def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any]:
        """Process a single MCP JSON-RPC 2.0 message."""
        if not isinstance(request, dict):
            return {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
        req_id = request.get("id")
        method = request.get("method")
        if not isinstance(method, str) or not method.strip():
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32600,
                    "message": "Invalid Request: 'method' must be a non-empty string",
                },
            }
        raw_params = request.get("params")
        if "params" in request and raw_params is not None and not isinstance(raw_params, dict):
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32602,
                    "message": "Invalid params: 'params' must be an object",
                },
            }
        params: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "antigravity-google-home-bridge",
                        "version": "0.1.0",
                    },
                },
            }

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.get_tools_manifest()},
            }

        if method == "tools/call":
            tool_name = params.get("name")
            if not isinstance(tool_name, str) or not tool_name.strip():
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: 'name' must be a non-empty string",
                    },
                }
            raw_args = params.get("arguments")
            if "arguments" in params and raw_args is not None and not isinstance(raw_args, dict):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: 'arguments' must be an object",
                    },
                }
            args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}

            if tool_name == "speak_on_google_home":
                message = args.get("message", "")
                room_arg = args.get("room")
                volume_arg = args.get("volume")
                if not isinstance(message, str):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Invalid params: 'message' must be a string"},
                    }
                if room_arg is not None and not isinstance(room_arg, str):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Invalid params: 'room' must be a string"},
                    }
                if volume_arg is not None and (
                    isinstance(volume_arg, bool)
                    or not isinstance(volume_arg, (int, float))
                    or not math.isfinite(volume_arg)
                ):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": "Invalid params: 'volume' must be a finite number"},
                    }
                try:
                    res = await self.cast_manager.speak_on_room(
                        text=message,
                        room_name=room_arg,
                        volume=volume_arg,
                    )
                except ValueError as exc:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": str(exc)},
                    }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                    },
                }

            if tool_name == "switch_active_google_home_room":
                room_arg = args.get("room", "")
                if not isinstance(room_arg, str) or not room_arg.strip():
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params: 'room' must be a non-empty string",
                        },
                    }
                if not normalize_room_key(room_arg):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params: 'room' must contain at least one alphanumeric character",
                        },
                    }
                try:
                    room = self.cast_manager.set_active_room(room_arg)
                except ValueError as exc:
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32602, "message": str(exc)},
                    }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": f"Active Google Home room set to '{room}'.",
                            }
                        ]
                    },
                }

            if tool_name == "list_google_home_speakers":
                refresh_arg = args.get("refresh", False)
                if not isinstance(refresh_arg, bool):
                    return {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": -32602,
                            "message": "Invalid params: 'refresh' must be a boolean",
                        },
                    }
                if refresh_arg:
                    await self.cast_manager.discover_speakers()
                payload = {
                    "active_room": self.cast_manager.active_room,
                    "routing_mode": self.cast_manager.routing_mode,
                    "speakers": self.cast_manager.list_speakers(),
                }
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {"type": "text", "text": json.dumps(payload, indent=2)}
                        ]
                    },
                }

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {},
        }


async def run_stdio_mcp_loop(mcp_server: BridgeMCPServer) -> None:
    """Run MCP server over stdin/stdout for local Antigravity IDE / CLI registration."""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        text = line.decode("utf-8").strip()
        if not text:
            continue
        try:
            def _reject_constant(val: str) -> Any:
                raise ValueError(f"Non-standard JSON constant {val!r} is not allowed")

            req = json.loads(text, parse_constant=_reject_constant)
            if "id" not in req:
                continue
            resp = await mcp_server.handle_jsonrpc(req)
            sys.stdout.write(json.dumps(resp, allow_nan=False) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stderr.write(f"MCP stdio error: {exc}\n")
            sys.stderr.flush()
