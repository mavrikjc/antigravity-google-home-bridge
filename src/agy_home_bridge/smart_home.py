"""Google Home Cloud-to-Cloud Smart Home Fulfillment (`action.devices.SYNC`, `QUERY`, `EXECUTE`, `DISCONNECT`).

Follows the official Google Home Developer Cloud-to-Cloud specification:
- https://developers.home.google.com/cloud-to-cloud/primer/intents
- https://developers.home.google.com/cloud-to-cloud/intents/sync
"""

from __future__ import annotations

import asyncio
from typing import Any

from agy_home_bridge.antigravity_agent import AntigravityAgentEngine
from agy_home_bridge.cast_manager import GoogleHomeCastManager, normalize_room_key


class SmartHomeFulfillment:
    """Implements Google Home Cloud-to-Cloud fulfillment intents for controlling Antigravity via 'Hey Google'."""

    def __init__(
        self,
        agent_engine: AntigravityAgentEngine,
        cast_manager: GoogleHomeCastManager,
    ) -> None:
        self.agent_engine = agent_engine
        self.cast_manager = cast_manager
        self._live_session_active: bool = True

    def build_sync_devices(self) -> list[dict[str, Any]]:
        """Return virtual Antigravity devices/scenes exposed to the Google Home Graph."""
        devices: list[dict[str, Any]] = [
            {
                "id": "agy-live-assistant",
                "type": "action.devices.types.SWITCH",
                "traits": ["action.devices.traits.OnOff"],
                "name": {
                    "defaultNames": ["Antigravity Live Assistant"],
                    "name": "Antigravity Live",
                    "nicknames": ["Antigravity", "Antigravity Voice Chat"],
                },
                "willReportState": False,
                "roomHint": self.cast_manager.active_room,
                "deviceInfo": {
                    "manufacturer": "Google Antigravity Community",
                    "model": "agy-proxmox-bridge-v1",
                    "hwVersion": "1.0",
                    "swVersion": "0.1.0",
                },
            },
            {
                "id": "agy-scene-server-status",
                "type": "action.devices.types.SCENE",
                "traits": ["action.devices.traits.Scene"],
                "name": {
                    "defaultNames": ["Antigravity Server Status"],
                    "name": "Antigravity Status Report",
                    "nicknames": ["Homelab Status", "Check Proxmox Status"],
                },
                "willReportState": False,
                "attributes": {"sceneReversible": False},
                "deviceInfo": {
                    "manufacturer": "Google Antigravity Community",
                    "model": "agy-scene-v1",
                    "hwVersion": "1.0",
                    "swVersion": "0.1.0",
                },
            },
        ]

        # Expose a "Follow Me to <Room>" scene for each discovered Google Home room speaker
        for speaker in self.cast_manager.list_speakers():
            room_name = speaker["name"]
            slug = (
                room_name.lower()
                .replace(" ", "-")
                .replace("_", "-")
            )
            devices.append(
                {
                    "id": f"agy-follow-{slug}",
                    "type": "action.devices.types.SCENE",
                    "traits": ["action.devices.traits.Scene"],
                    "name": {
                        "defaultNames": [f"Antigravity Follow Me to {room_name}"],
                        "name": f"Antigravity in {room_name}",
                        "nicknames": [
                            f"Follow me to {room_name}",
                            f"Antigravity {room_name}",
                        ],
                    },
                    "willReportState": False,
                    "roomHint": room_name,
                    "attributes": {"sceneReversible": False},
                    "customData": {"targetRoom": room_name},
                    "deviceInfo": {
                        "manufacturer": "Google Antigravity Community",
                        "model": "agy-room-follower-v1",
                        "hwVersion": "1.0",
                        "swVersion": "0.1.0",
                    },
                }
            )
        return devices

    async def handle_intent(self, body: dict[str, Any]) -> dict[str, Any]:
        """Process an incoming Google Home Cloud-to-Cloud fulfillment payload."""
        if not isinstance(body, dict):
            raise ValueError("Fulfillment body must be a JSON object")
        request_id = body.get("requestId", "req-default")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("requestId must be a string")
        inputs = body.get("inputs", [])
        if not isinstance(inputs, list):
            raise ValueError("inputs must be a list")
        if not inputs:
            return {"requestId": request_id, "payload": {}}

        intent_obj = inputs[0]
        if not isinstance(intent_obj, dict):
            raise ValueError("inputs[0] must be an object")
        intent_name = intent_obj.get("intent", "")
        if not isinstance(intent_name, str):
            raise ValueError("intent must be a string")
        payload = intent_obj.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("payload must be an object")

        if intent_name == "action.devices.SYNC":
            return {
                "requestId": request_id,
                "payload": {
                    "agentUserId": "antigravity-proxmox-owner",
                    "devices": self.build_sync_devices(),
                },
            }

        if intent_name == "action.devices.QUERY":
            query_devices = payload.get("devices", [])
            if not isinstance(query_devices, list):
                raise ValueError("payload.devices must be a list")
            device_states: dict[str, Any] = {}
            for dev in query_devices:
                if not isinstance(dev, dict):
                    raise ValueError("Each device in payload.devices must be an object")
                dev_id = dev.get("id", "")
                if not isinstance(dev_id, str):
                    raise ValueError("Device id must be a string")
                device_states[dev_id] = {
                    "online": True,
                    "status": "SUCCESS",
                    "on": self._live_session_active,
                }
            return {
                "requestId": request_id,
                "payload": {"devices": device_states},
            }

        if intent_name == "action.devices.EXECUTE":
            commands = payload.get("commands", [])
            if not isinstance(commands, list):
                raise ValueError("payload.commands must be a list")

            # Validate the entire command structure before executing side effects.
            parsed_commands: list[tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any]]]]] = []
            for cmd in commands:
                if not isinstance(cmd, dict):
                    raise ValueError("Each command in payload.commands must be an object")
                devices = cmd.get("devices", [])
                executions = cmd.get("execution", [])
                if not isinstance(devices, list) or not isinstance(executions, list):
                    raise ValueError("Command devices and execution must be lists")

                parsed_devices: list[tuple[str, dict[str, Any]]] = []
                for dev in devices:
                    if not isinstance(dev, dict):
                        raise ValueError("Each device in command.devices must be an object")
                    dev_id = dev.get("id", "")
                    custom_data = dev.get("customData", {})
                    if not isinstance(dev_id, str):
                        raise ValueError("Device id must be a string")
                    if custom_data is None:
                        custom_data = {}
                    elif not isinstance(custom_data, dict):
                        raise ValueError("Device customData must be an object")
                    if dev_id.startswith("agy-follow-"):
                        raw_target_room = custom_data.get("targetRoom")
                        if raw_target_room is not None and not isinstance(raw_target_room, str):
                            raise ValueError("Device customData.targetRoom must be a string")
                        if isinstance(raw_target_room, str) and raw_target_room != "":
                            if not raw_target_room.strip() or not normalize_room_key(raw_target_room):
                                raise ValueError(
                                    "Device customData.targetRoom must contain at least one alphanumeric character"
                                )
                        else:
                            slug_room = dev_id.replace("agy-follow-", "", 1).replace("-", " ").strip()
                            if not slug_room or not normalize_room_key(slug_room):
                                raise ValueError(
                                    "Follow scene device id must specify a valid target room"
                                )
                    parsed_devices.append((dev_id, custom_data))

                parsed_execs: list[tuple[str, dict[str, Any]]] = []
                for exec_item in executions:
                    if not isinstance(exec_item, dict):
                        raise ValueError("Each execution item must be an object")
                    exec_cmd = exec_item.get("command", "")
                    params = exec_item.get("params", {})
                    if not isinstance(exec_cmd, str):
                        raise ValueError("Execution command must be a string")
                    if params is None:
                        params = {}
                    elif not isinstance(params, dict):
                        raise ValueError("Execution params must be an object")
                    parsed_execs.append((exec_cmd, params))

                parsed_commands.append((parsed_devices, parsed_execs))

            executed_ids: list[str] = []

            for parsed_devices, parsed_execs in parsed_commands:
                for dev_id, custom_data in parsed_devices:
                    executed_ids.append(dev_id)

                    for exec_cmd, params in parsed_execs:
                        if dev_id == "agy-live-assistant" and exec_cmd == "action.devices.commands.OnOff":
                            self._live_session_active = bool(params.get("on", True))
                            greeting = (
                                f"Antigravity Live is active in {self.cast_manager.active_room}. How can I help?"
                                if self._live_session_active
                                else "Antigravity Live audio output paused."
                            )
                            asyncio.create_task(
                                self.cast_manager.speak_on_room(greeting)
                            )

                        elif dev_id == "agy-scene-server-status":
                            asyncio.create_task(
                                self.agent_engine.ask(
                                    "Check our Proxmox home server status and give a brief 2-sentence voice summary.",
                                    cast_response=True,
                                )
                            )

                        elif dev_id.startswith("agy-follow-"):
                            raw_target_room = custom_data.get("targetRoom")
                            if isinstance(raw_target_room, str) and raw_target_room.strip():
                                target_room = raw_target_room.strip()
                            else:
                                target_room = (
                                    dev_id.replace("agy-follow-", "", 1)
                                    .replace("-", " ")
                                    .title()
                                )
                            new_room = self.cast_manager.set_active_room(target_room)
                            asyncio.create_task(
                                self.cast_manager.speak_on_room(
                                    f"Antigravity Live is now following you in {new_room}.",
                                    room_name=new_room,
                                )
                            )

            return {
                "requestId": request_id,
                "payload": {
                    "commands": [
                        {
                            "ids": executed_ids,
                            "status": "SUCCESS",
                            "states": {
                                "online": True,
                                "on": self._live_session_active,
                            },
                        }
                    ]
                },
            }

        if intent_name == "action.devices.DISCONNECT":
            return {"requestId": request_id, "payload": {}}

        return {
            "requestId": request_id,
            "payload": {"errorCode": "protocolError"},
        }
