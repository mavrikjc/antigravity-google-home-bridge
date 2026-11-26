"""Google Home Cloud-to-Cloud Smart Home Fulfillment (`action.devices.SYNC`, `QUERY`, `EXECUTE`, `DISCONNECT`).

Follows the official Google Home Developer Cloud-to-Cloud specification:
- https://developers.home.google.com/cloud-to-cloud/primer/intents
- https://developers.home.google.com/cloud-to-cloud/intents/sync
"""

from __future__ import annotations

import asyncio
from typing import Any

from agy_home_bridge.antigravity_agent import AntigravityAgentEngine
from agy_home_bridge.cast_manager import GoogleHomeCastManager


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
        request_id = body.get("requestId", "req-default")
        inputs = body.get("inputs", [])
        if not inputs:
            return {"requestId": request_id, "payload": {}}

        intent_obj = inputs[0]
        intent_name = intent_obj.get("intent", "")
        payload = intent_obj.get("payload", {})

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
            device_states: dict[str, Any] = {}
            for dev in query_devices:
                dev_id = dev.get("id", "")
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
            executed_ids: list[str] = []

            for cmd in commands:
                devices = cmd.get("devices", [])
                executions = cmd.get("execution", [])
                for dev in devices:
                    dev_id = dev.get("id", "")
                    custom_data = dev.get("customData", {})
                    executed_ids.append(dev_id)

                    for exec_item in executions:
                        exec_cmd = exec_item.get("command", "")
                        params = exec_item.get("params", {})

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
                            target_room = custom_data.get("targetRoom") or dev_id.replace(
                                "agy-follow-", ""
                            ).replace("-", " ").title()
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
