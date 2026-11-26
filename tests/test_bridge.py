"""Automated test suite for Antigravity Google Home Bridge."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from agy_home_bridge.config import BridgeSettings
from agy_home_bridge.hardware_mic import compute_pcm16_rms
from agy_home_bridge.server import create_app


@pytest.fixture()
def test_settings(tmp_path: Path) -> BridgeSettings:
    return BridgeSettings(
        GEMINI_API_KEY="",
        ANTIGRAVITY_LS_ADDRESS="",
        ANTIGRAVITY_CSRF_TOKEN="",
        AGY_WORKSPACE_DIR=str(tmp_path),
        AGY_SAVE_DIR=str(tmp_path / "conversations"),
        AGY_APP_DATA_DIR=str(tmp_path / "brain"),
        HOST_LAN_IP="192.168.1.50",
        BRIDGE_PORT=8765,
        DEFAULT_CAST_ROOM="Office",
        DEFAULT_AUDIO_ROUTING="both",
    )


@pytest.fixture()
def client(test_settings: BridgeSettings) -> TestClient:
    app = create_app(test_settings)
    # Seed mock Google Home speakers on the LAN
    cast_mgr = app.state.cast_manager
    cast_mgr.register_mock_speaker("Office", host="192.168.1.101")
    cast_mgr.register_mock_speaker("Kitchen", host="192.168.1.102")
    cast_mgr.register_mock_speaker(
        "Whole Home",
        model_name="Google Cast Group",
        cast_type="group",
        host="192.168.1.101",
    )
    return TestClient(app)


def test_health_and_pwa_index(client: TestClient) -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["active_room"] == "Office"
    assert data["speakers_count"] == 3
    assert data["lan_url"] == "http://192.168.1.50:8765"

    html_resp = client.get("/")
    assert html_resp.status_code == 200
    assert "Antigravity Live" in html_resp.text
    assert 'closedby="any"' in html_resp.text
    assert 'name="color-scheme" content="light dark"' in html_resp.text


def test_speaker_listing_and_room_switching(client: TestClient) -> None:
    resp = client.get("/api/speakers")
    assert resp.status_code == 200
    speakers = resp.json()["speakers"]
    assert len(speakers) == 3

    # Fuzzy match "kitchen" -> "Kitchen"
    sel_resp = client.post(
        "/api/cast/select",
        json={"room": "kitchen", "routing_mode": "cast"},
    )
    assert sel_resp.status_code == 200
    payload = sel_resp.json()
    assert payload["active_room"] == "Kitchen"
    assert payload["routing_mode"] == "cast"


def test_cast_speak_and_tts_serving(client: TestClient, tmp_path: Path) -> None:
    fake_mp3 = tmp_path / "brain" / ".." / "tts_cache" / "tts_dummy.mp3"
    fake_mp3 = fake_mp3.resolve()
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"ID3-FAKE-MP3-BYTES")

    with patch.object(
        client.app.state.cast_manager,
        "generate_tts_file",
        return_value=fake_mp3,
    ):
        resp = client.post(
            "/api/cast/speak",
            json={"text": "Antigravity task completed.", "room": "Kitchen"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["room"] == "Kitchen"
        assert data["media_url"].endswith("/api/tts/tts_dummy.mp3")

        audio_resp = client.get("/api/tts/tts_dummy.mp3")
        assert audio_resp.status_code == 200
        assert audio_resp.content == b"ID3-FAKE-MP3-BYTES"


def test_google_home_cloud_to_cloud_fulfillment(client: TestClient) -> None:
    # 1. SYNC intent
    sync_resp = client.post(
        "/api/google-home/fulfillment",
        json={
            "requestId": "req-sync-1",
            "inputs": [{"intent": "action.devices.SYNC"}],
        },
    )
    assert sync_resp.status_code == 200
    sync_data = sync_resp.json()
    assert sync_data["requestId"] == "req-sync-1"
    device_ids = [d["id"] for d in sync_data["payload"]["devices"]]
    assert "agy-live-assistant" in device_ids
    assert "agy-follow-kitchen" in device_ids

    # 2. QUERY intent
    query_resp = client.post(
        "/api/google-home/fulfillment",
        json={
            "requestId": "req-query-1",
            "inputs": [
                {
                    "intent": "action.devices.QUERY",
                    "payload": {"devices": [{"id": "agy-live-assistant"}]},
                }
            ],
        },
    )
    assert query_resp.status_code == 200
    q_state = query_resp.json()["payload"]["devices"]["agy-live-assistant"]
    assert q_state["online"] is True
    assert q_state["status"] == "SUCCESS"

    # 3. EXECUTE intent (Follow Me to Kitchen scene)
    with patch.object(
        client.app.state.cast_manager,
        "generate_tts_file",
        return_value=Path(__file__),
    ):
        exec_resp = client.post(
            "/api/google-home/fulfillment",
            json={
                "requestId": "req-exec-1",
                "inputs": [
                    {
                        "intent": "action.devices.EXECUTE",
                        "payload": {
                            "commands": [
                                {
                                    "devices": [
                                        {
                                            "id": "agy-follow-kitchen",
                                            "customData": {"targetRoom": "Kitchen"},
                                        }
                                    ],
                                    "execution": [
                                        {
                                            "command": "action.devices.commands.ActivateScene",
                                            "params": {"deactivate": False},
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ],
            },
        )
        assert exec_resp.status_code == 200
        assert client.app.state.cast_manager.active_room == "Kitchen"


def test_mcp_jsonrpc_endpoints(client: TestClient) -> None:
    # tools/list
    list_resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert list_resp.status_code == 200
    tools = list_resp.json()["result"]["tools"]
    tool_names = {t["name"] for t in tools}
    assert "speak_on_google_home" in tool_names
    assert "switch_active_google_home_room" in tool_names

    # tools/call -> switch_active_google_home_room
    call_resp = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "switch_active_google_home_room",
                "arguments": {"room": "Whole Home"},
            },
        },
    )
    assert call_resp.status_code == 200
    assert client.app.state.cast_manager.active_room == "Whole Home"


def test_full_duplex_websocket_live_bridge(client: TestClient) -> None:
    with patch.object(
        client.app.state.cast_manager,
        "generate_tts_file",
        return_value=Path(__file__),
    ):
        with client.websocket_connect("/ws/live") as ws:
            initial = ws.receive_json()
            assert initial["type"] == "status"
            assert initial["state"] == "connected"

            # Switch room over live WebSocket
            ws.send_json({"type": "set_room", "room": "Kitchen"})
            room_status = ws.receive_json()
            assert room_status["active_room"] == "Kitchen"

            # Test barge-in interruption signal
            ws.send_json({"type": "barge_in"})
            interrupted = ws.receive_json()
            assert interrupted["type"] == "interrupted"
            listening = ws.receive_json()
            assert listening["state"] == "listening"

            # Send live voice phrase to switch room via natural language fallback
            ws.send_json({"type": "text", "text": "Follow me to the office"})
            user_tr = ws.receive_json()
            assert user_tr["type"] == "transcript"
            assert user_tr["role"] == "user"

            thinking = ws.receive_json()
            assert thinking["state"] == "thinking"

            tool_ev = ws.receive_json()
            assert tool_ev["type"] == "tool_call"
            assert tool_ev["name"] == "switch_active_google_home_room"
            assert tool_ev["active_room"] == "Office"

            agent_tr = ws.receive_json()
            assert agent_tr["type"] == "transcript"
            assert agent_tr["role"] == "agent"
            assert "Office" in agent_tr["text"]


def test_pcm16_rms_calculation() -> None:
    assert compute_pcm16_rms(b"") == 0.0
    silence = b"\x00\x00" * 160
    assert compute_pcm16_rms(silence) == 0.0
    loud = b"\xff\x7f" * 160
    assert compute_pcm16_rms(loud) > 0.99
