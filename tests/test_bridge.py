"""Comprehensive unit and integration test suite for Antigravity Google Home Bridge."""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import types
import wave
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from agy_home_bridge.antigravity_agent import AntigravityAgentEngine
from agy_home_bridge.cast_manager import GoogleHomeCastManager
from agy_home_bridge.cli import main as cli_main
from agy_home_bridge.config import BridgeSettings, detect_lan_ip
from agy_home_bridge.gemini_live import FullDuplexLiveBridge
from agy_home_bridge.hardware_mic import HardwareMicDaemon, compute_pcm16_rms
from agy_home_bridge.mcp_server import BridgeMCPServer, run_stdio_mcp_loop
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
        KNOWN_CAST_HOSTS="192.168.1.101, 192.168.1.102",
    )


@pytest.fixture()
def client(test_settings: BridgeSettings) -> TestClient:
    app = create_app(test_settings)
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


# ============================================================================
# 1. Config & LAN IP Detection Tests
# ============================================================================
def test_config_and_lan_ip_detection(test_settings: BridgeSettings) -> None:
    ip = detect_lan_ip()
    assert isinstance(ip, str) and len(ip) >= 7
    assert test_settings.parsed_known_cast_hosts == ["192.168.1.101", "192.168.1.102"]

    empty_cfg = BridgeSettings(HOST_LAN_IP="", KNOWN_CAST_HOSTS="", HW_MIC_DEVICE_INDEX="")
    assert empty_cfg.parsed_known_cast_hosts == []
    assert empty_cfg.hw_mic_device_index is None
    assert empty_cfg.resolved_lan_ip == ip

    with patch("socket.socket", side_effect=OSError("no network")):
        assert detect_lan_ip() == "127.0.0.1"


# ============================================================================
# 2. FastAPI Endpoints & Lifespan Tests
# ============================================================================
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


def test_speaker_listing_refresh_and_room_switching(client: TestClient) -> None:
    with patch.object(
        client.app.state.cast_manager,
        "discover_speakers",
        new_callable=AsyncMock,
    ) as mock_disc:
        resp = client.get("/api/speakers?refresh=true")
        assert resp.status_code == 200
        mock_disc.assert_awaited_once()

    sel_resp = client.post(
        "/api/cast/select",
        json={"room": "kitchen", "routing_mode": "cast"},
    )
    assert sel_resp.status_code == 200
    payload = sel_resp.json()
    assert payload["active_room"] == "Kitchen"
    assert payload["routing_mode"] == "cast"

    # Invalid routing mode returns 400
    bad_resp = client.post(
        "/api/cast/select",
        json={"room": "Kitchen", "routing_mode": "invalid_mode"},
    )
    assert bad_resp.status_code == 400


def test_cast_speak_and_tts_serving(client: TestClient, tmp_path: Path) -> None:
    fake_mp3 = (tmp_path / "tts_cache" / "tts_dummy.mp3").resolve()
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"ID3-FAKE-MP3-BYTES")

    with patch.object(
        client.app.state.cast_manager,
        "generate_tts_file",
        return_value=fake_mp3,
    ):
        resp = client.post(
            "/api/cast/speak",
            json={"text": "Antigravity task completed.", "room": "Kitchen", "volume": 0.5},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["room"] == "Kitchen"
        assert data["media_url"].endswith("/api/tts/tts_dummy.mp3")

        audio_resp = client.get("/api/tts/tts_dummy.mp3")
        assert audio_resp.status_code == 200
        assert audio_resp.content == b"ID3-FAKE-MP3-BYTES"

    # Missing file returns 404
    missing_resp = client.get("/api/tts/does_not_exist.mp3")
    assert missing_resp.status_code == 404


def test_agent_ask_endpoint(client: TestClient, tmp_path: Path) -> None:
    fake_mp3 = (tmp_path / "tts_cache" / "tts_agent.mp3").resolve()
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"ID3")

    with patch.object(
        client.app.state.cast_manager,
        "generate_tts_file",
        return_value=fake_mp3,
    ):
        resp = client.post(
            "/api/agent/ask",
            json={"prompt": "Follow me to the kitchen", "cast_response": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "Kitchen" in data["text"]
        assert data["active_room"] == "Kitchen"


# ============================================================================
# 3. Google Home Cast Manager Unit Tests (Discovery, WAV, PyChromecast)
# ============================================================================
@pytest.mark.asyncio
async def test_cast_manager_discovery_wav_and_playback(test_settings: BridgeSettings) -> None:
    cast_mgr = GoogleHomeCastManager(test_settings)
    assert cast_mgr.resolve_room_name("   ") is None

    # Save PCM as WAV and verify header
    pcm_data = b"\x00\x01" * 2400
    wav_path = cast_mgr.save_pcm_as_wav(pcm_data, sample_rate=24000)
    assert wav_path.exists()
    with wave.open(str(wav_path), "rb") as wf:
        assert wf.getframerate() == 24000
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2

    # Cast PCM audio when no device is present
    pcm_res = await cast_mgr.cast_pcm_audio(pcm_data, sample_rate=24000, room_name="Office")
    assert pcm_res["status"] == "queued_no_cast_device"

    # Test generate_tts_file validation and gTTS call
    with pytest.raises(ValueError, match="Cannot synthesize empty text"):
        cast_mgr.generate_tts_file("   ")

    with patch("agy_home_bridge.cast_manager.gTTS") as mock_gtts:
        inst = MagicMock()
        mock_gtts.return_value = inst

        def _fake_save(path_str: str) -> None:
            Path(path_str).write_bytes(b"MP3DATA")

        inst.save.side_effect = _fake_save
        p1 = cast_mgr.generate_tts_file("Hello Google Home")
        assert p1.exists()
        # Second call uses cache without calling gTTS again
        p2 = cast_mgr.generate_tts_file("Hello Google Home")
        assert p1 == p2
        assert mock_gtts.call_count == 1

    # Mock pychromecast discovery and active media controller playback
    fake_cc = MagicMock()
    fake_cc.cast_info.friendly_name = "Living Room Speaker"
    fake_cc.cast_info.uuid = "uuid-1234"
    fake_cc.cast_info.model_name = "Nest Audio"
    fake_cc.cast_info.cast_type = "audio"
    fake_cc.cast_info.host = "192.168.1.120"
    fake_cc.cast_info.port = 8009
    fake_browser = MagicMock()

    with patch("pychromecast.get_chromecasts", return_value=([fake_cc], fake_browser)):
        discovered = await cast_mgr.discover_speakers(timeout=0.5)
        assert len(discovered) == 1
        assert discovered[0]["name"] == "Living Room Speaker"
        assert cast_mgr.active_room == "Living Room Speaker"

        # Cast URL to discovered Chromecast device
        play_res = cast_mgr._cast_url_sync(
            "living room speaker",
            "http://192.168.1.50:8765/api/tts/test.mp3",
            "audio/mp3",
            volume=0.65,
        )
        assert play_res["status"] == "casting"
        fake_cc.set_volume.assert_called_once_with(0.65)
        fake_cc.media_controller.play_media.assert_called_once()

        # Simulate Chromecast playback error
        fake_cc.wait.side_effect = RuntimeError("Cast socket timeout")
        err_res = cast_mgr._cast_url_sync(
            "Living Room Speaker",
            "http://192.168.1.50:8765/api/tts/test.mp3",
            "audio/mp3",
        )
        assert err_res["status"] == "error"

        # Test warm_room & stop_room helpers
        fake_cc.wait.side_effect = None
        fake_cc.app_id = None
        await cast_mgr.warm_room("Living Room Speaker")
        assert (test_settings.tts_cache_dir / "warmup_silence.wav").exists()
        await cast_mgr.stop_room("Living Room Speaker")
        fake_cc.media_controller.stop.assert_called_once()


# ============================================================================
# 4. Antigravity Agent Engine Tests (Custom Tools, SDK, GenAI, LS Tiers)
# ============================================================================
@pytest.mark.asyncio
async def test_antigravity_agent_tiers_and_tools(
    test_settings: BridgeSettings, tmp_path: Path
) -> None:
    cast_mgr = GoogleHomeCastManager(test_settings)
    cast_mgr.register_mock_speaker("Kitchen")
    engine = AntigravityAgentEngine(test_settings, cast_mgr)

    # Persistence save & reload
    engine._save_persisted_conversation_id("conv-persisted-99")
    engine2 = AntigravityAgentEngine(test_settings, cast_mgr)
    assert engine2.conversation_id == "conv-persisted-99"

    # Execute custom tools directly
    tools = {fn.__name__: fn for fn in engine.build_custom_tools()}
    assert "Kitchen" in tools["list_google_home_speakers"]()
    assert "Kitchen" in tools["switch_active_google_home_room"]("Kitchen")
    assert "Uptime" in tools["check_homelab_status"]()

    fake_mp3 = (tmp_path / "tts_cache" / "tool.mp3").resolve()
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"MP3")
    with patch.object(cast_mgr, "generate_tts_file", return_value=fake_mp3):
        # Run speak_on_google_home outside a running event loop via thread
        spoken_json = await asyncio.to_thread(
            tools["speak_on_google_home"], "Dinner is ready", "Kitchen"
        )
        assert "Kitchen" in spoken_json

    # Tier 1: Mock official `google.antigravity` SDK
    class FakeResponseStream:
        def __aiter__(self):
            async def _gen():
                yield "Hello from "
                yield "Antigravity SDK!"

            return _gen()

    class FakeAgent:
        def __init__(self, cfg: Any) -> None:
            self.cfg = cfg
            self.conversation_id = "agy-sdk-conv-123"

        async def __aenter__(self) -> FakeAgent:
            return self

        async def __aexit__(self, *args: Any) -> None:
            pass

        async def chat(self, prompt: str) -> FakeResponseStream:
            return FakeResponseStream()

    fake_agy_mod = types.ModuleType("google.antigravity")
    fake_agy_mod.Agent = FakeAgent  # type: ignore
    fake_agy_mod.LocalAgentConfig = lambda **kw: kw  # type: ignore
    fake_agy_mod.CapabilitiesConfig = lambda **kw: kw  # type: ignore

    with patch.dict(sys.modules, {"google.antigravity": fake_agy_mod}):
        res_sdk = await engine.ask("Run diagnostics", cast_response=False)
        assert res_sdk.backend == "google-antigravity-sdk"
        assert res_sdk.text == "Hello from Antigravity SDK!"
        assert res_sdk.conversation_id == "agy-sdk-conv-123"

    # Tier 2: Mock `google-genai` fallback when `google.antigravity` is not installed
    test_settings.gemini_api_key = "fake-gemini-key"
    fake_genai_client = MagicMock()
    fake_genai_resp = MagicMock()
    fake_genai_resp.text = "Hello from Gemini 2.5 Flash!"
    fake_genai_client.models.generate_content.return_value = fake_genai_resp

    with patch("google.genai.Client", return_value=fake_genai_client):
        res_genai = await engine.ask("What is the weather?", cast_response=False)
        assert res_genai.backend == "google-genai"
        assert res_genai.text == "Hello from Gemini 2.5 Flash!"

    # Tier 3: Local Antigravity Language Server ConversationService
    test_settings.gemini_api_key = ""
    test_settings.antigravity_ls_address = "127.0.0.1:42100"
    test_settings.antigravity_csrf_token = "csrf-secret"

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"conversation": {"id": "ls-999"}}
        mock_resp.raise_for_status.return_value = None
        mock_post.return_value = mock_resp

        res_ls = await engine.ask("Check server logs", cast_response=False)
        assert res_ls.backend == "antigravity-ls-conversation-service"
        assert res_ls.conversation_id == "ls-999"


# ============================================================================
# 5. Google Home Cloud-to-Cloud Smart Home Fulfillment Tests
# ============================================================================
def test_google_home_cloud_to_cloud_fulfillment(
    client: TestClient, tmp_path: Path
) -> None:
    fake_mp3 = (tmp_path / "tts_cache" / "sh.mp3").resolve()
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"MP3")

    # Empty inputs
    empty_resp = client.post(
        "/api/google-home/fulfillment",
        json={"requestId": "req-0", "inputs": []},
    )
    assert empty_resp.status_code == 200

    # 1. SYNC intent
    sync_resp = client.post(
        "/api/google-home/fulfillment",
        json={
            "requestId": "req-sync-1",
            "inputs": [{"intent": "action.devices.SYNC"}],
        },
    )
    assert sync_resp.status_code == 200
    device_ids = [d["id"] for d in sync_resp.json()["payload"]["devices"]]
    assert "agy-live-assistant" in device_ids
    assert "agy-scene-server-status" in device_ids
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
    assert query_resp.json()["payload"]["devices"]["agy-live-assistant"]["online"] is True

    # 3. EXECUTE intents (OnOff, Server Status Scene, Follow-Me Scene)
    with patch.object(
        client.app.state.cast_manager,
        "generate_tts_file",
        return_value=fake_mp3,
    ):
        for dev_id, cmd_name, params in [
            ("agy-live-assistant", "action.devices.commands.OnOff", {"on": True}),
            ("agy-scene-server-status", "action.devices.commands.ActivateScene", {}),
            ("agy-follow-kitchen", "action.devices.commands.ActivateScene", {}),
        ]:
            exec_resp = client.post(
                "/api/google-home/fulfillment",
                json={
                    "requestId": f"req-exec-{dev_id}",
                    "inputs": [
                        {
                            "intent": "action.devices.EXECUTE",
                            "payload": {
                                "commands": [
                                    {
                                        "devices": [{"id": dev_id}],
                                        "execution": [
                                            {"command": cmd_name, "params": params}
                                        ],
                                    }
                                ]
                            },
                        }
                    ],
                },
            )
            assert exec_resp.status_code == 200

    # 4. DISCONNECT intent & unknown intent
    disc_resp = client.post(
        "/api/google-home/fulfillment",
        json={
            "requestId": "req-disc",
            "inputs": [{"intent": "action.devices.DISCONNECT"}],
        },
    )
    assert disc_resp.status_code == 200

    unk_resp = client.post(
        "/api/google-home/fulfillment",
        json={
            "requestId": "req-unk",
            "inputs": [{"intent": "action.devices.UNKNOWN"}],
        },
    )
    assert unk_resp.json()["payload"]["errorCode"] == "protocolError"

    # Bearer token enforcement when non-default token is configured
    client.app.state.settings.smart_home_webhook_token = "super-secret-token"
    unauth = client.post(
        "/api/google-home/fulfillment",
        json={"requestId": "req-auth", "inputs": []},
    )
    assert unauth.status_code == 401
    auth_ok = client.post(
        "/api/google-home/fulfillment",
        headers={"Authorization": "Bearer super-secret-token"},
        json={"requestId": "req-auth", "inputs": []},
    )
    assert auth_ok.status_code == 200


# ============================================================================
# 6. MCP Server (HTTP & Stdio) Tests
# ============================================================================
@pytest.mark.asyncio
async def test_mcp_jsonrpc_endpoints_and_stdio(
    client: TestClient, tmp_path: Path
) -> None:
    fake_mp3 = (tmp_path / "tts_cache" / "mcp.mp3").resolve()
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"MP3")

    # initialize
    init_resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 0, "method": "initialize"},
    )
    assert init_resp.status_code == 200
    assert (
        init_resp.json()["result"]["serverInfo"]["name"]
        == "antigravity-google-home-bridge"
    )

    # tools/list
    list_resp = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
    )
    assert list_resp.status_code == 200
    tool_names = {t["name"] for t in list_resp.json()["result"]["tools"]}
    assert "speak_on_google_home" in tool_names

    # tools/call -> speak_on_google_home, switch_active_google_home_room, list_google_home_speakers, unknown
    with patch.object(
        client.app.state.cast_manager,
        "generate_tts_file",
        return_value=fake_mp3,
    ):
        speak_call = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "speak_on_google_home",
                    "arguments": {"message": "Hello from MCP", "room": "Office"},
                },
            },
        )
        assert speak_call.status_code == 200

    list_call = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "list_google_home_speakers",
                "arguments": {"refresh": False},
            },
        },
    )
    assert list_call.status_code == 200

    unk_call = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        },
    )
    assert "error" in unk_call.json()


# ============================================================================
# 7. Full-Duplex Gemini Live WebSocket Bridge Tests
# ============================================================================
def test_full_duplex_websocket_live_bridge(
    client: TestClient, tmp_path: Path
) -> None:
    fake_mp3 = (tmp_path / "tts_cache" / "ws.mp3").resolve()
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"MP3")

    with patch.object(
        client.app.state.cast_manager,
        "generate_tts_file",
        return_value=fake_mp3,
    ):
        with client.websocket_connect("/ws/live") as ws:
            initial = ws.receive_json()
            assert initial["type"] == "status"
            assert initial["state"] == "connected"

            ws.send_json({"type": "set_room", "room": "Kitchen"})
            room_status = ws.receive_json()
            assert room_status["active_room"] == "Kitchen"

            ws.send_json({"type": "set_routing", "mode": "direct"})
            route_status = ws.receive_json()
            assert route_status["routing_mode"] == "direct"

            ws.send_json({"type": "barge_in"})
            interrupted = ws.receive_json()
            assert interrupted["type"] == "interrupted"
            listening = ws.receive_json()
            assert listening["state"] == "listening"

            ws.send_json({"type": "text", "text": "Follow me to the office"})
            user_tr = ws.receive_json()
            assert user_tr["role"] == "user"

            thinking = ws.receive_json()
            assert thinking["state"] == "thinking"

            tool_ev = ws.receive_json()
            assert tool_ev["type"] == "tool_call"
            assert tool_ev["active_room"] == "Office"

            agent_tr = ws.receive_json()
            assert agent_tr["role"] == "agent"


@pytest.mark.asyncio
async def test_gemini_live_native_audio_forwarding_and_tools(
    test_settings: BridgeSettings, tmp_path: Path
) -> None:
    cast_mgr = GoogleHomeCastManager(test_settings)
    cast_mgr.register_mock_speaker("Kitchen")
    agent_eng = AntigravityAgentEngine(test_settings, cast_mgr)
    bridge = FullDuplexLiveBridge(test_settings, agent_eng, cast_mgr)

    fake_ws = AsyncMock()
    pcm_chunk_b64 = base64.b64encode(b"\x01\x00" * 160).decode("ascii")
    fake_ws.receive_text = AsyncMock(
        side_effect=[
            json.dumps({"type": "audio_chunk", "pcm_b64": pcm_chunk_b64}),
            json.dumps({"type": "text", "text": "Switch to Kitchen"}),
            json.dumps({"type": "barge_in"}),
            json.dumps({"type": "set_room", "room": "Kitchen"}),
            json.dumps({"type": "set_routing", "mode": "both"}),
            asyncio.CancelledError(),
        ]
    )

    fake_session = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await bridge._forward_client_to_gemini(fake_ws, fake_session)
    fake_session.send_realtime_input.assert_awaited_once()
    fake_session.send_client_content.assert_awaited_once()

    # Test _forward_gemini_to_client with audio inline_data, text, interruption, and tool_call
    fake_part_audio = MagicMock()
    fake_part_audio.inline_data.data = b"\x02\x00" * 240
    fake_part_audio.text = "Audio transcript chunk"

    resp_interrupted = MagicMock()
    resp_interrupted.server_content.interrupted = True
    resp_interrupted.tool_call = None

    resp_turn = MagicMock()
    resp_turn.server_content.interrupted = False
    resp_turn.server_content.model_turn.parts = [fake_part_audio]
    resp_turn.server_content.turn_complete = True

    fc1 = MagicMock()
    fc1.id = "fc-1"
    fc1.name = "switch_google_home_room"
    fc1.args = {"room": "Kitchen"}

    fc2 = MagicMock()
    fc2.id = "fc-2"
    fc2.name = "delegate_to_antigravity_agent"
    fc2.args = {"task": "Check status"}

    resp_turn.tool_call.function_calls = [fc1, fc2]

    class FakeSessionReceiver:
        def __init__(self) -> None:
            self.send_tool_response = AsyncMock()
            self._calls = 0

        def receive(self):
            # Mirror google-genai: each receive() call yields one turn, then a
            # final call yields nothing when the session has closed.
            self._calls += 1
            turns = [[resp_interrupted], [resp_turn]]
            items = turns[self._calls - 1] if self._calls <= len(turns) else []

            async def _gen():
                for item in items:
                    yield item

            return _gen()

    receiver_session = FakeSessionReceiver()
    with patch.object(cast_mgr, "cast_pcm_audio", new_callable=AsyncMock):
        await bridge._forward_gemini_to_client(fake_ws, receiver_session)
    assert receiver_session._calls == 3  # kept receiving across turns
    assert cast_mgr.active_room == "Kitchen"
    assert receiver_session.send_tool_response.await_count == 1


# ============================================================================
# 8. Hardware Wireless Mic Daemon & CLI Tests
# ============================================================================
def test_pcm16_rms_and_hardware_mic_listing(test_settings: BridgeSettings) -> None:
    assert compute_pcm16_rms(b"") == 0.0
    assert compute_pcm16_rms(b"\x00\x00" * 160) == 0.0
    assert compute_pcm16_rms(b"\xff\x7f" * 160) > 0.99

    daemon = HardwareMicDaemon(test_settings)
    fake_sd = types.ModuleType("sounddevice")
    fake_sd.query_devices = lambda: [  # type: ignore
        {"name": "Hollyland Lark M2 USB Receiver", "max_input_channels": 2, "default_samplerate": 48000},
        {"name": "HDMI Output", "max_input_channels": 0},
    ]
    with patch.dict(sys.modules, {"sounddevice": fake_sd}):
        devices = daemon.list_audio_input_devices()
        assert len(devices) == 1
        assert devices[0]["name"] == "Hollyland Lark M2 USB Receiver"
    daemon.stop()


def test_cli_entrypoints(tmp_path: Path) -> None:
    with patch(
        "agy_home_bridge.cast_manager.GoogleHomeCastManager.discover_speakers",
        new_callable=AsyncMock,
        return_value=[{"name": "Office", "host": "192.168.1.101"}],
    ):
        assert cli_main(["discover"]) == 0

    with patch(
        "agy_home_bridge.cast_manager.GoogleHomeCastManager.discover_speakers",
        new_callable=AsyncMock,
        return_value=[],
    ), patch(
        "agy_home_bridge.cast_manager.GoogleHomeCastManager.speak_on_room",
        new_callable=AsyncMock,
        return_value={"status": "casting", "room": "Office"},
    ):
        assert cli_main(["speak", "Hello from CLI", "--room", "Office"]) == 0

    assert cli_main(["hw-mic", "--list-devices"]) == 0

    with patch("uvicorn.run") as mock_uvicorn:
        assert (
            cli_main(["serve", "--host", "127.0.0.1", "--port", "9000", "--no-https"])
            == 0
        )
        mock_uvicorn.assert_called_once()


def test_tls_certificate_generation_and_ca_download(
    client: TestClient, tmp_path: Path
) -> None:
    resp = client.get("/api/tls/ca.pem")
    assert resp.status_code == 200
    assert b"BEGIN CERTIFICATE" in resp.content

    # Second call reuses existing cached certificate
    resp2 = client.get("/api/tls/ca.pem")
    assert resp2.status_code == 200
    assert resp2.content == resp.content

