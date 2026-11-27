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
from agy_home_bridge.hardware_mic import (
    HardwareMicDaemon,
    compute_pcm16_rms,
    downsample_pcm16_to_16k,
)
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
            json.dumps({"type": "client_hello", "client_type": "hardware_mic"}),
            json.dumps({"type": "audio_chunk", "pcm_b64": pcm_chunk_b64}),
            json.dumps({"type": "text", "text": "Switch to Kitchen"}),
            json.dumps({"type": "barge_in"}),
            json.dumps({"type": "set_room", "room": "Kitchen"}),
            json.dumps({"type": "set_routing", "mode": "both"}),
            asyncio.CancelledError(),
        ]
    )

    cast_mgr.extend_cast_mute(5.0)
    assert cast_mgr.is_casting_audio is True
    fake_session = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await bridge._forward_client_to_gemini(fake_ws, fake_session)
    fake_session.send_realtime_input.assert_awaited_once()
    sent_blob = fake_session.send_realtime_input.call_args.kwargs["audio"]
    assert sent_blob.data == bytes(320)  # Echo-gated to silence while speaker casts
    cast_mgr._cast_mute_until = 0.0
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
        daemon.config.hw_mic_device_name = "Lark M2"
        assert daemon._resolve_input_device(fake_sd) == 0

    import struct
    pcm_48k = struct.pack("<6h", 1000, 1000, 1000, -2000, -2000, -2000)
    pcm_16k = downsample_pcm16_to_16k(pcm_48k, 48000)
    assert struct.unpack("<2h", pcm_16k) == (1000, -2000)
    daemon.stop()


def test_voice_mute_and_hud_transcript_events(client: TestClient) -> None:
    """Verify /api/voice/mute (Issue #5) and transcript EventHub broadcasting (Issues #6 & #7)."""
    mute_resp = client.post("/api/voice/mute", json={"muted": True})
    assert mute_resp.status_code == 200
    assert mute_resp.json()["voice_muted"] is True
    assert mute_resp.json()["voice_status"] == "muted"
    assert client.get("/api/health").json()["voice_muted"] is True

    with client.websocket_connect("/ws/hud") as hud_ws:
        snap = hud_ws.receive_json()
        assert snap["type"] == "hud_snapshot"
        assert snap["voice_muted"] is True
        assert snap["voice_status"] == "muted"

    unmute_resp = client.post("/api/voice/mute", json={"muted": False})
    assert unmute_resp.status_code == 200
    assert unmute_resp.json()["voice_muted"] is False


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


# ============================================================================
# 9. M1 Config Extensions, EventHub, ProjectManager & Cast/TV Presence Tests
# ============================================================================
def test_config_room_device_map_and_projects_state_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Empty ROOM_DEVICE_MAP env var must not crash pydantic-settings
    monkeypatch.setenv("ROOM_DEVICE_MAP", "")
    monkeypatch.setenv("PROJECTS_STATE_FILE", "")
    cfg_empty = BridgeSettings(
        AGY_SAVE_DIR=str(tmp_path / "save_dir"),
        HOST_LAN_IP="192.168.1.77",
        BRIDGE_PORT=8765,
    )
    assert cfg_empty.room_device_map == {}
    assert cfg_empty.resolved_projects_state_file == (
        tmp_path / "save_dir" / "projects_state.json"
    )
    assert cfg_empty.hud_lan_url == "http://192.168.1.77:8765/hud"
    assert cfg_empty.default_agent_id == "101"
    assert cfg_empty.event_history_size == 200

    # JSON object string in ROOM_DEVICE_MAP and explicit PROJECTS_STATE_FILE
    monkeypatch.setenv(
        "ROOM_DEVICE_MAP",
        json.dumps({"office": {"speaker": "Office Mini", "tvs": ["Office TV"]}}),
    )
    explicit_state = tmp_path / "custom" / "state.json"
    monkeypatch.setenv("PROJECTS_STATE_FILE", str(explicit_state))
    cfg_json = BridgeSettings()
    assert cfg_json.room_device_map["office"]["speaker"] == "Office Mini"
    assert cfg_json.resolved_projects_state_file == explicit_state

    # Dict passed by field name (populate_by_name=True) & merged by cast manager
    monkeypatch.delenv("ROOM_DEVICE_MAP", raising=False)
    monkeypatch.delenv("PROJECTS_STATE_FILE", raising=False)
    cfg_dict = BridgeSettings(
        room_device_map={"office": {"speaker": "Custom Office", "tvs": ["Studio Monitor"]}},
        default_presence_room="office",
    )
    assert cfg_dict.room_device_map["office"]["speaker"] == "Custom Office"
    cast_mgr = GoogleHomeCastManager(cfg_dict)
    for required_room in (
        "office",
        "living_room",
        "kitchen",
        "game_room",
        "98_qled",
        "master_bedroom",
        "whole_home",
    ):
        assert required_room in cast_mgr._room_device_map
    assert cast_mgr.active_speaker == "Custom Office"
    assert cast_mgr.active_tv == "Studio Monitor"

    with pytest.raises(ValueError):
        BridgeSettings(ROOM_DEVICE_MAP='["not-an-object"]')
    with pytest.raises(ValueError):
        BridgeSettings(room_device_map=123)  # type: ignore[arg-type]
    assert BridgeSettings(room_device_map=None).room_device_map == {}  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_event_hub_publish_subscribe_and_history_bounds() -> None:
    from agy_home_bridge.event_hub import KNOWN_EVENT_TYPES, EventHub

    hub = EventHub(maxlen=200, subscriber_queue_size=256)
    q1 = hub.subscribe()
    q2 = hub.subscribe()
    assert hub.subscriber_count == 2

    all_types = [
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
    ]
    assert set(all_types) == KNOWN_EVENT_TYPES

    for idx, ev_type in enumerate(all_types):
        ev = await hub.publish(
            ev_type,
            project_id="clark-home-server",
            project_name="clark-home-server",
            short_name="Clark Home Server",
            agent_id="101",
            project_agent="project Clark Home Server, agent 101",
            conversation_id="conv-1",
            task_id=f"task-{idx}",
            detail=f"payload-{idx}",
        )
        assert ev["type"] == ev_type
        assert isinstance(ev["event_id"], str) and len(ev["event_id"]) > 0
        assert "T" in ev["timestamp"]
        assert q1.get_nowait()["task_id"] == f"task-{idx}"
        assert q2.get_nowait()["type"] == ev_type

    hub.unsubscribe(q2)
    assert hub.subscriber_count == 1

    # Verify bounded deque(maxlen=200)
    for i in range(210):
        await hub.publish("token", project_id="p2", agent_id="102", seq=i)
    recent_all = hub.recent_events(limit=500)
    assert len(recent_all) == 200
    assert recent_all[0]["seq"] == 10
    assert recent_all[-1]["seq"] == 209
    assert len(hub.recent_events(limit=5)) == 5
    assert len(hub.recent_events(limit=50, project_id="p2", event_type="token")) == 50

    hub.clear_history()
    assert hub.recent_events() == []


@pytest.mark.asyncio
async def test_event_hub_slow_subscriber_isolation_and_snapshot(
    test_settings: BridgeSettings, tmp_path: Path
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    hub = EventHub(maxlen=200, subscriber_queue_size=2)
    slow_q = hub.subscribe()
    fast_q = hub.subscribe()

    # Fill slow_q (capacity 2) and drain fast_q
    await hub.publish("thought", text="one")
    await hub.publish("thought", text="two")
    assert fast_q.get_nowait()["text"] == "one"
    assert fast_q.get_nowait()["text"] == "two"

    # Third publish hits QueueFull on slow_q -> slow_q evicted, fast_q still receives
    await hub.publish("thought", text="three")
    assert hub.subscriber_count == 1
    assert fast_q.get_nowait()["text"] == "three"
    assert slow_q.qsize() == 2

    # Threadsafe publish from worker thread
    await asyncio.to_thread(
        hub.publish_threadsafe,
        "tool_call",
        project_id="alpha",
        agent_id="101",
        tool="run_cmd",
    )
    await asyncio.sleep(0.02)
    assert fast_q.get_nowait()["tool"] == "run_cmd"

    # Snapshot with missing managers must not crash
    empty_snap = hub.build_snapshot(None, None, voice_status="idle")
    assert empty_snap["type"] == "hud_snapshot"
    assert empty_snap["projects"] == []
    assert empty_snap["active_room"] is None

    # Snapshot with real ProjectManager & GoogleHomeCastManager
    ws = tmp_path / "ws_snap"
    (ws / "alpha-app").mkdir(parents=True)
    pm = ProjectManager(ws, tmp_path / "save_snap")
    cm = GoogleHomeCastManager(test_settings)
    cm.register_mock_speaker("Office")
    cm.set_presence_room("office", source="frigate_ha")

    snap = hub.build_snapshot(pm, cm, voice_status="listening")
    assert snap["type"] == "hud_snapshot"
    assert snap["voice_status"] == "listening"
    assert snap["active_room"] == "Office"
    assert snap["active_speaker"] == "Office"
    assert snap["active_presence_room"] == "office"
    assert snap["active_tv"] == "Office - Samsung Q60CD 65"
    assert snap["active_project_id"] == "alpha-app"
    assert snap["active_agent_id"] == "101"
    assert snap["active_project_agent"] == "project Alpha App, agent 101"
    assert len(snap["projects"]) == 1
    assert len(snap["recent_events"]) >= 4


def test_project_manager_scan_git_metadata_and_fallback(tmp_path: Path) -> None:
    import subprocess
    from agy_home_bridge.project_manager import ProjectManager

    ws = tmp_path / "workspace"
    ws.mkdir()
    save_dir = ws / "conversations"  # intentionally nested inside workspace to test exclusion
    save_dir.mkdir()

    # Ignored directories (avoid creating .git at workspace root, which triggers single-repo mode)
    for ignored in (".venv", "venv", "__pycache__", "node_modules", ".hidden_proj"):
        (ws / ignored).mkdir()

    # Real git project with branch + uncommitted change
    git_proj = ws / "antigravity-google-home-bridge"
    git_proj.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=git_proj, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=git_proj, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=git_proj, check=True)
    (git_proj / "README.md").write_text("# Bridge\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=git_proj, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=git_proj, check=True, capture_output=True)
    (git_proj / "dirty.txt").write_text("uncommitted", encoding="utf-8")

    # Plain non-git directory
    plain_proj = ws / "clark-home-server"
    plain_proj.mkdir()

    pm = ProjectManager(ws, save_dir)
    projects = pm.scan_projects()
    ids = [p["id"] for p in projects]
    assert ids == ["antigravity-google-home-bridge", "clark-home-server"]

    bridge_p = projects[0]
    assert bridge_p["short_name"] == "Google Home Bridge"
    assert bridge_p["git_branch"] == "main"
    assert bridge_p["uncommitted_changes"] == 1
    assert bridge_p["agents"][0]["agent_id"] == "101"
    assert bridge_p["agents"][0]["project_agent"] == "project Google Home Bridge, agent 101"

    clark_p = projects[1]
    assert clark_p["short_name"] == "Clark Home Server"
    assert clark_p["git_branch"] is None
    assert clark_p["uncommitted_changes"] == 0

    # Empty / README.md-only workspace fallback returns workspace root itself
    readme_ws = tmp_path / "opt-antigravity-workspace"
    readme_ws.mkdir()
    (readme_ws / "README.md").write_text("Workspace root\n", encoding="utf-8")
    pm_fallback = ProjectManager(readme_ws, tmp_path / "save_fb")
    fb_projects = pm_fallback.scan_projects()
    assert len(fb_projects) == 1
    assert Path(fb_projects[0]["path"]) == readme_ws.resolve()

    # Single-repo workspace (.git at root) returns workspace root itself even with subdirs
    single_repo = tmp_path / "single-repo"
    single_repo.mkdir()
    subprocess.run(["git", "init", "-b", "feature-x"], cwd=single_repo, check=True, capture_output=True)
    (single_repo / "src").mkdir()
    (single_repo / "tests").mkdir()
    pm_single = ProjectManager(single_repo, tmp_path / "save_single")
    single_projects = pm_single.scan_projects()
    assert len(single_projects) == 1
    assert Path(single_projects[0]["path"]) == single_repo.resolve()
    assert single_projects[0]["git_branch"] == "feature-x"


def test_project_manager_select_multi_agent_and_persistence(tmp_path: Path) -> None:
    from agy_home_bridge.project_manager import ProjectManager

    ws = tmp_path / "ws_multi"
    (ws / "alpha-service").mkdir(parents=True)
    (ws / "clark-home-server").mkdir(parents=True)
    save_dir = tmp_path / "save_multi"

    pm = ProjectManager(ws, save_dir)
    assert pm.get_state_summary()["active_project_id"] == "alpha-service"

    # Fuzzy / short_name / combined project_agent selection
    sel = pm.select_project(project_agent="project Clark Home Server, agent 102")
    assert sel["project_id"] == "clark-home-server"
    assert sel["agent_id"] == "102"
    assert sel["project_agent"] == "project Clark Home Server, agent 102"

    # Non-mutating get_target for concurrent background task on Alpha
    target_alpha = pm.get_target(project="alpha", agent="105")
    assert target_alpha["project_id"] == "alpha-service"
    assert target_alpha["agent_id"] == "105"
    # Active selection remains Clark Home Server agent 102
    summary = pm.get_state_summary()
    assert summary["active_project_id"] == "clark-home-server"
    assert summary["active_agent_id"] == "102"

    # Persist conversation_id & agent status with swarm_nodes
    pm.set_conversation_id("clark-home-server", "102", "conv-clark-102")
    pm.set_conversation_id("alpha-service", "105", "conv-alpha-105")
    pm.update_agent_status(
        "clark-home-server",
        "102",
        status="running",
        current_task="Inspect docker status",
        last_summary="All containers healthy.",
        swarm_nodes=[{"node_id": "sub-1", "role": "explorer", "status": "done"}],
    )

    state_file = save_dir / "projects_state.json"
    assert state_file.exists()

    # Reconstruct fresh ProjectManager and verify persistence
    pm2 = ProjectManager(ws, save_dir)
    s2 = pm2.get_state_summary()
    assert s2["active_project_id"] == "clark-home-server"
    assert s2["active_agent_id"] == "102"
    reloaded_clark = pm2.get_target(project="Clark Home Server", agent="102")
    assert reloaded_clark["conversation_id"] == "conv-clark-102"
    reloaded_alpha = pm2.get_target(project="alpha", agent="105")
    assert reloaded_alpha["conversation_id"] == "conv-alpha-105"

    with pytest.raises(ValueError, match="Unknown project"):
        pm2.select_project(project="does-not-exist")

    # Selecting only an agent keeps the active project
    sel_agent_only = pm2.select_project(agent="agent 104")
    assert sel_agent_only["project_id"] == "clark-home-server"
    assert sel_agent_only["agent_id"] == "104"

    # Corrupt projects_state.json recovers safely on startup
    state_file.write_text("{corrupt-json", encoding="utf-8")
    pm_corrupt = ProjectManager(ws, save_dir)
    assert len(pm_corrupt.scan_projects()) == 2

    # Short-name collision disambiguation (e.g. "app" vs "antigravity-app" or "My-App" vs "my_app")
    ws_col = tmp_path / "ws_col"
    (ws_col / "my-app").mkdir(parents=True)
    (ws_col / "my_app").mkdir(parents=True)
    pm_col = ProjectManager(ws_col, tmp_path / "save_col")
    col_projs = pm_col.scan_projects()
    assert len({p["id"] for p in col_projs}) == 2
    assert len({p["short_name"].lower() for p in col_projs}) == 2


@pytest.mark.asyncio
async def test_cast_manager_speaker_vs_tv_separation_and_tizen_seeding(
    test_settings: BridgeSettings,
) -> None:
    from agy_home_bridge.antigravity_agent import AntigravityAgentEngine
    from agy_home_bridge.smart_home import SmartHomeFulfillment

    cast_mgr = GoogleHomeCastManager(test_settings)

    def _make_cc(name: str, cast_type: str, model: str, host: str) -> MagicMock:
        cc = MagicMock()
        cc.cast_info.friendly_name = name
        cc.cast_info.uuid = f"uuid-{name}"
        cc.cast_info.model_name = model
        cc.cast_info.cast_type = cast_type
        cc.cast_info.host = host
        cc.cast_info.port = 8009
        return cc

    devices = [
        _make_cc("Office", "audio", "Nest Audio", "192.168.1.101"),
        _make_cc("All Speakers", "group", "Google Cast Group", "192.168.1.102"),
        _make_cc('98" QLED', "cast", "Google TV", "192.168.1.150"),
        _make_cc("Game Room TV", "cast", "Chromecast Ultra", "192.168.1.151"),
    ]

    with patch("pychromecast.get_chromecasts", return_value=(devices, MagicMock())):
        speakers = await cast_mgr.discover_speakers(timeout=0.1)

    assert len(speakers) == 2
    assert {s["name"] for s in speakers} == {"Office", "All Speakers"}
    assert len(cast_mgr.list_speakers(include_tvs=True)) > 2

    tv_names = {t["name"]: t for t in cast_mgr.list_tvs()}
    assert '98" QLED' in tv_names and tv_names['98" QLED']["protocol"] == "chromecast"
    assert "Game Room TV" in tv_names
    assert (
        "Office - Samsung Q60CD 65" in tv_names
        and tv_names["Office - Samsung Q60CD 65"]["protocol"] == "samsung_tizen"
    )
    assert "Living Room TV" in tv_names
    assert "Kitchen TV" in tv_names
    assert "The Frame" in tv_names

    # SmartHomeFulfillment must expose only speakers, never TVs
    engine = AntigravityAgentEngine(test_settings, cast_mgr)
    fulfillment = SmartHomeFulfillment(engine, cast_mgr)
    sync_ids = {d["id"] for d in fulfillment.build_sync_devices()}
    assert "agy-follow-office" in sync_ids
    assert "agy-follow-all-speakers" in sync_ids
    assert "agy-follow-game-room-tv" not in sync_ids
    assert 'agy-follow-98"-qled' not in sync_ids


def test_cast_manager_room_presence_normalization_and_mapping(
    test_settings: BridgeSettings,
) -> None:
    cast_mgr = GoogleHomeCastManager(test_settings)
    cast_mgr.register_mock_speaker("Office")
    cast_mgr.register_mock_speaker("Living Room speaker")
    cast_mgr.register_mock_speaker("All Speakers", cast_type="group")
    cast_mgr.register_mock_tv('98" QLED', chromecast=MagicMock())
    cast_mgr.register_mock_tv("Game Room TV", chromecast=MagicMock())

    with patch.object(cast_mgr, "cast_hud_to_tv") as mock_cast_hud:
        res_office = cast_mgr.set_presence_room("office", source="frigate_ha")
        assert res_office["presence_room"] == "office"
        assert res_office["active_speaker"] == "Office"
        assert res_office["active_tv"] == "Office - Samsung Q60CD 65"

        res_lr = cast_mgr.set_presence_room("living room")
        assert res_lr["presence_room"] == "living_room"
        assert res_lr["active_speaker"] == "Living Room speaker"
        # Prefers live/known Cast TV '98" QLED' when Living Room TV is non-Cast Tizen
        assert res_lr["active_tv"] == '98" QLED'

        res_lr_hyphen = cast_mgr.resolve_room_presence("living-room")
        assert res_lr_hyphen["room"] == "living_room"

        res_kitchen = cast_mgr.set_presence_room("kitchen")
        assert res_kitchen["active_speaker"] == "Living Room speaker"
        assert res_kitchen["active_tv"] == "Kitchen TV"

        # Kitchen falls back to All Speakers when Living Room speaker is absent
        cast_mgr._speakers.pop("Living Room speaker")
        res_kitchen_fb = cast_mgr.resolve_room_presence("kitchen")
        assert res_kitchen_fb["speaker"] == "All Speakers"

        for alias in ('98" QLED', "98_qled", "qled", "theater"):
            res_qled = cast_mgr.resolve_room_presence(alias)
            assert res_qled["room"] == "98_qled"
            assert res_qled["tv"] == '98" QLED'

        res_game = cast_mgr.resolve_room_presence("game_room")
        assert res_game["speaker"] == "All Speakers"
        assert res_game["tv"] == "Game Room TV"

        res_master = cast_mgr.resolve_room_presence("master_bedroom")
        assert res_master["tv"] == "The Frame"

        res_whole = cast_mgr.resolve_room_presence("all")
        assert res_whole["room"] == "whole_home"
        assert res_whole["speaker"] == "All Speakers"

        # Explicit overrides
        res_ovr = cast_mgr.set_presence_room(
            "office", speaker_override="All Speakers", tv_override='98" QLED'
        )
        assert res_ovr["active_speaker"] == "All Speakers"
        assert res_ovr["active_tv"] == '98" QLED'

        mock_cast_hud.assert_not_called()


@pytest.mark.asyncio
async def test_cast_manager_cast_hud_to_tv_dashcast_and_tizen_fallback(
    test_settings: BridgeSettings,
) -> None:
    cast_mgr = GoogleHomeCastManager(test_settings)
    fake_tv_cc = MagicMock()
    fake_tv_cc.cast_info.cast_type = "cast"
    cast_mgr.register_mock_tv('98" QLED', chromecast=fake_tv_cc)

    fake_dashcast = MagicMock()
    with patch(
        "pychromecast.controllers.dashcast.DashCastController",
        return_value=fake_dashcast,
    ):
        res_ok = await cast_mgr.cast_hud_to_tv(room_name="98_qled")
        assert res_ok["status"] == "casting"
        assert res_ok["protocol"] == "chromecast"
        assert res_ok["tv"] == '98" QLED'
        assert res_ok["hud_url"] == "http://192.168.1.50:8765/hud"
        fake_tv_cc.wait.assert_called_once()
        fake_tv_cc.register_handler.assert_called_once_with(fake_dashcast)
        fake_dashcast.load_url.assert_called_once_with(
            "http://192.168.1.50:8765/hud", force=True
        )

        # Error during DashCast launch returns structured error
        fake_dashcast.load_url.side_effect = RuntimeError("DashCast socket error")
        res_err = await cast_mgr.cast_hud_to_tv(tv_name='98" QLED')
        assert res_err["status"] == "error"

    # Samsung Tizen TVs return non-fatal unsupported_protocol with hud_url
    for tizen_tv in (
        "Office - Samsung Q60CD 65",
        "Living Room TV",
        "Kitchen TV",
        "The Frame",
    ):
        res_tizen = await cast_mgr.cast_hud_to_tv(tv_name=tizen_tv)
        assert res_tizen["status"] == "unsupported_protocol"
        assert res_tizen["protocol"] == "samsung_tizen"
        assert res_tizen["tv"] == tizen_tv
        assert res_tizen["hud_url"] == "http://192.168.1.50:8765/hud"

    # Offline Cast TV returns tv_offline (both explicit tv_name and room when no fallback TV is online)
    res_offline_explicit = await cast_mgr.cast_hud_to_tv(tv_name="Game Room TV")
    assert res_offline_explicit["status"] == "tv_offline"
    assert res_offline_explicit["tv"] == "Game Room TV"

    fresh_mgr = GoogleHomeCastManager(test_settings)
    res_offline_room = await fresh_mgr.cast_hud_to_tv(room_name="game_room")
    assert res_offline_room["status"] == "tv_offline"
    assert res_offline_room["tv"] == "Game Room TV"

    # Audio speaker explicitly targeted as TV returns unsupported_protocol
    audio_cc = MagicMock()
    audio_cc.cast_info.friendly_name = "Office"
    audio_cc.cast_info.uuid = "u-office-tv-test"
    audio_cc.cast_info.model_name = "Nest Audio"
    audio_cc.cast_info.cast_type = "audio"
    audio_cc.cast_info.host = "192.168.1.101"
    audio_cc.cast_info.port = 8009
    with patch("pychromecast.get_chromecasts", return_value=([audio_cc], MagicMock())):
        await fresh_mgr.discover_speakers(timeout=0.1)
    res_audio_as_tv = await fresh_mgr.cast_hud_to_tv(tv_name="Office")
    assert res_audio_as_tv["status"] == "unsupported_protocol"
    assert res_audio_as_tv["protocol"] == "chromecast"

    # Unmapped room with no TV configured returns no_tv_configured
    res_unmapped = await cast_mgr.cast_hud_to_tv(room_name="garage")
    assert res_unmapped["status"] == "no_tv_configured"
    assert res_unmapped["tv"] is None


@pytest.mark.asyncio
async def test_cast_manager_discovery_merge_resilience_and_warm_room(
    test_settings: BridgeSettings,
) -> None:
    cast_mgr = GoogleHomeCastManager(test_settings)

    cc_office = MagicMock()
    cc_office.cast_info.friendly_name = "Office"
    cc_office.cast_info.uuid = "u-office"
    cc_office.cast_info.model_name = "Nest Audio"
    cc_office.cast_info.cast_type = "audio"
    cc_office.cast_info.host = "192.168.1.101"
    cc_office.cast_info.port = 8009

    cc_lr = MagicMock()
    cc_lr.cast_info.friendly_name = "Living Room speaker"
    cc_lr.cast_info.uuid = "u-lr"
    cc_lr.cast_info.model_name = "Nest Hub"
    cc_lr.cast_info.cast_type = "audio"
    cc_lr.cast_info.host = "192.168.1.102"
    cc_lr.cast_info.port = 8009
    cc_lr.app_id = None

    with patch(
        "pychromecast.get_chromecasts",
        side_effect=[([cc_office], MagicMock()), ([cc_lr], MagicMock())],
    ):
        first = await cast_mgr.discover_speakers(timeout=0.1, merge=False)
        assert [s["name"] for s in first] == ["Office"]

        merged = await cast_mgr.discover_speakers(timeout=0.1, merge=True)
        assert {s["name"] for s in merged} == {"Office", "Living Room speaker"}

    # warm_room resolves room presence ("living_room" -> "Living Room speaker")
    await cast_mgr.warm_room("living_room")
    cc_lr.media_controller.play_media.assert_called_once()

    # Guard: warm_room never casts silence audio to a cast_type == "cast" TV
    tv_cc = MagicMock()
    tv_cc.cast_info.cast_type = "cast"
    cast_mgr.register_mock_tv('98" QLED', chromecast=tv_cc)
    await cast_mgr.warm_room('98" QLED')
    tv_cc.media_controller.play_media.assert_not_called()

    # Discovery resilience scheduling & cancellation
    task = cast_mgr.schedule_discovery_resilience(retry_delays=(0.01,))
    assert task is not None
    cast_mgr.cancel_discovery_resilience()
    await asyncio.sleep(0.02)

    # register_mock_speaker with cast_type="cast" routes to _tvs, not _speakers
    speaker_count_before = len(cast_mgr.list_speakers())
    cast_mgr.register_mock_speaker("Den TV", cast_type="cast")
    assert len(cast_mgr.list_speakers()) == speaker_count_before
    assert any(tv["name"] == "Den TV" for tv in cast_mgr.list_tvs())


def test_project_manager_ambiguous_queries_and_hyphenated_agent_dirs(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.project_manager import ProjectManager

    ws = tmp_path / "ws_ambig"
    (ws / "service-alpha").mkdir(parents=True)
    (ws / "service-beta").mkdir(parents=True)
    (ws / "alpha-core").mkdir(parents=True)
    (ws / "agent-sdk").mkdir(parents=True)
    (ws / "foo-agent-bar").mkdir(parents=True)

    pm = ProjectManager(ws, tmp_path / "save_ambig")

    # Ambiguous Tier 3 prefix match ("service" matches service-alpha & service-beta)
    with pytest.raises(ValueError, match="Ambiguous project query"):
        pm.select_project(project="service")

    # Unique Tier 3 prefix match ("alpha" prefixes alpha-core while only mid-string in service-alpha)
    unique_prefix = pm.select_project(project="alpha")
    assert unique_prefix["project_id"] == "alpha-core"

    # Ambiguous Tier 3 mid-string substring match (no prefix match)
    (ws / "x-widget-one").mkdir(parents=True)
    (ws / "y-widget-two").mkdir(parents=True)
    pm.scan_projects()
    with pytest.raises(ValueError, match="Ambiguous project query"):
        pm.select_project(project="widget")

    # Ambiguous Tier 4 token subset match (out-of-order tokens matching two projects)
    (ws / "my-alpha-box-one").mkdir(parents=True)
    (ws / "my-alpha-box-two").mkdir(parents=True)
    pm.scan_projects()
    with pytest.raises(ValueError, match="Ambiguous project query"):
        pm.select_project(project="box alpha")

    # Hyphenated agent-* directory names must not misparse agent_id="-sdk" or "-bar"
    pm.select_project(project="service-alpha", agent="102")
    sdk_target = pm.get_target(project="agent-sdk")
    assert sdk_target["project_id"] == "agent-sdk"
    assert sdk_target["agent_id"] == "101"

    sdk_sel = pm.select_project(project="agent-sdk")
    assert sdk_sel["project_id"] == "agent-sdk"
    assert sdk_sel["agent_id"] == "101"

    parsed_phrase = ProjectManager._parse_target_query(project="project foo agent-bar")
    assert parsed_phrase == ("foo agent-bar", None)


def test_project_manager_nested_save_dir_exclusion_and_swarm_persistence(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.project_manager import ProjectManager

    ws = tmp_path / "ws_nested"
    nested_save = ws / "agy_data" / "conversations"
    nested_save.mkdir(parents=True)
    nested_state_file = ws / "state_root" / "nested" / "projects_state.json"
    nested_state_file.parent.mkdir(parents=True)
    (ws / "real-project").mkdir()

    pm = ProjectManager(ws, nested_save, state_file=nested_state_file)
    ids = [p["id"] for p in pm.scan_projects()]
    assert ids == ["real-project"]

    # Persist swarm_nodes even without last_summary and verify reload survival
    swarm = [
        {"node_id": "n1", "role": "explorer", "status": "running"},
        {"node_id": "n2", "role": "implementer", "status": "done"},
    ]
    pm.update_agent_status(
        "real-project",
        "102",
        status="running",
        current_task="Analyze bridge",
        swarm_nodes=swarm,
    )

    pm_reloaded = ProjectManager(ws, nested_save, state_file=nested_state_file)
    reloaded_projs = {p["id"]: p for p in pm_reloaded.scan_projects()}
    agents = {a["agent_id"]: a for a in reloaded_projs["real-project"]["agents"]}
    assert agents["102"]["swarm_nodes"] == swarm


def test_inspect_git_metadata_timeout_fallback(tmp_path: Path) -> None:
    import subprocess
    from agy_home_bridge.project_manager import inspect_git_metadata

    proj = tmp_path / "git-timeout-proj"
    (proj / ".git").mkdir(parents=True)
    (proj / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    with patch(
        "agy_home_bridge.project_manager.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["git", "rev-parse"], timeout=2.0),
    ) as mock_run:
        branch, uncommitted = inspect_git_metadata(proj, timeout=0.1)
    mock_run.assert_called_once()
    assert branch is None
    assert uncommitted == 0


@pytest.mark.asyncio
async def test_cast_hud_to_tv_concurrent_dashcast_registration(
    test_settings: BridgeSettings,
) -> None:
    import threading
    import time

    cast_mgr = GoogleHomeCastManager(test_settings)
    fake_tv_cc = MagicMock()
    fake_tv_cc.cast_info.cast_type = "cast"

    reg_lock = threading.Lock()
    registered_controllers: list[Any] = []

    def _slow_register(ctrl: Any) -> None:
        time.sleep(0.03)
        with reg_lock:
            registered_controllers.append(ctrl)

    fake_tv_cc.register_handler.side_effect = _slow_register
    cast_mgr.register_mock_tv('98" QLED', chromecast=fake_tv_cc)

    created_controllers: list[MagicMock] = []

    def _make_dashcast() -> MagicMock:
        time.sleep(0.02)
        inst = MagicMock()
        created_controllers.append(inst)
        return inst

    with patch(
        "pychromecast.controllers.dashcast.DashCastController",
        side_effect=_make_dashcast,
    ):
        r1, r2 = await asyncio.gather(
            cast_mgr.cast_hud_to_tv(room_name="98_qled"),
            cast_mgr.cast_hud_to_tv(room_name="98_qled"),
        )

    assert r1["status"] == "casting"
    assert r2["status"] == "casting"
    assert len(created_controllers) == 1
    assert len(registered_controllers) == 1
    assert fake_tv_cc.register_handler.call_count == 1
    assert created_controllers[0].load_url.call_count == 2


def test_project_manager_agent_named_short_name_with_explicit_agent_and_no_slice_collision(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.project_manager import ProjectManager

    # Case 1: Project short_name ending in "Agent <token>" alongside explicit agent=
    ws_agents = tmp_path / "ws_agents"
    (ws_agents / "other-project").mkdir(parents=True)
    (ws_agents / "agent-2").mkdir(parents=True)
    (ws_agents / "alpha-agent-2").mkdir(parents=True)
    (ws_agents / "alpha-core").mkdir(parents=True)

    pm_agents = ProjectManager(ws_agents, tmp_path / "save_agents")
    pm_agents.select_project(project="other-project")

    sel_agent2 = pm_agents.select_project(project="Agent 2", agent="103")
    assert sel_agent2["project_id"] == "agent-2"
    assert sel_agent2["agent_id"] == "103"

    tgt_agent2 = pm_agents.get_target(project="Agent 2", agent="agent 104")
    assert tgt_agent2["project_id"] == "agent-2"
    assert tgt_agent2["agent_id"] == "104"

    sel_alpha_agent2 = pm_agents.select_project(project="Alpha Agent 2", agent="103")
    assert sel_alpha_agent2["project_id"] == "alpha-agent-2"
    assert sel_alpha_agent2["agent_id"] == "103"

    # Fuzzy query with explicit agent= must not strip or discard "Agent"
    pm_agents.select_project(project="other-project")
    sel_fuzzy_alpha = pm_agents.select_project(project="Alpha Agent", agent="105")
    assert sel_fuzzy_alpha["project_id"] == "alpha-agent-2"
    assert sel_fuzzy_alpha["agent_id"] == "105"

    sel_prefixed = pm_agents.select_project(project="project Alpha Agent", agent="106")
    assert sel_prefixed["project_id"] == "alpha-agent-2"
    assert sel_prefixed["agent_id"] == "106"

    # Case 2: Short project name ("ui") must not falsely collide with "build" via val in q
    ws_slice = tmp_path / "ws_slice"
    (ws_slice / "ui").mkdir(parents=True)
    (ws_slice / "my-build-tool").mkdir(parents=True)
    pm_slice = ProjectManager(ws_slice, tmp_path / "save_slice")

    sel_build = pm_slice.select_project(project="build")
    assert sel_build["project_id"] == "my-build-tool"

    # Reverse containment works only on whole-token boundaries, never mid-word slices
    sel_desc = pm_slice.select_project(project="my build tool repo")
    assert sel_desc["project_id"] == "my-build-tool"

    # Tier 2 normalized exact-match collision raises ValueError while Tier 1 exact still wins
    ws_tier2 = tmp_path / "ws_tier2"
    (ws_tier2 / "foo--bar").mkdir(parents=True)
    (ws_tier2 / "foo_bar").mkdir(parents=True)
    pm_tier2 = ProjectManager(ws_tier2, tmp_path / "save_tier2")
    assert pm_tier2.select_project(project="foo--bar")["project_id"] == "foo-bar"
    with pytest.raises(ValueError, match="Ambiguous project query"):
        pm_tier2.select_project(project="foo   bar")


def test_project_manager_match_first_target_resolution_matrix(tmp_path: Path) -> None:
    from agy_home_bridge.project_manager import ProjectManager

    # R-01..R-07: Workspace with other-project, agent-2, alpha-agent-2, alpha-core
    ws_r1 = tmp_path / "ws_r1"
    for name in ("other-project", "agent-2", "alpha-agent-2", "alpha-core"):
        (ws_r1 / name).mkdir(parents=True)
    pm_r1 = ProjectManager(ws_r1, tmp_path / "save_r1")

    # R-01
    pm_r1.select_project(project="other-project")
    r01 = pm_r1.select_project(project="project Alpha Agent 2")
    assert (r01["project_id"], r01["agent_id"]) == ("alpha-agent-2", "101")

    # R-02
    pm_r1.select_project(project="other-project")
    r02 = pm_r1.select_project(project="project Alpha Agent 2", agent="   ")
    assert (r02["project_id"], r02["agent_id"]) == ("alpha-agent-2", "101")

    # R-03
    pm_r1.select_project(project="other-project")
    r03 = pm_r1.select_project(project_agent="Alpha Agent 2")
    assert (r03["project_id"], r03["agent_id"]) == ("alpha-agent-2", "101")

    # R-04
    pm_r1.select_project(project="other-project")
    r04 = pm_r1.select_project(project_agent="project Alpha Agent 2")
    assert (r04["project_id"], r04["agent_id"]) == ("alpha-agent-2", "101")

    # R-05
    pm_r1.select_project(project="other-project")
    r05 = pm_r1.select_project(project="project Agent 2")
    assert (r05["project_id"], r05["agent_id"]) == ("agent-2", "101")

    # R-06: whitespace agent= preserves project's active_agent_id
    pm_r1.select_project(project="agent-2", agent="104")
    pm_r1.select_project(project="other-project")
    r06 = pm_r1.select_project(project="project Agent 2", agent=" ")
    assert (r06["project_id"], r06["agent_id"]) == ("agent-2", "104")

    # R-07: combo phrasing with Agent 2 / Alpha Agent 2 (and non-mutating get_target)
    r07_a = pm_r1.select_project(project="Agent 2, agent 105")
    assert (r07_a["project_id"], r07_a["agent_id"]) == ("agent-2", "105")
    r07_b = pm_r1.select_project(project="project Alpha Agent 2, agent 105")
    assert (r07_b["project_id"], r07_b["agent_id"]) == ("alpha-agent-2", "105")
    r07_c = pm_r1.get_target(project_agent="Alpha Agent 2 agent 106")
    assert (r07_c["project_id"], r07_c["agent_id"]) == ("alpha-agent-2", "106")
    assert pm_r1.get_state_summary()["active_project_id"] == "alpha-agent-2"
    assert pm_r1.get_state_summary()["active_agent_id"] == "105"

    # R-08: alpha-agent-2 only (no alpha-core, no agent-2) -> no agent_id="2" hijack
    ws_r8 = tmp_path / "ws_r8"
    (ws_r8 / "other-project").mkdir(parents=True)
    (ws_r8 / "alpha-agent-2").mkdir(parents=True)
    pm_r8 = ProjectManager(ws_r8, tmp_path / "save_r8")
    pm_r8.select_project(project="other-project")
    r08 = pm_r8.select_project(project="project Alpha Agent 2")
    assert (r08["project_id"], r08["agent_id"]) == ("alpha-agent-2", "101")

    # R-09 & R-10: Project-prefixed project name vs stripped fallback
    ws_r9 = tmp_path / "ws_r9"
    (ws_r9 / "project-x-service").mkdir(parents=True)
    (ws_r9 / "x-other").mkdir(parents=True)
    pm_r9 = ProjectManager(ws_r9, tmp_path / "save_r9")
    pm_r9.select_project(project="x-other")

    r09_b = pm_r9.get_target(project="Project X")
    assert (r09_b["project_id"], r09_b["agent_id"]) == ("project-x-service", "101")
    assert pm_r9.get_state_summary()["active_project_id"] == "x-other"
    r09_a = pm_r9.select_project(project="Project X", agent="102")
    assert (r09_a["project_id"], r09_a["agent_id"]) == ("project-x-service", "102")
    r09_c = pm_r9.select_project(project_agent="Project X, agent 104")
    assert (r09_c["project_id"], r09_c["agent_id"]) == ("project-x-service", "104")

    r10 = pm_r9.select_project(project="project X Other", agent="102")
    assert (r10["project_id"], r10["agent_id"]) == ("x-other", "102")

    # R-11: Ambiguous unstripped match must NOT fall back to stripped x-other
    ws_r11 = tmp_path / "ws_r11"
    for name in ("project-x-service", "project-x-tools", "x-other"):
        (ws_r11 / name).mkdir(parents=True)
    pm_r11 = ProjectManager(ws_r11, tmp_path / "save_r11")
    pm_r11.select_project(project="x-other")
    with pytest.raises(ValueError, match="Ambiguous project query"):
        pm_r11.select_project(project="Project X")

    # R-12: Agent <N> prefix match (agent-2-runner)
    ws_r12 = tmp_path / "ws_r12"
    (ws_r12 / "other-project").mkdir(parents=True)
    (ws_r12 / "agent-2-runner").mkdir(parents=True)
    pm_r12 = ProjectManager(ws_r12, tmp_path / "save_r12")
    pm_r12.select_project(project="other-project")
    r12_a = pm_r12.select_project(project="Agent 2")
    assert (r12_a["project_id"], r12_a["agent_id"]) == ("agent-2-runner", "101")
    r12_b = pm_r12.get_target(project="project Agent 2")
    assert (r12_b["project_id"], r12_b["agent_id"]) == ("agent-2-runner", "101")

    # R-13: Ambiguous Agent <N> prefix must NOT fall back to other-project, agent 2
    ws_r13 = tmp_path / "ws_r13"
    for name in ("other-project", "agent-2-runner", "agent-20"):
        (ws_r13 / name).mkdir(parents=True)
    pm_r13 = ProjectManager(ws_r13, tmp_path / "save_r13")
    pm_r13.select_project(project="other-project")
    with pytest.raises(ValueError, match="Ambiguous project query"):
        pm_r13.select_project(project="Agent 2")

    # R-14: Pure agent-only phrases not hijacked by mid-string "project agent" or "alpha-agent-2"
    ws_r14 = tmp_path / "ws_r14"
    for name in ("other-project", "my-project-agent-helper", "alpha-agent-2"):
        (ws_r14 / name).mkdir(parents=True)
    pm_r14 = ProjectManager(ws_r14, tmp_path / "save_r14")
    pm_r14.select_project(project="other-project")

    r14_a = pm_r14.select_project(project="project agent 102")
    assert (r14_a["project_id"], r14_a["agent_id"]) == ("other-project", "102")
    r14_b = pm_r14.select_project(project_agent="agent 103")
    assert (r14_b["project_id"], r14_b["agent_id"]) == ("other-project", "103")
    r14_c = pm_r14.select_project(project_agent="agent 2")
    assert (r14_c["project_id"], r14_c["agent_id"]) == ("other-project", "2")
    r14_d = pm_r14.select_project(project_agent="#105")
    assert (r14_d["project_id"], r14_d["agent_id"]) == ("other-project", "105")
    r14_e = pm_r14.select_project(project_agent="project, agent 106")
    assert (r14_e["project_id"], r14_e["agent_id"]) == ("other-project", "106")

    # R-15: Explicit agent= overrides embedded agent suffix
    ws_r15 = tmp_path / "ws_r15"
    (ws_r15 / "other-project").mkdir(parents=True)
    (ws_r15 / "agent-2").mkdir(parents=True)
    pm_r15 = ProjectManager(ws_r15, tmp_path / "save_r15")
    pm_r15.select_project(project="other-project")
    r15 = pm_r15.select_project(project="project Agent 2, agent 104", agent="103")
    assert (r15["project_id"], r15["agent_id"]) == ("agent-2", "103")

    # R-16: Direct _parse_target_query classmethod with projects=None
    assert ProjectManager._parse_target_query(project="project foo agent-bar") == (
        "foo agent-bar",
        None,
    )
    assert ProjectManager._parse_target_query(project="project agent 102") == (
        None,
        "102",
    )
    assert ProjectManager._parse_target_query(project="Agent 2", agent="103") == (
        "Agent 2",
        "103",
    )

    # Space-combo and hyphenated agent-sdk comma-combo with Clark Home Server
    ws_combo = tmp_path / "ws_combo"
    for name in ("clark-home-server", "agent-sdk", "other-project"):
        (ws_combo / name).mkdir(parents=True)
    pm_combo = ProjectManager(ws_combo, tmp_path / "save_combo")
    pm_combo.select_project(project="other-project")

    c1 = pm_combo.select_project(project="Clark Home Server agent 102")
    assert (c1["project_id"], c1["agent_id"]) == ("clark-home-server", "102")
    c2 = pm_combo.select_project(project="project agent-sdk, agent 103")
    assert (c2["project_id"], c2["agent_id"]) == ("agent-sdk", "103")
    c3 = pm_combo.select_project(project="project Clark Home Server agent 104")
    assert (c3["project_id"], c3["agent_id"]) == ("clark-home-server", "104")
    assert ProjectManager._parse_target_query(
        project="Clark Home Server agent 102", projects=pm_combo._projects
    ) == ("clark-home-server", "102")


def test_project_manager_pure_agent_hash_phrases_and_empty_projects_dict(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.project_manager import ProjectManager

    # 1. R-14 workspace: pure-agent phrases with '#' via both project= and project_agent=
    ws_r14 = tmp_path / "ws_r14_hash"
    for name in ("other-project", "my-project-agent-helper", "alpha-agent-2"):
        (ws_r14 / name).mkdir(parents=True)
    pm_r14 = ProjectManager(ws_r14, tmp_path / "save_r14_hash")
    pm_r14.select_project(project="other-project")

    for phrase, expected_agent in (
        ("agent #102", "102"),
        ("agent #105", "105"),
        ("agent # 102", "102"),
        ("project agent #102", "102"),
    ):
        res_proj = pm_r14.select_project(project=phrase)
        assert (res_proj["project_id"], res_proj["agent_id"]) == (
            "other-project",
            expected_agent,
        )
        res_pa = pm_r14.select_project(project_agent=phrase)
        assert (res_pa["project_id"], res_pa["agent_id"]) == (
            "other-project",
            expected_agent,
        )

    tgt_hash = pm_r14.get_target(project_agent="agent #109")
    assert (tgt_hash["project_id"], tgt_hash["agent_id"]) == ("other-project", "109")
    assert pm_r14.get_state_summary()["active_agent_id"] == "102"

    # 2. Workspace with agent-103-app: pure-agent '#' phrases must not hijack active project
    ws_103 = tmp_path / "ws_103_hash"
    for name in ("other-project", "agent-103-app"):
        (ws_103 / name).mkdir(parents=True)
    pm_103 = ProjectManager(ws_103, tmp_path / "save_103_hash")
    pm_103.select_project(project="other-project")

    for phrase, expected_agent in (
        ("agent #102", "102"),
        ("agent #105", "105"),
        ("agent # 102", "102"),
    ):
        res_proj = pm_103.select_project(project=phrase)
        assert (res_proj["project_id"], res_proj["agent_id"]) == (
            "other-project",
            expected_agent,
        )
        res_pa = pm_103.select_project(project_agent=phrase)
        assert (res_pa["project_id"], res_pa["agent_id"]) == (
            "other-project",
            expected_agent,
        )

    # 3. _parse_target_query with explicit empty projects={} must NOT fall back to syntactic mode
    with pytest.raises(ValueError, match="Unknown project"):
        ProjectManager._parse_target_query(project="other", projects={})
    assert ProjectManager._parse_target_query(project="agent #102", projects={}) == (
        None,
        "102",
    )
    assert ProjectManager._parse_target_query(project="other", projects=None) == (
        "other",
        None,
    )


# ============================================================================
# 10. Milestone M2 Tests (Async Engine, Gemini Live Tools, Server Routes & /ws/hud)
# ============================================================================
def _make_m2_settings(tmp_path: Path) -> BridgeSettings:
    ws = tmp_path / "workspace"
    (ws / "alpha-service").mkdir(parents=True, exist_ok=True)
    (ws / "beta-worker").mkdir(parents=True, exist_ok=True)
    return BridgeSettings(
        _env_file=None,
        GEMINI_API_KEY="",
        ANTIGRAVITY_LS_ADDRESS="",
        ANTIGRAVITY_CSRF_TOKEN="",
        AGY_WORKSPACE_DIR=str(ws),
        AGY_SAVE_DIR=str(tmp_path / "save"),
        AGY_APP_DATA_DIR=str(tmp_path / "brain"),
        HOST_LAN_IP="192.168.1.50",
        BRIDGE_PORT=8765,
        DEFAULT_CAST_ROOM="Office",
        DEFAULT_AUDIO_ROUTING="both",
        ENABLE_HTTPS=False,
    )


def test_m2_summarize_for_speech_helper() -> None:
    from agy_home_bridge.antigravity_agent import summarize_for_speech

    assert summarize_for_speech("   \n```python\nprint('hi')\n```\n   ") == ""
    raw = (
        "## Status Summary\n"
        "Checked [service](https://example.com) with `systemctl`.\n"
        "- **All units active**\n"
        "```bash\necho hidden\n```\n"
        + ("Healthy telemetry reported across nodes. " * 15)
    )
    spoken = summarize_for_speech(raw, max_chars=450)
    assert len(spoken) <= 450
    assert "```" not in spoken
    assert "##" not in spoken
    assert "echo hidden" not in spoken
    assert spoken.endswith("...")
    assert "service" in spoken


@pytest.mark.asyncio
async def test_m2_engine_sdk_chunks_hooks_swarm_and_concurrent_isolation(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    cfg.agy_enable_write_tools = True
    cfg.agy_allow_all_commands = False

    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub(maxlen=cfg.event_history_size)
    cast_mgr = GoogleHomeCastManager(cfg)
    cast_mgr.register_mock_speaker("Office", host="192.168.1.101")
    cast_mgr.register_mock_speaker("Living Room speaker", host="192.168.1.102")
    engine = AntigravityAgentEngine(
        cfg,
        cast_mgr,
        project_manager=pm,
        event_hub=hub,
    )

    instructions = engine.system_instructions
    assert "Alpha Service" in instructions
    assert "Beta Worker" in instructions
    assert "project Alpha Service, agent 101" in instructions

    class Thought:
        def __init__(self, text: str) -> None:
            self.text = text

    class ToolCall:
        def __init__(self, name: str, args: dict[str, Any], call_id: str) -> None:
            self.name = name
            self.args = args
            self.id = call_id
            self.step_id = f"step-{call_id}"

    class ToolResult:
        def __init__(self, name: str, call_id: str, result: str) -> None:
            self.name = name
            self.id = call_id
            self.step_id = f"step-{call_id}"
            self.result = result
            self.error = None
            self.exception = None

    class Text:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeStep:
        def __init__(self, traj_id: str, parent_id: str | None, depth: int) -> None:
            self.trajectory_id = traj_id
            self.parent_trajectory_id = parent_id
            self.depth = depth
            self.status = "completed"

    gate_alpha = asyncio.Event()
    gate_beta = asyncio.Event()
    captured_configs: list[dict[str, Any]] = []
    registered_post_hooks: list[Any] = []

    class FakeHooksReg:
        @staticmethod
        def post_tool_call(fn: Any) -> Any:
            registered_post_hooks.append(fn)
            return fn

        @staticmethod
        def on_tool_error(fn: Any) -> Any:
            return fn

    class FakePolicy:
        @staticmethod
        def allow_all() -> dict[str, str]:
            return {"policy": "allow_all"}

        @staticmethod
        def confirm_run_command() -> list[dict[str, str]]:
            return [{"deny": "run_command"}, {"allow": "*"}]

    class FakeResponse:
        def __init__(self, ws_name: str, hooks_list: list[Any]) -> None:
            self._ws_name = ws_name
            self._hooks = hooks_list

        @property
        def chunks(self):
            async def _gen():
                if "alpha" in self._ws_name:
                    yield Thought("Inspecting alpha")
                    await gate_alpha.wait()
                    tc = ToolCall("start_subagent", {"agent_name": "sub-alpha"}, "tc-a1")
                    yield tc
                    for h in self._hooks:
                        if h.__name__ == "_on_post_tool_call":
                            await h(ToolResult("start_subagent", "tc-a1", "subagent done"))
                    yield Text("Alpha task completed cleanly.")
                else:
                    yield Thought("Inspecting beta")
                    await gate_beta.wait()
                    tc = ToolCall("read_file", {"path": "README.md"}, "tc-b1")
                    yield tc
                    for h in self._hooks:
                        if h.__name__ == "_on_post_tool_call":
                            await h(ToolResult("read_file", "tc-b1", "readme ok"))
                    yield Text("Beta task completed cleanly.")

            return _gen()

    class FakeAgent:
        def __init__(self, ag_cfg: dict[str, Any]) -> None:
            self.ag_cfg = ag_cfg
            ws_first = Path(ag_cfg["workspaces"][0]).name
            self.conversation_id = ag_cfg.get("conversation_id") or f"conv-{ws_first}"
            self.conversation = types.SimpleNamespace(
                _steps=[FakeStep("traj-sub-1", "traj-root", 1)]
                if "alpha" in ws_first
                else []
            )

        async def __aenter__(self) -> FakeAgent:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def chat(self, prompt: str) -> FakeResponse:
            ws_first = Path(self.ag_cfg["workspaces"][0]).name
            return FakeResponse(ws_first, self.ag_cfg.get("hooks") or [])

    def _make_cfg(**kw: Any) -> dict[str, Any]:
        captured_configs.append(dict(kw))
        return dict(kw)

    fake_agy = types.ModuleType("google.antigravity")
    fake_agy.Agent = FakeAgent  # type: ignore[attr-defined]
    fake_agy.LocalAgentConfig = _make_cfg  # type: ignore[attr-defined]
    fake_agy.CapabilitiesConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]

    fake_hooks_mod = types.ModuleType("google.antigravity.hooks")
    fake_hooks_mod.policy = FakePolicy  # type: ignore[attr-defined]
    fake_hooks_mod.hooks = FakeHooksReg  # type: ignore[attr-defined]

    q = hub.subscribe()
    spoken_calls: list[tuple[str, str | None]] = []

    async def _fake_speak(text: str, room_name: str | None = None, volume: float | None = None) -> dict[str, Any]:
        spoken_calls.append((text, room_name))
        return {"status": "casting", "room": room_name}

    with (
        patch.dict(
            sys.modules,
            {"google.antigravity": fake_agy, "google.antigravity.hooks": fake_hooks_mod},
        ),
        patch.object(cast_mgr, "speak_on_room", side_effect=_fake_speak),
    ):
        cast_mgr.set_presence_room("office", source="frigate_ha")
        ack_a = engine.start_background_task("Run alpha", project="alpha-service", agent="101")
        ack_b = engine.start_background_task("Run beta", project="beta-worker", agent="102")
        assert ack_a["status"] == "started"
        assert ack_b["status"] == "started"
        assert ack_a["task_id"] != ack_b["task_id"]

        # Switch active room mid-task; completion speech must follow the new room.
        cast_mgr.set_presence_room("living_room", source="frigate_ha")
        gate_beta.set()
        gate_alpha.set()

        seen: list[dict[str, Any]] = []
        completed_ids: set[str] = set()
        while len(completed_ids) < 2:
            ev = await asyncio.wait_for(q.get(), timeout=3.0)
            seen.append(ev)
            if ev["type"] == "task_completed":
                completed_ids.add(ev["task_id"])

        await asyncio.sleep(0.05)

    hub.unsubscribe(q)
    await engine.shutdown()

    # Verify policy flattening & hooks wiring
    assert len(captured_configs) == 2
    assert all(cfg_item["policies"] == [{"deny": "run_command"}, {"allow": "*"}] for cfg_item in captured_configs)
    assert len(registered_post_hooks) >= 2

    # Verify isolated workspace roots & persisted conversation IDs
    summary = pm.get_state_summary()
    by_name = {p["name"]: p for p in summary["projects"]}
    alpha_101 = next(a for a in by_name["alpha-service"]["agents"] if a["agent_id"] == "101")
    beta_102 = next(a for a in by_name["beta-worker"]["agents"] if a["agent_id"] == "102")
    assert alpha_101["conversation_id"] == "conv-alpha-service"
    assert beta_102["conversation_id"] == "conv-beta-worker"
    assert len(alpha_101["swarm_nodes"]) >= 1

    # Verify scoped events & completion room change
    alpha_events = [e for e in seen if e.get("task_id") == ack_a["task_id"]]
    assert {"task_started", "thought", "tool_call", "tool_result", "token", "task_completed"}.issubset(
        {e["type"] for e in alpha_events}
    )
    assert all(e["project_id"] == "alpha-service" and e["agent_id"] == "101" for e in alpha_events)
    assert any(room == "Living Room speaker" for _, room in spoken_calls)

    # Existing swarm_nodes preserved when subsequent turn has swarm_nodes=None
    pm.update_agent_status("beta-worker", "102", status="idle", swarm_nodes=[{"node_id": "keep-me", "status": "idle"}])
    with patch.dict(sys.modules, {"google.antigravity": None}):
        await engine.ask("hello", cast_response=False, project="beta-worker", agent="102", emit_events=False)
    beta_after = next(
        a for a in pm.get_state_summary()["projects"][1]["agents"] if a["agent_id"] == "102"
    )
    assert beta_after["swarm_nodes"] == [{"node_id": "keep-me", "status": "idle"}]


@pytest.mark.asyncio
async def test_m2_background_task_sdk_crash_emits_task_failed_and_shutdown_cancels(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)

    gate = asyncio.Event()

    class CrashAgent:
        def __init__(self, ag_cfg: Any) -> None:
            self.conversation_id = "conv-crash"

        async def __aenter__(self) -> CrashAgent:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def chat(self, prompt: str):
            if "hang" in prompt:
                async def _hang():
                    await gate.wait()
                    yield "never"

                return _hang()
            raise RuntimeError("Boom SDK failure")

    fake_agy = types.ModuleType("google.antigravity")
    fake_agy.Agent = CrashAgent  # type: ignore[attr-defined]
    fake_agy.LocalAgentConfig = lambda **kw: kw  # type: ignore[attr-defined]
    fake_agy.CapabilitiesConfig = lambda **kw: kw  # type: ignore[attr-defined]

    q = hub.subscribe()
    try:
        with patch.dict(sys.modules, {"google.antigravity": fake_agy}):
            ack_crash = engine.start_background_task("crash now", project="alpha-service")
            failed_ev = None
            for _ in range(10):
                ev = await asyncio.wait_for(q.get(), timeout=2.0)
                if ev["type"] == "task_failed" and ev["task_id"] == ack_crash["task_id"]:
                    failed_ev = ev
                    break
            assert failed_ev is not None
            assert "Boom SDK failure" in failed_ev["error"]

            ack_hang = engine.start_background_task("hang forever", project="beta-worker")
            assert ack_hang["task_id"] in engine._background_tasks
            await engine.shutdown()
            assert engine._background_tasks == {}
    finally:
        hub.unsubscribe(q)


@pytest.mark.asyncio
async def test_m2_gemini_live_tools_nonblocking_switch_project_cast_hud_and_voice_status(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    cast_mgr.register_mock_speaker("Office", host="192.168.1.101")
    cast_mgr.register_mock_tv("Office TV", cast_type="cast", host="192.168.1.150")
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)
    bridge = FullDuplexLiveBridge(cfg, engine, cast_mgr, event_hub=hub, project_manager=pm)

    fake_ws = AsyncMock()
    fake_session = AsyncMock()
    gate = asyncio.Event()

    async def _slow_bg_turn(*args: Any, **kwargs: Any):
        await gate.wait()
        from agy_home_bridge.antigravity_agent import AgentTurnResult

        return AgentTurnResult(text="Done after gate", conversation_id="conv-bg"), None

    fc_switch = MagicMock(id="fc-1", args={"project_agent": "project Beta Worker, agent 202"})
    fc_switch.name = "switch_project"
    fc_delegate = MagicMock(id="fc-2", args={"task": "Build worker"})
    fc_delegate.name = "delegate_to_antigravity_agent"
    fc_cast = MagicMock(id="fc-3", args={"room": "office", "tv": "Office TV"})
    fc_cast.name = "cast_hud_to_tv"
    tool_call = MagicMock(function_calls=[fc_switch, fc_delegate, fc_cast])

    q = hub.subscribe()
    try:
        with (
            patch.object(engine, "_execute_agent_turn", side_effect=_slow_bg_turn),
            patch.object(
                cast_mgr,
                "cast_hud_to_tv",
                new_callable=AsyncMock,
                return_value={
                    "status": "casting",
                    "room": "office",
                    "tv": "Office TV",
                    "hud_url": "http://192.168.1.50:8765/hud",
                    "protocol": "chromecast",
                    "message": "Casting HUD",
                },
            ) as mock_cast_hud,
            patch.object(cast_mgr, "speak_on_room", new_callable=AsyncMock),
        ):
            await bridge._handle_live_tool_calls(fake_ws, fake_session, tool_call)
            # delegate_to_antigravity_agent returned BEFORE gate was set!
            assert not gate.is_set()
            assert fake_session.send_tool_response.await_count == 1
            responses = fake_session.send_tool_response.call_args.kwargs["function_responses"]
            by_name = {r.name: r.response for r in responses}
            assert by_name["switch_project"]["status"] == "switched"
            assert by_name["switch_project"]["agent_id"] == "202"
            assert by_name["delegate_to_antigravity_agent"]["status"] == "started"
            assert by_name["cast_hud_to_tv"]["status"] == "casting"
            mock_cast_hud.assert_awaited_once_with(room_name="office", tv_name="Office TV")

            gate.set()
            seen_types: set[str] = set()
            for _ in range(10):
                ev = await asyncio.wait_for(q.get(), timeout=2.0)
                seen_types.add(ev["type"])
                if {"voice_status", "project_changed", "task_completed"}.issubset(seen_types):
                    break
            assert {"voice_status", "project_changed", "task_completed"}.issubset(seen_types)
    finally:
        hub.unsubscribe(q)
        await engine.shutdown()


def test_m2_server_routes_presence_projects_hud_ws_and_cast_on_demand(
    tmp_path: Path,
) -> None:
    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience") as mock_sched,
        patch.object(app.state.cast_manager, "cancel_discovery_resilience") as mock_cancel,
        patch.object(app.state.cast_manager, "warm_room", new_callable=AsyncMock) as mock_warm,
        patch.object(app.state.cast_manager, "cast_hud_to_tv", new_callable=AsyncMock) as mock_cast_hud,
        TestClient(app) as client,
    ):
        mock_sched.assert_called_once()
        client.app.state.cast_manager.register_mock_speaker("Office", host="192.168.1.101")
        client.app.state.cast_manager.register_mock_speaker("Living Room speaker", host="192.168.1.102")
        client.app.state.cast_manager.register_mock_speaker("All Speakers", cast_type="group", host="192.168.1.103")
        client.app.state.cast_manager.register_mock_tv("Game Room TV", cast_type="cast", host="192.168.1.111")

        # GET /hud
        hud_resp = client.get("/hud")
        assert hud_resp.status_code == 200
        assert "text/html" in hud_resp.headers["content-type"]

        # GET /api/projects & POST /api/projects/select
        projs_resp = client.get("/api/projects")
        assert projs_resp.status_code == 200
        proj_names = {p["name"] for p in projs_resp.json()["projects"]}
        assert proj_names == {"alpha-service", "beta-worker"}

        bad_sel = client.post("/api/projects/select", json={"project": "does-not-exist"})
        assert bad_sel.status_code == 404

        sel_resp = client.post(
            "/api/projects/select",
            json={"project_agent": "project Beta Worker, agent 105"},
        )
        assert sel_resp.status_code == 200
        assert sel_resp.json()["project_id"] == "beta-worker"
        assert sel_resp.json()["agent_id"] == "105"

        # GET /api/health & GET /api/speakers include M2 fields
        health = client.get("/api/health").json()
        assert health["active_project"] == "beta-worker"
        assert health["active_project_agent"] == "project Beta Worker, agent 105"
        assert "active_presence_room" in health
        assert "active_tv" in health
        assert health["voice_status"] == "idle"

        speakers_data = client.get("/api/speakers").json()
        assert "tvs" in speakers_data
        assert any(t["name"] == "Game Room TV" for t in speakers_data["tvs"])

        # /ws/hud + /ws/live concurrent EventHub streaming + voice_status filtering on /ws/live
        with client.websocket_connect("/ws/hud") as hud_ws, client.websocket_connect("/ws/live") as live_ws:
            snap = hud_ws.receive_json()
            assert snap["type"] == "hud_snapshot"
            assert snap["active_project_id"] == "beta-worker"

            init_live = live_ws.receive_json()
            assert init_live["type"] == "status" and init_live["state"] == "connected"

            # POST /api/presence/room emits room_changed, warms room, never auto-casts TV
            pres_resp = client.post(
                "/api/presence/room",
                json={"room": "game_room", "source": "frigate_ha"},
            )
            assert pres_resp.status_code == 200
            assert pres_resp.json()["active_room"] == "All Speakers"
            assert pres_resp.json()["active_tv"] in ("Game Room TV", '98" QLED')
            mock_warm.assert_awaited()
            mock_cast_hud.assert_not_awaited()

            hud_ev = hud_ws.receive_json()
            while hud_ev.get("type") != "room_changed":
                hud_ev = hud_ws.receive_json()
            assert hud_ev["source"] == "frigate_ha"

            live_ev = live_ws.receive_json()
            while live_ev.get("type") != "room_changed":
                assert live_ev.get("type") != "voice_status"
                live_ev = live_ws.receive_json()
            assert live_ev["active_room"] == "All Speakers"

            # voice_status goes to /ws/hud on live text activity
            live_ws.send_json({"type": "text", "text": "Hello JARVIS"})
            got_voice_status = False
            for _ in range(10):
                msg = hud_ws.receive_json()
                if msg.get("type") == "voice_status":
                    got_voice_status = True
                    break
            assert got_voice_status

        # Empty room returns 400/422
        assert client.post("/api/presence/room", json={"room": "   "}).status_code == 400

        # On-demand POST /api/cast/hud invokes cast_hud_to_tv
        mock_cast_hud.return_value = {
            "status": "casting",
            "room": "game_room",
            "tv": "Game Room TV",
            "hud_url": "http://192.168.1.50:8765/hud",
            "protocol": "chromecast",
            "message": "ok",
        }
        cast_resp = client.post("/api/cast/hud", json={"room": "game_room", "tv": "Game Room TV"})
        assert cast_resp.status_code == 200
        assert cast_resp.json()["status"] == "casting"
        mock_cast_hud.assert_awaited_once_with(room_name="game_room", tv_name="Game Room TV")

    mock_cancel.assert_called()


def test_m2_ws_live_and_hud_abrupt_disconnect_cleanup_and_malformed_json(
    tmp_path: Path,
) -> None:
    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        TestClient(app) as client,
    ):
        # Challenger Finding 1 exact reproducer: 5 abrupt /ws/live disconnects without .close()
        for _ in range(5):
            with client.websocket_connect("/ws/live") as lws:
                init_msg = lws.receive_json()
                assert init_msg["type"] == "status"
                assert init_msg["state"] == "connected"

        assert app.state.event_hub.subscriber_count == 0
        assert app.state.live_bridge.voice_status == "idle"
        assert client.get("/api/health").json()["voice_status"] == "idle"

        # 5 abrupt /ws/hud disconnects without .close()
        for _ in range(5):
            with client.websocket_connect("/ws/hud") as hws:
                snap = hws.receive_json()
                assert snap["type"] == "hud_snapshot"
                assert snap["voice_status"] == "idle"

        assert app.state.event_hub.subscriber_count == 0

        # Malformed JSON & non-object JSON on /ws/live send structured error without crashing socket
        with client.websocket_connect("/ws/live") as lws:
            lws.receive_json()
            lws.send_text("{bad-json")
            err1 = lws.receive_json()
            assert err1["type"] == "error"
            assert "Invalid JSON" in err1["error"]

            lws.send_text("[1, 2, 3]")
            err2 = lws.receive_json()
            assert err2["type"] == "error"
            assert "JSON object" in err2["error"]

            # Socket remains alive for valid messages
            lws.send_json({"type": "set_routing", "mode": "direct"})
            status_msg = lws.receive_json()
            assert status_msg["type"] == "status"
            assert status_msg["routing_mode"] == "direct"

        assert app.state.event_hub.subscriber_count == 0
        assert app.state.live_bridge.voice_status == "idle"

        # _process_turn exception resets voice_status to "listening" (then "idle" on disconnect)
        with patch.object(
            app.state.agent_engine,
            "ask",
            new_callable=AsyncMock,
            side_effect=RuntimeError("simulated turn error"),
        ):
            with client.websocket_connect("/ws/live") as lws:
                lws.receive_json()
                lws.send_json({"type": "text", "text": "trigger error"})
                seen_types = []
                for _ in range(5):
                    msg = lws.receive_json()
                    seen_types.append(msg.get("type"))
                    if msg.get("type") == "status" and msg.get("state") == "listening":
                        break
                assert "error" in seen_types
                assert app.state.live_bridge.voice_status == "listening"

        assert app.state.event_hub.subscriber_count == 0
        assert app.state.live_bridge.voice_status == "idle"


@pytest.mark.asyncio
async def test_m2_tool_error_dedup_repeated_unnamed_tools_and_unnamed_subagent(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)

    class ToolCall:
        def __init__(self, name: str, args: dict[str, Any], call_id: str | None) -> None:
            self.name = name
            self.args = args
            self.id = call_id
            self.step_id = None

    class ToolResult:
        def __init__(
            self,
            name: str,
            call_id: str | None,
            result: str | None = None,
            error: str | None = None,
        ) -> None:
            self.name = name
            self.id = call_id
            self.step_id = None
            self.result = result
            self.error = error
            self.exception = None

    class Text:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeHooksReg:
        @staticmethod
        def post_tool_call(fn: Any) -> Any:
            return fn

        @staticmethod
        def on_tool_error(fn: Any) -> Any:
            return fn

    class FakeResponse:
        def __init__(self, hooks_list: list[Any]) -> None:
            self._hooks = hooks_list

        @property
        def chunks(self):
            async def _gen():
                # 1) Unnamed start_subagent (id=None) -> post_tool_call (id=None) -> transitions cleanly
                yield ToolCall("start_subagent", {"agent_name": "worker-x"}, None)
                for h in self._hooks:
                    if h.__name__ == "_on_post_tool_call":
                        await h(ToolResult("start_subagent", None, result="sub ok"))

                # 2) Duplicate error across @on_tool_error + ToolResult chunk for call_id="c2" -> only 1 tool_result
                yield ToolCall("run_command", {"cmd": "false"}, "c2")
                err_item = ToolResult("run_command", "c2", error="exit code 1")
                for h in self._hooks:
                    if h.__name__ == "_on_tool_error":
                        await h(err_item)
                yield err_item

                # 3) Two separate calls to the same tool name with id=None & step_id=None -> BOTH emitted
                yield ToolCall("read_file", {"path": "a.py"}, None)
                yield ToolResult("read_file", None, result="contents-a")
                yield ToolCall("read_file", {"path": "b.py"}, None)
                yield ToolResult("read_file", None, result="contents-b")

                yield Text("Done checking tools.")

            return _gen()

    class FakeAgent:
        def __init__(self, ag_cfg: dict[str, Any]) -> None:
            self.ag_cfg = ag_cfg
            self.conversation_id = "conv-dedup"
            self.conversation = types.SimpleNamespace(_steps=[])

        async def __aenter__(self) -> FakeAgent:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def chat(self, prompt: str) -> FakeResponse:
            return FakeResponse(self.ag_cfg.get("hooks") or [])

    fake_agy = types.ModuleType("google.antigravity")
    fake_agy.Agent = FakeAgent  # type: ignore[attr-defined]
    fake_agy.LocalAgentConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]
    fake_agy.CapabilitiesConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]

    fake_hooks_mod = types.ModuleType("google.antigravity.hooks")
    fake_hooks_mod.hooks = FakeHooksReg  # type: ignore[attr-defined]

    q = hub.subscribe()
    try:
        with patch.dict(
            sys.modules,
            {"google.antigravity": fake_agy, "google.antigravity.hooks": fake_hooks_mod},
        ):
            ack = engine.start_background_task("test tool dedup", project="alpha-service")
            events: list[dict[str, Any]] = []
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=3.0)
                if ev.get("task_id") == ack["task_id"]:
                    events.append(ev)
                    if ev["type"] == "task_completed":
                        break
            await asyncio.sleep(0.05)
    finally:
        hub.unsubscribe(q)
        await engine.shutdown()

    tool_results = [e for e in events if e["type"] == "tool_result"]
    c2_results = [e for e in tool_results if e.get("tool_call_id") == "c2"]
    assert len(c2_results) == 1
    assert c2_results[0]["error"] == "exit code 1"

    read_results = [e for e in tool_results if e.get("name") == "read_file"]
    assert [e["result"] for e in read_results] == ["contents-a", "contents-b"]

    alpha = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
    agent_101 = next(a for a in alpha["agents"] if a["agent_id"] == "101")
    assert len(agent_101["swarm_nodes"]) == 1
    assert agent_101["swarm_nodes"][0]["node_id"] == "subagent-1"
    assert agent_101["swarm_nodes"][0]["status"] == "completed"
    assert agent_101["swarm_nodes"][0]["label"] == "worker-x"


def test_m2_get_projects_runs_nonblocking_in_thread(tmp_path: Path) -> None:
    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        patch("agy_home_bridge.server.asyncio.to_thread", wraps=asyncio.to_thread) as mock_to_thread,
        TestClient(app) as client,
    ):
        resp = client.get("/api/projects")
        assert resp.status_code == 200
        assert {p["name"] for p in resp.json()["projects"]} == {"alpha-service", "beta-worker"}
        assert any(
            call.args and call.args[0] == app.state.project_manager.get_state_summary
            for call in mock_to_thread.call_args_list
        )


# ---------------------------------------------------------------------------
# 11. Milestone M3 Tests (JARVIS PWA Node View & /hud TV Dashboard)
# ---------------------------------------------------------------------------


def _strip_js_comments_m3(source: str) -> str:
    out: list[str] = []
    i = 0
    n = len(source)
    while i < n:
        ch = source[i]
        if ch in ("'", '"', "`"):
            quote = ch
            out.append(ch)
            i += 1
            while i < n:
                sc = source[i]
                out.append(sc)
                if sc == "\\" and i + 1 < n:
                    out.append(source[i + 1])
                    i += 2
                    continue
                if sc == quote:
                    i += 1
                    break
                i += 1
        elif ch == "/" and i + 1 < n and source[i + 1] == "/":
            i += 2
            while i < n and source[i] != "\n":
                i += 1
        elif ch == "/" and i + 1 < n and source[i + 1] == "*":
            i += 2
            while i + 1 < n and not (source[i] == "*" and source[i + 1] == "/"):
                if source[i] == "\n":
                    out.append("\n")
                i += 1
            i += 2
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def _collect_html_tags_m3(html_text: str) -> list[tuple[str, dict[str, str | None], str]]:
    from html.parser import HTMLParser

    tags: list[tuple[str, dict[str, str | None], str]] = []
    text_stack: list[list[str]] = []

    class _Parser(HTMLParser):
        def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
            attr_map = {k.lower(): v for k, v in attrs}
            buf: list[str] = []
            tags.append((tag.lower(), attr_map, ""))
            text_stack.append(buf)
            # Store index to update text on close
            buf.append(str(len(tags) - 1))

        def handle_data(self, data: str) -> None:
            for buf in text_stack:
                buf.append(data)

        def handle_endtag(self, tag: str) -> None:
            if not text_stack:
                return
            buf = text_stack.pop()
            idx = int(buf[0])
            tname, amap, _ = tags[idx]
            tags[idx] = (tname, amap, " ".join("".join(buf[1:]).split()))

    _Parser().feed(html_text)
    return tags


def test_m3_pwa_node_view_cast_hud_open_https_fix_and_static_contracts(
    tmp_path: Path,
) -> None:
    import re
    import shutil
    import subprocess

    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        TestClient(app) as client,
    ):
        index_resp = client.get("/")
        app_js_resp = client.get("/static/app.js")
        styles_resp = client.get("/static/styles.css")
        manifest_resp = client.get("/static/manifest.json")

        assert index_resp.status_code == 200
        assert app_js_resp.status_code == 200
        assert styles_resp.status_code == 200
        assert manifest_resp.status_code == 200

        html = index_resp.text
        assert "Antigravity Live" in html
        assert 'closedby="any"' in html
        assert 'name="color-scheme" content="light dark"' in html
        assert "aria-live" in html

        parsed = _collect_html_tags_m3(html)
        tag_names = {t for t, _, _ in parsed}
        assert {"header", "main", "section", "dialog"}.issubset(tag_names)

        assert any(
            t in {"section", "div", "ul", "ol", "nav", "aside", "article"}
            and "hidden" not in attrs
            and re.search(
                r"(project|node|agent|swarm|jarvis|workspace)",
                f"{attrs.get('id', '')} {attrs.get('class', '')} {attrs.get('aria-label', '')}",
                re.IGNORECASE,
            )
            for t, attrs, _ in parsed
        )
        assert any(
            t == "button"
            and "hidden" not in attrs
            and "disabled" not in attrs
            and "cast hud to room tv" in text.lower()
            for t, attrs, text in parsed
        )
        assert any(t == "a" and attrs.get("href") == "/hud" for t, attrs, _ in parsed)

        clean_js = _strip_js_comments_m3(app_js_resp.text)
        # openHttpsLink .home bug fix
        assert "openHttpsLink.href = httpsTargetUrl" in clean_js
        # Speaker vs TV filtering
        assert "isTvDevice" in clean_js
        # Required endpoints, WS, DOM ops & event strings
        assert "/api/health" in clean_js
        assert "/api/speakers" in clean_js
        assert "/api/projects" in clean_js
        assert "/api/projects/select" in clean_js
        assert "/api/cast/hud" in clean_js
        assert "/ws/live" in clean_js
        assert "JSON.parse" in clean_js
        assert "createElement" in clean_js
        assert "textContent" in clean_js
        for marker in (
            "short_name",
            "project_agent",
            "git_branch",
            "uncommitted_changes",
            "conversation_id",
            "agents",
            "swarm_nodes",
            "active_room",
            "active_tv",
            "voice_status",
            "status",
            "project_changed",
            "room_changed",
            "thought",
            "tool_call",
            "tool_result",
            "token",
            "task_started",
            "task_completed",
            "task_failed",
        ):
            assert marker in clean_js

        css_text = styles_resp.text
        assert "light-dark(" in css_text
        assert "prefers-color-scheme: dark" in css_text
        assert "container: jarvis-nodes / inline-size" in css_text
        assert "content-visibility: auto" in css_text
        assert "contain-intrinsic-size" in css_text
        assert "pointer: coarse" in css_text

    if shutil.which("node"):
        app_js_path = Path(__file__).resolve().parents[1] / "src/agy_home_bridge/static/app.js"
        res = subprocess.run(["node", "--check", str(app_js_path)], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr


def test_m3_hud_route_readonly_tv_ui_and_static_contracts(tmp_path: Path) -> None:
    import re
    import shutil
    import subprocess

    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        TestClient(app) as client,
    ):
        hud_resp = client.get("/hud")
        hud_js_resp = client.get("/static/hud.js")
        hud_css_resp = client.get("/static/hud.css")

        assert hud_resp.status_code == 200
        assert "text/html" in hud_resp.headers.get("content-type", "")
        assert hud_js_resp.status_code == 200
        assert "javascript" in hud_js_resp.headers.get("content-type", "")
        assert hud_css_resp.status_code == 200
        assert "text/css" in hud_css_resp.headers.get("content-type", "")

        hud_html = hud_resp.text
        assert "color-scheme" in hud_html
        assert "/static/hud.js" in hud_html
        assert "/static/hud.css" in hud_html
        assert "<script>" not in hud_html.lower()
        assert "contenteditable" not in hud_html.lower()

        parsed = _collect_html_tags_m3(hud_html)
        tag_names = {t for t, _, _ in parsed}
        assert not tag_names.intersection({"input", "textarea", "select", "form", "button"})
        assert any(
            t in {"section", "div", "ul", "ol", "nav", "aside", "article"}
            and "hidden" not in attrs
            and re.search(
                r"(project|node|agent|swarm|jarvis|workspace)",
                f"{attrs.get('id', '')} {attrs.get('class', '')}",
                re.IGNORECASE,
            )
            for t, attrs, _ in parsed
        )

        clean_hud_js = _strip_js_comments_m3(hud_js_resp.text)
        assert re.search(r"\bnew\s+WebSocket\s*\(", clean_hud_js)
        assert "/ws/hud" in clean_hud_js
        assert "/api/projects" in clean_hud_js
        assert "/api/health" in clean_hud_js
        assert "/api/speakers" in clean_hud_js
        assert "JSON.parse" in clean_hud_js
        assert "createElement" in clean_hud_js
        assert "textContent" in clean_hud_js
        assert "innerHTML" not in clean_hud_js

        for forbidden in (
            "/api/projects/select",
            "/api/cast/hud",
            "/api/agent/ask",
            "getUserMedia",
            "AudioWorklet",
        ):
            assert forbidden not in clean_hud_js

        for expected in (
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
            "active_room",
            "active_tv",
            "project_agent",
            "short_name",
            "agents",
            "swarm_nodes",
        ):
            assert expected in clean_hud_js

        hud_css = hud_css_resp.text
        assert "100dvh" in hud_css
        assert "100vw" not in hud_css
        assert "clamp(" in hud_css
        assert "color-scheme: dark light" in hud_css
        assert "content-visibility: auto" in hud_css
        assert "contain-intrinsic-size" in hud_css
        assert "overflow-wrap: anywhere" in hud_css
        assert "forced-colors: active" in hud_css

    if shutil.which("node"):
        hud_js_path = Path(__file__).resolve().parents[1] / "src/agy_home_bridge/static/hud.js"
        res = subprocess.run(["node", "--check", str(hud_js_path)], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr


def test_m3_api_speakers_tv_filtering_select_and_cast_hud_integration(
    tmp_path: Path,
) -> None:
    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        patch.object(app.state.cast_manager, "warm_room", new_callable=AsyncMock),
        TestClient(app) as client,
    ):
        client.app.state.cast_manager.register_mock_speaker("Office", host="192.168.1.101")
        client.app.state.cast_manager.register_mock_speaker("Living Room speaker", host="192.168.1.102")
        client.app.state.cast_manager.register_mock_tv("Game Room TV", cast_type="cast", host="192.168.1.111")

        # Speakers and TVs remain cleanly separated for PWA & HUD hydration
        sp_resp = client.get("/api/speakers").json()
        speaker_names = {s["name"] for s in sp_resp["speakers"]}
        tv_names = {t["name"] for t in sp_resp["tvs"]}
        assert {"Office", "Living Room speaker"}.issubset(speaker_names)
        assert "Game Room TV" not in speaker_names
        assert "Game Room TV" in tv_names
        assert "Office - Samsung Q60CD 65" in tv_names

        # Selecting project/agent updates state and emits project_changed to /ws/hud
        with client.websocket_connect("/ws/hud") as hud_ws:
            snap = hud_ws.receive_json()
            assert snap["type"] == "hud_snapshot"

            sel = client.post(
                "/api/projects/select",
                json={"project": "alpha-service", "agent": "303"},
            )
            assert sel.status_code == 200
            assert sel.json()["project_agent"] == "project Alpha Service, agent 303"

            ev = hud_ws.receive_json()
            while ev.get("type") != "project_changed":
                ev = hud_ws.receive_json()
            assert ev["agent_id"] == "303"
            assert ev["project_agent"] == "project Alpha Service, agent 303"

        # On-demand Samsung Tizen TV returns structured non-fatal fallback with hud_url
        tizen_cast = client.post("/api/cast/hud", json={"tv": "Office - Samsung Q60CD 65"})
        assert tizen_cast.status_code == 200
        tizen_body = tizen_cast.json()
        assert tizen_body["status"] == "unsupported_protocol"
        assert tizen_body["protocol"] == "samsung_tizen"
        assert tizen_body["hud_url"].endswith("/hud")


def test_m3_remediation_security_state_races_and_responsive_layout_contracts(
    tmp_path: Path,
) -> None:
    import json
    import re
    import shutil
    import subprocess

    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        TestClient(app) as client,
    ):
        app_js = _strip_js_comments_m3(client.get("/static/app.js").text)
        hud_js = _strip_js_comments_m3(client.get("/static/hud.js").text)
        styles_css = client.get("/static/styles.css").text
        hud_css = client.get("/static/hud.css").text

        # 1. renderCastHudFeedback validates http:/https: via new URL(..., window.location.origin)
        assert "new URL(" in app_js
        assert "window.location.origin" in app_js
        assert "'http:'" in app_js and "'https:'" in app_js

        # 2. hud.js guards replay mutations and HTTP hydration after WS snapshot
        assert "wsSnapshotReceived" in hud_js
        assert "if (replay) break;" in hud_js
        assert "if (!replay)" in hud_js

        # 3. app.js monotonic sequencing & authoritative project normalization
        assert "selectRequestSeq" in app_js
        assert "fetchProjectsSeq" in app_js
        assert "normalizeActiveProjects" in app_js

        # 4. try/catch around JSON.parse in both WS handlers
        assert re.search(r"try\s*\{[^}]*JSON\.parse\(event\.data\)", app_js)
        assert re.search(r"try\s*\{[^}]*JSON\.parse\(event\.data\)", hud_js)

        # 5. styles.css: container moved to .jarvis-nodes-panel, minmax(0, 1fr), touch target clamp, wrap rules
        panel_block = re.search(r"\.jarvis-nodes-panel\s*\{([^}]*)\}", styles_css)
        grid_block = re.search(r"\.projects-node-grid\s*\{([^}]*)\}", styles_css)
        close_btn_block = re.search(r"\.icon-close-btn\s*\{([^}]*)\}", styles_css)
        assert panel_block is not None
        assert grid_block is not None
        assert close_btn_block is not None
        assert "container: jarvis-nodes / inline-size" in panel_block.group(1)
        assert "container: jarvis-nodes / inline-size" not in grid_block.group(1)
        assert "minmax(0, 1fr)" in grid_block.group(1)
        assert "max(32px, var(--target-size))" in close_btn_block.group(1)
        for cls in (
            ".project-short-name",
            ".agent-verbal-name",
            ".swarm-chip",
            "#activeTargetSummary",
            "#activityScopeBadge",
        ):
            assert cls in styles_css

        # 6. hud.css: responsive minmax(min(280px, 100%), 1fr) and min-inline-size: 0
        hud_tree_block = re.search(r"\.hud-node-tree\s*\{([^}]*)\}", hud_css)
        assert hud_tree_block is not None
        assert "minmax(min(280px, 100%), 1fr)" in hud_tree_block.group(1)
        assert ".hud-project-short" in hud_css
        assert ".hud-agent-name" in hud_css

    # Execute actual JS handlers in Node mock DOM to verify runtime behavior
    if shutil.which("node"):
        repo_root = Path(__file__).resolve().parents[1]
        app_js_path = repo_root / "src/agy_home_bridge/static/app.js"
        hud_js_path = repo_root / "src/agy_home_bridge/static/hud.js"
        probe_script = f"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

function makeEl(tag = 'div') {{
  const el = {{
    tagName: tag.toUpperCase(),
    children: [],
    className: '',
    dataset: {{}},
    attributes: {{}},
    textContent: '',
    hidden: false,
    disabled: false,
    value: '',
    style: {{}},
    classList: {{ add() {{}}, remove() {{}} }},
    addEventListener() {{}},
    setAttribute(k, v) {{ this.attributes[k] = String(v); }},
    getAttribute(k) {{ return this.attributes[k] ?? null; }},
    removeAttribute(k) {{ delete this.attributes[k]; }},
    appendChild(c) {{ c.isConnected = true; this.children.push(c); return c; }},
    replaceChildren(...cs) {{ this.children = cs.map(c => {{ c.isConnected = true; return c; }}); }},
    querySelector(sel) {{
      if (sel.startsWith('.')) {{
        const cls = sel.slice(1);
        for (const c of this.children) {{
          if ((c.className || '').split(/\\s+/).includes(cls)) return c;
          const found = c.querySelector && c.querySelector(sel);
          if (found) return found;
        }}
      }}
      return null;
    }},
    querySelectorAll() {{ return []; }},
    getContext() {{ return {{ clearRect() {{}}, beginPath() {{}}, arc() {{}}, stroke() {{}} }}; }},
    get firstElementChild() {{ return this.children[0] || null; }},
    get lastElementChild() {{ return this.children[this.children.length - 1] || null; }},
    remove() {{}},
    isConnected: true,
  }};
  return el;
}}

// --- Probe app.js ---
const appEls = new Map();
const appFetchQueue = [];
let mockServerProj = 'p1';
class MockWS {{
  constructor() {{ this.readyState = 1; MockWS.last = this; }}
  send() {{}}
}}
MockWS.OPEN = 1;

const appCtx = {{
  URL,
  Map, Set, Array, Number, Boolean, String, JSON, Math, Date, Promise, Error,
  setTimeout: () => 0,
  setInterval: () => 0,
  requestAnimationFrame: () => 0,
  HTMLInputElement: {{ prototype: {{ switch: true }} }},
  HTMLDialogElement: {{ prototype: {{ closedBy: 'any' }} }},
  WebSocket: MockWS,
  navigator: {{}},
  window: {{
    isSecureContext: true,
    location: {{ hostname: 'bridge.home', protocol: 'https:', host: 'bridge.home', origin: 'https://bridge.home' }},
  }},
  document: {{
    getElementById(id) {{
      if (!appEls.has(id)) appEls.set(id, makeEl('div'));
      return appEls.get(id);
    }},
    createElement(tag) {{ return makeEl(tag); }},
    querySelectorAll() {{ return []; }},
  }},
  fetch: (url, opts) => {{
    if (url === '/api/projects/select') {{
      return new Promise((resolve) => appFetchQueue.push({{ url, body: JSON.parse(opts.body), resolve }}));
    }}
    if (url === '/api/health') {{
      return Promise.resolve({{
        ok: true,
        json: async () => ({{ active_project_id: 'p2', active_agent_id: '101', active_project_agent: 'project Two, agent 101' }}),
      }});
    }}
    if (url === '/api/projects') {{
      const activeId = mockServerProj;
      const activePa = activeId === 'p2' ? 'project Two, agent 101' : 'project One, agent 101';
      return Promise.resolve({{
        ok: true,
        json: async () => ({{
          active_project_id: activeId,
          active_agent_id: '101',
          active_project_agent: activePa,
          projects: [
            {{ id: 'p1', name: 'one', short_name: 'One', active: activeId === 'p1', active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }}] }},
            {{ id: 'p2', name: 'two', short_name: 'Two', active: activeId === 'p2', active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project Two, agent 101', status: 'idle' }}] }},
          ],
        }}),
      }});
    }}
    return Promise.resolve({{ ok: true, json: async () => ({{ speakers: [], tvs: [] }}) }});
  }},
}};
vm.createContext(appCtx);
vm.runInContext(fs.readFileSync({json.dumps(str(app_js_path))}, 'utf8'), appCtx);

(async () => {{
  // 1. javascript: URL sink blocked
  appCtx.renderCastHudFeedback({{ status: 'unsupported_protocol', tv: 'Living Room TV', hud_url: 'javascript:alert(1)', message: 'bad' }});
  const fb = appEls.get('castHudFeedback');
  assert.strictEqual(fb.children.some(c => c.tagName === 'A'), false, 'javascript: URL must not create <a>');
  appCtx.renderCastHudFeedback({{ status: 'unsupported_protocol', tv: 'Living Room TV', hud_url: 'http://192.168.1.249:8000/hud', message: 'ok' }});
  assert.strictEqual(fb.children.some(c => c.tagName === 'A' && c.href === 'http://192.168.1.249:8000/hud'), true);

  // 2. Stale p1.active normalized when fetchHealth sets p2
  await appCtx.fetchProjects();
  await appCtx.fetchHealth();
  const grid = appEls.get('projectsNodeGrid');
  const activeCards = grid.children.filter(c => c.dataset.active === 'true');
  assert.strictEqual(activeCards.length, 1, 'Expected exactly 1 active project card');
  assert.strictEqual(activeCards[0].dataset.projectId, 'p2');

  // 3. No div/p/ul inside <button>
  const firstCard = grid.children[0];
  const buttons = firstCard.children.filter(c => c.tagName === 'BUTTON');
  assert.ok(buttons.length >= 1);
  for (const btn of buttons) {{
    for (const child of btn.children) {{
      assert.notStrictEqual(child.tagName, 'DIV');
      assert.notStrictEqual(child.tagName, 'P');
      assert.notStrictEqual(child.tagName, 'UL');
    }}
  }}

  // 4. Out-of-order selectProjectTarget converges to latest click
  const p1Promise = appCtx.selectProjectTarget({{ project: 'p1', agent: '101' }});
  const p2Promise = appCtx.selectProjectTarget({{ project: 'p2', agent: '101' }});
  assert.strictEqual(appFetchQueue.length, 2);
  mockServerProj = 'p2';
  appFetchQueue[1].resolve({{ ok: true, json: async () => ({{ project_id: 'p2', agent_id: '101', project_agent: 'project Two, agent 101' }}) }});
  await p2Promise;
  appFetchQueue[0].resolve({{ ok: true, json: async () => ({{ project_id: 'p1', agent_id: '101', project_agent: 'project One, agent 101' }}) }});
  await p1Promise;
  assert.strictEqual(appEls.get('activeTargetSummary').textContent, 'Voice Target: project Two, agent 101');

  // 5. Invalid JSON WS frame does not throw
  MockWS.last.onmessage({{ data: 'not-json{{' }});

  // --- Probe hud.js ---
  const hudEls = new Map();
  const hudCtx = {{
    Map, Array, Number, Boolean, String, JSON, Math, Date, Promise,
    setTimeout: () => 0,
    setInterval: () => 0,
    WebSocket: MockWS,
    window: {{ location: {{ protocol: 'https:', host: 'bridge.home' }} }},
    document: {{
      getElementById(id) {{
        if (!hudEls.has(id)) hudEls.set(id, makeEl('div'));
        return hudEls.get(id);
      }},
      createElement(tag) {{ return makeEl(tag); }},
    }},
    fetch: () => Promise.resolve({{ ok: false }}),
  }};
  vm.createContext(hudCtx);
  vm.runInContext(fs.readFileSync({json.dumps(str(hud_js_path))}, 'utf8'), hudCtx);

  hudCtx.handleHudEvent({{
    type: 'hud_snapshot',
    active_room: 'Office',
    active_presence_room: 'office',
    active_speaker: 'Office',
    active_tv: 'Office TV',
    voice_status: 'idle',
    active_project_id: 'p2',
    active_agent_id: '101',
    active_project_agent: 'project Two, agent 101',
    projects: [
      {{ id: 'p1', name: 'one', short_name: 'One', active: false, active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }}] }},
      {{ id: 'p2', name: 'two', short_name: 'Two', active: true, active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project Two, agent 101', status: 'idle' }}] }},
    ],
    recent_events: [
      {{ type: 'room_changed', active_room: 'Living Room speaker', active_presence_room: 'living_room', active_tv: 'Living Room TV' }},
      {{ type: 'project_changed', project_id: 'p1', agent_id: '101', project_agent: 'project One, agent 101' }},
      {{ type: 'voice_status', voice_status: 'speaking', active_room: 'Kitchen' }},
      {{ type: 'task_started', project_id: 'p2', agent_id: '101', prompt: 'old task' }},
    ],
  }}, false);

  assert.strictEqual(hudEls.get('hudActiveRoom').textContent, 'office');
  assert.strictEqual(hudEls.get('hudActiveSpeaker').textContent, 'Office');
  assert.strictEqual(hudEls.get('hudActiveTv').textContent, 'Office TV');
  assert.strictEqual(hudEls.get('hudActiveProjectAgent').textContent, 'project Two, agent 101');
  assert.strictEqual(hudEls.get('hudVoiceLabel').textContent, 'Voice: IDLE');
  MockWS.last.onmessage({{ data: 'invalid-json' }});
}})().catch((e) => {{ console.error(e); process.exit(1); }});
"""
        res = subprocess.run(["node", "-e", probe_script], capture_output=True, text=True)
        assert res.returncode == 0, f"Node runtime probe failed:\n{res.stderr}\n{res.stdout}"


# ---------------------------------------------------------------------------
# 12. Milestone M4 Phase 1 Regression Tests
# ---------------------------------------------------------------------------


def test_m4_project_manager_git_detection_noise_dir_vs_head_vs_worktree_file(
    tmp_path: Path,
) -> None:
    import subprocess
    from agy_home_bridge.project_manager import ProjectManager, inspect_git_metadata

    # 1. Workspace root with .git/marker.txt (noise dir, no HEAD) must NOT trigger single-repo fallback
    ws_noise = tmp_path / "ws_noise"
    ws_noise.mkdir()
    (ws_noise / ".git").mkdir()
    (ws_noise / ".git" / "marker.txt").write_text("ignore", encoding="utf-8")

    real_proj = ws_noise / "alpha-app"
    real_proj.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=str(real_proj), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(real_proj), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=str(real_proj), check=True, capture_output=True)
    (real_proj / "README.md").write_text("hi\n", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=str(real_proj), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=str(real_proj), check=True, capture_output=True)

    noise_subproj = ws_noise / "plain-notes"
    noise_subproj.mkdir()
    (noise_subproj / ".git").mkdir()
    (noise_subproj / ".git" / "marker.txt").write_text("not a git repo", encoding="utf-8")

    # inspect_git_metadata on noise subdir must return (None, 0) without running git
    with patch("agy_home_bridge.project_manager.subprocess.run") as mock_run:
        assert inspect_git_metadata(noise_subproj) == (None, 0)
        mock_run.assert_not_called()

    save_dir = tmp_path / "save_noise"
    pm_noise = ProjectManager(ws_noise, save_dir)
    projs = {p["id"]: p for p in pm_noise.scan_projects()}
    assert set(projs.keys()) == {"alpha-app", "plain-notes"}
    assert projs["alpha-app"]["git_branch"] == "main"
    assert projs["plain-notes"]["git_branch"] is None
    assert projs["plain-notes"]["uncommitted_changes"] == 0

    # 2. Workspace root with real .git/HEAD triggers single-repo mode
    pm_single = ProjectManager(real_proj, tmp_path / "save_single")
    single_projs = pm_single.scan_projects()
    assert [p["id"] for p in single_projs] == ["alpha-app"]
    assert single_projs[0]["git_branch"] == "main"

    # 3. Git worktree (.git file pointing to gitdir) is recognized in both inspect_git_metadata and scan_projects
    wt_ws = tmp_path / "wt_ws"
    wt_proj = wt_ws / "wt-branch-proj"
    wt_ws.mkdir()
    subprocess.run(
        ["git", "worktree", "add", "-b", "wt-feat", str(wt_proj)],
        cwd=str(real_proj),
        check=True,
        capture_output=True,
    )
    assert (wt_proj / ".git").is_file()
    (wt_proj / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    assert inspect_git_metadata(wt_proj) == ("wt-feat", 1)

    pm_wt = ProjectManager(wt_proj, tmp_path / "save_wt")
    wt_projs = pm_wt.scan_projects()
    assert [p["id"] for p in wt_projs] == ["wt-branch-proj"]
    assert wt_projs[0]["git_branch"] == "wt-feat"
    assert wt_projs[0]["uncommitted_changes"] == 1


def test_m4_hud_find_active_project_prefers_active_project_id() -> None:
    import shutil
    import subprocess

    if not shutil.which("node"):
        pytest.skip("node not installed")

    repo_root = Path(__file__).resolve().parents[1]
    hud_js_path = repo_root / "src/agy_home_bridge/static/hud.js"
    probe_script = f"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

function makeEl(tag = 'div') {{
  return {{
    tagName: tag.toUpperCase(),
    children: [],
    className: '',
    dataset: {{}},
    attributes: {{}},
    textContent: '',
    hidden: false,
    style: {{}},
    classList: {{ add() {{}}, remove() {{}} }},
    addEventListener() {{}},
    setAttribute(k, v) {{ this.attributes[k] = String(v); }},
    getAttribute(k) {{ return this.attributes[k] ?? null; }},
    appendChild(c) {{ this.children.push(c); return c; }},
    replaceChildren(...cs) {{ this.children = cs; }},
    get firstElementChild() {{ return this.children[0] || null; }},
    remove() {{}},
    isConnected: true,
  }};
}}

const hudEls = new Map();
class MockWS {{ constructor() {{ this.readyState = 1; }} }}
const hudCtx = {{
  Map, Array, Number, Boolean, String, JSON, Math, Date, Promise,
  setTimeout: () => 0,
  setInterval: () => 0,
  WebSocket: MockWS,
  window: {{ location: {{ protocol: 'https:', host: 'bridge.home' }} }},
  document: {{
    getElementById(id) {{
      if (!hudEls.has(id)) hudEls.set(id, makeEl('div'));
      return hudEls.get(id);
    }},
    createElement(tag) {{ return makeEl(tag); }},
  }},
  fetch: () => Promise.resolve({{ ok: false }}),
}};
vm.createContext(hudCtx);
vm.runInContext(fs.readFileSync({json.dumps(str(hud_js_path))}, 'utf8'), hudCtx);

// Stale earlier p1.active === true while active_project_id points to later p2
hudCtx._assert = assert;
vm.runInContext(`
  hudState.active_project_id = 'p2';
  hudState.active_agent_id = '202';
  hudState.active_project_agent = '';
  hudState.projects = [
    {{ id: 'p1', name: 'one', short_name: 'One', active: true, active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }}] }},
    {{ id: 'p2', name: 'two', short_name: 'Two', active: false, active_agent_id: '202', agents: [{{ agent_id: '202', project_agent: 'project Two, agent 202', status: 'running' }}] }},
  ];
  const found = findActiveProjectAndAgent();
  _assert.strictEqual(found.proj.id, 'p2');
  _assert.strictEqual(found.agent.agent_id, '202');

  // Falls back to p.active when active_project_id is null
  hudState.active_project_id = null;
  hudState.active_agent_id = null;
  const fallbackFound = findActiveProjectAndAgent();
  _assert.strictEqual(fallbackFound.proj.id, 'p1');
  _assert.strictEqual(fallbackFound.agent.agent_id, '101');
`, hudCtx);
"""
    res = subprocess.run(["node", "-e", probe_script], capture_output=True, text=True)
    assert res.returncode == 0, f"HUD probe failed:\n{res.stderr}\n{res.stdout}"


@pytest.mark.asyncio
async def test_m4_duplicate_unnamed_subagent_tool_result_does_not_pop_second_subagent(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)

    class ToolCall:
        def __init__(self, name: str, args: dict[str, Any], call_id: str | None, step_id: str | None) -> None:
            self.name = name
            self.args = args
            self.id = call_id
            self.step_id = step_id

    class ToolResult:
        def __init__(
            self,
            name: str,
            call_id: str | None,
            step_id: str | None,
            result: str | None = None,
        ) -> None:
            self.name = name
            self.id = call_id
            self.step_id = step_id
            self.result = result
            self.error = None
            self.exception = None

    class Text:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeHooksReg:
        @staticmethod
        def post_tool_call(fn: Any) -> Any:
            return fn

        @staticmethod
        def on_tool_error(fn: Any) -> Any:
            return fn

    class FakeResponse:
        def __init__(self, hooks_list: list[Any]) -> None:
            self._hooks = hooks_list

        @property
        def chunks(self):
            async def _gen():
                # First unnamed start_subagent (id=None, step_id="s1") completes via @post_tool_call
                yield ToolCall("start_subagent", {"agent_name": "first-worker"}, None, "s1")
                for h in self._hooks:
                    if h.__name__ == "_on_post_tool_call":
                        await h(ToolResult("start_subagent", None, "s1", result="first ok"))

                # Second unnamed start_subagent starts (id=None, step_id="s2")
                yield ToolCall("start_subagent", {"agent_name": "second-worker"}, None, "s2")

                # Duplicate ToolResult for s1 arrives via stream chunk -> MUST NOT pop second-worker
                yield ToolResult("start_subagent", None, "s1", result="first ok dup")
                yield Text("First done, second still running.")

            return _gen()

    class FakeAgent:
        def __init__(self, ag_cfg: dict[str, Any]) -> None:
            self.ag_cfg = ag_cfg
            self.conversation_id = "conv-sub-dedup"
            self.conversation = types.SimpleNamespace(_steps=[])

        async def __aenter__(self) -> FakeAgent:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def chat(self, prompt: str) -> FakeResponse:
            return FakeResponse(self.ag_cfg.get("hooks") or [])

    fake_agy = types.ModuleType("google.antigravity")
    fake_agy.Agent = FakeAgent  # type: ignore[attr-defined]
    fake_agy.LocalAgentConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]
    fake_agy.CapabilitiesConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]

    fake_hooks_mod = types.ModuleType("google.antigravity.hooks")
    fake_hooks_mod.hooks = FakeHooksReg  # type: ignore[attr-defined]

    q = hub.subscribe()
    try:
        with patch.dict(
            sys.modules,
            {"google.antigravity": fake_agy, "google.antigravity.hooks": fake_hooks_mod},
        ):
            ack = engine.start_background_task("test unnamed subagent dedup", project="alpha-service")
            events: list[dict[str, Any]] = []
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=3.0)
                if ev.get("task_id") == ack["task_id"]:
                    events.append(ev)
                    if ev["type"] == "task_completed":
                        break
            await asyncio.sleep(0.05)
    finally:
        hub.unsubscribe(q)
        await engine.shutdown()

    sub_results = [
        e for e in events if e["type"] == "tool_result" and e.get("name") == "start_subagent"
    ]
    assert [e["step_id"] for e in sub_results] == ["s1"]

    alpha = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
    agent_101 = next(a for a in alpha["agents"] if a["agent_id"] == "101")
    assert len(agent_101["swarm_nodes"]) == 2
    by_label = {n["label"]: n for n in agent_101["swarm_nodes"]}
    assert by_label["first-worker"]["status"] == "completed"
    assert by_label["second-worker"]["status"] == "running"


@pytest.mark.asyncio
async def test_m4_ws_live_malformed_field_types_return_structured_errors(
    tmp_path: Path,
) -> None:
    from fastapi import WebSocketDisconnect
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        TestClient(app) as client,
    ):
        # 1. Fallback text/VAD duplex loop over /ws/live
        with client.websocket_connect("/ws/live") as lws:
            lws.receive_json()  # connected status

            lws.send_json({"type": "text", "text": 12345})
            err_text = lws.receive_json()
            assert err_text["type"] == "error"

            lws.send_json({"type": "set_room", "room": ["Office"]})
            err_room = lws.receive_json()
            assert err_room["type"] == "error"

            lws.send_json({"type": "set_routing", "mode": 99})
            err_mode_type = lws.receive_json()
            assert err_mode_type["type"] == "error"

            lws.send_json({"type": "set_routing", "mode": "bogus_mode"})
            err_mode_val = lws.receive_json()
            assert err_mode_val["type"] == "error"

            # Loop remains healthy after malformed payloads
            lws.send_json({"type": "set_routing", "mode": "cast"})
            ok_msg = lws.receive_json()
            while ok_msg.get("type") != "status" or "routing_mode" not in ok_msg:
                ok_msg = lws.receive_json()
            assert ok_msg["routing_mode"] == "cast"

    # 2. Native Gemini Live _forward_client_to_gemini loop (including pcm_b64)
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)
    bridge = FullDuplexLiveBridge(cfg, engine, cast_mgr, event_hub=hub, project_manager=pm)

    incoming = [
        json.dumps({"type": "audio_chunk"}),  # missing pcm_b64
        json.dumps({"type": "audio_chunk", "pcm_b64": 123}),  # non-str pcm_b64
        json.dumps({"type": "audio_chunk", "pcm_b64": "!!!not-base64!!!"}),  # invalid b64
        json.dumps({"type": "text", "text": None}),
        json.dumps({"type": "set_room", "room": {"bad": 1}}),
        json.dumps({"type": "set_routing", "mode": "invalid_mode"}),
        json.dumps({"type": "set_routing", "mode": "direct"}),
    ]
    sent_frames: list[dict[str, Any]] = []

    class FakeWS:
        async def receive_text(self) -> str:
            if not incoming:
                raise WebSocketDisconnect()
            return incoming.pop(0)

        async def send_text(self, raw: str) -> None:
            sent_frames.append(json.loads(raw))

    fake_session = AsyncMock()
    with pytest.raises(WebSocketDisconnect):
        await bridge._forward_client_to_gemini(FakeWS(), fake_session)  # type: ignore[arg-type]

    assert len(sent_frames) == 7
    for frame in sent_frames[:6]:
        assert frame["type"] == "error"
        assert isinstance(frame["error"], str) and frame["error"]
    assert sent_frames[6] == {"type": "status", "state": "listening", "routing_mode": "direct"}


def test_m4_r2_git_file_validation_rejects_empty_junk_broken_gitdir(tmp_path: Path) -> None:
    from agy_home_bridge.project_manager import (
        ProjectManager,
        _is_git_repo,
        inspect_git_metadata,
    )

    # 1. Empty .git file in workspace root must NOT mask real subdirectories
    ws_empty = tmp_path / "ws_empty_gitfile"
    ws_empty.mkdir()
    (ws_empty / ".git").write_text("", encoding="utf-8")
    (ws_empty / "real-sub-a").mkdir()
    (ws_empty / "real-sub-b").mkdir()
    assert _is_git_repo(ws_empty) is False
    assert inspect_git_metadata(ws_empty) == (None, 0)

    pm = ProjectManager(ws_empty, tmp_path / "save_empty")
    assert sorted(p["id"] for p in pm.scan_projects()) == ["real-sub-a", "real-sub-b"]

    # 2. Junk .git file and broken gitdir: pointers must return False
    junk_dir = tmp_path / "junk_gitfile"
    junk_dir.mkdir()
    (junk_dir / ".git").write_text("not a gitdir pointer\n", encoding="utf-8")
    assert _is_git_repo(junk_dir) is False

    broken_dir = tmp_path / "broken_gitdir"
    broken_dir.mkdir()
    (broken_dir / ".git").write_text("gitdir: /nonexistent/path/to/git\n", encoding="utf-8")
    assert _is_git_repo(broken_dir) is False

    no_head_target = tmp_path / "target_without_head"
    no_head_target.mkdir()
    missing_head_dir = tmp_path / "missing_head_gitdir"
    missing_head_dir.mkdir()
    (missing_head_dir / ".git").write_text(
        f"gitdir: {no_head_target}\n", encoding="utf-8"
    )
    assert _is_git_repo(missing_head_dir) is False

    # 3. Relative gitdir: pointer to a valid directory with HEAD returns True
    rel_target = tmp_path / "valid_rel_gitdir"
    rel_target.mkdir()
    (rel_target / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    rel_repo = tmp_path / "rel_worktree"
    rel_repo.mkdir()
    (rel_repo / ".git").write_text("gitdir: ../valid_rel_gitdir\n", encoding="utf-8")
    assert _is_git_repo(rel_repo) is True


def test_m4_r2_hud_node_tree_and_banner_consistency() -> None:
    import shutil
    import subprocess

    if not shutil.which("node"):
        pytest.skip("node not installed")

    repo_root = Path(__file__).resolve().parents[1]
    hud_js_path = repo_root / "src/agy_home_bridge/static/hud.js"
    probe_script = f"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

function makeEl(tag = 'div') {{
  return {{
    tagName: tag.toUpperCase(),
    children: [],
    className: '',
    dataset: {{}},
    attributes: {{}},
    textContent: '',
    hidden: false,
    style: {{}},
    classList: {{ add() {{}}, remove() {{}} }},
    addEventListener() {{}},
    setAttribute(k, v) {{ this.attributes[k] = String(v); }},
    getAttribute(k) {{ return this.attributes[k] ?? null; }},
    appendChild(c) {{ this.children.push(c); return c; }},
    replaceChildren(...cs) {{ this.children = cs; }},
    get firstElementChild() {{ return this.children[0] || null; }},
    querySelector() {{ return null; }},
    remove() {{}},
    isConnected: true,
  }};
}}

const hudEls = new Map();
class MockWS {{ constructor() {{ this.readyState = 1; }} }}
const hudCtx = {{
  Map, Array, Number, Boolean, String, JSON, Math, Date, Promise,
  setTimeout: () => 0,
  setInterval: () => 0,
  WebSocket: MockWS,
  window: {{ location: {{ protocol: 'https:', host: 'bridge.home' }} }},
  document: {{
    getElementById(id) {{
      if (!hudEls.has(id)) hudEls.set(id, makeEl('div'));
      return hudEls.get(id);
    }},
    createElement(tag) {{ return makeEl(tag); }},
  }},
  fetch: () => Promise.resolve({{ ok: false }}),
}};
vm.createContext(hudCtx);
vm.runInContext(fs.readFileSync({json.dumps(str(hud_js_path))}, 'utf8'), hudCtx);
hudCtx._assert = assert;
hudCtx._hudEls = hudEls;

vm.runInContext(`
  function getTreeActive() {{
    const cards = _hudEls.get('hudNodeTree').children;
    const activeCards = [];
    const activeAgents = [];
    for (const c of cards) {{
      if (c.dataset.active === 'true') {{
        activeCards.push(c.children[0].children[0].children[0].textContent);
      }}
      const ul = c.children[1];
      if (ul) {{
        for (const li of ul.children) {{
          if (li.dataset.active === 'true') {{
            activeAgents.push(li.children[0].children[0].textContent);
          }}
        }}
      }}
    }}
    return {{ activeCards, activeAgents }};
  }}

  const sampleProjects = () => [
    {{
      id: 'p1', name: 'one', short_name: 'One', active: false, active_agent_id: '101',
      agents: [{{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }}],
    }},
    {{
      id: 'p2', name: 'two', short_name: 'Two', active: true, active_agent_id: '202',
      agents: [
        {{ agent_id: '101', project_agent: 'project Two, agent 101', status: 'idle' }},
        {{ agent_id: '202', project_agent: 'project Two, agent 202', status: 'running' }},
      ],
    }},
  ];

  // Case A: empty active_project_id & active_agent_id uses proj.active + proj.active_agent_id
  hudState.active_project_id = '';
  hudState.active_agent_id = '';
  hudState.active_project_agent = '';
  hudState.projects = sampleProjects();
  renderActiveTargetBanner();
  renderNodeTree();
  _assert.strictEqual(_hudEls.get('hudActiveProjectAgent').textContent, 'project Two, agent 202');
  _assert.deepStrictEqual(getTreeActive(), {{ activeCards: ['Two'], activeAgents: ['project Two, agent 202'] }});

  // Case B: null active_project_id & no p.active falls back to projects[0] in BOTH banner and tree
  hudState.active_project_id = null;
  hudState.active_agent_id = null;
  hudState.projects = sampleProjects().map(p => ({{ ...p, active: false }}));
  renderActiveTargetBanner();
  renderNodeTree();
  _assert.strictEqual(_hudEls.get('hudActiveProjectAgent').textContent, 'project One, agent 101');
  _assert.deepStrictEqual(getTreeActive(), {{ activeCards: ['One'], activeAgents: ['project One, agent 101'] }});

  // Case C: stale active_project_id falls back to p.active in BOTH banner and tree
  hudState.active_project_id = 'stale-deleted-id';
  hudState.active_agent_id = '202';
  hudState.projects = sampleProjects();
  renderActiveTargetBanner();
  renderNodeTree();
  _assert.strictEqual(_hudEls.get('hudActiveProjectAgent').textContent, 'project Two, agent 202');
  _assert.deepStrictEqual(getTreeActive(), {{ activeCards: ['Two'], activeAgents: ['project Two, agent 202'] }});

  // Case D: project_changed omitting project_agent clears/derives active_project_agent
  handleHudEvent({{
    type: 'hud_snapshot',
    active_project_id: 'p1',
    active_agent_id: '101',
    active_project_agent: 'project One, agent 101',
    projects: sampleProjects(),
  }}, false);
  handleHudEvent({{
    type: 'project_changed',
    project_id: 'p2',
    agent_id: '202',
  }}, false);
  _assert.strictEqual(hudState.active_project_agent, 'project Two, agent 202');
  _assert.strictEqual(_hudEls.get('hudActiveProjectAgent').textContent, 'project Two, agent 202');
  _assert.deepStrictEqual(getTreeActive(), {{ activeCards: ['Two'], activeAgents: ['project Two, agent 202'] }});

  // Case E: hud_snapshot resetting active_project_id: null clears stale selection
  handleHudEvent({{
    type: 'hud_snapshot',
    active_project_id: null,
    active_agent_id: null,
    active_project_agent: null,
    projects: [],
  }}, false);
  _assert.strictEqual(hudState.active_project_id, null);
  _assert.strictEqual(hudState.active_project_agent, null);
  _assert.strictEqual(_hudEls.get('hudActiveProjectAgent').textContent, 'No project selected');
`, hudCtx);
"""
    res = subprocess.run(["node", "-e", probe_script], capture_output=True, text=True)
    assert res.returncode == 0, f"HUD consistency probe failed:\n{res.stderr}\n{res.stdout}"


@pytest.mark.asyncio
async def test_m4_r2_ws_live_binary_frame_empty_audio_and_blank_room(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        TestClient(app) as client,
    ):
        with client.websocket_connect("/ws/live") as lws:
            lws.receive_json()  # connected

            # Binary frame on /ws/live returns structured error without crashing loop
            lws.send_bytes(b"\x00\x01\x02\x03")
            bin_err = lws.receive_json()
            assert bin_err["type"] == "error"
            assert "binary" in bin_err["error"].lower()

            # Blank room rejected cleanly
            lws.send_json({"type": "set_room", "room": "   "})
            room_err = lws.receive_json()
            assert room_err["type"] == "error"
            assert "room" in room_err["error"].lower()

            # Loop remains healthy afterwards
            lws.send_json({"type": "set_routing", "mode": "direct"})
            ok_msg = lws.receive_json()
            while ok_msg.get("type") != "status" or "routing_mode" not in ok_msg:
                ok_msg = lws.receive_json()
            assert ok_msg["routing_mode"] == "direct"

    # Native Gemini Live _forward_client_to_gemini with binary frame, empty pcm_b64, blank room, and clean disconnect
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)
    bridge = FullDuplexLiveBridge(cfg, engine, cast_mgr, event_hub=hub, project_manager=pm)

    frames = [
        {"type": "websocket.receive", "bytes": b"\x00\x01\x02"},
        {"type": "websocket.receive", "text": json.dumps({"type": "audio_chunk", "pcm_b64": ""})},
        {"type": "websocket.receive", "text": json.dumps({"type": "set_room", "room": "  "})},
        {"type": "websocket.receive", "text": json.dumps({"type": "set_routing", "mode": "cast"})},
        {"type": "websocket.disconnect", "code": 1000},
    ]
    sent_frames: list[dict[str, Any]] = []

    class FakeFrameWS:
        async def receive(self) -> dict[str, Any]:
            return frames.pop(0)

        async def send_text(self, raw: str) -> None:
            sent_frames.append(json.loads(raw))

    fake_session = AsyncMock()
    await bridge._forward_client_to_gemini(FakeFrameWS(), fake_session)  # type: ignore[arg-type]

    assert len(sent_frames) == 4
    assert sent_frames[0]["type"] == "error"
    assert "binary" in sent_frames[0]["error"].lower()
    assert sent_frames[1] == {"type": "error", "error": "Empty audio_chunk pcm_b64"}
    assert sent_frames[2] == {"type": "error", "error": "room must be a non-empty string"}
    assert sent_frames[3] == {"type": "status", "state": "listening", "routing_mode": "cast"}
    fake_session.send_realtime_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_m4_r2_subagent_out_of_order_step_id_and_mixed_call_id_dedup(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)

    class ToolCall:
        def __init__(self, name: str, args: dict[str, Any], call_id: str | None, step_id: str | None) -> None:
            self.name = name
            self.args = args
            self.id = call_id
            self.step_id = step_id

    class ToolResult:
        def __init__(self, name: str, call_id: str | None, step_id: str | None, result: str | None = None) -> None:
            self.name = name
            self.id = call_id
            self.step_id = step_id
            self.result = result
            self.error = None
            self.exception = None

    class Text:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeHooksReg:
        @staticmethod
        def post_tool_call(fn: Any) -> Any:
            return fn

        @staticmethod
        def on_tool_error(fn: Any) -> Any:
            return fn

    class FakeResponse:
        def __init__(self, hooks_list: list[Any]) -> None:
            self._hooks = hooks_list

        @property
        def chunks(self):
            async def _gen():
                # 1. Start two unnamed subagents s1 (w1) and s2 (w2)
                yield ToolCall("start_subagent", {"agent_name": "w1"}, None, "s1")
                yield ToolCall("start_subagent", {"agent_name": "w2"}, None, "s2")

                # 2. Out-of-order completion: s2 finishes first via @post_tool_call
                for h in self._hooks:
                    if h.__name__ == "_on_post_tool_call":
                        await h(ToolResult("start_subagent", None, "s2", result="w2 ok"))

                # 3. Mixed call_id vs None for s1: hook reports call_id="c1", dup ToolResult has call_id=None
                for h in self._hooks:
                    if h.__name__ == "_on_post_tool_call":
                        await h(ToolResult("start_subagent", "c1", "s1", result="w1 ok"))
                yield ToolResult("start_subagent", None, "s1", result="w1 ok dup")

                # 4. Two distinct named tool calls sharing the same step_id="s3" BOTH emit
                yield ToolCall("read_file", {"path": "a.py"}, "r1", "s3")
                yield ToolCall("read_file", {"path": "b.py"}, "r2", "s3")
                yield ToolResult("read_file", "r1", "s3", result="a")
                yield ToolResult("read_file", "r2", "s3", result="b")

                yield Text("All done.")

            return _gen()

    class FakeAgent:
        def __init__(self, ag_cfg: dict[str, Any]) -> None:
            self.ag_cfg = ag_cfg
            self.conversation_id = "conv-r2-sub"
            self.conversation = types.SimpleNamespace(_steps=[])

        async def __aenter__(self) -> FakeAgent:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def chat(self, prompt: str) -> FakeResponse:
            return FakeResponse(self.ag_cfg.get("hooks") or [])

    fake_agy = types.ModuleType("google.antigravity")
    fake_agy.Agent = FakeAgent  # type: ignore[attr-defined]
    fake_agy.LocalAgentConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]
    fake_agy.CapabilitiesConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]

    fake_hooks_mod = types.ModuleType("google.antigravity.hooks")
    fake_hooks_mod.hooks = FakeHooksReg  # type: ignore[attr-defined]

    q = hub.subscribe()
    try:
        with patch.dict(
            sys.modules,
            {"google.antigravity": fake_agy, "google.antigravity.hooks": fake_hooks_mod},
        ):
            ack = engine.start_background_task("test r2 subagent correlation", project="alpha-service")
            events: list[dict[str, Any]] = []
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=3.0)
                if ev.get("task_id") == ack["task_id"]:
                    events.append(ev)
                    if ev["type"] == "task_completed":
                        break
            await asyncio.sleep(0.05)
    finally:
        hub.unsubscribe(q)
        await engine.shutdown()

    sub_results = [
        e for e in events if e["type"] == "tool_result" and e.get("name") == "start_subagent"
    ]
    assert [e["step_id"] for e in sub_results] == ["s2", "s1"]

    read_results = [
        e for e in events if e["type"] == "tool_result" and e.get("name") == "read_file"
    ]
    assert [e["tool_call_id"] for e in read_results] == ["r1", "r2"]

    alpha = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
    agent_101 = next(a for a in alpha["agents"] if a["agent_id"] == "101")
    assert len(agent_101["swarm_nodes"]) == 2
    by_label = {n["label"]: n for n in agent_101["swarm_nodes"]}
    assert by_label["w1"]["status"] == "completed"
    assert by_label["w2"]["status"] == "completed"


def test_m4_r3_app_node_view_banner_and_state_consistency() -> None:
    import shutil
    import subprocess

    if not shutil.which("node"):
        pytest.skip("node not installed")

    repo_root = Path(__file__).resolve().parents[1]
    app_js_path = repo_root / "src/agy_home_bridge/static/app.js"
    probe_script = f"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

function makeEl(tag = 'div') {{
  return {{
    tagName: tag.toUpperCase(),
    children: [],
    className: '',
    dataset: {{}},
    attributes: {{}},
    textContent: '',
    hidden: false,
    disabled: false,
    value: '',
    style: {{}},
    classList: {{ add() {{}}, remove() {{}} }},
    addEventListener() {{}},
    setAttribute(k, v) {{ this.attributes[k] = String(v); }},
    getAttribute(k) {{ return this.attributes[k] ?? null; }},
    removeAttribute(k) {{ delete this.attributes[k]; }},
    appendChild(c) {{ c.isConnected = true; this.children.push(c); return c; }},
    replaceChildren(...cs) {{ this.children = cs.map(c => {{ c.isConnected = true; return c; }}); }},
    querySelector() {{ return null; }},
    querySelectorAll() {{ return []; }},
    getContext() {{ return {{ clearRect() {{}}, beginPath() {{}}, arc() {{}}, stroke() {{}} }}; }},
    get firstElementChild() {{ return this.children[0] || null; }},
    get lastElementChild() {{ return this.children[this.children.length - 1] || null; }},
    remove() {{}},
    isConnected: true,
  }};
}}

const appEls = new Map();
class MockWS {{ constructor() {{ this.readyState = 1; MockWS.last = this; }} send() {{}} }}
MockWS.OPEN = 1;

const appCtx = {{
  URL,
  Map, Set, Array, Number, Boolean, String, JSON, Math, Date, Promise, Error,
  setTimeout: () => 0,
  setInterval: () => 0,
  requestAnimationFrame: () => 0,
  HTMLInputElement: {{ prototype: {{ switch: true }} }},
  HTMLDialogElement: {{ prototype: {{ closedBy: 'any' }} }},
  WebSocket: MockWS,
  navigator: {{}},
  window: {{
    isSecureContext: true,
    location: {{ hostname: 'bridge.home', protocol: 'https:', host: 'bridge.home', origin: 'https://bridge.home' }},
  }},
  document: {{
    getElementById(id) {{
      if (!appEls.has(id)) appEls.set(id, makeEl('div'));
      return appEls.get(id);
    }},
    createElement(tag) {{ return makeEl(tag); }},
    querySelectorAll() {{ return []; }},
  }},
  fetch: () => Promise.resolve({{ ok: false }}),
}};
vm.createContext(appCtx);
vm.runInContext(fs.readFileSync({json.dumps(str(app_js_path))}, 'utf8'), appCtx);
appCtx._assert = assert;
appCtx._appEls = appEls;

vm.runInContext(`
  function getGridActive() {{
    const cards = _appEls.get('projectsNodeGrid').children;
    const activeCards = [];
    const activeAgents = [];
    for (const c of cards) {{
      if (!c.dataset || !c.dataset.projectId) continue;
      if (c.dataset.active === 'true') {{
        activeCards.push(c.children[0].children[0].children[0].children[0].textContent);
      }}
      const ul = c.children[1];
      if (ul) {{
        for (const li of ul.children) {{
          const btn = li.children[0];
          if (btn && btn.getAttribute('aria-pressed') === 'true') {{
            activeAgents.push(btn.children[0].children[0].textContent);
          }}
        }}
      }}
    }}
    return {{
      summary: _appEls.get('activeTargetSummary').textContent,
      scopeBadge: _appEls.get('activityScopeBadge').textContent,
      activeCards,
      activeAgents,
    }};
  }}

  const sampleProjects = () => [
    {{
      id: 'p1', name: 'one', short_name: 'One', active: false, active_agent_id: '101',
      agents: [{{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }}],
    }},
    {{
      id: 'p2', name: 'two', short_name: 'Two', active: true, active_agent_id: '202',
      agents: [
        {{ agent_id: '101', project_agent: 'project Two, agent 101', status: 'idle' }},
        {{ agent_id: '202', project_agent: 'project Two, agent 202', status: 'running' }},
      ],
    }},
  ];

  // Case 2: empty active_project_id & active_agent_id uses p.active + p.active_agent_id
  projectsState.active_project_id = '';
  projectsState.active_agent_id = '';
  projectsState.active_project_agent = '';
  projectsState.projects = sampleProjects();
  renderProjectsNodeView();
  _assert.deepStrictEqual(getGridActive(), {{
    summary: 'Voice Target: project Two, agent 202',
    scopeBadge: 'project Two, agent 202',
    activeCards: ['Two'],
    activeAgents: ['project Two, agent 202'],
  }});

  // Case 3: null active_project_id & no p.active yields None selected and 0 active cards
  projectsState.active_project_id = null;
  projectsState.active_agent_id = null;
  projectsState.active_project_agent = null;
  projectsState.projects = sampleProjects().map(p => ({{ ...p, active: false }}));
  renderProjectsNodeView();
  _assert.deepStrictEqual(getGridActive(), {{
    summary: 'Voice Target: None selected',
    scopeBadge: 'None selected',
    activeCards: [],
    activeAgents: [],
  }});

  // Case 4a: stale active_project_id with p2.active=true falls back consistently
  projectsState.active_project_id = 'stale-deleted-id';
  projectsState.active_agent_id = '101';
  projectsState.active_project_agent = 'project Stale, agent 202';
  projectsState.projects = sampleProjects();
  renderProjectsNodeView();
  _assert.deepStrictEqual(getGridActive(), {{
    summary: 'Voice Target: project Two, agent 202',
    scopeBadge: 'project Two, agent 202',
    activeCards: ['Two'],
    activeAgents: ['project Two, agent 202'],
  }});

  // Case 4b: stale active_project_id with all p.active=false clears stale active_project_agent
  projectsState.active_project_id = 'stale-deleted-id';
  projectsState.active_agent_id = '202';
  projectsState.active_project_agent = 'project Stale, agent 202';
  projectsState.projects = sampleProjects().map(p => ({{ ...p, active: false }}));
  renderProjectsNodeView();
  _assert.deepStrictEqual(getGridActive(), {{
    summary: 'Voice Target: None selected',
    scopeBadge: 'None selected',
    activeCards: [],
    activeAgents: [],
  }});

  // Case 5: project_changed omitting project_agent derives/resets active_project_agent
  handleEventHubMessage({{
    type: 'hud_snapshot',
    active_project_id: 'p1',
    active_agent_id: '101',
    active_project_agent: 'project One, agent 101',
    projects: sampleProjects(),
  }});
  handleEventHubMessage({{
    type: 'project_changed',
    project_id: 'p2',
    agent_id: '202',
  }});
  _assert.strictEqual(projectsState.active_project_agent, 'project Two, agent 202');
  _assert.deepStrictEqual(getGridActive(), {{
    summary: 'Voice Target: project Two, agent 202',
    scopeBadge: 'project Two, agent 202',
    activeCards: ['Two'],
    activeAgents: ['project Two, agent 202'],
  }});

  // Case 6: hud_snapshot resetting active_project_id: null clears stale selection
  handleEventHubMessage({{
    type: 'hud_snapshot',
    active_project_id: null,
    active_agent_id: null,
    active_project_agent: null,
    projects: [],
  }});
  _assert.strictEqual(projectsState.active_project_id, null);
  _assert.strictEqual(projectsState.active_project_agent, null);
  _assert.strictEqual(_appEls.get('activeTargetSummary').textContent, 'Voice Target: None selected');
  _assert.strictEqual(_appEls.get('activityScopeBadge').textContent, 'None selected');
`, appCtx);
"""
    res = subprocess.run(["node", "-e", probe_script], capture_output=True, text=True)
    assert res.returncode == 0, f"APP consistency probe failed:\n{res.stderr}\n{res.stdout}"


@pytest.mark.asyncio
async def test_m4_r3_eager_hooks_before_stream_shared_step_id_3c2(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)

    class ToolCall:
        def __init__(self, name: str, args: dict[str, Any], call_id: str | None, step_id: str | None) -> None:
            self.name = name
            self.args = args
            self.id = call_id
            self.step_id = step_id

    class ToolResult:
        def __init__(self, name: str, call_id: str | None, step_id: str | None, result: str | None = None) -> None:
            self.name = name
            self.id = call_id
            self.step_id = step_id
            self.result = result
            self.error = None
            self.exception = None

    class Text:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeHooksReg:
        @staticmethod
        def post_tool_call(fn: Any) -> Any:
            return fn

        @staticmethod
        def on_tool_error(fn: Any) -> Any:
            return fn

    class FakeResponse:
        @property
        def chunks(self):
            async def _gen():
                yield ToolCall("start_subagent", {"agent_name": "w1"}, "c1", "s1")
                yield ToolCall("start_subagent", {"agent_name": "w2"}, "c2", "s1")
                yield Text("Done.")

            return _gen()

    class FakeAgent:
        def __init__(self, ag_cfg: dict[str, Any]) -> None:
            self.ag_cfg = ag_cfg
            self.conversation_id = "conv-r3-3c2"
            self.conversation = types.SimpleNamespace(_steps=[])

        async def __aenter__(self) -> FakeAgent:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def chat(self, prompt: str) -> FakeResponse:
            for h in self.ag_cfg.get("hooks") or []:
                if h.__name__ == "_on_post_tool_call":
                    await h(ToolResult("start_subagent", "c1", "s1", result="A"))
                    await h(ToolResult("start_subagent", "c2", "s1", result="B"))
            return FakeResponse()

    fake_agy = types.ModuleType("google.antigravity")
    fake_agy.Agent = FakeAgent  # type: ignore[attr-defined]
    fake_agy.LocalAgentConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]
    fake_agy.CapabilitiesConfig = lambda **kw: dict(kw)  # type: ignore[attr-defined]

    fake_hooks_mod = types.ModuleType("google.antigravity.hooks")
    fake_hooks_mod.hooks = FakeHooksReg  # type: ignore[attr-defined]

    q = hub.subscribe()
    try:
        with patch.dict(
            sys.modules,
            {"google.antigravity": fake_agy, "google.antigravity.hooks": fake_hooks_mod},
        ):
            ack = engine.start_background_task("test 3c2 eager hooks", project="alpha-service")
            events: list[dict[str, Any]] = []
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=3.0)
                if ev.get("task_id") == ack["task_id"]:
                    events.append(ev)
                    if ev["type"] == "task_completed":
                        break
            await asyncio.sleep(0.05)
    finally:
        hub.unsubscribe(q)
        await engine.shutdown()

    sub_results = [
        e for e in events if e["type"] == "tool_result" and e.get("name") == "start_subagent"
    ]
    assert [e["tool_call_id"] for e in sub_results] == ["c1", "c2"]

    alpha = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
    agent_101 = next(a for a in alpha["agents"] if a["agent_id"] == "101")
    assert len(agent_101["swarm_nodes"]) == 2
    by_id = {n["node_id"]: n for n in agent_101["swarm_nodes"]}
    assert by_id["c1"]["status"] == "completed"
    assert by_id["c1"]["label"] == "w1"
    assert by_id["c2"]["status"] == "completed"
    assert by_id["c2"]["label"] == "w2"


# ============================================================================
# 12. Tier-5 Adversarial Hardening Tests (BUG-1 through BUG-11)
# ============================================================================
@pytest.mark.asyncio
async def test_t5_cast_manager_concurrent_discovery_threadsafe_no_iteration_crash(
    test_settings: BridgeSettings,
) -> None:
    import threading

    cast_mgr = GoogleHomeCastManager(test_settings)
    cast_mgr.register_mock_speaker("Office", host="192.168.1.101")
    cast_mgr.register_mock_speaker("Kitchen", host="192.168.1.102")

    def _make_cc(name: str, cast_type: str, host: str) -> MagicMock:
        cc = MagicMock()
        cc.cast_info.friendly_name = name
        cc.cast_info.uuid = f"uuid-{name}"
        cc.cast_info.model_name = "Nest Audio" if cast_type == "audio" else "Google TV"
        cc.cast_info.cast_type = cast_type
        cc.cast_info.host = host
        cc.cast_info.port = 8009
        return cc

    fake_devices = [
        _make_cc("Office", "audio", "192.168.1.101"),
        _make_cc("Kitchen", "audio", "192.168.1.102"),
        _make_cc("Office TV", "cast", "192.168.1.110"),
    ]
    fake_browser = MagicMock()

    stop_flag = threading.Event()
    errors: list[BaseException] = []
    empty_speaker_reads = 0

    def _reader_loop() -> None:
        nonlocal empty_speaker_reads
        try:
            while not stop_flag.is_set():
                speakers = cast_mgr.list_speakers(include_tvs=True)
                audio_speakers = [s for s in speakers if s.get("device_role") == "speaker"]
                if not audio_speakers:
                    empty_speaker_reads += 1
                cast_mgr.set_active_room("Office")
                cast_mgr.resolve_room_name("Kitchen")
        except BaseException as exc:  # pragma: no cover - test assertion
            errors.append(exc)

    reader = threading.Thread(target=_reader_loop, daemon=True)
    reader.start()
    try:
        with patch(
            "pychromecast.get_chromecasts",
            return_value=(fake_devices, fake_browser),
        ):
            for _ in range(40):
                await cast_mgr.discover_speakers(timeout=0.05, merge=False)
    finally:
        stop_flag.set()
        reader.join(timeout=3.0)

    assert errors == []
    assert empty_speaker_reads == 0


@pytest.mark.asyncio
async def test_t5_ws_live_barge_in_cancels_ask_and_resets_agent_idle(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)

    # 1. Direct ask() cancellation resets status to idle and emits task_failed
    started_gate = asyncio.Event()
    release_gate = asyncio.Event()

    async def _slow_exec(*args: Any, **kwargs: Any) -> Any:
        started_gate.set()
        await release_gate.wait()
        return (MagicMock(), None)

    q = hub.subscribe()
    try:
        with patch.object(engine, "_execute_agent_turn", side_effect=_slow_exec):
            ask_task = asyncio.create_task(engine.ask("long running ask", emit_events=True))
            await asyncio.wait_for(started_gate.wait(), timeout=2.0)

            alpha = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
            assert alpha["agents"][0]["status"] == "running"
            assert alpha["agents"][0]["current_task"] == "long running ask"

            ask_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await ask_task

        alpha_after = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
        assert alpha_after["agents"][0]["status"] == "idle"
        assert alpha_after["agents"][0]["current_task"] is None

        seen_events: list[dict[str, Any]] = []
        while not q.empty():
            seen_events.append(q.get_nowait())
        failed = [e for e in seen_events if e.get("type") == "task_failed"]
        assert len(failed) == 1
        assert "cancel" in failed[0]["error"].lower()
    finally:
        hub.unsubscribe(q)

    # 2. /ws/live text followed by barge_in cancels ask() and leaves agent idle
    import threading

    app = create_app(cfg)
    ws_started = threading.Event()

    async def _slow_exec_ws(*args: Any, **kwargs: Any) -> Any:
        ws_started.set()
        await asyncio.sleep(10.0)
        return (MagicMock(), None)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        patch.object(app.state.agent_engine, "_execute_agent_turn", side_effect=_slow_exec_ws),
        TestClient(app) as client,
    ):
        with client.websocket_connect("/ws/live") as lws:
            lws.receive_json()  # connected
            lws.send_json({"type": "text", "text": "slow task via ws"})
            # Drain until thinking status arrives and _execute_agent_turn is running
            for _ in range(5):
                msg = lws.receive_json()
                if msg.get("type") == "status" and msg.get("state") == "thinking":
                    break
            assert ws_started.wait(timeout=2.0)

            proj_mid = client.get("/api/projects").json()["projects"][0]
            assert proj_mid["agents"][0]["status"] == "running"

            lws.send_json({"type": "barge_in"})
            got_interrupted = False
            for _ in range(5):
                msg = lws.receive_json()
                if msg.get("type") == "interrupted":
                    got_interrupted = True
                    break
            assert got_interrupted

        proj_after = client.get("/api/projects").json()["projects"][0]
        assert proj_after["agents"][0]["status"] == "idle"
        assert proj_after["agents"][0]["current_task"] is None


def test_t5_start_background_task_without_running_loop_does_not_stick_running(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)

    with pytest.raises(RuntimeError, match="no running event loop"):
        engine.start_background_task("hello without loop")

    alpha = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
    assert alpha["agents"][0]["status"] == "idle"
    assert alpha["agents"][0]["current_task"] is None
    assert hub.recent_events() == []


def test_t5_generate_tts_file_concurrent_atomic_write(
    test_settings: BridgeSettings,
) -> None:
    import concurrent.futures
    import threading
    import time

    cast_mgr = GoogleHomeCastManager(test_settings)
    expected_bytes = b"PART1-PART2-COMPLETE-MP3-PAYLOAD"
    start_barrier = threading.Barrier(2)

    with patch("agy_home_bridge.cast_manager.gTTS") as mock_gtts:
        def _slow_save(path_str: str) -> None:
            with open(path_str, "wb") as f:
                f.write(b"PART1-")
                f.flush()
                time.sleep(0.03)
                f.write(b"PART2-COMPLETE-MP3-PAYLOAD")

        inst = MagicMock()
        inst.save.side_effect = _slow_save
        mock_gtts.return_value = inst

        def _worker() -> bytes:
            start_barrier.wait(timeout=2.0)
            p = cast_mgr.generate_tts_file("Atomic concurrent speech")
            return p.read_bytes()

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(_worker)
            f2 = pool.submit(_worker)
            res1 = f1.result(timeout=5.0)
            res2 = f2.result(timeout=5.0)

    assert res1 == expected_bytes
    assert res2 == expected_bytes
    leftover_tmps = list(test_settings.tts_cache_dir.glob(".tmp_*"))
    assert leftover_tmps == []


@pytest.mark.asyncio
async def test_t5_gemini_live_tool_error_and_none_parts_resilience(
    test_settings: BridgeSettings,
) -> None:
    cast_mgr = GoogleHomeCastManager(test_settings)
    cast_mgr.register_mock_speaker("Office")
    engine = AntigravityAgentEngine(test_settings, cast_mgr)
    bridge = FullDuplexLiveBridge(test_settings, engine, cast_mgr)
    fake_ws = AsyncMock()

    # Turn 1: model_turn.parts is None (must not raise TypeError)
    resp_none_parts = MagicMock()
    resp_none_parts.server_content.interrupted = False
    resp_none_parts.server_content.input_transcription = None
    resp_none_parts.server_content.output_transcription = None
    resp_none_parts.server_content.model_turn.parts = None
    resp_none_parts.server_content.turn_complete = False
    resp_none_parts.tool_call = None

    # Turn 2: blank broadcast_to_google_home + blank switch_google_home_room
    fc_blank_bcast = MagicMock()
    fc_blank_bcast.id = "fc-err-1"
    fc_blank_bcast.name = "broadcast_to_google_home"
    fc_blank_bcast.args = {"message": "   "}

    fc_blank_switch = MagicMock()
    fc_blank_switch.id = "fc-err-2"
    fc_blank_switch.name = "switch_google_home_room"
    fc_blank_switch.args = {"room": "   "}

    resp_tool_err = MagicMock()
    resp_tool_err.server_content = None
    resp_tool_err.tool_call.function_calls = [fc_blank_bcast, fc_blank_switch]

    # Turn 3: valid text after error turns proves loop stayed alive
    part_ok = MagicMock()
    part_ok.inline_data = None
    part_ok.text = "Still alive after tool error."
    resp_ok = MagicMock()
    resp_ok.server_content.interrupted = False
    resp_ok.server_content.input_transcription = None
    resp_ok.server_content.output_transcription = None
    resp_ok.server_content.model_turn.parts = [part_ok]
    resp_ok.server_content.turn_complete = True
    resp_ok.tool_call = None

    class FakeSession:
        def __init__(self) -> None:
            self.send_tool_response = AsyncMock()
            self._calls = 0

        def receive(self):
            self._calls += 1
            turns = [[resp_none_parts, resp_tool_err, resp_ok]]
            items = turns[self._calls - 1] if self._calls <= len(turns) else []

            async def _gen():
                for item in items:
                    yield item

            return _gen()

    sess = FakeSession()
    await bridge._forward_gemini_to_client(fake_ws, sess)

    sess.send_tool_response.assert_awaited_once()
    sent_resps = sess.send_tool_response.await_args.kwargs["function_responses"]
    assert len(sent_resps) == 2
    assert sent_resps[0].response["status"] == "error"
    assert "empty text" in sent_resps[0].response["error"].lower()
    assert sent_resps[1].response["status"] == "error"

    sent_payloads = [json.loads(call.args[0]) for call in fake_ws.send_text.await_args_list]
    assert any(
        p.get("type") == "transcript" and p.get("text") == "Still alive after tool error."
        for p in sent_payloads
    )


def test_t5_server_tts_dot_and_malformed_json_payloads(client: TestClient) -> None:
    # /api/tts/%2e and /api/tts/. return 404 instead of 500
    assert client.get("/api/tts/%2e").status_code == 404
    assert client.get("/api/tts/.").status_code == 404

    # /api/google-home/fulfillment returns 400 on invalid JSON or non-dict JSON
    r_bad_ff = client.post(
        "/api/google-home/fulfillment",
        content="{not-valid-json",
        headers={"Content-Type": "application/json"},
    )
    assert r_bad_ff.status_code == 400

    r_list_ff = client.post(
        "/api/google-home/fulfillment",
        json=["not", "a", "dict"],
    )
    assert r_list_ff.status_code == 400

    # /mcp returns 400 with JSON-RPC -32700 on parse error and -32600 on non-dict
    r_bad_mcp = client.post(
        "/mcp",
        content="{not-valid-json",
        headers={"Content-Type": "application/json"},
    )
    assert r_bad_mcp.status_code == 400
    assert r_bad_mcp.json()["error"]["code"] == -32700

    r_list_mcp = client.post("/mcp", json=[1, 2, 3])
    assert r_list_mcp.status_code == 400
    assert r_list_mcp.json()["error"]["code"] == -32600


@pytest.mark.asyncio
async def test_t5_speak_on_room_cast_pcm_and_stop_resolve_presence_room_keys(
    test_settings: BridgeSettings, tmp_path: Path
) -> None:
    cast_mgr = GoogleHomeCastManager(test_settings)
    fake_cc = MagicMock()
    cast_mgr.register_mock_speaker("Living Room speaker", host="192.168.1.120")
    cast_mgr._chromecasts["Living Room speaker"] = fake_cc

    fake_mp3 = (tmp_path / "tts_cache" / "living.mp3").resolve()
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"MP3")

    with patch.object(cast_mgr, "generate_tts_file", return_value=fake_mp3):
        res_speak = await cast_mgr.speak_on_room("Hello living room", room_name="living_room")
    assert res_speak["status"] == "casting"
    assert res_speak["room"] == "Living Room speaker"

    res_pcm = await cast_mgr.cast_pcm_audio(b"\x01\x00" * 160, room_name="kitchen")
    assert res_pcm["status"] == "casting"
    assert res_pcm["room"] == "Living Room speaker"

    await cast_mgr.stop_room("living_room")
    fake_cc.media_controller.stop.assert_called_once()


@pytest.mark.asyncio
async def test_t5_speak_on_google_home_tool_called_inside_running_loop_no_deadlock(
    test_settings: BridgeSettings, tmp_path: Path
) -> None:
    cast_mgr = GoogleHomeCastManager(test_settings)
    cast_mgr.register_mock_speaker("Office", host="192.168.1.101")
    engine = AntigravityAgentEngine(test_settings, cast_mgr)
    tools = {fn.__name__: fn for fn in engine.build_custom_tools()}

    fake_mp3 = (tmp_path / "tts_cache" / "loop_tool.mp3").resolve()
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"MP3")

    with patch.object(cast_mgr, "generate_tts_file", return_value=fake_mp3):
        # Directly invoke synchronous tool on the active event loop thread
        out_json = tools["speak_on_google_home"]("No deadlock on loop thread", "Office")

    parsed = json.loads(out_json)
    assert parsed["room"] == "Office"
    assert parsed["media_url"].endswith("/api/tts/loop_tool.mp3")


def test_t5_frontend_app_and_hud_state_reconciliation() -> None:
    import shutil
    import subprocess

    if not shutil.which("node"):
        pytest.skip("node not installed")

    repo_root = Path(__file__).resolve().parents[1]
    app_js_path = repo_root / "src/agy_home_bridge/static/app.js"
    hud_js_path = repo_root / "src/agy_home_bridge/static/hud.js"

    probe_script = f"""
const fs = require('fs');
const vm = require('vm');
const assert = require('assert');

function makeEl(tag = 'div') {{
  return {{
    tagName: tag.toUpperCase(),
    children: [],
    className: '',
    dataset: {{}},
    attributes: {{}},
    textContent: '',
    hidden: false,
    disabled: false,
    value: '',
    style: {{}},
    classList: {{ add() {{}}, remove() {{}} }},
    addEventListener() {{}},
    setAttribute(k, v) {{ this.attributes[k] = String(v); }},
    getAttribute(k) {{ return this.attributes[k] ?? null; }},
    removeAttribute(k) {{ delete this.attributes[k]; }},
    appendChild(c) {{ c.isConnected = true; this.children.push(c); return c; }},
    replaceChildren(...cs) {{ this.children = cs.map(c => {{ c.isConnected = true; return c; }}); }},
    querySelector() {{ return null; }},
    querySelectorAll() {{ return []; }},
    getContext() {{ return {{ clearRect() {{}}, beginPath() {{}}, arc() {{}}, stroke() {{}} }}; }},
    get firstElementChild() {{ return this.children[0] || null; }},
    get lastElementChild() {{ return this.children[this.children.length - 1] || null; }},
    remove() {{}},
    isConnected: true,
  }};
}}

// ---- Test app.js (F1, F2, F3, F8) ----
const appEls = new Map();
class MockWS {{ constructor() {{ this.readyState = 1; }} send() {{}} }}
MockWS.OPEN = 1;

const appCtx = {{
  URL, Map, Set, Array, Number, Boolean, String, JSON, Math, Date, Promise, Error,
  setTimeout: () => 0,
  setInterval: () => 0,
  requestAnimationFrame: () => 0,
  HTMLInputElement: {{ prototype: {{ switch: true }} }},
  HTMLDialogElement: {{ prototype: {{ closedBy: 'any' }} }},
  WebSocket: MockWS,
  navigator: {{}},
  window: {{
    isSecureContext: true,
    location: {{ hostname: 'bridge.home', protocol: 'https:', host: 'bridge.home', origin: 'https://bridge.home' }},
  }},
  document: {{
    getElementById(id) {{
      if (!appEls.has(id)) appEls.set(id, makeEl('div'));
      return appEls.get(id);
    }},
    createElement(tag) {{ return makeEl(tag); }},
    querySelectorAll() {{ return []; }},
  }},
  fetch: () => Promise.resolve({{
    ok: true,
    json: async () => ({{ project_id: 'p1', agent_id: '101', project_agent: 'project One, agent 101', conversation_id: 'conv-null-guard' }}),
  }}),
}};
vm.createContext(appCtx);
vm.runInContext(fs.readFileSync({json.dumps(str(app_js_path))}, 'utf8'), appCtx);
appCtx._assert = assert;
appCtx._appEls = appEls;

vm.runInContext(`
  const mkProjs = () => [
    {{
      id: 'p1', name: 'one', short_name: 'One', active: false, active_agent_id: '202',
      agents: [
        {{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }},
        {{ agent_id: '202', project_agent: 'project One, agent 202', status: 'idle' }},
      ],
    }},
    {{
      id: 'p2', name: 'two', short_name: 'Two', active: true, active_agent_id: '101',
      agents: [
        {{ agent_id: '101', project_agent: 'project Two, agent 101', status: 'idle' }},
      ],
    }},
  ];

  // F1: hud_snapshot with active_project_id: null omitting projects clears p.active and shows None selected
  handleEventHubMessage({{
    type: 'hud_snapshot',
    active_project_id: 'p2',
    active_agent_id: '101',
    active_project_agent: 'project Two, agent 101',
    projects: mkProjs(),
  }});
  handleEventHubMessage({{
    type: 'hud_snapshot',
    active_project_id: null,
    active_agent_id: null,
  }});
  _assert.strictEqual(_appEls.get('activeTargetSummary').textContent, 'Voice Target: None selected');
  _assert.strictEqual(projectsState.projects.some(p => p && p.active), false);

  // F2: project_changed omitting agent_id resolves from target project active_agent_id (not prev project)
  handleEventHubMessage({{
    type: 'hud_snapshot',
    active_project_id: 'p1',
    active_agent_id: '202',
    active_project_agent: 'project One, agent 202',
    projects: mkProjs(),
  }});
  handleEventHubMessage({{
    type: 'project_changed',
    project_id: 'p2',
  }});
  _assert.strictEqual(projectsState.active_agent_id, '101');
  _assert.strictEqual(_appEls.get('activeTargetSummary').textContent, 'Voice Target: project Two, agent 101');

  // F8: tool_result renders 0 and object JSON properly (including error: 0)
  handleEventHubMessage({{ type: 'tool_result', name: 'exit_code', result: 0 }});
  handleEventHubMessage({{ type: 'tool_result', name: 'check_obj', result: {{ ok: true }} }});
  handleEventHubMessage({{ type: 'tool_result', name: 'err_zero', error: 0 }});
  const feedItems = _appEls.get('nodeActivityFeed').children;
  const bodies = feedItems.map(li => li.children[1].textContent);
  _assert.ok(bodies.some(b => b.includes('exit_code: 0')), 'Expected 0 result in app.js feed, got: ' + JSON.stringify(bodies));
  _assert.ok(bodies.some(b => b.includes('{{"ok":true}}')), 'Expected JSON result in app.js feed, got: ' + JSON.stringify(bodies));
  _assert.ok(bodies.some(b => b.includes('err_zero: 0')), 'Expected error 0 in app.js feed, got: ' + JSON.stringify(bodies));

  // NEW-BUG-2 (app.js): project_changed to an uncached project_id does not wipe active_project_id or revert to stale p1
  handleEventHubMessage({{
    type: 'hud_snapshot',
    active_project_id: 'p1',
    active_agent_id: '101',
    active_project_agent: 'project One, agent 101',
    projects: mkProjs().map(p => ({{ ...p, active: p.id === 'p1' }})),
  }});
  handleEventHubMessage({{
    type: 'project_changed',
    project_id: 'p_new',
    short_name: 'New Proj',
    agent_id: '101',
    project_agent: 'project New Proj, agent 101',
  }});
  _assert.strictEqual(projectsState.active_project_id, 'p_new');
  _assert.strictEqual(projectsState.active_project_agent, 'project New Proj, agent 101');
  _assert.strictEqual(projectsState.projects.some(p => p && p.active), false);
  _assert.strictEqual(_appEls.get('activeTargetSummary').textContent, 'Voice Target: project New Proj, agent 101');
`, appCtx);

// F3 + NEW-BUG-2 (app.js fetchHealth explicit_none transition & uncached resolution)
(async () => {{
  await vm.runInContext(`
    projectsState.projects = [null, {{
      id: 'p1', name: 'one', short_name: 'One', active: true, active_agent_id: '101',
      agents: [null, {{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }}],
    }}];
    selectProjectTarget({{ project: 'p1', agent: '101' }});
  `, appCtx);
  vm.runInContext(`
    _assert.strictEqual(projectsState.projects[1].agents[1].conversation_id, 'conv-null-guard');
  `, appCtx);

  // Transition explicit_none: true -> fetchHealth returns valid project p2 -> clears explicit_none
  await vm.runInContext(`
    (async () => {{
      const origFetch = fetch;
      projectsState.projects = [
        {{ id: 'p1', name: 'one', short_name: 'One', active: false, active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }}] }},
        {{ id: 'p2', name: 'two', short_name: 'Two', active: false, active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project Two, agent 101', status: 'idle' }}] }},
      ];
      projectsState.active_project_id = null;
      projectsState.explicit_none = true;
      projectsState.pending_project_id = null;
      projectsState.active_project_agent = null;
      fetch = async () => ({{
        ok: true,
        json: async () => ({{
          active_project_id: 'p2',
          active_agent_id: '101',
          active_project_agent: 'project Two, agent 101',
        }}),
      }});
      await fetchHealth();
      _assert.strictEqual(projectsState.explicit_none, false);
      _assert.strictEqual(projectsState.active_project_id, 'p2');
      _assert.strictEqual(_appEls.get('activeTargetSummary').textContent, 'Voice Target: project Two, agent 101');

      // And fetchHealth returning active_project_id: null sets explicit_none: true
      fetch = async () => ({{
        ok: true,
        json: async () => ({{
          active_project_id: null,
          active_agent_id: null,
          active_project_agent: null,
        }}),
      }});
      await fetchHealth();
      _assert.strictEqual(projectsState.explicit_none, true);
      _assert.strictEqual(projectsState.active_project_id, null);
      _assert.strictEqual(_appEls.get('activeTargetSummary').textContent, 'Voice Target: None selected');

      // R3-GAP-2 (app.js): fetchHealth changing to uncached project without active_project_agent clears stale label
      projectsState.projects = [
        {{ id: 'p1', name: 'one', short_name: 'One', active: false, active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }}] }},
        {{ id: 'p2', name: 'two', short_name: 'Two', active: false, active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project Two, agent 101', status: 'idle' }}] }},
      ];
      projectsState.active_project_id = 'p_new';
      projectsState.pending_project_id = 'p_new';
      projectsState.explicit_none = false;
      projectsState.active_project_agent = 'project New, agent 101';
      fetch = async () => ({{
        ok: true,
        json: async () => ({{
          active_project_id: 'p_other_uncached',
          active_agent_id: '102',
        }}),
      }});
      await fetchHealth();
      _assert.strictEqual(projectsState.active_project_id, 'p_other_uncached');
      _assert.strictEqual(projectsState.pending_project_id, 'p_other_uncached');
      _assert.strictEqual(projectsState.active_project_agent, null);
      _assert.strictEqual(_appEls.get('activeTargetSummary').textContent, 'Voice Target: p_other_uncached');

      // And fetchHealth changing to cached project without active_project_agent recomputes label
      fetch = async () => ({{
        ok: true,
        json: async () => ({{
          active_project_id: 'p1',
          active_agent_id: '101',
        }}),
      }});
      await fetchHealth();
      _assert.strictEqual(projectsState.pending_project_id, null);
      _assert.strictEqual(projectsState.active_project_agent, 'project One, agent 101');
      _assert.strictEqual(_appEls.get('activeTargetSummary').textContent, 'Voice Target: project One, agent 101');

      fetch = origFetch;
    }})()
  `, appCtx);
}})().then(async () => {{
  // ---- Test hud.js (F6, F7, F8 + NEW-BUG-2 hydrateHudInitial explicit_none) ----
  const hudEls = new Map();
  const hudCtx = {{
    Map, Array, Number, Boolean, String, JSON, Math, Date, Promise,
    setTimeout: () => 0,
    setInterval: () => 0,
    WebSocket: MockWS,
    window: {{ location: {{ protocol: 'https:', host: 'bridge.home' }} }},
    document: {{
      getElementById(id) {{
        if (!hudEls.has(id)) hudEls.set(id, makeEl('div'));
        return hudEls.get(id);
      }},
      createElement(tag) {{ return makeEl(tag); }},
    }},
    fetch: (url) => {{
      if (url === '/api/projects') {{
        return Promise.resolve({{
          ok: true,
          json: async () => ({{
            active_project_id: null,
            active_agent_id: null,
            active_project_agent: null,
            projects: [
              {{ id: 'p1', name: 'one', short_name: 'One', active: true, active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }}] }},
              {{ id: 'p2', name: 'two', short_name: 'Two', active: false, active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project Two, agent 101', status: 'idle' }}] }},
            ],
          }}),
        }});
      }}
      if (url === '/api/health') {{
        return Promise.resolve({{
          ok: true,
          json: async () => ({{ active_room: 'Office', active_project_id: null, active_project_agent: null }}),
        }});
      }}
      return Promise.resolve({{ ok: false }});
    }},
  }};
  vm.createContext(hudCtx);
  vm.runInContext(fs.readFileSync({json.dumps(str(hud_js_path))}, 'utf8'), hudCtx);
  hudCtx._assert = assert;
  hudCtx._hudEls = hudEls;

  // Wait for initial hydrateHudInitial() called on script load
  await vm.runInContext(`
    (async () => {{
      await hydrateHudInitial();
      _assert.strictEqual(hudState.explicit_none, true);
      _assert.strictEqual(hudState.projects.some(p => p && p.active), false);
      _assert.strictEqual(_hudEls.get('hudActiveProjectAgent').textContent, 'No project selected');
    }})()
  `, hudCtx);

  vm.runInContext(`
    const mkHudProjs = () => [
      {{
        id: 'p1', name: 'one', short_name: 'One', active: false, active_agent_id: '202',
        agents: [
          {{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }},
          {{ agent_id: '202', project_agent: 'project One, agent 202', status: 'idle' }},
        ],
      }},
      {{
        id: 'p2', name: 'two', short_name: 'Two', active: true, active_agent_id: '101',
        agents: [
          {{ agent_id: '101', project_agent: 'project Two, agent 101', status: 'idle' }},
        ],
      }},
    ];

    // F6: hud_snapshot with active_project_id: null omitting projects shows No project selected
    handleHudEvent({{
      type: 'hud_snapshot',
      active_project_id: 'p2',
      active_agent_id: '101',
      active_project_agent: 'project Two, agent 101',
      projects: mkHudProjs(),
    }}, false);
    handleHudEvent({{
      type: 'hud_snapshot',
      active_project_id: null,
      active_agent_id: null,
    }}, false);
    _assert.strictEqual(_hudEls.get('hudActiveProjectAgent').textContent, 'No project selected');
    _assert.strictEqual(hudState.projects.some(p => p && p.active), false);

    // F7: project_changed omitting agent_id uses target project active_agent_id
    handleHudEvent({{
      type: 'hud_snapshot',
      active_project_id: 'p1',
      active_agent_id: '202',
      active_project_agent: 'project One, agent 202',
      projects: mkHudProjs(),
    }}, false);
    handleHudEvent({{
      type: 'project_changed',
      project_id: 'p2',
    }}, false);
    _assert.strictEqual(hudState.active_agent_id, '101');
    _assert.strictEqual(_hudEls.get('hudActiveProjectAgent').textContent, 'project Two, agent 101');

    // F8: tool_result renders 0 and object JSON properly in hud.js (including error: 0)
    handleHudEvent({{ type: 'tool_result', name: 'exit_code', result: 0 }}, false);
    handleHudEvent({{ type: 'tool_result', name: 'check_obj', result: {{ ok: true }} }}, false);
    handleHudEvent({{ type: 'tool_result', name: 'err_zero', error: 0 }}, false);
    const hudStreamBodies = _hudEls.get('hudActivityStream').children.map(li => li.children[1].textContent);
    _assert.ok(hudStreamBodies.some(b => b.includes('exit_code → 0')), 'Expected 0 result in hud.js stream, got: ' + JSON.stringify(hudStreamBodies));
    _assert.ok(hudStreamBodies.some(b => b.includes('{{"ok":true}}')), 'Expected JSON result in hud.js stream, got: ' + JSON.stringify(hudStreamBodies));
    _assert.ok(hudStreamBodies.some(b => b.includes('ERROR err_zero: 0')), 'Expected error 0 in hud.js stream, got: ' + JSON.stringify(hudStreamBodies));

    // R3-GAP-2 (hud.js): project_changed to uncached project suppresses projects[0] fallback
    handleHudEvent({{
      type: 'hud_snapshot',
      active_project_id: 'p1',
      active_agent_id: '101',
      active_project_agent: 'project One, agent 101',
      projects: mkHudProjs(),
    }}, false);
    handleHudEvent({{
      type: 'project_changed',
      project_id: 'p_new',
      short_name: 'New',
      agent_id: '101',
      project_agent: 'project New, agent 101',
    }}, false);
    _assert.strictEqual(hudState.active_project_id, 'p_new');
    _assert.strictEqual(hudState.pending_project_id, 'p_new');
    _assert.strictEqual(_hudEls.get('hudActiveProjectAgent').textContent, 'project New, agent 101');
    const treeCards = _hudEls.get('hudNodeTree').children;
    _assert.strictEqual(treeCards.some(c => c.dataset && c.dataset.active === 'true'), false);
  `, hudCtx);

  // R3-GAP-2 (hud.js): fetchProjectsHydration returning active_project_id: null without active_project_agent clears stale label
  await vm.runInContext(`
    (async () => {{
      const origFetch = fetch;
      fetch = async () => ({{
        ok: true,
        json: async () => ({{
          active_project_id: null,
          active_agent_id: null,
          projects: [
            {{ id: 'p1', name: 'one', short_name: 'One', active: true, active_agent_id: '101', agents: [{{ agent_id: '101', project_agent: 'project One, agent 101', status: 'idle' }}] }},
          ],
        }}),
      }});
      await fetchProjectsHydration();
      _assert.strictEqual(hudState.explicit_none, true);
      _assert.strictEqual(hudState.pending_project_id, null);
      _assert.strictEqual(hudState.active_project_agent, null);
      _assert.strictEqual(_hudEls.get('hudActiveProjectAgent').textContent, 'No project selected');
      _assert.strictEqual(hudState.projects.some(p => p && p.active), false);
      fetch = origFetch;
    }})()
  `, hudCtx);
}}).catch(err => {{ console.error(err); process.exit(1); }});
"""
    res = subprocess.run(["node", "-e", probe_script], capture_output=True, text=True)
    assert res.returncode == 0, f"Tier-5 frontend probe failed:\n{res.stderr}\n{res.stdout}"


def test_t5_presence_punctuation_rejected_and_cast_select_atomic_sync(
    tmp_path: Path,
) -> None:
    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)
    app.state.cast_manager.register_mock_speaker("Office", host="192.168.1.101")
    app.state.cast_manager.register_mock_speaker("Kitchen", host="192.168.1.102")

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        TestClient(app) as client,
    ):
        # 1. Punctuation-only room rejected with 400 and leaves state untouched
        r_punct = client.post("/api/presence/room", json={"room": "!!!"})
        assert r_punct.status_code == 400
        assert app.state.cast_manager.active_room == "Office"

        # 2. Invalid routing_mode on /api/cast/select does NOT mutate active_room
        r_bad_select = client.post(
            "/api/cast/select",
            json={"room": "Kitchen", "routing_mode": "invalid_mode"},
        )
        assert r_bad_select.status_code == 400
        assert app.state.cast_manager.active_room == "Office"

        # 3. Valid /api/cast/select and /ws/live set_room both emit room_changed to /ws/hud
        with client.websocket_connect("/ws/hud") as hud_ws, client.websocket_connect("/ws/live") as live_ws:
            _ = hud_ws.receive_json()  # hud_snapshot
            _ = live_ws.receive_json()  # connected
            # Drain initial voice_status from live_ws connection on hud_ws
            for _ in range(5):
                m = hud_ws.receive_json()
                if m.get("type") == "voice_status":
                    break

            r_ok_select = client.post(
                "/api/cast/select",
                json={"room": "Kitchen", "routing_mode": "cast"},
            )
            assert r_ok_select.status_code == 200
            ev_select = hud_ws.receive_json()
            assert ev_select["type"] == "room_changed"
            assert ev_select["active_room"] == "Kitchen"
            # Consume forwarded room_changed on live_ws from REST call
            ev_live_select = live_ws.receive_json()
            assert ev_live_select["type"] == "room_changed"

            # /ws/live set_room emits room_changed to /ws/hud without disrupting /ws/live status
            live_ws.send_json({"type": "set_room", "room": "Office"})
            live_status = live_ws.receive_json()
            assert live_status["type"] == "status"
            assert live_status["active_room"] == "Office"

            # Subsequent /ws/live command still gets its direct status cleanly (no self-echo room_changed)
            live_ws.send_json({"type": "set_routing", "mode": "direct"})
            live_route = live_ws.receive_json()
            assert live_route["type"] == "status"
            assert live_route["routing_mode"] == "direct"

            ev_ws_room = hud_ws.receive_json()
            assert ev_ws_room["type"] == "room_changed"
            assert ev_ws_room["active_room"] == "Office"


@pytest.mark.asyncio
async def test_t5_ws_event_forwarder_survives_non_json_payload_and_cast_tasks_cleanup(
    tmp_path: Path,
) -> None:
    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        TestClient(app) as client,
    ):
        with client.websocket_connect("/ws/hud") as hud_ws, client.websocket_connect("/ws/live") as live_ws:
            _ = hud_ws.receive_json()  # hud_snapshot
            _ = live_ws.receive_json()  # connected
            # Drain initial voice_status from live_ws connection on hud_ws
            for _ in range(5):
                m = hud_ws.receive_json()
                if m.get("type") == "voice_status":
                    break

            # Publish non-JSON-native object (Path) followed by normal event
            app.state.event_hub.publish_threadsafe("thought", text="with path", extra_path=Path("/tmp/x"))
            app.state.event_hub.publish_threadsafe("thought", text="after path")

            hud_ev1 = hud_ws.receive_json()
            hud_ev2 = hud_ws.receive_json()
            assert hud_ev1["extra_path"] == "/tmp/x"
            assert hud_ev2["text"] == "after path"

            live_ev1 = live_ws.receive_json()
            live_ev2 = live_ws.receive_json()
            assert live_ev1["extra_path"] == "/tmp/x"
            assert live_ev2["text"] == "after path"

    # Also verify pending cast_tasks in _forward_gemini_to_client are cancelled on exit
    cast_mgr = GoogleHomeCastManager(cfg)
    cast_mgr.register_mock_speaker("Office")
    engine = AntigravityAgentEngine(cfg, cast_mgr)
    bridge = FullDuplexLiveBridge(cfg, engine, cast_mgr)
    fake_ws = AsyncMock()

    cast_started = asyncio.Event()
    cast_cancelled = asyncio.Event()

    async def _blocking_cast(*args: Any, **kwargs: Any) -> dict[str, Any]:
        cast_started.set()
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            cast_cancelled.set()
            raise
        return {"status": "casting"}

    part_audio = MagicMock()
    part_audio.inline_data.data = b"\x02\x00" * 240
    part_audio.text = None
    resp_audio = MagicMock()
    resp_audio.server_content.interrupted = False
    resp_audio.server_content.input_transcription = None
    resp_audio.server_content.output_transcription = None
    resp_audio.server_content.model_turn.parts = [part_audio]
    resp_audio.server_content.turn_complete = True
    resp_audio.tool_call = None

    # Keep session alive briefly so cast_started fires before _forward_gemini_to_client exits
    class DelayedEndSession:
        def __init__(self) -> None:
            self._calls = 0

        def receive(self):
            self._calls += 1

            async def _gen():
                if self._calls == 1:
                    yield resp_audio
                    await asyncio.wait_for(cast_started.wait(), timeout=2.0)

            return _gen()

    with patch.object(cast_mgr, "cast_pcm_audio", side_effect=_blocking_cast):
        await bridge._forward_gemini_to_client(fake_ws, DelayedEndSession())

    assert cast_started.is_set()
    assert cast_cancelled.is_set()


def test_t5_project_manager_deleted_active_project_preserves_fallback_agent_slot(
    tmp_path: Path,
) -> None:
    import shutil
    from agy_home_bridge.project_manager import ProjectManager

    ws = tmp_path / "ws"
    proj_a = ws / "proj-a"
    proj_b = ws / "proj-b"
    proj_a.mkdir(parents=True)
    proj_b.mkdir(parents=True)

    pm = ProjectManager(ws, tmp_path / "save")
    pm.select_project(project="proj-a", agent="102")
    # Ensure agent "103" also exists in proj-a
    pm.get_target(project="proj-a", agent="103")
    # Select proj-b with agent "103"
    pm.select_project(project="proj-b", agent="103")

    # Delete active proj-b and rescan
    shutil.rmtree(proj_b)
    projects = pm.scan_projects()

    assert len(projects) == 1
    assert projects[0]["id"] == "proj-a"
    assert projects[0]["active_agent_id"] == "102"
    summary = pm.get_state_summary()
    assert summary["active_project_id"] == "proj-a"
    assert summary["active_agent_id"] == "102"


@pytest.mark.asyncio
async def test_t5_ws_live_rapid_second_text_does_not_clobber_running_status(
    tmp_path: Path,
) -> None:
    import threading
    from agy_home_bridge.antigravity_agent import AgentTurnResult
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)

    # 1. Direct ProjectManager task-token guard prevents stale task from clobbering newer task
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    assert pm.update_agent_status("alpha-service", "101", status="running", current_task="first", task_id="tok-1")
    assert pm.update_agent_status("alpha-service", "101", status="running", current_task="second", task_id="tok-2")
    assert pm.update_agent_status("alpha-service", "101", status="idle", current_task=None, task_id="tok-1") is False
    alpha = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
    assert alpha["agents"][0]["status"] == "running"
    assert alpha["agents"][0]["current_task"] == "second"
    assert pm.update_agent_status("alpha-service", "101", status="idle", current_task=None, last_summary="done", task_id="tok-2") is True

    # 2. Rapid /ws/live text -> text where turn 1 yields during cancellation cleanup
    app = create_app(cfg)
    turn1_started = threading.Event()
    turn2_started = threading.Event()
    release_turn2 = threading.Event()
    call_count = 0

    async def _exec_turns(prompt_text: str, **kwargs: Any) -> tuple[AgentTurnResult, None]:
        nonlocal call_count
        call_count += 1
        idx = call_count
        if idx == 1:
            turn1_started.set()
            try:
                await asyncio.sleep(10.0)
            finally:
                # Simulate async SDK context-manager cleanup yielding
                await asyncio.sleep(0.05)
        else:
            turn2_started.set()
            while not release_turn2.is_set():
                await asyncio.sleep(0.01)
            return (AgentTurnResult(text="done-2", conversation_id="conv-2"), None)
        return (AgentTurnResult(text="unreachable", conversation_id="conv-1"), None)

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        patch.object(app.state.agent_engine, "_execute_agent_turn", side_effect=_exec_turns),
        TestClient(app) as client,
    ):
        with client.websocket_connect("/ws/live") as lws:
            lws.receive_json()  # connected
            lws.send_json({"type": "text", "text": "first slow task"})
            assert turn1_started.wait(timeout=2.0)

            lws.send_json({"type": "text", "text": "second task"})
            assert turn2_started.wait(timeout=2.0)

            # Give turn 1 cleanup window to finish if it had raced
            await asyncio.sleep(0.08)
            mid_proj = next(
                p for p in client.get("/api/projects").json()["projects"] if p["id"] == "alpha-service"
            )
            assert mid_proj["agents"][0]["status"] == "running"
            assert mid_proj["agents"][0]["current_task"] == "second task"

            release_turn2.set()
            got_done = False
            for _ in range(10):
                msg = lws.receive_json()
                if msg.get("type") == "transcript" and msg.get("role") == "agent" and msg.get("text") == "done-2":
                    got_done = True
                    break
            assert got_done

        after_proj = next(
            p for p in client.get("/api/projects").json()["projects"] if p["id"] == "alpha-service"
        )
        assert after_proj["agents"][0]["status"] == "idle"
        assert after_proj["agents"][0]["current_task"] is None
        assert after_proj["agents"][0]["last_summary"] == "done-2"


def test_t5_fulfillment_and_mcp_nested_type_validation(client: TestClient) -> None:
    # Smart Home fulfillment malformed nested types -> 400 (never 500)
    malformed_fulfillment_payloads = [
        {"inputs": "bad"},
        {"inputs": [123]},
        {"inputs": [{"intent": 123}]},
        {"inputs": [{"intent": "action.devices.QUERY", "payload": "bad"}]},
        {"inputs": [{"intent": "action.devices.QUERY", "payload": {"devices": "bad"}}]},
        {"inputs": [{"intent": "action.devices.QUERY", "payload": {"devices": [123]}}]},
        {"inputs": [{"intent": "action.devices.EXECUTE", "payload": {"commands": "bad"}}]},
        {"inputs": [{"intent": "action.devices.EXECUTE", "payload": {"commands": [123]}}]},
        {
            "inputs": [
                {
                    "intent": "action.devices.EXECUTE",
                    "payload": {"commands": [{"devices": ["bad"], "execution": []}]},
                }
            ]
        },
        {
            "inputs": [
                {
                    "intent": "action.devices.EXECUTE",
                    "payload": {"commands": [{"devices": [{"id": "agy-live-assistant"}], "execution": "bad"}]},
                }
            ]
        },
        # R3-GAP-1: customData.targetRoom non-string or invalid/whitespace on agy-follow-* -> 400
        {
            "inputs": [
                {
                    "intent": "action.devices.EXECUTE",
                    "payload": {
                        "commands": [
                            {
                                "devices": [{"id": "agy-follow-office", "customData": {"targetRoom": 123}}],
                                "execution": [{"command": "action.devices.commands.ActivateScene", "params": {}}],
                            }
                        ]
                    },
                }
            ]
        },
        {
            "inputs": [
                {
                    "intent": "action.devices.EXECUTE",
                    "payload": {
                        "commands": [
                            {
                                "devices": [{"id": "agy-follow-office", "customData": {"targetRoom": {"r": "office"}}}],
                                "execution": [{"command": "action.devices.commands.ActivateScene", "params": {}}],
                            }
                        ]
                    },
                }
            ]
        },
        {
            "inputs": [
                {
                    "intent": "action.devices.EXECUTE",
                    "payload": {
                        "commands": [
                            {
                                "devices": [{"id": "agy-follow-office", "customData": {"targetRoom": "   "}}],
                                "execution": [{"command": "action.devices.commands.ActivateScene", "params": {}}],
                            }
                        ]
                    },
                }
            ]
        },
        {
            "inputs": [
                {
                    "intent": "action.devices.EXECUTE",
                    "payload": {
                        "commands": [
                            {
                                "devices": [{"id": "agy-follow-office", "customData": {"targetRoom": "!!!"}}],
                                "execution": [{"command": "action.devices.commands.ActivateScene", "params": {}}],
                            }
                        ]
                    },
                }
            ]
        },
        {
            "inputs": [
                {
                    "intent": "action.devices.EXECUTE",
                    "payload": {
                        "commands": [
                            {
                                "devices": [{"id": "agy-follow-!!!"}],
                                "execution": [{"command": "action.devices.commands.ActivateScene", "params": {}}],
                            }
                        ]
                    },
                }
            ]
        },
    ]
    for payload in malformed_fulfillment_payloads:
        resp = client.post("/api/google-home/fulfillment", json=payload)
        assert resp.status_code == 400, f"Expected 400 for {payload!r}, got {resp.status_code}"

    # Empty string targetRoom falls back cleanly to agy-follow- slug
    r_follow_empty = client.post(
        "/api/google-home/fulfillment",
        json={
            "inputs": [
                {
                    "intent": "action.devices.EXECUTE",
                    "payload": {
                        "commands": [
                            {
                                "devices": [{"id": "agy-follow-office", "customData": {"targetRoom": ""}}],
                                "execution": [{"command": "action.devices.commands.ActivateScene", "params": {}}],
                            }
                        ]
                    },
                }
            ]
        },
    )
    assert r_follow_empty.status_code == 200
    assert client.app.state.cast_manager.active_room == "Office"  # type: ignore[attr-defined]

    # Raw JSON NaN in fulfillment body -> 400
    r_nan_ff = client.post(
        "/api/google-home/fulfillment",
        content='{"inputs": [], "x": NaN}',
        headers={"Content-Type": "application/json"},
    )
    assert r_nan_ff.status_code == 400

    # MCP malformed nested types and ValueError tool params -> 400 with structured JSON-RPC error
    r_bad_params = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": "bad"},
    )
    assert r_bad_params.status_code == 400
    assert r_bad_params.json()["error"]["code"] == -32602

    r_bad_method = client.post(
        "/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": 123, "params": {}},
    )
    assert r_bad_method.status_code == 400
    assert r_bad_method.json()["error"]["code"] == -32600

    r_bad_args = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "speak_on_google_home", "arguments": "bad"},
        },
    )
    assert r_bad_args.status_code == 400
    assert r_bad_args.json()["error"]["code"] == -32602

    r_blank_msg = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "speak_on_google_home", "arguments": {"message": "   "}},
        },
    )
    assert r_blank_msg.status_code == 400
    assert r_blank_msg.json()["error"]["code"] == -32602

    # R3-GAP-1 MCP parity: NaN volume rejected (-32700 over HTTP, -32602 in handler), punctuation room rejected (-32602)
    r_nan_mcp = client.post(
        "/mcp",
        content='{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"speak_on_google_home","arguments":{"message":"hi","volume":NaN}}}',
        headers={"Content-Type": "application/json"},
    )
    assert r_nan_mcp.status_code == 400
    assert r_nan_mcp.json()["error"]["code"] == -32700

    r_punct_mcp = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {"name": "switch_active_google_home_room", "arguments": {"room": "!!!"}},
        },
    )
    assert r_punct_mcp.status_code == 400
    assert r_punct_mcp.json()["error"]["code"] == -32602


def test_t5_ws_forwarders_survive_circular_nan_and_tuple_key_events(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.antigravity_agent import _safe_json_value

    # 1. _safe_json_value handles circular refs, shared DAG refs, NaN/Inf, tuple keys, and max_depth
    circ: dict[str, Any] = {"ok": 1}
    circ["self"] = circ
    shared = {"x": 42}
    dag = {"a": shared, "b": shared}
    safe_circ = _safe_json_value(circ)
    assert safe_circ["ok"] == 1
    assert safe_circ["self"] == "<circular reference>"
    assert _safe_json_value(dag) == {"a": {"x": 42}, "b": {"x": 42}}
    assert _safe_json_value(float("nan")) == "nan"
    assert _safe_json_value(float("inf")) == "inf"
    assert _safe_json_value({(1, 2): "v"}) == {"(1, 2)": "v"}

    deep: dict[str, Any] = {"leaf": "end"}
    for _ in range(12):
        deep = {"nest": deep}
    json.dumps(_safe_json_value(deep), allow_nan=False)

    # 2. /ws/hud and /ws/live forwarders survive circular refs, tuple keys, and NaN with strict JSON
    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)

    def _reject_nan(val: str) -> Any:
        raise ValueError(f"Non-standard JSON constant {val!r}")

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        TestClient(app) as client,
    ):
        with client.websocket_connect("/ws/hud") as hud_ws, client.websocket_connect("/ws/live") as live_ws:
            _ = hud_ws.receive_json()  # hud_snapshot
            _ = live_ws.receive_json()  # connected
            for _ in range(5):
                m = hud_ws.receive_json()
                if m.get("type") == "voice_status":
                    break

            app.state.event_hub.publish_threadsafe("thought", text="circ", bad=circ)
            app.state.event_hub.publish_threadsafe("thought", text="tuplekey", bad={(1, 2): "v"})
            app.state.event_hub.publish_threadsafe("thought", text="nan", bad=float("nan"))
            app.state.event_hub.publish_threadsafe("thought", text="still alive")

            hud_texts = [hud_ws.receive_text() for _ in range(4)]
            live_texts = [live_ws.receive_text() for _ in range(4)]

            hud_decoded = [json.loads(t, parse_constant=_reject_nan) for t in hud_texts]
            live_decoded = [json.loads(t, parse_constant=_reject_nan) for t in live_texts]

            assert [e["text"] for e in hud_decoded] == ["circ", "tuplekey", "nan", "still alive"]
            assert [e["text"] for e in live_decoded] == ["circ", "tuplekey", "nan", "still alive"]
            assert hud_decoded[0]["bad"]["self"] == "<circular reference>"
            assert hud_decoded[1]["bad"] == {"(1, 2)": "v"}
            assert hud_decoded[2]["bad"] == "nan"


def test_t5_cast_select_and_ws_set_room_reject_punctuation_only_room(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.gemini_live import _BoundedEventIdSet

    cfg = _make_m2_settings(tmp_path)
    app = create_app(cfg)
    app.state.cast_manager.register_mock_speaker("Office", host="192.168.1.101")

    with (
        patch.object(app.state.cast_manager, "discover_speakers", new_callable=AsyncMock, return_value=[]),
        patch.object(app.state.cast_manager, "schedule_discovery_resilience"),
        patch.object(app.state.cast_manager, "cancel_discovery_resilience"),
        TestClient(app) as client,
    ):
        r_select = client.post("/api/cast/select", json={"room": "!!!"})
        assert r_select.status_code == 400
        assert app.state.cast_manager.active_room == "Office"

        with client.websocket_connect("/ws/live") as live_ws:
            _ = live_ws.receive_json()  # connected
            live_ws.send_json({"type": "set_room", "room": "!!!"})
            err_msg = live_ws.receive_json()
            assert err_msg["type"] == "error"
            assert "alphanumeric" in err_msg["error"].lower()
            assert app.state.cast_manager.active_room == "Office"

    # Bounded local_emitted_event_ids set never exceeds maxlen on unconsumed evictions
    bounded = _BoundedEventIdSet(maxlen=256)
    for i in range(500):
        bounded.add(f"ev-{i}")
    assert len(bounded) == 256
    assert "ev-0" not in bounded
    assert "ev-499" in bounded


@pytest.mark.asyncio
async def test_t5_concurrent_background_tasks_out_of_order_completion_and_ask_start_cancel(
    tmp_path: Path,
) -> None:
    from agy_home_bridge.antigravity_agent import AgentTurnResult
    from agy_home_bridge.event_hub import EventHub
    from agy_home_bridge.mcp_server import BridgeMCPServer
    from agy_home_bridge.project_manager import ProjectManager

    cfg = _make_m2_settings(tmp_path)
    pm = ProjectManager(cfg.agy_workspace_dir, cfg.agy_save_dir)
    hub = EventHub()
    cast_mgr = GoogleHomeCastManager(cfg)
    engine = AntigravityAgentEngine(cfg, cast_mgr, project_manager=pm, event_hub=hub)

    # Also verify BridgeMCPServer direct NaN rejection (-32602)
    mcp_srv = BridgeMCPServer(cast_mgr, engine)
    nan_resp = await mcp_srv.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {
                "name": "speak_on_google_home",
                "arguments": {"message": "hi", "volume": float("nan")},
            },
        }
    )
    assert nan_resp["error"]["code"] == -32602

    # 1. R3-GAP-3: task1 (slow) and task2 (fast) run concurrently on (alpha-service, 101).
    # task2 completes first; task1 fails later -> task1 MUST NOT overwrite task2's idle/last_summary.
    t1_started = asyncio.Event()
    release_t1_fail = asyncio.Event()
    t2_done = asyncio.Event()
    call_idx = 0

    async def _exec_out_of_order(prompt: str, **kwargs: Any) -> tuple[AgentTurnResult, None]:
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            t1_started.set()
            await release_t1_fail.wait()
            raise RuntimeError("task1 stale failure")
        return (AgentTurnResult(text="task2 succeeded", conversation_id="conv-t2"), None)

    q = hub.subscribe()
    try:
        with (
            patch.object(engine, "_execute_agent_turn", side_effect=_exec_out_of_order),
            patch.object(cast_mgr, "speak_on_room", new_callable=AsyncMock, return_value={"status": "ok"}),
        ):
            ack1 = engine.start_background_task("slow task 1", project="alpha-service", agent="101")
            await asyncio.wait_for(t1_started.wait(), timeout=2.0)

            ack2 = engine.start_background_task("fast task 2", project="alpha-service", agent="101")
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=2.0)
                if ev.get("task_id") == ack2["task_id"] and ev.get("type") == "task_completed":
                    t2_done.set()
                    break
            assert t2_done.is_set()
            await asyncio.sleep(0.02)

            alpha_mid = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
            assert alpha_mid["agents"][0]["status"] == "idle"
            assert alpha_mid["agents"][0]["current_task"] is None
            assert alpha_mid["agents"][0]["last_summary"] == "task2 succeeded"

            # Now let slow task1 fail
            release_t1_fail.set()
            while True:
                ev = await asyncio.wait_for(q.get(), timeout=2.0)
                if ev.get("task_id") == ack1["task_id"] and ev.get("type") == "task_failed":
                    break
            await asyncio.sleep(0.02)

            alpha_after = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
            assert alpha_after["agents"][0]["status"] == "idle"
            assert alpha_after["agents"][0]["current_task"] is None
            assert alpha_after["agents"][0]["last_summary"] == "task2 succeeded"
    finally:
        hub.unsubscribe(q)
        await engine.shutdown()

    # 2. R3-GAP-3: ask() cancelled while awaiting _emit_event("task_started") resets status="idle"
    emit_entered = asyncio.Event()
    emit_release = asyncio.Event()

    async def _blocking_emit(event_type: str, **kwargs: Any) -> None:
        if event_type == "task_started":
            emit_entered.set()
            await emit_release.wait()

    with patch.object(engine, "_emit_event", side_effect=_blocking_emit):
        ask_task = asyncio.create_task(engine.ask("cancel during task_started", emit_events=True))
        await asyncio.wait_for(emit_entered.wait(), timeout=2.0)
        alpha_running = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
        assert alpha_running["agents"][0]["status"] == "running"

        ask_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await ask_task

    alpha_cancelled = next(p for p in pm.get_state_summary()["projects"] if p["id"] == "alpha-service")
    assert alpha_cancelled["agents"][0]["status"] == "idle"
    assert alpha_cancelled["agents"][0]["current_task"] is None


@pytest.mark.asyncio
async def test_speak_on_room_prefers_gemini_voice_and_falls_back_to_gtts(
    test_settings: BridgeSettings, tmp_path: Path
) -> None:
    assert test_settings.gemini_live_voice == "Giro"
    cast_mgr = GoogleHomeCastManager(test_settings)
    cast_mgr.register_mock_speaker("Office speaker", host="192.168.1.121")
    cast_mgr._chromecasts["Office speaker"] = MagicMock()

    with (
        patch.object(
            cast_mgr,
            "synthesize_gemini_pcm",
            new_callable=AsyncMock,
            return_value=b"\x01\x00" * 240,
        ) as mock_synth,
        patch.object(
            cast_mgr,
            "cast_pcm_audio",
            new_callable=AsyncMock,
            return_value={
                "status": "casting",
                "room": "Office speaker",
                "media_url": "http://192.168.1.50:8765/api/tts/resp.wav",
                "detail": "ok",
            },
        ) as mock_cast_pcm,
        patch.object(cast_mgr, "generate_tts_file") as mock_gtts,
    ):
        res = await cast_mgr.speak_on_room("Task completed.", room_name="office")

    mock_synth.assert_awaited_once_with("Task completed.")
    mock_cast_pcm.assert_awaited_once()
    mock_gtts.assert_not_called()
    assert res["status"] == "casting"
    assert res["voice"] == "Giro"

    fake_mp3 = (tmp_path / "tts_cache" / "fallback.mp3").resolve()
    fake_mp3.parent.mkdir(parents=True, exist_ok=True)
    fake_mp3.write_bytes(b"MP3")

    with (
        patch.object(
            cast_mgr,
            "synthesize_gemini_pcm",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch.object(cast_mgr, "generate_tts_file", return_value=fake_mp3) as mock_gtts_fb,
    ):
        res_fb = await cast_mgr.speak_on_room("Fallback test", room_name="office")

    mock_gtts_fb.assert_called_once_with("Fallback test")
    assert res_fb["status"] == "casting"
    assert res_fb["media_url"].endswith("/fallback.mp3")


@pytest.mark.asyncio
async def test_voice_selector_api_hud_snapshot_and_static_markup(
    test_settings: BridgeSettings,
) -> None:
    import httpx

    app = create_app(test_settings)
    q = app.state.event_hub.subscribe()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            r_health = await client.get("/api/health")
            assert r_health.status_code == 200
            h = r_health.json()
            assert h["voice_name"] == "Giro"
            assert "Giro" in h["available_voices"]
            assert "Puck" in h["available_voices"]

            r_get = await client.get("/api/voice")
            assert r_get.status_code == 200
            assert r_get.json()["voice_name"] == "Giro"

            r_bad = await client.post("/api/voice", json={"voice_name": "NotARealVoice"})
            assert r_bad.status_code == 400

            r_set = await client.post("/api/voice", json={"voice_name": "kore"})
            assert r_set.status_code == 200
            assert r_set.json()["voice_name"] == "Kore"
            assert test_settings.gemini_live_voice == "Kore"

            ev = await asyncio.wait_for(q.get(), timeout=2.0)
            assert ev["type"] == "voice_changed"
            assert ev["voice_name"] == "Kore"

            snap = app.state.event_hub.build_snapshot(
                app.state.project_manager,
                app.state.cast_manager,
                voice_status=app.state.live_bridge.voice_status,
                voice_muted=app.state.live_bridge.voice_muted,
                voice_name=app.state.live_bridge.voice_name,
            )
            assert snap["voice_name"] == "Kore"

            r_pwa = await client.get("/")
            assert r_pwa.status_code == 200
            assert 'id="voiceSelect"' in r_pwa.text

            r_hud = await client.get("/hud")
            assert r_hud.status_code == 200
            assert 'id="hudVoiceNameLabel"' in r_hud.text
            assert "<select" not in r_hud.text
    finally:
        app.state.event_hub.unsubscribe(q)


@pytest.mark.asyncio
async def test_user_prompt_audio_saved_and_linked_to_transcript(
    test_settings: BridgeSettings,
) -> None:
    import struct

    app = create_app(test_settings)
    bridge = app.state.live_bridge
    q = app.state.event_hub.subscribe()

    # Build a 16kHz PCM buffer with leading silence + active tone
    silence = b"\x00\x00" * 3200
    tone = struct.pack("<h", 4000) * 3200
    user_pcm_ring = bytearray(silence + tone + silence)

    fake_ws = MagicMock()
    sent_payloads: list[dict[str, Any]] = []

    async def _fake_send_text(text: str) -> None:
        sent_payloads.append(json.loads(text))

    fake_ws.send_text = AsyncMock(side_effect=_fake_send_text)

    class _FakeResp:
        def __init__(self, input_text: str | None = None, output_text: str | None = None, turn_complete: bool = False):
            sc = MagicMock()
            sc.interrupted = False
            sc.input_transcription = MagicMock(text=input_text) if input_text else None
            sc.output_transcription = MagicMock(text=output_text) if output_text else None
            sc.model_turn = None
            sc.turn_complete = turn_complete
            self.server_content = sc
            self.tool_call = None

    class _FakeSession:
        def __init__(self) -> None:
            self._done = False

        async def receive(self):
            if self._done:
                return
            self._done = True
            yield _FakeResp(input_text="Can you hear me?")
            yield _FakeResp(output_text="Yes, loud and clear.", turn_complete=True)

    try:
        await bridge._forward_gemini_to_client(
            fake_ws,
            _FakeSession(),
            user_pcm_ring=user_pcm_ring,
        )
        audio_msgs = [m for m in sent_payloads if m.get("type") == "transcript_audio"]
        assert len(audio_msgs) == 1
        audio_url = audio_msgs[0]["audio_url"]
        assert audio_url.startswith("/api/tts/prompt_") and audio_url.endswith(".wav")

        # Verify the saved WAV file exists and is non-empty
        wav_name = audio_url.rsplit("/", 1)[-1]
        wav_path = test_settings.tts_cache_dir / wav_name
        assert wav_path.is_file()
        assert wav_path.stat().st_size > 44

        # Verify EventHub also received transcript_audio for other PWA tabs
        hub_events: list[dict[str, Any]] = []
        while not q.empty():
            hub_events.append(q.get_nowait())
        assert any(
            e.get("type") == "transcript_audio" and e.get("audio_url") == audio_url
            for e in hub_events
        )
    finally:
        app.state.event_hub.unsubscribe(q)


@pytest.mark.asyncio
async def test_streaming_resampler_and_audio_stream_end_forwarding(
    test_settings: BridgeSettings,
) -> None:
    import math
    import struct
    from agy_home_bridge.hardware_mic import StreamingResampler, compute_pcm16_rms

    # 40ms of 1kHz sine at 48kHz (1920 samples) at low amplitude (~2000)
    in_samples = [
        int(2000 * math.sin(2.0 * math.pi * 1000.0 * i / 48000.0))
        for i in range(1920)
    ]
    pcm_48k = struct.pack(f"<{len(in_samples)}h", *in_samples)

    resampler = StreamingResampler(in_rate=48000, out_rate=16000, gain=4.0)
    out_16k = resampler.process(pcm_48k)
    assert len(out_16k) == 640 * 2  # 40ms @ 16kHz
    assert compute_pcm16_rms(out_16k) > compute_pcm16_rms(pcm_48k) * 2.5

    # Verify audio_stream_end forwards audio_stream_end=True to Gemini Live session
    app = create_app(test_settings)
    bridge = app.state.live_bridge
    fake_ws = MagicMock()
    frames = iter(
        [
            {"type": "websocket.receive", "text": json.dumps({"type": "audio_stream_end"})},
            {"type": "websocket.disconnect"},
        ]
    )
    fake_ws.receive = AsyncMock(side_effect=lambda: next(frames))
    fake_ws.send_text = AsyncMock()
    fake_session = MagicMock()
    fake_session.send_realtime_input = AsyncMock()

    await bridge._forward_client_to_gemini(fake_ws, fake_session)
    fake_session.send_realtime_input.assert_awaited_once_with(audio_stream_end=True)

















