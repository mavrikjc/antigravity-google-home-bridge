"""Configuration management for Antigravity Google Home Bridge."""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def detect_lan_ip() -> str:
    """Detect the primary LAN IPv4 address of this host so Google Home speakers can fetch audio."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"


AVAILABLE_GEMINI_VOICES: tuple[str, ...] = (
    "Giro",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Aoede",
    "Leda",
    "Orus",
    "Zephyr",
)


class BridgeSettings(BaseSettings):
    """Application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Gemini & Antigravity SDK
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_live_model: str = Field(
        default="gemini-3.8-live", alias="GEMINI_LIVE_MODEL"
    )
    # Set true only for Vertex AI Express Mode keys; AI Studio keys (AIza... or AQ...) use the Gemini Developer API.
    gemini_use_vertex: bool = Field(default=False, alias="GEMINI_USE_VERTEX")
    gemini_live_voice: str = Field(default="Giro", alias="GEMINI_LIVE_VOICE")
    # How long a pause ends your turn. Lower = snappier replies, but may cut you off mid-thought.
    gemini_live_silence_ms: int = Field(default=1000, alias="GEMINI_LIVE_SILENCE_MS")

    agy_workspace_dir: str = Field(
        default_factory=lambda: os.getcwd(), alias="AGY_WORKSPACE_DIR"
    )
    agy_save_dir: str = Field(
        default_factory=lambda: str(Path.cwd() / ".agy_home_data" / "conversations"),
        alias="AGY_SAVE_DIR",
    )
    agy_app_data_dir: str = Field(
        default_factory=lambda: str(Path.cwd() / ".agy_home_data" / "brain"),
        alias="AGY_APP_DATA_DIR",
    )
    agy_enable_write_tools: bool = Field(default=True, alias="AGY_ENABLE_WRITE_TOOLS")
    # "Turbo mode": let the agent run any shell command without confirmation (policy.allow_all()).
    # Off by default; voice sessions have no one to approve commands, so they'd otherwise be denied.
    agy_allow_all_commands: bool = Field(default=False, alias="AGY_ALLOW_ALL_COMMANDS")
    agy_enable_sandbox: bool = Field(default=False, alias="AGY_ENABLE_SANDBOX")

    # Local Antigravity Desktop / Language Server bridge (optional)
    antigravity_ls_address: str = Field(default="", alias="ANTIGRAVITY_LS_ADDRESS")
    antigravity_csrf_token: str = Field(default="", alias="ANTIGRAVITY_CSRF_TOKEN")

    # Server networking
    host_lan_ip: str = Field(default="", alias="HOST_LAN_IP")
    bridge_host: str = Field(default="0.0.0.0", alias="BRIDGE_HOST")
    bridge_port: int = Field(default=8765, alias="BRIDGE_PORT")
    bridge_https_port: int = Field(default=8443, alias="BRIDGE_HTTPS_PORT")
    enable_https: bool = Field(default=True, alias="ENABLE_HTTPS")

    # Google Home / Cast multi-room settings
    default_cast_room: str = Field(default="Whole Home", alias="DEFAULT_CAST_ROOM")
    default_audio_routing: Literal["direct", "cast", "both"] = Field(
        default="cast", alias="DEFAULT_AUDIO_ROUTING"
    )
    known_cast_hosts: str = Field(default="", alias="KNOWN_CAST_HOSTS")

    # Google Home Cloud-to-Cloud Smart Home webhook auth
    smart_home_webhook_token: str = Field(
        default="change-me-to-a-secure-token", alias="SMART_HOME_WEBHOOK_TOKEN"
    )

    # Proxmox Hardware Wireless Mic Daemon
    hw_mic_enabled: bool = Field(default=False, alias="HW_MIC_ENABLED")
    hw_mic_device_index: int | None = Field(default=None, alias="HW_MIC_DEVICE_INDEX")
    hw_mic_device_name: str = Field(default="", alias="HW_MIC_DEVICE_NAME")
    hw_mic_sample_rate: int = Field(default=16000, alias="HW_MIC_SAMPLE_RATE")
    hw_mic_gain: float = Field(default=2.0, ge=0.5, le=16.0, alias="HW_MIC_GAIN")
    hw_mic_vad_threshold: float = Field(default=0.025, alias="HW_MIC_VAD_THRESHOLD")
    hw_mic_silence_duration_sec: float = Field(
        default=1.4, alias="HW_MIC_SILENCE_DURATION_SEC"
    )

    # JARVIS Mode: Room presence, TV casting, Project Manager & EventHub
    room_device_map: Annotated[dict[str, Any], NoDecode] = Field(
        default_factory=dict,
        alias="ROOM_DEVICE_MAP",
    )
    default_presence_room: str = Field(
        default="",
        alias="DEFAULT_PRESENCE_ROOM",
    )
    cast_discovery_timeout: float = Field(
        default=6.0,
        ge=0.5,
        alias="CAST_DISCOVERY_TIMEOUT",
    )
    cast_tv_timeout: float = Field(
        default=5.0,
        ge=0.5,
        alias="CAST_TV_TIMEOUT",
    )
    hud_cast_force: bool = Field(
        default=True,
        alias="HUD_CAST_FORCE",
    )
    projects_state_file: str = Field(
        default="",
        alias="PROJECTS_STATE_FILE",
    )
    default_agent_id: str = Field(
        default="101",
        alias="DEFAULT_AGENT_ID",
    )
    event_history_size: int = Field(
        default=200,
        ge=1,
        alias="EVENT_HISTORY_SIZE",
    )

    @field_validator("hw_mic_device_index", mode="before")
    @classmethod
    def _empty_str_to_none(cls, val: Any) -> Any:
        if val == "" or (isinstance(val, str) and not val.strip()):
            return None
        return val

    @field_validator("room_device_map", mode="before")
    @classmethod
    def _parse_room_device_map(cls, val: Any) -> dict[str, Any]:
        if val is None:
            return {}
        if isinstance(val, str):
            raw = val.strip()
            if not raw:
                return {}
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("ROOM_DEVICE_MAP JSON must decode to an object")
            return parsed
        if isinstance(val, dict):
            return dict(val)
        raise ValueError("room_device_map must be a dict or JSON object string")

    @property
    def resolved_lan_ip(self) -> str:
        """Return explicit HOST_LAN_IP or auto-detected LAN IP."""
        return self.host_lan_ip.strip() or detect_lan_ip()

    @property
    def base_lan_url(self) -> str:
        """Return base HTTP URL reachable by Google Home speakers on the LAN."""
        return f"http://{self.resolved_lan_ip}:{self.bridge_port}"

    @property
    def https_lan_url(self) -> str:
        """Return HTTPS URL for mobile browsers (iOS Safari / Android Chrome) that require a Secure Context for mic access."""
        return f"https://{self.resolved_lan_ip}:{self.bridge_https_port}"

    @property
    def certs_dir(self) -> Path:
        """Directory for storing the auto-generated LAN Root CA and TLS certificates."""
        path = Path(self.agy_app_data_dir).parent / "certs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def tts_cache_dir(self) -> Path:
        """Directory for storing generated TTS and raw audio files served to Google Home."""
        path = Path(self.agy_app_data_dir).parent / "tts_cache"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def parsed_known_cast_hosts(self) -> list[str]:
        """Parse comma-separated list of known Google Home IP addresses."""
        if not self.known_cast_hosts.strip():
            return []
        return [
            host.strip()
            for host in self.known_cast_hosts.split(",")
            if host.strip()
        ]

    @property
    def resolved_projects_state_file(self) -> Path:
        """Return explicit PROJECTS_STATE_FILE or default `<agy_save_dir>/projects_state.json`."""
        if self.projects_state_file.strip():
            return Path(self.projects_state_file.strip())
        return Path(self.agy_save_dir) / "projects_state.json"

    @property
    def hud_lan_url(self) -> str:
        """Return plain HTTP LAN URL for `/hud` (required for Chromecast DashCast / TV browsers)."""
        return f"{self.base_lan_url}/hud"


settings = BridgeSettings()
