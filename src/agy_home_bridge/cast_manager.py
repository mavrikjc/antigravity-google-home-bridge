"""Multi-room Google Home & Chromecast discovery and audio/TTS streamer."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import time
import uuid
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from gtts import gTTS

from agy_home_bridge.config import BridgeSettings

logger = logging.getLogger(__name__)


@dataclass
class CastSpeakerInfo:
    """Metadata for a discovered Google Home speaker or Cast speaker group."""

    name: str
    uuid: str
    model_name: str
    cast_type: str
    host: str
    port: int
    is_active_target: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GoogleHomeCastManager:
    """Discovers Google Home speakers and speaker groups on the LAN and casts speech or PCM audio."""

    def __init__(self, config: BridgeSettings) -> None:
        self.config = config
        self._speakers: dict[str, CastSpeakerInfo] = {}
        self._chromecasts: dict[str, Any] = {}
        self._browser: Any = None
        self._active_room: str = config.default_cast_room
        self._routing_mode: str = config.default_audio_routing
        self._lock = asyncio.Lock()

    @property
    def active_room(self) -> str:
        return self._active_room

    @property
    def routing_mode(self) -> str:
        return self._routing_mode

    def set_routing_mode(self, mode: str) -> str:
        if mode not in ("direct", "cast", "both"):
            raise ValueError("routing_mode must be 'direct', 'cast', or 'both'")
        self._routing_mode = mode
        return self._routing_mode

    def set_active_room(self, room_name: str) -> str:
        """Switch the active room or speaker group (fuzzy matched against discovered speakers)."""
        matched = self.resolve_room_name(room_name)
        self._active_room = matched or room_name
        for speaker in self._speakers.values():
            speaker.is_active_target = (
                speaker.name.lower() == self._active_room.lower()
            )
        logger.info("Active Google Home target room set to: %s", self._active_room)
        return self._active_room

    def resolve_room_name(self, query: str) -> str | None:
        """Fuzzy-match a requested room name against discovered Google Home speakers/groups."""
        q = query.strip().lower()
        if not q:
            return None
        # Exact match first
        for name in self._speakers:
            if name.lower() == q:
                return name
        # Substring match next (e.g. "kitchen" matches "Kitchen Speaker")
        for name in self._speakers:
            if q in name.lower() or name.lower() in q:
                return name
        return None

    def list_speakers(self) -> list[dict[str, Any]]:
        """Return all discovered Google Home speakers and speaker groups."""
        results = []
        for s in sorted(self._speakers.values(), key=lambda x: x.name.lower()):
            s.is_active_target = s.name.lower() == self._active_room.lower()
            results.append(s.to_dict())
        return results

    async def discover_speakers(self, timeout: float = 4.0) -> list[dict[str, Any]]:
        """Run mDNS / Known-Host discovery for Google Home speakers on the LAN."""
        async with self._lock:
            return await asyncio.to_thread(self._discover_sync, timeout)

    def _discover_sync(self, timeout: float) -> list[dict[str, Any]]:
        try:
            import pychromecast
        except ImportError:
            logger.warning("pychromecast not installed; skipping LAN Cast discovery.")
            return self.list_speakers()

        try:
            if self._browser is not None:
                try:
                    self._browser.stop_discovery()
                except Exception:
                    pass

            known_hosts = self.config.parsed_known_cast_hosts or None
            chromecasts, browser = pychromecast.get_chromecasts(
                timeout=timeout,
                known_hosts=known_hosts,
            )
            self._browser = browser
            self._speakers.clear()
            self._chromecasts.clear()

            for cc in chromecasts:
                info = cc.cast_info
                name = info.friendly_name or f"Google Home ({info.host})"
                speaker = CastSpeakerInfo(
                    name=name,
                    uuid=str(info.uuid),
                    model_name=info.model_name or "Google Cast Device",
                    cast_type=info.cast_type or "audio",
                    host=info.host,
                    port=info.port,
                    is_active_target=(name.lower() == self._active_room.lower()),
                )
                self._speakers[name] = speaker
                self._chromecasts[name] = cc

            # If default room didn't match any speaker yet and we found speakers, pick best match
            if self._speakers and not self.resolve_room_name(self._active_room):
                first_name = next(iter(self._speakers.keys()))
                self._active_room = first_name

            logger.info(
                "Discovered %d Google Home / Cast device(s): %s",
                len(self._speakers),
                list(self._speakers.keys()),
            )
        except Exception as exc:
            logger.warning("Google Home LAN discovery encountered an error: %s", exc)

        return self.list_speakers()

    def register_mock_speaker(
        self,
        name: str,
        model_name: str = "Google Nest Audio",
        cast_type: str = "audio",
        host: str = "192.168.1.100",
        port: int = 8009,
    ) -> CastSpeakerInfo:
        """Register a virtual/mock Google Home speaker (helpful for testing or manual overrides)."""
        speaker = CastSpeakerInfo(
            name=name,
            uuid=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}-{host}")),
            model_name=model_name,
            cast_type=cast_type,
            host=host,
            port=port,
            is_active_target=(name.lower() == self._active_room.lower()),
        )
        self._speakers[name] = speaker
        return speaker

    def generate_tts_file(self, text: str, lang: str = "en") -> Path:
        """Synthesize clean speech MP3 for casting to Google Home speakers."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text")

        digest = hashlib.sha256(f"{lang}:{clean_text}".encode("utf-8")).hexdigest()[:20]
        filename = f"tts_{digest}.mp3"
        filepath = self.config.tts_cache_dir / filename

        if not filepath.exists():
            tts = gTTS(text=clean_text, lang=lang, slow=False)
            tts.save(str(filepath))
        return filepath

    def save_pcm_as_wav(
        self, pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1
    ) -> Path:
        """Wrap raw 16-bit PCM audio bytes (e.g. from Gemini Live audio output) into a castable WAV file."""
        filename = f"live_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.wav"
        filepath = self.config.tts_cache_dir / filename
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit PCM
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return filepath

    async def speak_on_room(
        self,
        text: str,
        room_name: str | None = None,
        volume: float | None = None,
    ) -> dict[str, Any]:
        """Generate TTS audio and cast it to the specified (or active) Google Home room speaker."""
        target_query = room_name or self._active_room
        resolved_name = self.resolve_room_name(target_query) or target_query

        tts_path = await asyncio.to_thread(self.generate_tts_file, text)
        media_url = f"{self.config.base_lan_url}/api/tts/{tts_path.name}"

        cast_result = await asyncio.to_thread(
            self._cast_url_sync, resolved_name, media_url, "audio/mp3", volume
        )
        return {
            "status": cast_result["status"],
            "room": resolved_name,
            "text": text,
            "media_url": media_url,
            "detail": cast_result.get("detail", ""),
        }

    async def cast_pcm_audio(
        self,
        pcm_bytes: bytes,
        sample_rate: int = 24000,
        room_name: str | None = None,
    ) -> dict[str, Any]:
        """Cast raw 16-bit PCM audio response from Gemini Live directly to a Google Home speaker."""
        target_query = room_name or self._active_room
        resolved_name = self.resolve_room_name(target_query) or target_query

        wav_path = await asyncio.to_thread(
            self.save_pcm_as_wav, pcm_bytes, sample_rate
        )
        media_url = f"{self.config.base_lan_url}/api/tts/{wav_path.name}"

        cast_result = await asyncio.to_thread(
            self._cast_url_sync, resolved_name, media_url, "audio/wav", None
        )
        return {
            "status": cast_result["status"],
            "room": resolved_name,
            "media_url": media_url,
            "detail": cast_result.get("detail", ""),
        }

    def _cast_url_sync(
        self,
        speaker_name: str,
        media_url: str,
        content_type: str,
        volume: float | None = None,
    ) -> dict[str, str]:
        cc = self._chromecasts.get(speaker_name)
        if cc is None:
            # Try case-insensitive lookup
            for name, dev in self._chromecasts.items():
                if name.lower() == speaker_name.lower():
                    cc = dev
                    break

        if cc is None:
            return {
                "status": "queued_no_cast_device",
                "detail": (
                    f"Speaker '{speaker_name}' not currently connected on LAN; "
                    f"audio prepared at {media_url}"
                ),
            }

        try:
            cc.wait(timeout=5.0)
            if volume is not None and 0.0 <= volume <= 1.0:
                cc.set_volume(volume)
            mc = cc.media_controller
            mc.play_media(
                media_url,
                content_type,
                title="Antigravity Live",
                stream_type="BUFFERED",
            )
            mc.block_until_active(timeout=5.0)
            return {
                "status": "casting",
                "detail": f"Playing {media_url} on {speaker_name}",
            }
        except Exception as exc:
            logger.error("Failed to cast audio to %s: %s", speaker_name, exc)
            return {
                "status": "error",
                "detail": str(exc),
            }
