"""Multi-room Google Home & Chromecast discovery, room-presence mapping, and audio/TV casting."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from gtts import gTTS

from agy_home_bridge.config import BridgeSettings

logger = logging.getLogger(__name__)

SAMSUNG_TIZEN_TV_SPECS: dict[str, dict[str, str]] = {
    "Office - Samsung Q60CD 65": {
        "model_name": "Samsung Q60CD 65 (Tizen)",
        "protocol": "samsung_tizen",
    },
    "Living Room TV": {
        "model_name": "Samsung Tizen Smart TV",
        "protocol": "samsung_tizen",
    },
    "Kitchen TV": {
        "model_name": "Samsung Tizen Smart TV",
        "protocol": "samsung_tizen",
    },
    "The Frame": {
        "model_name": "Samsung The Frame (Tizen)",
        "protocol": "samsung_tizen",
    },
    "Master Bedroom Frame": {
        "model_name": "Samsung The Frame (Tizen)",
        "protocol": "samsung_tizen",
    },
}

ROOM_ALIASES: dict[str, str] = {
    "office": "office",
    "office_tv": "office",
    "office_samsung_q60cd_65": "office",
    "living_room": "living_room",
    "living_room_speaker": "living_room",
    "living_room_tv": "living_room",
    "kitchen": "kitchen",
    "kitchen_tv": "kitchen",
    "game_room": "game_room",
    "game_room_tv": "game_room",
    "98_qled": "98_qled",
    "98qled": "98_qled",
    "qled": "98_qled",
    "theater": "98_qled",
    "home_theater": "98_qled",
    "master_bedroom": "master_bedroom",
    "bedroom": "master_bedroom",
    "master": "master_bedroom",
    "master_bedroom_tv": "master_bedroom",
    "the_frame": "master_bedroom",
    "master_bedroom_frame": "master_bedroom",
    "whole_home": "whole_home",
    "all": "whole_home",
    "all_speakers": "whole_home",
    "house": "whole_home",
    "home": "whole_home",
    "everywhere": "whole_home",
}

DEFAULT_ROOM_DEVICE_MAP: dict[str, dict[str, Any]] = {
    "office": {
        "speaker": "Office",
        "speaker_fallbacks": ["All Speakers", "Whole Home"],
        "tvs": ["Office TV", "Office - Samsung Q60CD 65"],
    },
    "living_room": {
        "speaker": "Living Room speaker",
        "speaker_fallbacks": ["All Speakers", "Whole Home"],
        "tvs": ["Living Room TV", '98" QLED'],
    },
    "kitchen": {
        "speaker": "Living Room speaker",
        "speaker_fallbacks": ["All Speakers", "Whole Home"],
        "tvs": ["Kitchen TV"],
    },
    "game_room": {
        "speaker": "All Speakers",
        "speaker_fallbacks": ["Whole Home"],
        "tvs": ["Game Room TV", '98" QLED'],
    },
    "98_qled": {
        "speaker": "All Speakers",
        "speaker_fallbacks": ["Whole Home"],
        "tvs": ['98" QLED'],
    },
    "master_bedroom": {
        "speaker": "All Speakers",
        "speaker_fallbacks": ["Whole Home"],
        "tvs": ["The Frame", "Master Bedroom Frame", "Master Bedroom TV"],
    },
    "whole_home": {
        "speaker": "All Speakers",
        "speaker_fallbacks": ["Whole Home"],
        "tvs": ['98" QLED', "Living Room TV", "Office TV", "Game Room TV"],
    },
}


def normalize_room_key(room: str) -> str:
    """Normalize free-form room names/aliases (e.g. `98\" QLED`, `living-room`) to canonical keys."""
    cleaned = room.strip().lower()
    for qchar in ('"', "'", "\u201c", "\u201d", "\u2019"):
        cleaned = cleaned.replace(qchar, "")
    cleaned = re.sub(r"[\s\-]+", "_", cleaned)
    cleaned = re.sub(r"[^a-z0-9_]", "", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return ROOM_ALIASES.get(cleaned, cleaned)


@dataclass
class CastDeviceInfo:
    """Metadata for a discovered Google Home speaker, speaker group, or TV display target."""

    name: str
    uuid: str
    model_name: str
    cast_type: str
    host: str
    port: int
    is_active_target: bool = False
    device_role: str = "speaker"
    protocol: str = "chromecast"
    capabilities: list[str] = field(default_factory=lambda: ["audio"])

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["capabilities"] = list(self.capabilities)
        return data


CastSpeakerInfo = CastDeviceInfo


class GoogleHomeCastManager:
    """Discovers Google Home speakers/TVs on the LAN, tracks room presence, and casts audio or /hud."""

    def __init__(self, config: BridgeSettings) -> None:
        self.config = config
        self._speakers: dict[str, CastDeviceInfo] = {}
        self._tvs: dict[str, CastDeviceInfo] = {}
        self._chromecasts: dict[str, Any] = {}
        self._dashcast_controllers: dict[str, Any] = {}
        self._browser: Any = None
        self._active_room: str = config.default_cast_room
        self._routing_mode: str = config.default_audio_routing
        self._lock = asyncio.Lock()
        self._registry_lock = threading.RLock()
        self._cast_lock = threading.RLock()

        self._room_device_map = self._build_room_device_map(config)
        explicit_presence = getattr(config, "default_presence_room", "").strip()
        default_presence = explicit_presence or config.default_cast_room
        self._active_presence_room: str = normalize_room_key(default_presence) or "whole_home"
        self._active_tv: str | None = None
        self._presence_source: str | None = None
        self._presence_updated_at: str | None = None
        self._cast_mute_until: float = 0.0
        self._retry_task: asyncio.Task[Any] | None = None

        self._seed_known_tvs()
        initial_presence = self.resolve_room_presence(self._active_presence_room)
        if explicit_presence and initial_presence["speaker"]:
            self._active_room = initial_presence["speaker"]
        self._active_tv = initial_presence["tv"]
        self._sync_tv_active_flags()

    @property
    def is_casting_audio(self) -> bool:
        """True while a Google Home speaker is expected to be playing cast audio."""
        return time.monotonic() < self._cast_mute_until

    def extend_cast_mute(self, duration_sec: float, append: bool = False) -> None:
        """Extend the echo-gate window while Google Home is speaking."""
        now = time.monotonic()
        base = max(self._cast_mute_until, now) if append else now
        self._cast_mute_until = max(self._cast_mute_until, base + max(0.0, duration_sec))

    @staticmethod
    def _build_room_device_map(config: BridgeSettings) -> dict[str, dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {
            key: {
                "speaker": val["speaker"],
                "speaker_fallbacks": list(val.get("speaker_fallbacks", [])),
                "tvs": list(val.get("tvs", [])),
            }
            for key, val in DEFAULT_ROOM_DEVICE_MAP.items()
        }
        overrides = getattr(config, "room_device_map", None) or {}
        if isinstance(overrides, dict):
            for raw_key, raw_val in overrides.items():
                norm_key = normalize_room_key(str(raw_key))
                if not norm_key or not isinstance(raw_val, dict):
                    continue
                base = dict(
                    merged.get(
                        norm_key,
                        {"speaker": "", "speaker_fallbacks": [], "tvs": []},
                    )
                )
                if "speaker" in raw_val and raw_val["speaker"] is not None:
                    base["speaker"] = str(raw_val["speaker"]).strip()
                if "speaker_fallbacks" in raw_val and isinstance(
                    raw_val["speaker_fallbacks"], list
                ):
                    base["speaker_fallbacks"] = [
                        str(s).strip()
                        for s in raw_val["speaker_fallbacks"]
                        if str(s).strip()
                    ]
                if "tvs" in raw_val and isinstance(raw_val["tvs"], list):
                    base["tvs"] = [
                        str(t).strip() for t in raw_val["tvs"] if str(t).strip()
                    ]
                elif "tv" in raw_val and raw_val["tv"]:
                    base["tvs"] = [str(raw_val["tv"]).strip()]
                merged[norm_key] = base
        return merged

    def _build_known_tvs(self) -> dict[str, CastDeviceInfo]:
        tvs: dict[str, CastDeviceInfo] = {}
        active_lower = self._active_tv.lower() if self._active_tv else ""
        for name, spec in SAMSUNG_TIZEN_TV_SPECS.items():
            tvs[name] = CastDeviceInfo(
                name=name,
                uuid=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"tizen-{name}")),
                model_name=spec["model_name"],
                cast_type="tizen",
                host="",
                port=8001,
                is_active_target=bool(active_lower and name.lower() == active_lower),
                device_role="tv",
                protocol=spec["protocol"],
                capabilities=["tizen_browser"],
            )
        return tvs

    def _seed_known_tvs(self) -> None:
        with self._registry_lock:
            self._tvs.update(self._build_known_tvs())

    def _sync_tv_active_flags(self) -> None:
        with self._registry_lock:
            active_lower = self._active_tv.lower() if self._active_tv else ""
            for tv in self._tvs.values():
                tv.is_active_target = bool(active_lower and tv.name.lower() == active_lower)

    @staticmethod
    def _lookup_device_name(
        registry: dict[str, CastDeviceInfo], query: str
    ) -> str | None:
        q = query.strip().lower()
        if not q:
            return None
        for name in registry:
            if name.lower() == q:
                return name
        return None

    def _lookup_tv_info(self, tv_name: str) -> CastDeviceInfo | None:
        with self._registry_lock:
            matched = self._lookup_device_name(self._tvs, tv_name)
            return self._tvs.get(matched) if matched else None

    @property
    def active_room(self) -> str:
        return self._active_room

    @property
    def active_speaker(self) -> str:
        return self._active_room

    @property
    def active_presence_room(self) -> str:
        return self._active_presence_room

    @property
    def active_tv(self) -> str | None:
        return self._active_tv

    @property
    def presence_source(self) -> str | None:
        return self._presence_source

    @property
    def presence_updated_at(self) -> str | None:
        return self._presence_updated_at

    @property
    def routing_mode(self) -> str:
        return self._routing_mode

    def set_routing_mode(self, mode: str) -> str:
        if mode not in ("direct", "cast", "both"):
            raise ValueError("routing_mode must be 'direct', 'cast', or 'both'")
        self._routing_mode = mode
        return self._routing_mode

    def set_active_room(self, room_name: str) -> str:
        """Switch the active speaker room or speaker group (fuzzy matched against discovered speakers)."""
        with self._registry_lock:
            matched = self.resolve_room_name(room_name)
            self._active_room = matched or room_name
            for speaker in self._speakers.values():
                speaker.is_active_target = (
                    speaker.name.lower() == self._active_room.lower()
                )
            active = self._active_room
        logger.info("Active Google Home target room set to: %s", active)
        return active

    def resolve_room_name(self, query: str) -> str | None:
        """Fuzzy-match a requested room name against discovered Google Home speakers/groups."""
        q = query.strip().lower()
        if not q:
            return None
        with self._registry_lock:
            for name in self._speakers:
                if name.lower() == q:
                    return name
            for name in self._speakers:
                if q in name.lower() or name.lower() in q:
                    return name
        return None

    def resolve_room_presence(
        self,
        room: str,
        speaker_override: str | None = None,
        tv_override: str | None = None,
    ) -> dict[str, Any]:
        """Resolve a Home Assistant room update into active speaker and TV targets."""
        norm_room = normalize_room_key(room)
        mapping = self._room_device_map.get(norm_room)

        with self._registry_lock:
            if speaker_override and speaker_override.strip():
                selected_speaker = (
                    self.resolve_room_name(speaker_override)
                    or speaker_override.strip()
                )
            elif mapping is not None:
                candidates = [
                    c
                    for c in [
                        mapping.get("speaker", ""),
                        *(mapping.get("speaker_fallbacks") or []),
                    ]
                    if c
                ]
                selected_speaker = ""
                for cand in candidates:
                    matched = self._lookup_device_name(self._speakers, cand)
                    if matched:
                        selected_speaker = matched
                        break
                if not selected_speaker:
                    selected_speaker = (
                        str(mapping.get("speaker") or "").strip() or self._active_room
                    )
            else:
                selected_speaker = (
                    self.resolve_room_name(room) or room.strip() or self._active_room
                )

            tv_candidates = (
                [str(t) for t in (mapping.get("tvs") or []) if str(t).strip()]
                if mapping is not None
                else []
            )
            selected_tv: str | None = None
            if tv_override and tv_override.strip():
                selected_tv = (
                    self._lookup_device_name(self._tvs, tv_override)
                    or tv_override.strip()
                )
            elif tv_candidates:
                # Prefer a live discovered Cast TV first, then any known TV (incl. Samsung Tizen), then first candidate.
                for cand in tv_candidates:
                    matched = self._lookup_device_name(self._tvs, cand)
                    if (
                        matched
                        and self._tvs[matched].protocol == "chromecast"
                        and self._lookup_chromecast(matched) is not None
                    ):
                        selected_tv = matched
                        break
                if selected_tv is None:
                    for cand in tv_candidates:
                        matched = self._lookup_device_name(self._tvs, cand)
                        if matched:
                            selected_tv = matched
                            break
                if selected_tv is None:
                    selected_tv = tv_candidates[0]

        return {
            "room": norm_room or room.strip(),
            "mapped": mapping is not None,
            "speaker": selected_speaker,
            "tv": selected_tv,
            "tv_candidates": tv_candidates,
        }

    def set_presence_room(
        self,
        room: str,
        source: str | None = None,
        speaker_override: str | None = None,
        tv_override: str | None = None,
    ) -> dict[str, Any]:
        """Update active presence room, speaker, and TV target without auto-casting the TV."""
        with self._registry_lock:
            resolved = self.resolve_room_presence(
                room,
                speaker_override=speaker_override,
                tv_override=tv_override,
            )
            self._active_presence_room = resolved["room"]
            self.set_active_room(resolved["speaker"])
            self._active_tv = resolved["tv"]
            self._sync_tv_active_flags()
            self._presence_source = source
            self._presence_updated_at = datetime.now(timezone.utc).isoformat()
            return {
                "presence_room": self._active_presence_room,
                "active_room": self._active_room,
                "active_speaker": self._active_room,
                "active_tv": self._active_tv,
                "source": self._presence_source,
                "updated_at": self._presence_updated_at,
                "mapped": resolved["mapped"],
            }

    def list_speakers(self, include_tvs: bool = False) -> list[dict[str, Any]]:
        """Return discovered Google Home speakers/groups (and optionally TVs)."""
        with self._registry_lock:
            results = []
            for s in sorted(self._speakers.values(), key=lambda x: x.name.lower()):
                s.is_active_target = s.name.lower() == self._active_room.lower()
                results.append(s.to_dict())
            if include_tvs:
                results.extend(self.list_tvs())
            return results

    def list_tvs(self) -> list[dict[str, Any]]:
        """Return known and discovered TV display targets."""
        with self._registry_lock:
            self._sync_tv_active_flags()
            return [
                tv.to_dict()
                for tv in sorted(self._tvs.values(), key=lambda x: x.name.lower())
            ]

    async def discover_speakers(
        self, timeout: float | None = None, merge: bool = False
    ) -> list[dict[str, Any]]:
        """Run mDNS / Known-Host discovery for Google Home speakers and Cast TVs on the LAN."""
        effective_timeout = (
            float(timeout)
            if timeout is not None
            else float(getattr(self.config, "cast_discovery_timeout", 6.0))
        )
        async with self._lock:
            return await asyncio.to_thread(
                self._discover_sync, effective_timeout, merge
            )

    def _discover_sync(
        self, timeout: float, merge: bool = False
    ) -> list[dict[str, Any]]:
        try:
            import pychromecast
        except ImportError:
            logger.warning("pychromecast not installed; skipping LAN Cast discovery.")
            return self.list_speakers()

        try:
            old_browser = self._browser
            old_chromecasts = list(self._chromecasts.values())

            known_hosts = self.config.parsed_known_cast_hosts or None
            chromecasts, browser = pychromecast.get_chromecasts(
                timeout=timeout,
                known_hosts=known_hosts,
            )
            self._browser = browser

            if old_browser is not None and old_browser is not browser:
                for old_cc in old_chromecasts:
                    try:
                        old_cc.disconnect(blocking=False)
                    except Exception:
                        pass
                try:
                    old_browser.stop_discovery()
                except Exception:
                    pass

            with self._registry_lock:
                if merge:
                    next_speakers = dict(self._speakers)
                    next_tvs = dict(self._tvs)
                    next_chromecasts = dict(self._chromecasts)
                else:
                    next_speakers = {}
                    next_tvs = self._build_known_tvs()
                    next_chromecasts = {}
                    with self._cast_lock:
                        self._dashcast_controllers.clear()

                for cc in chromecasts:
                    info = cc.cast_info
                    name = info.friendly_name or f"Google Home ({info.host})"
                    cast_type = (info.cast_type or "audio").lower()
                    is_tv = cast_type == "cast"

                    if is_tv:
                        tv_info = CastDeviceInfo(
                            name=name,
                            uuid=str(info.uuid),
                            model_name=info.model_name or "Google Cast TV",
                            cast_type=cast_type,
                            host=info.host,
                            port=info.port,
                            is_active_target=(
                                self._active_tv is not None
                                and name.lower() == self._active_tv.lower()
                            ),
                            device_role="tv",
                            protocol="chromecast",
                            capabilities=["hud", "cast"],
                        )
                        next_tvs[name] = tv_info
                    else:
                        speaker = CastDeviceInfo(
                            name=name,
                            uuid=str(info.uuid),
                            model_name=info.model_name or "Google Cast Device",
                            cast_type=cast_type,
                            host=info.host,
                            port=info.port,
                            is_active_target=(name.lower() == self._active_room.lower()),
                            device_role="speaker",
                            protocol="chromecast",
                            capabilities=["audio"],
                        )
                        next_speakers[name] = speaker

                    prev_cc = next_chromecasts.get(name)
                    if prev_cc is not cc:
                        with self._cast_lock:
                            self._dashcast_controllers.pop(name, None)
                    next_chromecasts[name] = cc

                self._speakers = next_speakers
                self._tvs = next_tvs
                self._chromecasts = next_chromecasts

                # Only auto-select from speakers, never from TVs
                if self._speakers and not self.resolve_room_name(self._active_room):
                    first_name = next(iter(self._speakers.keys()))
                    self._active_room = first_name

                # Reconcile active_tv for active_presence_room so newly discovered Cast TVs take priority
                resolved = self.resolve_room_presence(self._active_presence_room)
                if resolved["tv"] is not None:
                    self._active_tv = resolved["tv"]
                self._sync_tv_active_flags()

                speaker_count = len(self._speakers)
                speaker_names = list(self._speakers.keys())
                tv_count = len(self._tvs)
                tv_names = list(self._tvs.keys())

            logger.info(
                "Discovered %d speaker(s) %s and %d TV target(s) %s",
                speaker_count,
                speaker_names,
                tv_count,
                tv_names,
            )
        except Exception as exc:
            logger.warning("Google Home LAN discovery encountered an error: %s", exc)

        return self.list_speakers()

    def schedule_discovery_resilience(
        self, retry_delays: tuple[float, ...] = (5.0, 15.0)
    ) -> asyncio.Task[Any] | None:
        """Schedule background merge=True discovery retries if key configured speakers are missing."""
        self.cancel_discovery_resilience()

        expected_speakers = {
            str(entry.get("speaker")).strip().lower()
            for entry in self._room_device_map.values()
            if entry.get("speaker")
        }

        async def _retry_loop() -> None:
            for delay in retry_delays:
                await asyncio.sleep(delay)
                with self._registry_lock:
                    discovered_lower = {name.lower() for name in self._speakers}
                if expected_speakers and expected_speakers.issubset(discovered_lower):
                    return
                await self.discover_speakers(
                    timeout=float(getattr(self.config, "cast_discovery_timeout", 6.0)),
                    merge=True,
                )

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return None
        self._retry_task = loop.create_task(_retry_loop())
        return self._retry_task

    def cancel_discovery_resilience(self) -> None:
        """Cancel any pending background discovery retry task."""
        if self._retry_task is not None and not self._retry_task.done():
            self._retry_task.cancel()
        self._retry_task = None

    def register_mock_speaker(
        self,
        name: str,
        model_name: str = "Google Nest Audio",
        cast_type: str = "audio",
        host: str = "192.168.1.100",
        port: int = 8009,
    ) -> CastDeviceInfo:
        """Register a virtual/mock Google Home speaker (or Cast TV if `cast_type == 'cast'`)."""
        is_tv = cast_type.lower() == "cast"
        with self._registry_lock:
            device = CastDeviceInfo(
                name=name,
                uuid=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{name}-{host}")),
                model_name=model_name,
                cast_type=cast_type,
                host=host,
                port=port,
                is_active_target=(
                    bool(self._active_tv and name.lower() == self._active_tv.lower())
                    if is_tv
                    else (name.lower() == self._active_room.lower())
                ),
                device_role="tv" if is_tv else "speaker",
                protocol="chromecast",
                capabilities=["hud", "cast"] if is_tv else ["audio"],
            )
            if is_tv:
                self._tvs[name] = device
            else:
                self._speakers[name] = device
            return device

    def register_mock_tv(
        self,
        name: str,
        model_name: str = "Google TV",
        cast_type: str = "cast",
        host: str = "192.168.1.150",
        port: int = 8009,
        protocol: str = "chromecast",
        chromecast: Any = None,
    ) -> CastDeviceInfo:
        """Register a virtual/mock TV display target (Chromecast or Samsung Tizen)."""
        caps = ["tizen_browser"] if protocol == "samsung_tizen" else ["hud", "cast"]
        with self._registry_lock:
            tv = CastDeviceInfo(
                name=name,
                uuid=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"tv-{name}-{host}")),
                model_name=model_name,
                cast_type=cast_type,
                host=host,
                port=port,
                is_active_target=(
                    self._active_tv is not None and name.lower() == self._active_tv.lower()
                ),
                device_role="tv",
                protocol=protocol,
                capabilities=caps,
            )
            self._tvs[name] = tv
            if chromecast is not None:
                self._chromecasts[name] = chromecast
                with self._cast_lock:
                    self._dashcast_controllers.pop(name, None)
            return tv

    def generate_tts_file(self, text: str, lang: str = "en") -> Path:
        """Synthesize clean speech MP3 for casting to Google Home speakers."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Cannot synthesize empty text")

        digest = hashlib.sha256(f"{lang}:{clean_text}".encode("utf-8")).hexdigest()[:20]
        filename = f"tts_{digest}.mp3"
        filepath = self.config.tts_cache_dir / filename

        if not filepath.exists():
            self.config.tts_cache_dir.mkdir(parents=True, exist_ok=True)
            tts = gTTS(text=clean_text, lang=lang, slow=False)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=self.config.tts_cache_dir,
                prefix=f".tmp_{digest}_",
                suffix=".mp3",
            )
            os.close(tmp_fd)
            try:
                tts.save(tmp_path)
                os.replace(tmp_path, filepath)
            except Exception:
                if os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                raise
        return filepath

    def save_pcm_as_wav(
        self, pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1
    ) -> Path:
        """Wrap raw 16-bit PCM audio bytes (e.g. from Gemini Live audio output) into a castable WAV file."""
        filename = f"live_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.wav"
        filepath = self.config.tts_cache_dir / filename
        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return filepath

    def _resolve_speaker_name(self, room_name: str | None = None) -> str:
        """Resolve a speaker name or canonical presence room key (e.g. 'living_room') to a speaker."""
        with self._registry_lock:
            target_query = (room_name or "").strip() or self._active_room
            resolved_name = self.resolve_room_name(target_query)
            if resolved_name:
                return resolved_name
            norm_key = normalize_room_key(target_query)
            if norm_key in self._room_device_map:
                return self.resolve_room_presence(target_query)["speaker"]
            return target_query

    async def speak_on_room(
        self,
        text: str,
        room_name: str | None = None,
        volume: float | None = None,
    ) -> dict[str, Any]:
        """Generate TTS audio and cast it to the specified (or active) Google Home room speaker."""
        resolved_name = self._resolve_speaker_name(room_name)

        tts_path = await asyncio.to_thread(self.generate_tts_file, text)
        media_url = f"{self.config.base_lan_url}/api/tts/{tts_path.name}"

        cast_result = await asyncio.to_thread(
            self._cast_url_sync, resolved_name, media_url, "audio/mp3", volume
        )
        if cast_result.get("status") == "casting":
            est_sec = max(2.0, len(text.split()) * 0.42) + 3.0
            self.extend_cast_mute(est_sec, append=False)
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
        enqueue: bool = False,
    ) -> dict[str, Any]:
        """Cast raw 16-bit PCM audio response from Gemini Live directly to a Google Home speaker."""
        resolved_name = self._resolve_speaker_name(room_name)

        wav_path = await asyncio.to_thread(
            self.save_pcm_as_wav, pcm_bytes, sample_rate
        )
        media_url = f"{self.config.base_lan_url}/api/tts/{wav_path.name}"

        cast_result = await asyncio.to_thread(
            self._cast_url_sync,
            resolved_name,
            media_url,
            "audio/wav",
            None,
            enqueue,
        )
        if cast_result.get("status") == "casting":
            clip_sec = len(pcm_bytes) / float(max(1, sample_rate) * 2)
            tail = 1.0 if enqueue else 3.0
            self.extend_cast_mute(clip_sec + tail, append=enqueue)
        return {
            "status": cast_result["status"],
            "room": resolved_name,
            "media_url": media_url,
            "detail": cast_result.get("detail", ""),
        }

    async def warm_room(self, room_name: str | None = None) -> None:
        """Pre-warm the Google Home Default Media Receiver so subsequent casts start in ~0.4s."""
        resolved_name = self._resolve_speaker_name(room_name)
        await asyncio.to_thread(self._warm_room_sync, resolved_name)

    def _warm_room_sync(self, speaker_name: str) -> None:
        # Guard: never warm TVs with silence audio
        with self._registry_lock:
            is_known_speaker = speaker_name in self._speakers
        if not is_known_speaker:
            tv_info = self._lookup_tv_info(speaker_name)
            if tv_info is not None and tv_info.cast_type == "cast":
                return

        cc = self._lookup_chromecast(speaker_name)
        if cc is None:
            return

        cc_cast_type = getattr(getattr(cc, "cast_info", None), "cast_type", None)
        if isinstance(cc_cast_type, str) and cc_cast_type.lower() == "cast":
            return

        try:
            cc.wait(timeout=4.0)
            if cc.app_id == "CC1AD845":
                return
            silence_path = self.config.tts_cache_dir / "warmup_silence.wav"
            if not silence_path.exists():
                self.config.tts_cache_dir.mkdir(parents=True, exist_ok=True)
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=self.config.tts_cache_dir,
                    prefix=".tmp_warmup_",
                    suffix=".wav",
                )
                os.close(tmp_fd)
                try:
                    with wave.open(tmp_path, "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(b"\x00\x00" * 12000)
                    os.replace(tmp_path, silence_path)
                except Exception:
                    if os.path.exists(tmp_path):
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
                    raise
            media_url = f"{self.config.base_lan_url}/api/tts/{silence_path.name}"
            mc = cc.media_controller
            mc.play_media(
                media_url,
                "audio/wav",
                title="Antigravity Live Ready",
                stream_type="BUFFERED",
            )
            mc.block_until_active(timeout=5.0)
            logger.info("Pre-warmed Google Home receiver for '%s'", speaker_name)
        except Exception as exc:
            logger.debug("Receiver pre-warm skipped for '%s': %s", speaker_name, exc)

    async def stop_room(self, room_name: str | None = None) -> None:
        """Stop current playback on the Google Home speaker (used for instant barge-in)."""
        self._cast_mute_until = 0.0
        resolved_name = self._resolve_speaker_name(room_name)
        await asyncio.to_thread(self._stop_room_sync, resolved_name)

    def _stop_room_sync(self, speaker_name: str) -> None:
        cc = self._lookup_chromecast(speaker_name)
        if cc is None:
            return
        try:
            cc.media_controller.stop()
        except Exception as exc:
            logger.debug("Stop cast failed for '%s': %s", speaker_name, exc)

    async def cast_hud_to_tv(
        self, room_name: str | None = None, tv_name: str | None = None
    ) -> dict[str, Any]:
        """Strictly on-demand: cast the `/hud` dashboard to a room TV via DashCast or return fallback info."""
        hud_url = self.config.hud_lan_url

        if room_name and room_name.strip():
            resolved = self.resolve_room_presence(room_name, tv_override=tv_name)
            room_label = resolved["room"]
            target_tv = resolved["tv"]
        elif tv_name and tv_name.strip():
            with self._registry_lock:
                room_label = self._active_presence_room
                target_tv = (
                    self._lookup_device_name(self._tvs, tv_name)
                    or self._lookup_device_name(self._speakers, tv_name)
                    or tv_name.strip()
                )
        else:
            with self._registry_lock:
                room_label = self._active_presence_room
                target_tv = (
                    self._active_tv
                    or self.resolve_room_presence(self._active_presence_room)["tv"]
                )

        if not target_tv:
            return {
                "status": "no_tv_configured",
                "room": room_label,
                "tv": None,
                "hud_url": hud_url,
                "protocol": None,
                "message": f"No TV configured for room '{room_label}'.",
            }

        tv_info = self._lookup_tv_info(target_tv)
        canonical_tv_name = tv_info.name if tv_info is not None else target_tv
        cc = self._lookup_chromecast(canonical_tv_name)

        if cc is not None:
            raw_cast_type = (
                tv_info.cast_type
                if tv_info is not None
                else getattr(getattr(cc, "cast_info", None), "cast_type", "cast")
            )
            if isinstance(raw_cast_type, str) and raw_cast_type.lower() in (
                "audio",
                "group",
            ):
                return {
                    "status": "unsupported_protocol",
                    "room": room_label,
                    "tv": canonical_tv_name,
                    "hud_url": hud_url,
                    "protocol": "chromecast",
                    "message": (
                        f"Device '{canonical_tv_name}' is an audio target "
                        f"({raw_cast_type}), not a video TV."
                    ),
                }
            return await asyncio.to_thread(
                self._cast_hud_sync, canonical_tv_name, hud_url, room_label
            )

        is_tizen = (
            (tv_info is not None and tv_info.protocol == "samsung_tizen")
            or canonical_tv_name in SAMSUNG_TIZEN_TV_SPECS
        )
        if is_tizen:
            return {
                "status": "unsupported_protocol",
                "room": room_label,
                "tv": canonical_tv_name,
                "hud_url": hud_url,
                "protocol": "samsung_tizen",
                "message": (
                    f"'{canonical_tv_name}' is a Samsung Tizen TV without native Chromecast. "
                    f"Open {hud_url} in Samsung Internet on the TV or attach a Chromecast/Google TV dongle."
                ),
            }

        return {
            "status": "tv_offline",
            "room": room_label,
            "tv": canonical_tv_name,
            "hud_url": hud_url,
            "protocol": "chromecast",
            "message": (
                f"TV '{canonical_tv_name}' is offline or not discovered via mDNS on the LAN. "
                f"You can also open {hud_url} directly in a TV browser."
            ),
        }

    def _cast_hud_sync(
        self, tv_name: str, hud_url: str, room_label: str
    ) -> dict[str, Any]:
        cc = self._lookup_chromecast(tv_name)
        if cc is None:
            return {
                "status": "tv_offline",
                "room": room_label,
                "tv": tv_name,
                "hud_url": hud_url,
                "protocol": "chromecast",
                "message": f"TV '{tv_name}' is offline or not discovered.",
            }

        try:
            from pychromecast.controllers.dashcast import DashCastController

            cc.wait(timeout=float(getattr(self.config, "cast_tv_timeout", 5.0)))
            with self._cast_lock:
                controller = self._dashcast_controllers.get(tv_name)
                if controller is None:
                    controller = DashCastController()
                    cc.register_handler(controller)
                    self._dashcast_controllers[tv_name] = controller

            force = bool(getattr(self.config, "hud_cast_force", True))
            controller.load_url(hud_url, force=force)
            logger.info("Cast HUD %s to TV '%s'", hud_url, tv_name)
            return {
                "status": "casting",
                "room": room_label,
                "tv": tv_name,
                "hud_url": hud_url,
                "protocol": "chromecast",
                "message": f"Casting JARVIS HUD ({hud_url}) to '{tv_name}'.",
            }
        except Exception as exc:
            logger.error("Failed to cast HUD to TV '%s': %s", tv_name, exc)
            return {
                "status": "error",
                "room": room_label,
                "tv": tv_name,
                "hud_url": hud_url,
                "protocol": "chromecast",
                "message": str(exc),
            }

    def _lookup_chromecast(self, speaker_name: str) -> Any:
        with self._registry_lock:
            cc = self._chromecasts.get(speaker_name)
            if cc is not None:
                return cc
            for name, dev in self._chromecasts.items():
                if name.lower() == speaker_name.lower():
                    return dev
            return None

    def _cast_url_sync(
        self,
        speaker_name: str,
        media_url: str,
        content_type: str,
        volume: float | None = None,
        enqueue: bool = False,
    ) -> dict[str, str]:
        cc = self._lookup_chromecast(speaker_name)

        if cc is None:
            logger.warning(
                "Cast target '%s' is not a discovered Google Home device; audio not played",
                speaker_name,
            )
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
                enqueue=enqueue,
            )
            if not enqueue:
                mc.block_until_active(timeout=5.0)
            logger.info(
                "Casting %s to Google Home '%s' (enqueue=%s)",
                media_url,
                speaker_name,
                enqueue,
            )
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
