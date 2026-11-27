"""Proxmox Hardware Wireless Mic Daemon (USB 2.4GHz Coin Mic Receiver / Bluetooth ALSA/PipeWire input).

Allows plugging a long-range wireless lapel/coin microphone receiver (e.g., Hollyland Lark M2,
DJI Mic Mini, AnkerWork M650) directly into a USB port on your Proxmox host and walking
room-to-room completely phone-free while Antigravity responds out of your Google Home speakers.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import struct
from typing import Any

import websockets

from agy_home_bridge.config import BridgeSettings

logger = logging.getLogger(__name__)


def compute_pcm16_rms(pcm_bytes: bytes) -> float:
    """Compute normalized RMS amplitude (0.0 to 1.0) of 16-bit signed mono PCM audio."""
    if len(pcm_bytes) < 2:
        return 0.0
    sample_count = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{sample_count}h", pcm_bytes[: sample_count * 2])
    sum_sq = sum((s / 32768.0) ** 2 for s in samples)
    return math.sqrt(sum_sq / sample_count)


def downsample_pcm16_to_16k(pcm_bytes: bytes, in_rate: int) -> bytes:
    """Downsample 16-bit signed mono PCM from `in_rate` (e.g. 48000 Hz) to 16000 Hz."""
    if in_rate == 16000 or len(pcm_bytes) < 2:
        return pcm_bytes
    sample_count = len(pcm_bytes) // 2
    samples = struct.unpack(f"<{sample_count}h", pcm_bytes[: sample_count * 2])
    ratio = in_rate / 16000.0
    out_len = int(sample_count / ratio)
    if out_len <= 0:
        return b""
    out_samples = []
    for o in range(out_len):
        start = int(o * ratio)
        end = min(sample_count, int((o + 1) * ratio))
        if end <= start:
            end = start + 1
        avg = sum(samples[start:end]) // (end - start)
        out_samples.append(max(-32768, min(32767, avg)))
    return struct.pack(f"<{len(out_samples)}h", *out_samples)


class HardwareMicDaemon:
    """Captures 16kHz mono PCM from a USB wireless microphone connected to Proxmox and streams to `/ws/live`."""

    def __init__(self, config: BridgeSettings) -> None:
        self.config = config
        self._running = False

    @staticmethod
    def list_audio_input_devices() -> list[dict[str, Any]]:
        """List available hardware microphone inputs on this host."""
        try:
            import sounddevice as sd  # type: ignore

            devices = sd.query_devices()
            inputs = []
            for idx, dev in enumerate(devices):
                if dev.get("max_input_channels", 0) > 0:
                    inputs.append(
                        {
                            "index": idx,
                            "name": dev.get("name", f"Device {idx}"),
                            "channels": dev.get("max_input_channels", 1),
                            "default_samplerate": dev.get("default_samplerate", 16000),
                        }
                    )
            return inputs
        except Exception as exc:
            logger.debug("sounddevice not available for hardware mic query: %s", exc)
            return []

    async def run(self, ws_url: str | None = None) -> None:
        """Continuously stream hardware mic audio with energy VAD barge-in to the Live WebSocket bridge."""
        try:
            import sounddevice as sd  # type: ignore
        except ImportError:
            logger.warning(
                "sounddevice is not installed. Run `pip install 'antigravity-google-home-bridge[hardware-mic]'` "
                "to enable direct Proxmox USB wireless mic capture."
            )
            return

        target_ws = (
            ws_url
            or f"ws://127.0.0.1:{self.config.bridge_port}/ws/live"
        )
        device = self._resolve_input_device(sd)
        capture_rate = self._choose_capture_rate(sd, device, self.config.hw_mic_sample_rate)
        block_size = int(capture_rate * 0.04)  # 40ms chunks for low-latency streaming
        loop = asyncio.get_running_loop()
        audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)

        def _audio_callback(
            indata: bytes, frames: int, time_info: Any, status: Any
        ) -> None:
            if status:
                logger.debug("Audio input status: %s", status)
            raw_bytes = bytes(indata)
            loop.call_soon_threadsafe(
                lambda: audio_queue.put_nowait(raw_bytes)
                if not audio_queue.full()
                else None
            )

        self._running = True
        logger.info(
            "Starting Proxmox Hardware Wireless Mic Daemon (device=%s, capture=%dHz -> 16000Hz) -> %s",
            device,
            capture_rate,
            target_ws,
        )

        with sd.RawInputStream(
            samplerate=capture_rate,
            blocksize=block_size,
            device=device,
            channels=1,
            dtype="int16",
            callback=_audio_callback,
        ):
            while self._running:
                try:
                    async with websockets.connect(target_ws) as ws:
                        # Force cast output so replies play on the active Google Home speaker
                        await ws.send(
                            json.dumps({"type": "set_routing", "mode": "cast"})
                        )
                        while self._running:
                            raw_chunk = await audio_queue.get()
                            pcm_chunk = downsample_pcm16_to_16k(raw_chunk, capture_rate)
                            if not pcm_chunk:
                                continue
                            # Stream continuously and let Gemini Live's server-side VAD
                            # (`server_content.interrupted`) handle barge-in without
                            # hair-trigger local RMS stops cutting off Google Home casts.
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "audio_chunk",
                                        "pcm_b64": base64.b64encode(pcm_chunk).decode(
                                            "ascii"
                                        ),
                                    }
                                )
                            )
                except Exception as exc:
                    logger.warning(
                        "Hardware mic daemon reconnecting in 3s (%s)", exc
                    )
                    await asyncio.sleep(3.0)

    def _resolve_input_device(self, sd: Any) -> int | None:
        """Resolve input device by HW_MIC_DEVICE_NAME substring first, then HW_MIC_DEVICE_INDEX."""
        if self.config.hw_mic_device_name.strip():
            target = self.config.hw_mic_device_name.strip().lower()
            for dev in self.list_audio_input_devices():
                if target in dev["name"].lower():
                    return int(dev["index"])
        return self.config.hw_mic_device_index

    @staticmethod
    def _choose_capture_rate(sd: Any, device: int | None, preferred_rate: int) -> int:
        """Pick preferred_rate (16000) if supported, else fall back to 48000/44100/device default."""
        candidates = [preferred_rate, 16000, 48000, 44100]
        seen: set[int] = set()
        for rate in candidates:
            if rate in seen:
                continue
            seen.add(rate)
            try:
                sd.check_input_settings(
                    device=device, samplerate=rate, channels=1, dtype="int16"
                )
                return rate
            except Exception:
                continue
        return preferred_rate

    def stop(self) -> None:
        self._running = False
