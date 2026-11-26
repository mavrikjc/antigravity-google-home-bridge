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
        sample_rate = self.config.hw_mic_sample_rate
        block_size = int(sample_rate * 0.1)  # 100ms chunks
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
            "Starting Proxmox Hardware Wireless Mic Daemon (device=%s, rate=%dHz) -> %s",
            self.config.hw_mic_device_index,
            sample_rate,
            target_ws,
        )

        with sd.RawInputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            device=self.config.hw_mic_device_index,
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
                        is_speaking = False
                        while self._running:
                            pcm_chunk = await audio_queue.get()
                            rms = compute_pcm16_rms(pcm_chunk)
                            if rms >= self.config.hw_mic_vad_threshold and not is_speaking:
                                is_speaking = True
                                await ws.send(json.dumps({"type": "barge_in"}))
                            elif rms < self.config.hw_mic_vad_threshold * 0.6:
                                is_speaking = False

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

    def stop(self) -> None:
        self._running = False
