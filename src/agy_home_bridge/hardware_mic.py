"""Proxmox Hardware Wireless Mic Daemon (USB 2.4GHz Coin Mic Receiver / Bluetooth ALSA/PipeWire input).

Allows plugging a long-range wireless lapel/coin microphone receiver (e.g., Hollyland Lark M2,
DJI Mic Mini, AnkerWork M650) directly into a USB port on your Proxmox host and walking
room-to-room completely phone-free while Antigravity responds out of your Google Home speakers.
"""

from __future__ import annotations

import asyncio
import base64
from collections import deque
import json
import logging
import math
import struct
import time
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


class StreamingResampler:
    """Stateful windowed-sinc anti-aliased resampler + soft-limited gain for 16-bit mono PCM."""

    def __init__(self, in_rate: int, out_rate: int = 16000, gain: float = 1.0) -> None:
        self.in_rate = in_rate
        self.out_rate = out_rate
        self.gain = max(0.5, float(gain))
        self._np: Any = None
        self._taps: Any = None
        self._tail: Any = None
        try:
            import numpy as np  # type: ignore

            self._np = np
            if in_rate > out_rate:
                # 31-tap Blackman-windowed sinc low-pass filter at 0.45 * out_rate (7.2 kHz for 16 kHz)
                num_taps = 31
                fc = (0.45 * out_rate) / float(in_rate)
                n = np.arange(num_taps) - (num_taps - 1) / 2.0
                h = np.sinc(2.0 * fc * n) * np.blackman(num_taps)
                h = h / np.sum(h)
                self._taps = h.astype(np.float32)
                self._tail = np.zeros(num_taps - 1, dtype=np.float32)
        except ImportError:
            self._np = None

    def process(self, pcm_bytes: bytes) -> bytes:
        if len(pcm_bytes) < 2:
            return b""
        if self._np is None:
            down = downsample_pcm16_to_16k(pcm_bytes, self.in_rate)
            if self.gain == 1.0 or not down:
                return down
            count = len(down) // 2
            samples = struct.unpack(f"<{count}h", down[: count * 2])
            scaled = [
                max(-32768, min(32767, round(s * self.gain))) for s in samples
            ]
            return struct.pack(f"<{count}h", *scaled)

        np = self._np
        even_len = len(pcm_bytes) - (len(pcm_bytes) % 2)
        x = np.frombuffer(pcm_bytes[:even_len], dtype="<i2").astype(np.float32) / 32768.0

        if self.in_rate == self.out_rate:
            y = x
        elif self._taps is not None:
            # Overlap-save convolution so 40ms chunk boundaries have zero clicks
            padded = np.concatenate((self._tail, x))
            self._tail = padded[-(len(self._taps) - 1) :].copy()
            filtered = np.convolve(padded, self._taps, mode="valid")
            if self.in_rate % self.out_rate == 0:
                step = self.in_rate // self.out_rate
                y = filtered[::step]
            else:
                out_len = int(len(filtered) * self.out_rate / self.in_rate)
                if out_len <= 0:
                    return b""
                src_idx = np.linspace(0.0, len(filtered) - 1, out_len, dtype=np.float32)
                y = np.interp(src_idx, np.arange(len(filtered), dtype=np.float32), filtered).astype(np.float32)
        else:
            out_len = int(len(x) * self.out_rate / self.in_rate)
            if out_len <= 0:
                return b""
            src_idx = np.linspace(0.0, len(x) - 1, out_len, dtype=np.float32)
            y = np.interp(src_idx, np.arange(len(x), dtype=np.float32), x).astype(np.float32)

        if self.gain != 1.0:
            y = y * self.gain
            # Smooth soft-limiter above 0.85 full-scale to prevent harsh digital clipping
            over = np.abs(y) > 0.85
            if np.any(over):
                y = np.where(over, np.sign(y) * (0.85 + 0.15 * np.tanh((np.abs(y) - 0.85) / 0.15)), y)

        out_i16 = np.clip(np.rint(y * 32767.0), -32768, 32767).astype("<i2")
        return out_i16.tobytes()


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
        avg = round(sum(samples[start:end]) / (end - start))
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
        resampler = StreamingResampler(
            in_rate=capture_rate,
            out_rate=16000,
            gain=self.config.hw_mic_gain,
        )
        vad_threshold = max(0.0, float(self.config.hw_mic_vad_threshold))
        hangover_sec = max(0.4, float(self.config.hw_mic_silence_duration_sec))
        pre_roll: deque[bytes] = deque(maxlen=8)  # 8 * 40ms = 320ms pre-roll
        gate_open = False
        last_speech_at = 0.0

        logger.info(
            "Starting Proxmox Hardware Wireless Mic Daemon "
            "(device=%s, capture=%dHz -> 16000Hz, gain=%.1fx, vad=%.4f, hangover=%.2fs) -> %s",
            device,
            capture_rate,
            self.config.hw_mic_gain,
            vad_threshold,
            hangover_sec,
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
                    async with websockets.connect(
                        target_ws,
                        ping_interval=None,
                    ) as ws:
                        # Identify as a hardware mic without AEC and force cast output
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "client_hello",
                                    "client_type": "hardware_mic",
                                }
                            )
                        )
                        await ws.send(
                            json.dumps({"type": "set_routing", "mode": "cast"})
                        )
                        gate_open = False
                        pre_roll.clear()

                        async def _send_chunk(chunk: bytes) -> None:
                            await ws.send(
                                json.dumps(
                                    {
                                        "type": "audio_chunk",
                                        "pcm_b64": base64.b64encode(chunk).decode(
                                            "ascii"
                                        ),
                                    }
                                )
                            )

                        while self._running:
                            raw_chunk = await audio_queue.get()
                            pcm_chunk = resampler.process(raw_chunk)
                            if not pcm_chunk:
                                continue

                            rms = compute_pcm16_rms(pcm_chunk)
                            now = time.monotonic()

                            if vad_threshold <= 0.0:
                                await _send_chunk(pcm_chunk)
                                continue

                            if rms >= vad_threshold:
                                last_speech_at = now
                                if not gate_open:
                                    gate_open = True
                                    for buffered in pre_roll:
                                        await _send_chunk(buffered)
                                    pre_roll.clear()
                                await _send_chunk(pcm_chunk)
                            elif gate_open:
                                await _send_chunk(pcm_chunk)
                                if (now - last_speech_at) >= hangover_sec:
                                    gate_open = False
                                    await ws.send(
                                        json.dumps({"type": "audio_stream_end"})
                                    )
                            else:
                                pre_roll.append(pcm_chunk)
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
