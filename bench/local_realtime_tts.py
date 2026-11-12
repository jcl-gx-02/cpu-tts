#!/usr/bin/env python3
"""Call the SSE TTS endpoint and play each returned WAV chunk immediately.

The server emits newline-delimited SSE events. Every event contains one complete
WAV chunk encoded as base64, so this client does not request raw PCM or send the
streaming parameters used by other TTS services.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
import wave
from dataclasses import dataclass
from typing import Iterable

import requests

try:
    import pyaudio
except ImportError:
    pyaudio = None


DEFAULT_TEXT = "现在的时间是3：55，请问您需要我为您播报什么内容呢？"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


@dataclass(frozen=True)
class WavChunk:
    channels: int
    sample_width: int
    sample_rate: int
    frames: bytes
    duration_s: float


class AudioPlayer:
    """Lazily open a PyAudio stream using the WAV parameters from the server."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._pyaudio = None
        self._stream = None
        self._params: tuple[int, int, int] | None = None

    def _format_for_width(self, sample_width: int) -> int:
        formats = {
            1: pyaudio.paUInt8,
            2: pyaudio.paInt16,
            3: pyaudio.paInt24,
            4: pyaudio.paInt32,
        }
        if sample_width not in formats:
            raise ValueError(f"不支持的采样位宽: {sample_width * 8} bit")
        return formats[sample_width]

    def play(self, chunk: WavChunk) -> None:
        if not self.enabled:
            return
        if pyaudio is None:
            raise RuntimeError(
                "未安装 PyAudio。请安装后重试（如 pip install pyaudio），"
                "或使用 --no-play 仅接收并保存音频。"
            )

        params = (chunk.channels, chunk.sample_width, chunk.sample_rate)
        if self._stream is None or params != self._params:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
            if self._pyaudio is None:
                self._pyaudio = pyaudio.PyAudio()
            self._stream = self._pyaudio.open(
                format=self._format_for_width(chunk.sample_width),
                channels=chunk.channels,
                rate=chunk.sample_rate,
                output=True,
                frames_per_buffer=1024,
            )
            self._params = params

        self._stream.write(chunk.frames, exception_on_underflow=False)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pyaudio is not None:
            self._pyaudio.terminate()
            self._pyaudio = None


def decode_wav_chunk(encoded: str) -> WavChunk:
    wav_bytes = base64.b64decode(encoded)
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE":
            raise ValueError(f"不支持压缩 WAV: {wav_file.getcomptype()}")
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        frames = wav_file.readframes(frame_count)
    return WavChunk(
        channels=channels,
        sample_width=sample_width,
        sample_rate=sample_rate,
        frames=frames,
        duration_s=frame_count / sample_rate if sample_rate else 0.0,
    )


def iter_sse_payloads(response: requests.Response) -> Iterable[dict]:
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        raw = line[5:].strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法解析 SSE 数据: {raw[:160]}") from exc


def save_wav(
    output_path: str,
    chunks: list[WavChunk],
) -> None:
    if not chunks:
        return
    params = {(c.channels, c.sample_width, c.sample_rate) for c in chunks}
    if len(params) != 1:
        raise ValueError(f"分块音频参数不一致，无法合并: {sorted(params)}")
    first = chunks[0]
    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(first.channels)
        wav_file.setsampwidth(first.sample_width)
        wav_file.setframerate(first.sample_rate)
        for chunk in chunks:
            wav_file.writeframes(chunk.frames)


def check_health(base_url: str, timeout: float) -> None:
    try:
        response = requests.get(base_url.rstrip("/") + "/health", timeout=timeout)
        response.raise_for_status()
        status = response.json()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"无法访问 TTS 服务 {base_url}/health: {exc}\n"
            "请检查 SSH 隧道，例如:\n"
            "ssh -N -L 8000:127.0.0.1:8000 root@172.18.190.100"
        ) from exc
    except ValueError as exc:
        raise RuntimeError(f"健康检查返回了非 JSON 内容: {exc}") from exc
    if status.get("status") != "ok":
        raise RuntimeError(f"TTS 服务未就绪: {status}")


def play_sse_tts(args: argparse.Namespace) -> int:
    if not args.no_play and pyaudio is None:
        print(
            "未安装 PyAudio。请执行 pip install -r requirements-local-playback.txt，"
            "或使用 --no-play 仅接收并保存音频。",
            file=sys.stderr,
        )
        return 1

    check_health(args.base_url, args.timeout)
    endpoint = args.base_url.rstrip("/") + "/api/tts_sse"
    payload = {
        "text": args.text,
        "voice": args.voice,
        "sid": args.sid,
        "speed": args.speed,
    }

    print(f"请求流式 TTS: {endpoint}")
    print(f"voice={args.voice} speed={args.speed} play={not args.no_play}")
    started = time.perf_counter()
    player = AudioPlayer(enabled=not args.no_play)
    chunks: list[WavChunk] = []
    first_chunk_seconds: float | None = None

    try:
        with requests.post(
            endpoint,
            json=payload,
            stream=True,
            timeout=args.timeout,
        ) as response:
            if response.status_code != 200:
                raise RuntimeError(
                    f"HTTP {response.status_code}: {response.text[:500]}"
                )
            for event in iter_sse_payloads(response):
                if first_chunk_seconds is None:
                    first_chunk_seconds = time.perf_counter() - started
                chunk = decode_wav_chunk(event["audio_base64"])
                chunks.append(chunk)
                player.play(chunk)
                print(
                    f"[chunk {event.get('chunk_index', len(chunks) - 1)}] "
                    f"voice={event.get('voice_used', args.voice)} "
                    f"duration={chunk.duration_s:.3f}s "
                    f"text={event.get('text', '')}"
                )
    except KeyboardInterrupt:
        print("\n播放已手动中断。")
    except (requests.RequestException, RuntimeError, ValueError, KeyError) as exc:
        print(f"\n调用失败: {exc}", file=sys.stderr)
        return 1
    finally:
        player.close()

    total_duration = sum(chunk.duration_s for chunk in chunks)
    elapsed = time.perf_counter() - started
    if not chunks:
        print("服务未返回任何音频分块。", file=sys.stderr)
        return 1
    print(
        f"接收{'并播放' if not args.no_play else ''}完成: "
        f"chunks={len(chunks)} audio={total_duration:.3f}s "
        f"elapsed={elapsed:.3f}s"
    )
    if first_chunk_seconds is not None:
        print(f"首块到达耗时: {first_chunk_seconds:.3f}s")

    if not args.no_save:
        try:
            save_wav(args.output, chunks)
        except (OSError, ValueError) as exc:
            print(f"保存 WAV 失败: {exc}", file=sys.stderr)
            return 1
        print(f"已保存 WAV: {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="调用 sherpa-tts 的 SSE 接口并实时播放音频分块"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument(
        "--voice",
        default="female",
        choices=[
            "female",
            "male",
            "en_female",
            "en_male",
            "auto",
            "auto:female",
            "auto:male",
        ],
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sid", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output", default="local_tts_output.wav")
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="只接收并保存音频，不打开本机扬声器",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="实时播放但不保存 WAV",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.text.strip():
        print("--text 不能为空", file=sys.stderr)
        return 2
    return play_sse_tts(args)


if __name__ == "__main__":
    raise SystemExit(main())
