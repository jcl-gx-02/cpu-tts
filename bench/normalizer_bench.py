#!/usr/bin/env python3
"""Measure preprocessing overhead and optional end-to-end TTS RTF."""

from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
from pathlib import Path

import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_app"))

from text_routing import normalize_domain_text  # noqa: E402


ALERT = (
    "2026年11月12日8：35，红岭变220kV红荷线1198F6开关跳闸，"
    "相间距离I段、零序过流III段保护动作，故障相别AB相，"
    "故障测距3.012km，TA二次A相8.014A、B相7.369A。"
    "负荷率30%，电压±1.5%，请值班员立即核对10kV母线I段电压并汇报。"
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def bench_preprocessor(iterations: int) -> None:
    samples = {
        "clock": "8：35",
        "alert_137": ALERT,
        "alert_x20": ALERT * 20,
    }
    for name, text in samples.items():
        for _ in range(20):
            normalize_domain_text(text)
        timings = []
        for _ in range(iterations):
            started = time.perf_counter()
            normalize_domain_text(text)
            timings.append((time.perf_counter() - started) * 1000)
        print(
            f"{name:<12} chars={len(text):>4} "
            f"p50={statistics.median(timings):.3f}ms "
            f"p95={percentile(timings, 0.95):.3f}ms"
        )


def bench_api(url: str, repeats: int) -> None:
    normalized = normalize_domain_text(ALERT)
    results = {"raw": [], "normalized": []}
    for index in range(repeats):
        order = (("raw", ALERT), ("normalized", normalized))
        if index % 2:
            order = tuple(reversed(order))
        for name, text in order:
            started = time.perf_counter()
            response = requests.post(
                url,
                json={"text": text, "voice": "female", "speed": 1.0},
                timeout=120,
            )
            wall_s = time.perf_counter() - started
            response.raise_for_status()
            audio, sample_rate = sf.read(
                io.BytesIO(response.content), dtype="float32"
            )
            results[name].append((wall_s, len(audio) / sample_rate))

    for name, rows in results.items():
        wall_s = statistics.median(row[0] for row in rows)
        audio_s = statistics.median(row[1] for row in rows)
        print(
            f"api_{name:<10} wall={wall_s:.4f}s audio={audio_s:.3f}s "
            f"RTF={wall_s / audio_s:.5f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--api-url", default="")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    bench_preprocessor(args.iterations)
    if args.api_url:
        bench_api(args.api_url, args.repeats)


if __name__ == "__main__":
    main()
