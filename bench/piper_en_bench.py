#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark a piper English VITS voice (for the hybrid routing option)."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import sherpa_onnx

import kokoro_bench as kb

D = os.environ.get("PIPER_EN_DIR", "/en/vits-piper-en_US-amy-medium")
ONNX = os.environ.get("PIPER_EN_ONNX", "en_US-amy-medium.onnx")

EN_SHORT = "The rescue team arrived at the core area and started the search operation at 9:30 in the morning."
EN_LONG = (
    "It is a new generation high performance intelligent terminal built for critical command and dispatch scenarios. "
    "With a large model and an agent as its technical core, it carries a lightweight on-device intelligence engine, "
    "combines multi-modal perception with intelligent interaction, and builds an efficient collaborative "
    "command workflow. The product adopts on-device intelligence with cloud collaboration, and relies on the "
    "lightweight engine to achieve local real-time response for core business."
)


def build(num_threads=10):
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f"{D}/{ONNX}",
                tokens=f"{D}/tokens.txt",
                data_dir=f"{D}/espeak-ng-data",
            ),
            num_threads=num_threads,
            provider="cpu",
        ),
    )
    if not cfg.validate():
        raise RuntimeError("invalid config")
    t0 = time.time()
    tts = sherpa_onnx.OfflineTts(cfg)
    print(f"[load] piper-en T={num_threads} in {time.time()-t0:.2f}s", flush=True)
    return tts


def timed(tts, text, repeats=3):
    tts.generate(text, sid=0)
    rows = []
    for _ in range(repeats):
        w0, c0 = time.perf_counter(), time.process_time()
        a = tts.generate(text, sid=0)
        rows.append((time.perf_counter() - w0, time.process_time() - c0, len(a.samples) / a.sample_rate))
    return (float(np.median([r[i] for r in rows])) for i in range(3))


if __name__ == "__main__":
    tts = build(10)
    print(f"num_speakers={tts.num_speakers}", flush=True)
    for lbl, text in (("en_short", EN_SHORT), ("en_long", EN_LONG), ("zh", kb.ZH_SENT)):
        wall, cpu, dur = timed(tts, text)
        smp, sr, _, _ = kb.synth(tts, text)
        sf.write(f"{kb.OUT}/piper_en_{lbl}.wav", smp, sr)
        print(
            f"piper-en {lbl:<9} chars={len(text):>4} audio={dur:5.2f}s wall={wall:6.3f}s cpu={cpu:5.2f}s "
            f"RTF={wall/dur:.4f} cpu/audio={cpu/dur:.2f}",
            flush=True,
        )
    pieces = kb.split_text_by_punctuation(EN_LONG)
    for threads, workers in ((10, 10), (4, 10)):
        eng = build(threads)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            ex.submit(kb.synth, eng, "warm up.").result()
            walls, durs = [], []
            for _ in range(3):
                t0 = time.perf_counter()
                out = list(ex.map(lambda s: kb.synth(eng, s), pieces))
                walls.append(time.perf_counter() - t0)
                durs.append(sum(o[3] for o in out))
        wall, dur = float(np.median(walls)), float(np.median(durs))
        print(
            f"piper-en prod threads={threads} workers={workers} chunks={len(pieces)} audio={dur:5.2f}s "
            f"wall={wall:6.3f}s RTF={wall/dur:.4f} xRT={dur/wall:5.1f}",
            flush=True,
        )
