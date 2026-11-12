#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wall time vs CPU time for fp32 / int8 Kokoro, plus piper zh on English text."""

import os
import time

import numpy as np
import soundfile as sf
import sherpa_onnx

import kokoro_bench as kb

ZH = kb.ZH_LONG
EN = kb.EN_SENT


def diag(name, tts, text, sid, repeats=2):
    tts.generate(text, sid=sid)
    rows = []
    for _ in range(repeats):
        w0, c0 = time.perf_counter(), time.process_time()
        a = tts.generate(text, sid=sid)
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        dur = len(a.samples) / a.sample_rate
        rows.append((wall, cpu, dur))
    wall, cpu, dur = (float(np.median([r[i] for r in rows])) for i in range(3))
    print(
        f"{name:<28} audio={dur:6.2f}s wall={wall:7.3f}s cpu={cpu:7.3f}s "
        f"RTF={wall/dur:6.4f} cpu/audio={cpu/dur:6.3f} parallelism={cpu/wall:5.2f}",
        flush=True,
    )
    return wall, cpu, dur


def piper_en_test():
    """Does the existing piper zh model do anything sane with English?"""
    for voice in ("vits-piper-zh_CN-xiao_ya-medium", "vits-piper-zh_CN-chaowen-medium"):
        onnx = voice.replace("vits-piper-zh_CN-", "zh_CN-") + ".onnx"
        tts = kb.make_vits(voice, onnx, 8)
        for lbl, text in (("zh", "救援队已抵达核心位置。"), ("en", "The rescue team arrived at the core area.")):
            try:
                smp, sr, _, dur = kb.synth(tts, text, sid=0)
                peak = float(np.abs(smp).max()) if len(smp) else 0.0
                print(f"[piper {voice}] {lbl}: dur={dur:5.2f}s peak={peak:.3f} samples={len(smp)}", flush=True)
                sf.write(f"{kb.OUT}/piper_{voice.split('-')[-2]}_{lbl}.wav", smp, sr)
            except Exception as e:  # noqa: BLE001
                print(f"[piper {voice}] {lbl}: FAIL {str(e)[:70]}", flush=True)


if __name__ == "__main__":
    model = os.environ.get("KOKORO_MODEL", "model.onnx")
    threads = int(os.environ.get("T", "10"))
    tts = kb.make_kokoro(threads, model_file=model, lang="en")
    diag(f"kokoro {model} T={threads}", tts, ZH, 45)
    diag(f"kokoro {model} T={threads}", tts, EN, 0)
    if os.environ.get("PIPER"):
        piper_en_test()
