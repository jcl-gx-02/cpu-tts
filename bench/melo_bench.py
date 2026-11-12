#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Benchmark sherpa-onnx vits-melo-tts-zh_en (single bilingual VITS model)."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import sherpa_onnx

import kokoro_bench as kb

D = os.environ.get("MELO_DIR", "/melo/vits-melo-tts-zh_en")

ZH = "据中央广播电视总台报道，按照现场指挥部部署，救援队已抵达吉隆口岸救援核心位置，开展排查搜救工作。"
EN = "The rescue team arrived at the core area and started the search operation at 9:30 in the morning."
MIX = "你好，这里是智能语音合成服务。The service supports both Chinese and English."


def build(num_threads=10, model_file="model.onnx"):
    try:
        vits = sherpa_onnx.OfflineTtsVitsModelConfig(
            model=f"{D}/{model_file}",
            lexicon=f"{D}/lexicon.txt",
            tokens=f"{D}/tokens.txt",
            dict_dir=f"{D}/dict",
        )
    except TypeError as e:
        print("vits cfg TypeError:", e, flush=True)
        print(sherpa_onnx.OfflineTtsVitsModelConfig.__init__.__doc__, flush=True)
        raise
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=num_threads, provider="cpu"),
        rule_fsts=f"{D}/phone.fst,{D}/date.fst,{D}/number.fst,{D}/new_heteronym.fst",
    )
    if not cfg.validate():
        raise RuntimeError("invalid config")
    t0 = time.time()
    tts = sherpa_onnx.OfflineTts(cfg)
    print(f"[load] melo {model_file} T={num_threads} in {time.time()-t0:.2f}s", flush=True)
    return tts


def timed(tts, text, sid=0, repeats=3):
    tts.generate(text, sid=sid)
    rows = []
    for _ in range(repeats):
        w0, c0 = time.perf_counter(), time.process_time()
        a = tts.generate(text, sid=sid)
        rows.append((time.perf_counter() - w0, time.process_time() - c0, len(a.samples) / a.sample_rate))
    wall, cpu, dur = (float(np.median([r[i] for r in rows])) for i in range(3))
    return wall, cpu, dur


def main():
    tts = build(10)
    print(f"num_speakers={tts.num_speakers} sample_rate={tts.sample_rate}", flush=True)
    for lbl, text in (("zh", ZH), ("en", EN), ("mix", MIX)):
        wall, cpu, dur = timed(tts, text)
        smp, sr, _, _ = kb.synth(tts, text)
        sf.write(f"{kb.OUT}/melo_{lbl}.wav", smp, sr)
        print(
            f"melo {lbl:<4} chars={len(text):>3} audio={dur:5.2f}s wall={wall:6.3f}s cpu={cpu:6.2f}s "
            f"RTF={wall/dur:.4f} cpu/audio={cpu/dur:.2f}",
            flush=True,
        )
    # production-style parallel request
    pieces = kb.split_text_by_punctuation(kb.PROD_TEXT)
    for threads, workers in ((10, 10), (4, 10), (10, 20)):
        eng = build(threads)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            ex.submit(kb.synth, eng, "预热。", 0).result()
            walls, durs = [], []
            for _ in range(3):
                t0 = time.perf_counter()
                out = list(ex.map(lambda s: kb.synth(eng, s, sid=0), pieces))
                walls.append(time.perf_counter() - t0)
                durs.append(sum(o[3] for o in out))
        wall, dur = float(np.median(walls)), float(np.median(durs))
        print(
            f"melo prod threads={threads} workers={workers} chunks={len(pieces)} audio={dur:5.2f}s "
            f"wall={wall:6.3f}s RTF={wall/dur:.4f} xRT={dur/wall:5.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
