#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare manual chunking (ThreadPool over sentences) with sherpa-onnx
internal batching via OfflineTtsConfig(max_num_sentences=N)."""

import os
import time

import numpy as np
import sherpa_onnx
from concurrent.futures import ThreadPoolExecutor

import kokoro_bench as kb

MM = kb.MM


def make(num_threads, max_num_sentences, model_file):
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=f"{MM}/{model_file}",
                voices=f"{MM}/voices.bin",
                tokens=f"{MM}/tokens.txt",
                lexicon=f"{MM}/lexicon-us-en.txt,{MM}/lexicon-zh.txt",
                data_dir=f"{MM}/espeak-ng-data",
                lang="en",
            ),
            num_threads=num_threads,
            debug=False,
            provider="cpu",
        ),
        rule_fsts=kb.ZH_RULE_FSTS,
        max_num_sentences=max_num_sentences,
    )
    return sherpa_onnx.OfflineTts(cfg)


def bench_batch(tts, text, sid, workers, repeats=3):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ex.submit(kb.synth, tts, "预热。", sid).result()
        walls, durs = [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            outs = list(ex.map(lambda _: kb.synth(tts, text, sid=sid), range(workers)))
            walls.append(time.perf_counter() - t0)
            durs.append(sum(o[3] for o in outs))
    return float(np.median(walls)), float(np.median(durs))


if __name__ == "__main__":
    model = os.environ.get("KOKORO_MODEL", "model.onnx")
    text = kb.PROD_TEXT
    sid = int(os.environ.get("SID", "45"))
    print(f"model={model} text_chars={len(text)} sentences={len(kb.split_text_by_punctuation(text))}", flush=True)
    for max_sent in (1, 4, 11, 16):
        for threads in (8, 16, 32, 60):
            for workers in (1, 4):
                try:
                    tts = make(threads, max_sent, model)
                except Exception as e:  # noqa: BLE001
                    print(f"max_sent={max_sent} T={threads} load-fail {str(e)[:60]}", flush=True)
                    continue
                wall, dur = bench_batch(tts, text, sid, workers)
                print(
                    f"max_sent={max_sent:<3} T={threads:<3} W={workers:<2} audio/req={dur/workers:5.2f}s "
                    f"wall={wall:6.3f}s RTF={wall/dur:.4f} xRT={dur/wall:6.1f}",
                    flush=True,
                )
                del tts
