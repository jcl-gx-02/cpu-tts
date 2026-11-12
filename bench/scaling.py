#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thread/worker scaling matrix for the production-style chunked request."""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import sherpa_onnx

import kokoro_bench as kb

KB = kb


def run_matrix(model_file, combos, text=KB.PROD_TEXT, sid=45, repeats=2, label=""):
    print(f"\n=== {label or model_file} (sid={sid}) ===", flush=True)
    sentences = KB.split_text_by_punctuation(text)
    for threads, workers in combos:
        try:
            tts = KB.make_kokoro(threads, model_file=model_file)
        except Exception as e:  # noqa: BLE001
            print(f"T={threads:<3} W={workers:<3} LOAD-FAIL {e}", flush=True)
            continue
        with ThreadPoolExecutor(max_workers=workers) as ex:
            ex.submit(KB.synth, tts, "预热。", sid=sid).result()
            walls, durs = [], []
            for _ in range(repeats):
                t0 = time.perf_counter()
                chunks = list(ex.map(lambda s: KB.synth(tts, s, sid=sid), sentences))
                walls.append(time.perf_counter() - t0)
                durs.append(sum(c[3] for c in chunks))
        wall = float(np.median(walls))
        dur = float(np.median(durs))
        print(
            f"T={threads:<3} W={workers:<3} chunks={len(sentences)} audio={dur:5.2f}s wall={wall:6.3f}s "
            f"RTF={wall/dur:.4f} xRT={dur/wall:6.1f}",
            flush=True,
        )
        del tts


if __name__ == "__main__":
    model = os.environ.get("KOKORO_MODEL", "model.onnx")
    combos = [(1, 10), (4, 10), (10, 10), (20, 10), (40, 10), (10, 20), (20, 20)]
    run_matrix(model, combos, label=f"kokoro {model}")
