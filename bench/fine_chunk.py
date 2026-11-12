#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does finer text chunking unlock better CPU utilisation for Kokoro?

Current server splits only on sentence punctuation, so an 11-chunk request
finishes when its longest chunk finishes. Splitting further should shorten
the critical path.
"""

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import kokoro_bench as kb

STRONG = re.compile(r"([，。！？；,\.\!\?\;：:]+)")
WEAK = re.compile(r"([、\s]+)")


def split_fine(text, max_len):
    pieces = []
    for sent in kb.split_text_by_punctuation(text):
        if len(sent) <= max_len:
            pieces.append(sent)
            continue
        # split on weak punctuation first, then hard-wrap on length
        parts = [p for p in WEAK.split(sent) if p.strip()]
        cur = ""
        for p in parts:
            if len(cur) + len(p) <= max_len:
                cur += p
            else:
                if cur:
                    pieces.append(cur)
                while len(p) > max_len:
                    pieces.append(p[:max_len])
                    p = p[max_len:]
                cur = p
        if cur:
            pieces.append(cur)
    return [p for p in pieces if p.strip("，。、,. ")]


def run(tts, text, sid, pieces, workers, repeats=2):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ex.submit(kb.synth, tts, "预热。", sid).result()
        walls, durs = [], []
        for _ in range(repeats):
            t0 = time.perf_counter()
            out = list(ex.map(lambda s: kb.synth(tts, s, sid=sid), pieces))
            walls.append(time.perf_counter() - t0)
            durs.append(sum(o[3] for o in out))
    wall, dur = float(np.median(walls)), float(np.median(durs))
    return wall, dur


if __name__ == "__main__":
    model = os.environ.get("KOKORO_MODEL", "model.onnx")
    sid = int(os.environ.get("SID", "45"))
    text = kb.PROD_TEXT
    print(f"model={model} chars={len(text)}", flush=True)
    for max_len in (8, 12, 20, 999):
        pieces = split_fine(text, max_len)
        for threads, workers in ((4, 16), (8, 12), (4, 32), (8, 24), (16, 10)):
            tts = kb.make_kokoro(threads, model_file=model, lang="en")
            wall, dur = run(tts, text, sid, pieces, workers)
            print(
                f"max_len={max_len:<4} nchunks={len(pieces):<3} T={threads:<3} W={workers:<3} "
                f"audio={dur:5.2f}s wall={wall:6.3f}s RTF={wall/dur:.4f} xRT={dur/wall:5.1f}",
                flush=True,
            )
            del tts
