#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-utterance RTF for the int8 model (zh / en / mixed)."""

import os

import kokoro_bench as kb

model = os.environ.get("KOKORO_MODEL", "model.int8.onnx")
tts = kb.make_kokoro(int(os.environ.get("T", "10")), model_file=model, lang="en")
print(f"num_speakers={tts.num_speakers}", flush=True)
for sid, tag in ((45, "zh-voice"), (0, "en-voice")):
    kb.bench_single(tts, kb.ZH_LONG, sid, f"zh_long sid={sid}({tag})", repeats=3)
    kb.bench_single(tts, kb.EN_SENT, sid, f"en_sent sid={sid}({tag})", repeats=3)
    kb.bench_single(tts, kb.MIXED_SENT, sid, f"mixed sid={sid}({tag})", repeats=3)
