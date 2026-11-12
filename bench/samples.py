#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write listening samples: Chinese voices, English voices, mixed text."""

import time

import numpy as np
import soundfile as sf

import kokoro_bench as kb

ZH_VOICES = {45: "zf_xiaobei", 46: "zf_xiaoni", 47: "zf_xiaoxiao", 48: "zf_xiaoyi",
             49: "zm_yunjian", 50: "zm_yunxi", 51: "zm_yunxia", 52: "zm_yunyang"}
EN_VOICES = {0: "af_alloy", 3: "af_heart", 11: "am_adam", 20: "bf_alice"}

zh_text = "据中央广播电视总台报道，按照现场指挥部部署，救援队已抵达吉隆口岸救援核心位置，开展排查搜救工作。今天气温是零下5度，其中百分之三十的区域需要重点排查。"
en_text = "The rescue team arrived at the core area and started the search operation at 9:30 in the morning."
mix_text = "你好，这里是智能语音合成服务。The service supports both Chinese and English, and the RTF is about 0.1."


def main():
    tts = kb.make_kokoro(10, model_file="model.onnx", lang="en")
    print(f"num_speakers={tts.num_speakers}", flush=True)
    for sid, name in ZH_VOICES.items():
        smp, sr, dt, dur = kb.synth(tts, zh_text, sid=sid)
        sf.write(f"{kb.OUT}/zh_{sid}_{name}.wav", smp, sr)
        print(f"zh {sid:>2} {name:<12} dur={dur:5.2f}s infer={dt:5.2f}s RTF={dt/dur:.3f}", flush=True)
    for sid, name in EN_VOICES.items():
        smp, sr, dt, dur = kb.synth(tts, en_text, sid=sid)
        sf.write(f"{kb.OUT}/en_{sid}_{name}.wav", smp, sr)
        print(f"en {sid:>2} {name:<12} dur={dur:5.2f}s infer={dt:5.2f}s RTF={dt/dur:.3f}", flush=True)
    for sid in (46, 0):
        smp, sr, dt, dur = kb.synth(tts, mix_text, sid=sid)
        sf.write(f"{kb.OUT}/mixed_{sid}.wav", smp, sr)
        print(f"mix {sid:>2} dur={dur:5.2f}s infer={dt:5.2f}s RTF={dt/dur:.3f}", flush=True)
    # full passage, same text as the production test.py
    text = kb.PROD_TEXT
    t0 = time.perf_counter()
    smp, sr, dt, dur = kb.synth(tts, text, sid=46)
    sf.write(f"{kb.OUT}/prod_zh_onecall_sid46.wav", smp, sr)
    print(f"prod one-call sid=46 dur={dur:.2f}s infer={dt:.2f}s wall={time.perf_counter()-t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
