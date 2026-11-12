#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check whether the `lang` field changes phonemization for zh vs en text."""

import numpy as np
import soundfile as sf
import sherpa_onnx

MM = "/models/kokoro-multi-lang-v1_0"
ZH = "救援队已抵达核心位置。"
EN = "The rescue team arrived at the core area."
MIX = "你好，welcome to the system。"


def build(lang, lex_as_list=True):
    lex = f"{MM}/lexicon-us-en.txt,{MM}/lexicon-zh.txt" if lex_as_list else f"{MM}/lexicon-zh.txt"
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=f"{MM}/model.onnx",
                voices=f"{MM}/voices.bin",
                tokens=f"{MM}/tokens.txt",
                lexicon=lex,
                data_dir=f"{MM}/espeak-ng-data",
                lang=lang,
            ),
            num_threads=6,
            provider="cpu",
        ),
        rule_fsts=f"{MM}/phone-zh.fst,{MM}/date-zh.fst,{MM}/number-zh.fst",
    )
    return sherpa_onnx.OfflineTts(cfg)


def gen(tts, text, sid=45):
    a = tts.generate(text, sid=sid)
    return np.asarray(a.samples, dtype=np.float32), a.sample_rate


def cmp(a, b):
    n = min(len(a), len(b))
    if n == 0:
        return "len-a=0/len-b=0"
    x, y = a[:n], b[:n]
    denom = np.sqrt((x**2).sum() * (y**2).sum())
    corr = float((x * y).sum() / denom) if denom else 0.0
    return f"len {len(a)/1000:.1f}k/{len(b)/1000:.1f}k corr={corr:.4f}"


engines = {}
for lang in ["en", "zh", "cmn", ""]:
    try:
        engines[lang] = build(lang)
        print(f"loaded lang={lang!r}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"load lang={lang!r} failed: {str(e)[:70]}", flush=True)

for label, text in (("zh", ZH), ("en", EN), ("mix", MIX)):
    base = None
    for lang, tts in engines.items():
        try:
            smp, sr = gen(tts, text, sid=45)
            if lang == "en":
                base = smp
                print(f"{label:<4} lang={lang!r:<6} OK dur={len(smp)/sr:.2f}s (reference)", flush=True)
                sf.write(f"/bench/out/lang_{label}_lang_en.wav", smp, sr)
            else:
                print(f"{label:<4} lang={lang!r:<6} OK dur={len(smp)/sr:.2f}s vs en: {cmp(smp, base)}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{label:<4} lang={lang!r:<6} FAIL {str(e)[:60]}", flush=True)

# lexicon list vs zh-only lexicon
t_list, t_zh = build("en", True), build("en", False)
for label, text in (("zh", ZH), ("en", EN)):
    a, sr = gen(t_list, text)
    b, sr = gen(t_zh, text)
    print(f"lexicon us-en+zh vs zh-only on {label}: {cmp(a, b)}", flush=True)
