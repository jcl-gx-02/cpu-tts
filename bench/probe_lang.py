#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe which lang / lexicon combinations work for zh, en and mixed text."""

import time

import numpy as np
import sherpa_onnx

MM = "/models/kokoro-multi-lang-v1_0"
ZH = "救援队已抵达核心位置。"
EN = "The rescue team arrived at the core area."
MIX = "你好，welcome to the system。"


def build(lang, lexicon, data_dir=True, lex_en=True, num_threads=8):
    lex = []
    if lex_en:
        lex.append(f"{MM}/lexicon-us-en.txt")
    if lexicon:
        lex.append(f"{MM}/lexicon-zh.txt")
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=f"{MM}/model.onnx",
                voices=f"{MM}/voices.bin",
                tokens=f"{MM}/tokens.txt",
                lexicon=",".join(lex),
                data_dir=f"{MM}/espeak-ng-data" if data_dir else "",
                dict_dir=f"{MM}/dict",
                lang=lang,
            ),
            num_threads=num_threads,
            debug=False,
            provider="cpu",
        ),
        rule_fsts=f"{MM}/phone-zh.fst,{MM}/date-zh.fst,{MM}/number-zh.fst",
    )
    return cfg


def try_cfg(name, **kw):
    try:
        cfg = build(**kw)
        if not cfg.validate():
            print(f"{name:<38} CONFIG-INVALID")
            return
        tts = sherpa_onnx.OfflineTts(cfg)
        out = []
        for lbl, txt, sid in (("zh", ZH, 0), ("en", EN, 0), ("mix", MIX, 0)):
            try:
                a = tts.generate(txt, sid=sid)
                dur = len(a.samples) / a.sample_rate
                out.append(f"{lbl}=OK({dur:.2f}s)")
            except Exception as e:  # noqa: BLE001
                out.append(f"{lbl}=FAIL({str(e)[:48]})")
        print(f"{name:<38} " + "  ".join(out), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"{name:<38} LOAD-FAIL({str(e)[:60]})", flush=True)


for lang in ["zh", "en", "", "cmn", "en-us"]:
    for lex_en in [True, False]:
        for data_dir in [True, False]:
            try_cfg(f"lang={lang!r} lex_en={lex_en} espeak={data_dir}", lang=lang, lexicon=True,
                    data_dir=data_dir, lex_en=lex_en)
