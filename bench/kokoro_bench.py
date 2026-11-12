#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Feasibility benchmark: kokoro-multi-lang-v1_0 vs current vits-piper zh models.

Runs inside the existing tts-sherpa image:
  docker run --rm \
    -v $PWD/model_weight:/models:ro -v $PWD/bench:/bench \
    --entrypoint python <IMAGE> /bench/kokoro_bench.py --mode kokoro
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import sherpa_onnx

MM = os.environ.get("KOKORO_DIR", "/models/kokoro-multi-lang-v1_0")
OUT = os.environ.get("BENCH_OUT", "/bench/out")

ZH_RULE_FSTS = f"{MM}/phone-zh.fst,{MM}/date-zh.fst,{MM}/number-zh.fst"

ZH_SENT = "按照现场指挥部部署，救援队已抵达核心位置。"
ZH_LONG = (
    "据中央广播电视总台国家应急广播8月29日报道，按照现场指挥部部署，"
    "应急管理部、国家消防救援局调派的中国救援队与拉萨市消防救援机动支队抵达吉隆口岸救援核心位置后，开展排查搜救工作。"
)
EN_SENT = "The rescue team arrived at the core area and started the search operation."
MIXED_SENT = "你好，欢迎使用智能语音合成服务，this service supports both Chinese and English."
PROD_TEXT = (
    "它是面向关键指挥调度场景打造的新一代高性能智能终端，以大模型加智能体为技术核心，搭载内置轻量化智能引擎，"
    "融合多模态感知与智能交互能力，构建高效协同的指挥作业新范式。产品采用端侧智能，云端协同架构，"
    "依托轻量引擎实现核心业务本地实时响应，离线运行，部署高效，便于规模化推广。"
)


def split_text_by_punctuation(text):
    pattern = r"([，。！？；,\.\!\?\;]+)"
    chunks = re.split(pattern, text)
    sentences = []
    for i in range(0, len(chunks) - 1, 2):
        s = chunks[i].strip() + chunks[i + 1].strip()
        if s:
            sentences.append(s)
    if len(chunks) % 2 != 0 and chunks[-1].strip():
        sentences.append(chunks[-1].strip())
    return sentences


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def make_kokoro(num_threads, model_file="model.onnx", lang="zh", lex_en=True):
    lexicons = [f"{MM}/lexicon-zh.txt"]
    if lex_en:
        lexicons.append(f"{MM}/lexicon-us-en.txt")
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                model=f"{MM}/{model_file}",
                voices=f"{MM}/voices.bin",
                tokens=f"{MM}/tokens.txt",
                lexicon=",".join(lexicons),
                data_dir=f"{MM}/espeak-ng-data",
                dict_dir=f"{MM}/dict",
                lang=lang,
            ),
            num_threads=num_threads,
            debug=False,
            provider="cpu",
        ),
        rule_fsts=ZH_RULE_FSTS,
    )
    if not cfg.validate():
        raise RuntimeError("invalid kokoro config")
    t0 = time.time()
    tts = sherpa_onnx.OfflineTts(cfg)
    print(f"[load] kokoro {model_file} threads={num_threads} lang={lang} in {time.time()-t0:.2f}s", flush=True)
    return tts


def make_vits(voice_dir, onnx_name, num_threads):
    d = f"/models/{voice_dir}"
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=f"{d}/{onnx_name}",
                tokens=f"{d}/tokens.txt",
                lexicon=f"{d}/lexicon.txt",
            ),
            num_threads=num_threads,
            debug=False,
            provider="cpu",
        ),
        rule_fsts=f"{d}/phone.fst,{d}/date.fst,{d}/number.fst",
    )
    if not cfg.validate():
        raise RuntimeError("invalid vits config")
    t0 = time.time()
    tts = sherpa_onnx.OfflineTts(cfg)
    print(f"[load] vits {onnx_name} threads={num_threads} in {time.time()-t0:.2f}s", flush=True)
    return tts


def synth(tts, text, sid=0, speed=1.0):
    t0 = time.perf_counter()
    audio = tts.generate(text, sid=sid, speed=speed)
    dt = time.perf_counter() - t0
    samples = np.asarray(audio.samples, dtype=np.float32)
    dur = len(samples) / audio.sample_rate if audio.sample_rate else 0.0
    return samples, audio.sample_rate, dt, dur


def bench_single(tts, text, sid, label, repeats=3, save=None):
    synth(tts, text, sid=sid)  # warm up
    rows = []
    for _ in range(repeats):
        _, sr, dt, dur = synth(tts, text, sid=sid)
        rows.append((dt, dur, dt / dur if dur else float("inf")))
    dt = float(np.median([r[0] for r in rows]))
    dur = float(np.median([r[1] for r in rows]))
    rtf = dt / dur if dur else float("inf")
    print(
        f"[single] {label:<22} chars={len(text):>4} audio={dur:6.2f}s infer={dt:6.3f}s RTF={rtf:.4f} xRT={1/rtf:6.1f}",
        flush=True,
    )
    if save:
        samples, sr, _, _ = synth(tts, text, sid=sid)
        sf.write(f"{OUT}/{save}", samples, sr)
    return {"label": label, "audio_s": dur, "infer_s": dt, "rtf": rtf}


def bench_parallel(engine, text, sid, workers, label, repeats=3, save=None):
    sentences = split_text_by_punctuation(text)
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        ex.submit(synth, engine, "预热。", sid).result()  # warm up
        for _ in range(repeats):
            t0 = time.perf_counter()
            chunks = list(ex.map(lambda s: synth(engine, s, sid=sid), sentences))
            wall = time.perf_counter() - t0
            dur = sum(c[3] for c in chunks)
            results.append((wall, dur))
    wall = float(np.median([r[0] for r in results]))
    dur = float(np.median([r[1] for r in results]))
    rtf = wall / dur if dur else float("inf")
    print(
        f"[parallel] {label:<22} chunks={len(sentences):>3} audio={dur:6.2f}s wall={wall:6.3f}s "
        f"RTF={rtf:.4f} xRT={1/rtf:6.1f} workers={workers}",
        flush=True,
    )
    if save:
        parts = []
        sr = 24000
        for s in sentences:
            smp, sr, _, _ = synth(engine, s, sid=sid)
            parts.append(smp)
        sf.write(f"{OUT}/{save}", np.concatenate(parts), sr)
    return {"label": label, "chunks": len(sentences), "audio_s": dur, "wall_s": wall, "rtf": rtf}


def scan_sids(tts, limit=None):
    n = tts.num_speakers
    print(f"[sids] num_speakers={n}", flush=True)
    rows = []
    for sid in range(n if limit is None else min(limit, n)):
        try:
            _, sr, _, dur = synth(tts, "你好，世界。", sid=sid)
            sync = dur
            _, _, _, dur_en = synth(tts, "Hello world, this is a test.", sid=sid)
            rows.append((sid, round(sync, 2), round(dur_en, 2)))
        except Exception as e:  # noqa: BLE001
            rows.append((sid, f"ERR {type(e).__name__}", ""))
    for sid, a, b in rows:
        print(f"  sid={sid:>3} zh_dur={a:<12} en_dur={b}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["kokoro", "vits", "both"], default="kokoro")
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--sid", type=int, default=None)
    ap.add_argument("--lang", default="en")
    ap.add_argument("--scan-sids", action="store_true")
    ap.add_argument("--model-file", default="model.onnx")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    summary = {"mode": args.mode, "threads": args.threads, "workers": args.workers}

    if args.mode in ("kokoro", "both"):
        tts = make_kokoro(args.threads, model_file=args.model_file, lang=args.lang)
        summary["kokoro_rss_mb"] = round(rss_mb(), 1)
        summary["kokoro_num_speakers"] = tts.num_speakers
        if args.scan_sids:
            scan_sids(tts)
        sid = args.sid if args.sid is not None else 0
        sfx = args.suffix
        summary["kokoro"] = [
            bench_single(tts, ZH_SENT, sid, f"zh_sent sid={sid}", save=f"kokoro_zh_{sid}{sfx}.wav"),
            bench_single(tts, EN_SENT, sid, f"en_sent sid={sid}", save=f"kokoro_en_{sid}{sfx}.wav"),
            bench_single(tts, MIXED_SENT, sid, f"mixed sid={sid}", save=f"kokoro_mixed_{sid}{sfx}.wav"),
            bench_parallel(tts, PROD_TEXT, sid, args.workers, f"prod_text sid={sid}", save=f"kokoro_prod_{sid}{sfx}.wav"),
        ]
        summary["kokoro_rss_mb_after"] = round(rss_mb(), 1)
        del tts

    if args.mode in ("vits", "both"):
        tts = make_vits("vits-piper-zh_CN-xiao_ya-medium", "zh_CN-xiao_ya-medium.onnx", args.threads)
        summary["vits_rss_mb"] = round(rss_mb(), 1)
        summary["vits"] = [
            bench_single(tts, ZH_SENT, 0, "vits zh_sent"),
            bench_parallel(tts, PROD_TEXT, 0, args.workers, "vits prod_text"),
        ]

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
