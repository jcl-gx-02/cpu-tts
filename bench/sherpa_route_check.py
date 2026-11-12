#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证 sherpa 服务（piper zh + piper en 双模型路由）的正确性与性能。

服务接口：POST /api/tts  {"text": ..., "voice": "auto|male|female|en_male|en_female", "sid": 0, "speed": 1.0}
"""

import io
import json
import statistics
import time

import requests
import soundfile as sf

URL = "http://127.0.0.1:8000/api/tts"
OUT = "../bench/out_sherpa_route"

CASES = {
    "zh": ("female", "古埠变 15F7 古新线 TV 断线告警，相间距离 I 段、零序过流 III 段保护动作，QF 跳闸，重合成功，故障相别 BC 相，故障测距 2.284km (古埠变侧)，TA 二次 B 相 7.935A,C 相 9.041A."),
    "en": ("auto", "The rescue team arrived at the core area and started the search operation at 9:30."),
    "male": ("auto:male", "ABC三相电正常。在B处设置一个启动动作。新丰变 ,19F5 新丰北线 ,UPS 电源切换，相间距离 III 段、零序过流 II 段保护动作，QF 跳闸，重合成功，故障相别, AB 相，故障测距 5.219km ,新丰变侧，TA 二次, A 相 7.142A,B 相 8.853A."),
    "mix": ("auto", "龙潭变 16F4 龙泉线 MOA 计数器动作，相间距离 III 段、零序过流 III 段保护动作，QF 跳闸，重合成功，故障相别 AC 相，故障测距 4.628km (龙潭变侧)，TA 二次 A 相 7.418A,C 相 8.762A."),
    "mix_male": ("auto:male", "青园变 6F1 青东线 BAT 单体电压越下限，UPS 自动切旁路，一次设备 QF，QS 位置无变位，CT，PT 采样正常，无故障跳闸。"),
    "power": ("auto", "乐园变11F5园开线相间距离I段、零序过流段保护动作，开关跳闸，重合成功，故障相别，AC相，故障测距1.359km，距离乐园变侧，故障电流二次值A相7.594A,C相9.313A."),
    "power300": ("auto","红岭变 1198F6 红荷线 GIS 气室 SF6 气压低告警，相间距离 I 段、零序过流 II 段保护动作，QF 跳闸，重合成功，故障相别，AB 相，故障测距 3.012km，TA 二次 A 相 8.014A,B 相 7.369A."),
    "testabc":("auto:male","跟着我来读一下ABC，打开开关6631f5，继续ILKLMNOPQRSTUVWXYZ")
}


def call(voice, text, repeats=3):
    rows = []
    for _ in range(repeats):
        payload = {"text": text, "voice": voice, "sid": 0, "speed": 1.0}
        t0 = time.time()
        r = requests.post(URL, json=payload, timeout=600)
        wall = time.time() - t0
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        d, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        rows.append((wall, len(d) / sr, r.content))
    wall = statistics.median(x[0] for x in rows)
    dur = statistics.median(x[1] for x in rows)
    return wall, dur, rows[-1][2]


def main():
    import os

    os.makedirs(OUT, exist_ok=True)
    print(f"目标: {URL}\n" + "-" * 88)
    for name, (voice, text) in CASES.items():
        try:
            wall, dur, content = call(voice, text)
        except Exception as e:  # noqa: BLE001
            print(f"{name:<10} voice={voice:<9} FAILED: {str(e)[:90]}", flush=True)
            continue
        with open(f"{OUT}/{name}.wav", "wb") as f:
            f.write(content)
        print(
            f"{name:<10} voice={voice:<9} chars={len(text):>4} audio={dur:6.2f}s wall={wall:6.3f}s "
            f"RTF={wall/dur:.4f} xRT={dur/wall:5.1f}",
            flush=True,
        )
    print("-" * 88)
    print("音频已保存到", OUT)


if __name__ == "__main__":
    main()
