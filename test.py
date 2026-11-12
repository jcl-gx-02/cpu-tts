#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import requests
import io
import json
import base64
import soundfile as sf
import numpy as np

# 服务端 API 地址
API_URL_FULL = "http://127.0.0.1:8000/api/tts"
API_URL_SSE = "http://127.0.0.1:8000/api/tts_sse"

# 测试长文本
# TEXT = "它是面向关键指挥调度场景打造的新一代高性能智能终端，以大模型加智能体为技术核心，搭载内置轻量化智能引擎，融合多模态感知与智能交互能力，构建高效协同的指挥作业新范式。产品采用端侧智能，云端协同架构，依托轻量引擎实现核心业务本地实时响应，离线运行，部署高效，便于规模化推广。接入应用平台即可拓展任务协同，智能问答，数据交互与决策辅助，兼顾便捷部署与持续进化。终端搭载高算力模块，六麦协同拾音阵列技术，配备高清工业触控屏，接口丰富，扩展性强，可适配多行业指挥调度场景。具备语音唤醒，语音操控，实时语音转写，智能数据采集，自动日志生成等核心智能能力，配套完备调度功能，全面满足高安全，高可靠的运行标准。"

TEXT = "现在的时间是8：35，据中央广播电视总台国家应急广播8月29日报道，按照现场指挥部部署，应急管理部、国家消防救援局调派的中国救援队与拉萨市消防救援机动支队抵达吉隆口岸救援核心位置后，开展排查搜救工作。中国救援队与拉萨市消防救援机动支队陆地搜救队等联合编组，经过约9个小时的连续急行军，抵达受灾严重的吉隆口岸救援核心位置，开展排查搜救。而尼泊尔方面相关研究机构专家向总台记者表示，“是尼泊尔一侧发生冰崩，导致了此次泥石流灾害的发生”。　同时，海叔注意到国内多家媒体报道、转载了一则新闻，指称“尼泊尔改变立场，向中国等国救援队开绿灯”。但尼泊尔总理巴伦德拉·沙阿却又在社交媒体发帖称，之前有媒体所称尼泊尔“拒绝外国援助的说法具有误导性”。 "

def print_metrics(title, text_len, audio_duration, infer_time, first_chunk_time=None):
    """统一打印指标结果的工具函数"""
    rtf = infer_time / audio_duration if audio_duration > 0 else 0
    speed = audio_duration / infer_time if infer_time > 0 else 0

    print("\n" + "="*60)
    print(f"[{title}] 性能测试报告")
    print("-" * 60)
    print(f"合成文本字数：{text_len} 字")
    print(f"音频总时长：{audio_duration:.2f} 秒")
    if first_chunk_time:
        print(f"首句响应耗时 (TTFB)：{first_chunk_time:.3f} 秒  <-- (流式接口的核心优势!)")
    print(f"模型总耗时：{infer_time:.3f} 秒")
    print(f"整体 RTF (实时率)：{rtf:.4f}")
    print(f"整体合成速度：{speed:.2f} x 实时 (每秒生成 {speed:.2f} 秒音频)")
    print("="*60 + "\n")

def test_full_api():
    print(f"正在请求完整接口 (等待所有句子合成完毕): {API_URL_FULL} ...")
    # payload = {"text": TEXT, "sid": 0, "speed": 1}
    payload = {"text": TEXT, "voice": "male", "sid": 0, "speed": 1}

    start_time = time.time()
    response = requests.post(API_URL_FULL, json=payload)
    infer_time = time.time() - start_time

    if response.status_code != 200:
        print(f"Error: {response.text}")
        return

    # 从内存读取二进制音频数据计算时长
    audio_data, sample_rate = sf.read(io.BytesIO(response.content))
    audio_duration = len(audio_data) / sample_rate

    # 保存文件
    save_path = "output_full.wav"
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"音频已保存至: {save_path}")

    print_metrics("完整返回接口 (/api/tts)", len(TEXT), audio_duration, infer_time)

def test_sse_api():
    print(f"正在请求 SSE 流式推送接口 (按句返回): {API_URL_SSE} ...")
    payload = {"text": TEXT, "sid": 0, "speed": 1.0}

    start_time = time.time()
    # 使用 stream=True 保持长连接接收 SSE 推送
    response = requests.post(API_URL_SSE, json=payload, stream=True)

    if response.status_code != 200:
        print(f"Error: {response.text}")
        return

    first_chunk_time = None
    all_audio_data = []
    final_sample_rate = 22050

    # 逐行解析 SSE 数据流
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                # 如果是第一句话返回，记录 TTFB 首字节时间
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time

                # 解析 JSON payload
                json_str = decoded_line[6:] # 去掉 "data: " 前缀
                data = json.loads(json_str)

                chunk_index = data["chunk_index"]
                chunk_text = data["text"]
                audio_base64 = data["audio_base64"]

                # 解码 Base64 获取这段音频
                audio_bytes = base64.b64decode(audio_base64)
                chunk_audio, sample_rate = sf.read(io.BytesIO(audio_bytes))
                final_sample_rate = sample_rate
                all_audio_data.append(chunk_audio)

                print(f"[收到片段 {chunk_index}] 耗时: {time.time()-start_time:.3f}s | 文本: {chunk_text}")

    infer_time = time.time() - start_time

    # 拼接所有片段以计算总时长并保存验证
    final_audio = np.concatenate(all_audio_data)
    audio_duration = len(final_audio) / final_sample_rate

    save_path = "output_sse_combined.wav"
    sf.write(save_path, final_audio, samplerate=final_sample_rate)
    print(f"所有拼接后的音频已保存至: {save_path}")

    print_metrics("SSE流式接口 (/api/tts_sse)", len(TEXT), audio_duration, infer_time, first_chunk_time)

if __name__ == "__main__":
    print(">>> 1. 测试基础完整文件接口")
    test_full_api()

    # time.sleep(2) # 歇一会

    # print(">>> 2. 测试极低延迟的流式接口")
    # test_sse_api()
