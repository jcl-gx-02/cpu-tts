####版本3升级，模型权重共享global memory，减少内存占用，支持多实例并发推理
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import re
import os
import io
import base64
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import logging
from logging.handlers import RotatingFileHandler

import numpy as np
import soundfile as sf
import sherpa_onnx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# 规范化后至少要有汉字/字母/数字才算「有可读内容」
SPEAKABLE_RE = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

from text_routing import (
    normalize_domain_text,
    resolve_voice_spans,
    AUTO_MIN_SPAN_CHARS,
    DOMAIN_NORMALIZE,
    LETTER_NAMES_ZH,
)

# ==========================================
# 0. 日志配置 (新增)
# ==========================================
# 配置日志同时输出到控制台和文件，按 10MB 大小滚动保存，最多留 5 个备份
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vits_tts_server")
logger.setLevel(logging.INFO)
file_handler = RotatingFileHandler("/logs/tts_server.log", maxBytes=10*1024*1024, backupCount=5, encoding="utf-8")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ==========================================
# 1. 服务与多模型配置区
# ==========================================
#MAX_CONCURRENT_WORKERS = 10 
#ONNX_NUM_THREADS = 10 

MAX_CONCURRENT_WORKERS = int(
    os.getenv("MAX_CONCURRENT_WORKERS", "10")
)

ONNX_NUM_THREADS = int(
    os.getenv("ONNX_NUM_THREADS", "4")
)

# 拼接质量相关：去掉每个分块首尾的自带静音，再按标点插入受控停顿
TRIM_SILENCE = os.getenv("TRIM_SILENCE", "true").lower() not in ("0", "false", "no")
SILENCE_KEEP_MS = int(os.getenv("SILENCE_KEEP_MS", "25"))       # 裁剪后保留的边距
SENTENCE_GAP_MS = int(os.getenv("SENTENCE_GAP_MS", "120"))      # 。！？ 后的停顿
CLAUSE_GAP_MS = int(os.getenv("CLAUSE_GAP_MS", "50"))           # ，、；： 后的停顿
DEFAULT_GAP_MS = int(os.getenv("DEFAULT_GAP_MS", "20"))         # 句中切分（多为语种切换）只留极小间隙
FADE_MS = int(os.getenv("FADE_MS", "8"))                        # 分块边缘淡化，消除拼接click
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "80"))       # 每个合成块的字符上限（越大拼接点越少）

MODEL_ROOT = "/models"

# 多音色模型字典：
#   male / female          -> piper 中文（chaowen 男声 / xiao_ya 女声）
#   en_male / en_female    -> piper 英文（ryan / amy），22.05kHz，与中文同采样率可直接拼接
# zh 模型用 lexicon + 中文 rule_fsts；en 模型用 espeak-ng-data（无 lexicon/rule_fsts）
VOICE_MODELS = {
    "male": {
        "dir": f"{MODEL_ROOT}/vits-piper-zh_CN-chaowen-medium",
        "onnx_name": "zh_CN-chaowen-medium.onnx",
        "kind": "zh",
    },
    "female": {
        "dir": f"{MODEL_ROOT}/vits-piper-zh_CN-xiao_ya-medium",
        "onnx_name": "zh_CN-xiao_ya-medium.onnx",
        "kind": "zh",
    },
    "en_female": {
        "dir": f"{MODEL_ROOT}/vits-piper-en_US-amy-medium",
        "onnx_name": "en_US-amy-medium.onnx",
        "kind": "en",
    },
    "en_male": {
        "dir": f"{MODEL_ROOT}/vits-piper-en_US-ryan-medium",
        "onnx_name": "en_US-ryan-medium.onnx",
        "kind": "en",
    },
}

# 默认音色：auto = 按语种在 zh/en 双模型间路由
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "female")

# ==========================================
# 2. 全局模型管理 (替换原有的 Thread-Local)
# ==========================================
# 核心优化：全局共享的模型字典，所有线程复用这同一份内存权重
GLOBAL_TTS_ENGINES = {}
executor = None

def load_all_models():
    """在主进程中一次性加载所有模型到全局内存"""
    logger.info("开始加载 TTS 模型到全局共享内存...")
    for voice_id, config in VOICE_MODELS.items():
        try:
            model_dir = config["dir"]
            onnx_name = config["onnx_name"]
            kind = config.get("kind", "zh")

            if kind == "en":
                # piper 英文模型：espeak-ng 前端，无 lexicon / rule_fsts
                vits = sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=f"{model_dir}/{onnx_name}",
                    tokens=f"{model_dir}/tokens.txt",
                    data_dir=f"{model_dir}/espeak-ng-data",
                )
                tts_config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=vits, num_threads=ONNX_NUM_THREADS, debug=False
                    ),
                )
            else:
                tts_config = sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=f"{model_dir}/{onnx_name}",
                            tokens=f"{model_dir}/tokens.txt",
                            lexicon=f"{model_dir}/lexicon.txt",
                        ),
                        num_threads=ONNX_NUM_THREADS,
                        debug=False,
                    ),
                    rule_fsts=f"{model_dir}/phone.fst,{model_dir}/date.fst,{model_dir}/number.fst",
                )
            if not tts_config.validate():
                raise ValueError(f"VITS TTS Config validation failed for {voice_id}.")

            GLOBAL_TTS_ENGINES[voice_id] = sherpa_onnx.OfflineTts(tts_config)
            logger.info(f"[Model Load] 成功加载音色: '{voice_id}'")
        except Exception as e:
            logger.error(f"[Model Load Error] 加载音色 '{voice_id}' 失败: {e}")

# ==========================================
# 3. API 数据模型定义
# ==========================================
class TTSRequest(BaseModel):
    text: str = Field(..., description="需要合成的目标文本")
    voice: str = Field(
        default=DEFAULT_VOICE,
        description="音色：male/female(中文)、en_male/en_female(英文)、auto/auto:male/auto:female(按语种路由)",
    )
    sid: int = Field(default=0, description="说话人 ID (Speaker ID)")
    speed: float = Field(default=1.0, description="语速调节系数")

# ==========================================
# 4. FastAPI 应用生命周期管理
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global executor
    
    # --- 1. 启动前加载全局模型 ---
    load_all_models()
    
    # --- 2. 初始化线程池 ---
    # 注意：这里不再需要 initializer 函数，因为所有 Worker 将共享 GLOBAL_TTS_ENGINES
    logger.info(f"Starting ThreadPoolExecutor with {MAX_CONCURRENT_WORKERS} workers...")
    executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)
    
    # --- 3. 模型预热 (Model Warm-up) ---
    logger.info(f"开始预热模型... 提交 {MAX_CONCURRENT_WORKERS} 个并发任务...")
    start_warmup = time.time()
    loop = asyncio.get_running_loop()
    
    warmup_tasks = []
    warmup_text = "引擎预热。"
    
    # 为每一个加载的音色都执行预热
    for _ in range(MAX_CONCURRENT_WORKERS):
        for voice_id in VOICE_MODELS.keys():
            warmup_tasks.append(
                loop.run_in_executor(
                    executor, _generate_chunk_sync, warmup_text, voice_id, 0, 1.0
                )
            )
    
    await asyncio.gather(*warmup_tasks)
    logger.info(f"双音色模型预热完成！耗时: {time.time() - start_warmup:.3f} 秒. 服务完全就绪。")

    yield
    
    logger.info("Shutting down ThreadPoolExecutor...")
    executor.shutdown(wait=True)

app = FastAPI(title="VITS TTS Multi-Voice API", lifespan=lifespan)

# ==========================================
# 5. 核心处理逻辑 (文本切分与推理任务)
# ==========================================
def split_text_by_punctuation(text: str) -> list:
    """按中英文常用标点符号切分文本，保留标点，确保短句均匀"""
    pattern = r'([，。！？；,\.\!\?\;]+)'
    chunks = re.split(pattern, text)
    sentences = []
    for i in range(0, len(chunks) - 1, 2):
        sentence = chunks[i].strip() + chunks[i+1].strip()
        if sentence:
            sentences.append(sentence)
    if len(chunks) % 2 != 0 and chunks[-1].strip():
        sentences.append(chunks[-1].strip())
    return sentences

def _generate_chunk_sync(text: str, voice: str, sid: int, speed: float):
    """在工作线程中执行指定音色的 VITS 推理 (使用全局共享模型)"""
    if voice not in GLOBAL_TTS_ENGINES:
        raise ValueError(f"Voice '{voice}' is not supported or not loaded.")
        
    # 直接从全局字典中获取唯一的模型实例进行推理 (ONNX Runtime 原生支持多线程并发)
    tts_engine = GLOBAL_TTS_ENGINES[voice]
    audio = tts_engine.generate(text, sid=sid, speed=speed)
    
    if len(audio.samples) == 0:
        return None, 0

    samples = np.array(audio.samples, dtype=np.float32)
    if TRIM_SILENCE:
        samples = trim_silence(samples, audio.sample_rate)
        samples = apply_edge_fade(samples, audio.sample_rate)
    return samples, audio.sample_rate


def apply_edge_fade(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """分块首尾各做一次短淡化，避免拼接处出现咔哒声/突兀感。"""
    n = int(sample_rate * FADE_MS / 1000)
    if n <= 1 or samples.size < 2 * n:
        return samples
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    out = samples.copy()
    out[:n] *= ramp
    out[-n:] *= ramp[::-1]
    return out


def trim_silence(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """裁掉分块首尾的静音，保留 SILENCE_KEEP_MS 的边距。

    piper 每个分块自带 0.1~0.3s 首尾静音，直接拼接会导致中英/句间停顿过长（实测 0.2~0.4s）。
    这里只裁「边缘静音」，分块内部的自然停顿不动。
    """
    if samples.size == 0:
        return samples
    peak = float(np.abs(samples).max())
    if peak <= 0:
        return samples
    threshold = max(peak * 0.01, 1e-4)
    loud = np.abs(samples) > threshold
    idx = np.flatnonzero(loud)
    if idx.size == 0:
        return samples
    keep = int(sample_rate * SILENCE_KEEP_MS / 1000)
    start = max(0, int(idx[0]) - keep)
    end = min(samples.size, int(idx[-1]) + keep)
    return samples[start:end]


def gap_samples(sentence: str, sample_rate: int) -> int:
    """按句子末尾标点决定停顿长度。"""
    s = (sentence or "").rstrip()
    if not s:
        return int(sample_rate * DEFAULT_GAP_MS / 1000)
    if s[-1] in "。！？.!?":
        ms = SENTENCE_GAP_MS
    elif s[-1] in "，、；：,;:":
        ms = CLAUSE_GAP_MS
    else:
        ms = DEFAULT_GAP_MS
    return int(sample_rate * ms / 1000)


def build_chunk_plan(text: str, voice: str, sid: int) -> list:
    """把文本展开成 [(voice_id, sid, sentence)] 的合成计划（保持原文顺序）。

    voice="auto" 时先按语种切分（zh -> piper zh，en -> piper en），
    再把每个片段按标点切句；同一片段内的句子可以并发合成。
    """
    text = normalize_domain_text(text)
    # 规范化后没有任何可读内容（纯标点/纯符号）直接报错，避免返回一段静音
    if not SPEAKABLE_RE.search(text):
        raise HTTPException(status_code=400, detail="Text cannot be empty.")
    plan = []
    for voice_id, _lang, span_text in resolve_voice_spans(text, voice):
        if voice_id not in VOICE_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Voice '{voice_id}' is not supported. Supported: {list(VOICE_MODELS.keys())} + auto",
            )
        # 合并相邻句子成一个较大的块（≤ CHUNK_MAX_CHARS），减少拼接点、让块内韵律由模型自己处理；
        # 语种片段之间不合并（否则会交给错误的音色）。
        buf = ""
        for sentence in split_text_by_punctuation(span_text):
            if buf and len(buf) + len(sentence) > CHUNK_MAX_CHARS:
                plan.append((voice_id, sid, buf))
                buf = sentence
            else:
                buf += sentence
        if buf:
            plan.append((voice_id, sid, buf))
    return plan

# ==========================================
# 6. 完整文件合成接口
# ==========================================
@app.post("/api/tts")
async def generate_tts(request: TTSRequest):
    if not request.voice.startswith("auto") and request.voice not in VOICE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice. Supported: {list(VOICE_MODELS.keys())} + auto/auto:male/auto:female",
        )

    plan = build_chunk_plan(request.text, request.voice, request.sid)
    if not plan:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    start_time = time.time()
    loop = asyncio.get_running_loop()
    all_samples = []
    out_sample_rate = 22050

    # 并发提交所有分块（zh/en 模型混排在同一线程池里排队），结果按原顺序拼接
    tasks = [
        loop.run_in_executor(
            executor,
            _generate_chunk_sync,
            sentence,
            voice_id,
            sid,
            request.speed,
        )
        for voice_id, sid, sentence in plan
    ]

    results = await asyncio.gather(*tasks)

    # 逐块拼接：块内首尾静音已在 _generate_chunk_sync 裁掉，这里按标点插入受控停顿
    for (samples, sr), (_voice_id, _sid, sentence) in zip(results, plan):
        if samples is None or len(samples) == 0:
            continue
        if all_samples:
            all_samples.append(np.zeros(gap_samples(sentence, sr), dtype=np.float32))
        all_samples.append(samples)
        out_sample_rate = sr

    if not all_samples:
        raise HTTPException(status_code=500, detail="Failed to generate any audio.")

    final_audio = np.concatenate(all_samples)
    elapsed = time.time() - start_time
    duration = len(final_audio) / out_sample_rate

    voices_used = sorted({voice_id for voice_id, _, _ in plan})
    logger.info(
        f"[Complete TTS | {request.voice} -> {voices_used}] Chunks: {len(plan)} "
        f"| RTF: {elapsed/duration:.3f}"
    )

    buffer = io.BytesIO()
    sf.write(buffer, final_audio, samplerate=out_sample_rate, subtype="PCM_16", format="WAV")
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="audio/wav")

# ==========================================
# 7. 分段 WAV 推送接口 (SSE + JSON + Base64)
# ==========================================
@app.post("/api/tts_sse")
async def generate_tts_sse(request: TTSRequest):
    if not request.voice.startswith("auto") and request.voice not in VOICE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid voice. Supported: {list(VOICE_MODELS.keys())} + auto/auto:male/auto:female",
        )

    plan = build_chunk_plan(request.text, request.voice, request.sid)
    if not plan:
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    async def sse_generator():
        import json
        loop = asyncio.get_running_loop()
        for i, (voice_id, sid, sentence) in enumerate(plan):
            start = time.time()
            samples, sr = await loop.run_in_executor(
                executor,
                _generate_chunk_sync,
                sentence,
                voice_id,
                sid,
                request.speed,
            )

            if samples is not None and len(samples) > 0:
                buffer = io.BytesIO()
                sf.write(buffer, samples, samplerate=sr, subtype="PCM_16", format="WAV")
                buffer.seek(0)

                wav_base64 = base64.b64encode(buffer.read()).decode('utf-8')

                logger.info(f"[SSE Chunk {i} | {voice_id}] Synthesized: '{sentence[:10]}...'")

                payload = {
                    "chunk_index": i,
                    "text": sentence,
                    "voice_used": voice_id,
                    "audio_base64": wav_base64,
                    "is_final": i == (len(plan) - 1),
                }
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


# ==========================================
# 7.5. 健康检查接口 (Health Check)
# ==========================================
@app.get("/health")
async def health_check():
    """健康检查与探针接口，返回服务当前状态与内存驻留的模型信息"""
    voice_info = {}
    
    # 检查全局内存中的实际加载状态
    for voice_id, config in VOICE_MODELS.items():
        voice_info[voice_id] = {
            "status": "ready" if voice_id in GLOBAL_TTS_ENGINES else "unloaded",
            "model_path": config["onnx_name"],
            "shared_instance": 1 # 标志当前使用了全局单例共享
        }

    return {
        "status": "ok",
        "timestamp": time.time(),
        "config": {
            "max_workers": MAX_CONCURRENT_WORKERS,
            "onnx_num_threads": ONNX_NUM_THREADS,
            "architecture": "global_shared_memory",
            "default_voice": DEFAULT_VOICE,
            "auto_min_span_chars": AUTO_MIN_SPAN_CHARS,
            "language_routing": "zh -> male/female, en -> en_male/en_female (voice=auto)",
            "text_processing": {
                "domain_normalize": DOMAIN_NORMALIZE,
                "letter_names_zh": LETTER_NAMES_ZH,
                "chunk_max_chars": CHUNK_MAX_CHARS,
                "sentence_gap_ms": SENTENCE_GAP_MS,
                "clause_gap_ms": CLAUSE_GAP_MS,
                "default_gap_ms": DEFAULT_GAP_MS,
                "silence_keep_ms": SILENCE_KEEP_MS,
                "fade_ms": FADE_MS,
                "trim_silence": TRIM_SILENCE,
            },
        },
        "voices": voice_info
    }


# ==========================================
# 8. 启动脚本
# ==========================================
if __name__ == "__main__":
    uvicorn.run(
        "vits_tts_server:app", 
        host="0.0.0.0", 
        port=8000, 
        workers=1,
        log_level="info"
    )
