# CPU-Only TTS Service

纯 CPU 环境下的**中英混读**语音合成服务：sherpa-onnx + piper VITS **双模型语种路由**，
并针对电力调度 / 告警文本做了读法规范化。

- **快** —— 300 字中文实测 **≈1.1–1.3 s**（RTF 0.02，约 **50× 实时**）
- **准** —— 中英混读自动选音色；编号、单位、罗马数字、符号、缩写都有专门读法
- **省** —— 无 GPU 依赖，权重全局共享，单容器部署

```
文本 ──► 领域规范化 ──► 语种切分 ──► 分块并行合成 ──► 静音裁剪 / 停顿拼接 ──► WAV
        (编号/单位/符号)   (中 ↔ 英)     (10 worker)       (句末120ms/逗号50ms/句中20ms)
```

---

## 特性

| 能力 | 说明 |
| --- | --- |
| 中英语种路由 | `voice=auto` 按语种切分：中文走 piper zh，英文走 piper en；两者同为 22.05 kHz，可直接拼接 |
| 高吞吐 | 相邻句子合并成 ≤80 字的块后并行合成（默认 10 并发 × 10 ONNX 线程） |
| 领域文本规范化 | 时刻/日期、编号逐位读、数值+单位、保护段号（I/II/III 段）、比例、电话、符号（℃ `%` ± Ω …）、HTML 实体清理 |
| 缩写读音 | 大写字母/缩写（AC、QF、QS、SCADA、GPU…）用**中文近似音**读出，音色不跳动；只有真正的英文单词/句子才切英文音色 |
| 拼接质量 | 分块首尾静音裁剪 + 按标点分级停顿（句末 120 ms / 逗号 50 ms / 句中 20 ms）+ 边缘淡化 |
| 接口 | `POST /api/tts`（整段 WAV）、`POST /api/tts_sse`（流式 SSE）、`GET /health` |

---

## 快速开始

```bash
# 1) 准备模型权重（见「模型权重」章节，放入 model_weight/）

# 2) 启动服务（默认监听 8000）
docker compose up -d
curl http://127.0.0.1:8000/health

# 3) 合成一条告警文本（中英混读自动选音色）
curl -X POST http://127.0.0.1:8000/api/tts \
  -H 'Content-Type: application/json' \
  -d '{"text":"红岭变 1198F6 红荷线开关跳闸，请派人检查。","voice":"auto"}' \
  -o out.wav
```

```python
import requests

r = requests.post(
    "http://127.0.0.1:8000/api/tts",
    json={"text": "一次设备 QF、QS 位置异常。Please check immediately.",
          "voice": "auto", "speed": 1.0},
    timeout=60,
)
open("out.wav", "wb").write(r.content)      # 22050 Hz / 单声道 / 16-bit WAV
```

> 若拉不到 `docker-compose.yaml` 中预设的镜像，可本地构建后替换 `image` 字段：
> ```bash
> docker build -t tts-sherpa:latest .
> ```

---

## API

三个接口；完整字段、错误码与 SSE 解析示例见 **[docs/sherpa-tts-api.md](docs/sherpa-tts-api.md)**。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/tts` | 整段合成，返回 WAV 二进制 |
| POST | `/api/tts_sse` | 逐块流式推送（SSE，每块含 Base64 WAV） |
| GET | `/health` | 健康检查 + 当前配置 + 各音色加载状态 |

## 本地实时播放测试

在本地安装播放依赖：

```bash
pip install -r requirements-local-playback.txt
```

若服务位于远端，先建立 SSH 隧道。下面把远端 `8000` 转发到本地 `18000`：

```bash
ssh -N -L 18000:127.0.0.1:8000 root@172.18.190.100
```

另开一个终端，逐块接收并实时播放：

```bash
python bench/local_realtime_tts.py \
  --base-url http://127.0.0.1:18000 \
  --text "现在的时间是3：55，请问您需要我为您播报什么内容呢？" \
  --voice female \
  --output local_tts_output.wav
```

脚本调用 `/api/tts_sse`，收到每个 WAV 分块后立即交给 PyAudio 播放，并累计保存为普通 WAV。
没有音频设备或未安装 PyAudio 时，可先只验证接收和保存：

```bash
python bench/local_realtime_tts.py \
  --base-url http://127.0.0.1:18000 \
  --no-play \
  --output local_tts_output.wav
```

请求体：

```json
{ "text": "需要合成的文本", "voice": "auto", "sid": 0, "speed": 1.0 }
```

| `voice` | 语言 | 模型 |
| --- | --- | --- |
| `female` / `male` | 中 | piper zh xiao_ya / chaowen |
| `en_female` / `en_male` | 英 | piper en amy / ryan |
| **`auto`** / `auto:male` / `auto:female` | 中 + 英 | 按语种自动路由（推荐） |

---

## 读法规范化示例

只作用于中文语境，纯英文句子不受影响：

| 输入 | 实际读法 |
| --- | --- |
| `红岭变 1198F6 红荷线` | 红岭变 **一一九八 艾弗 六** 红荷线（编号逐位读） |
| `10kV出线1198号开关` | 10千伏出线**一一九八号**开关 |
| `相间距离 I 段、零序过流 III 段` | 相间距离**一段**、零序过流**三段** |
| `一次设备 QF、QS 位置` | 一次设备 **丘艾弗、丘艾丝** 位置 |
| `故障相别AC相` | 故障相别**诶西**相（字母用中文近似音，不切音色） |
| `故障测距2.284km` / `TA二次A相7.935A` | 故障测距**二点二八四千米** / TA二次A相**七点九三五安** |
| `25.3℃` / `30%` / `±1.5%` / `#2主变` | 二十五点三摄氏度 / 百分之三十 / 正负百分之一点五 / 2号主变 |
| `8：35` / `8：35至12：30` | 八点三十五分 / 八点三十五分至十二点三十分 |
| `2026年11月12日8：35` | 二零二六年十一月十二日八点三十五分 |
| `Please check the breaker status.` | 原样（走英文音色） |

字母读音表在 [`model_app/letter_names.json`](model_app/letter_names.json)，可逐字调整后重启生效。

---

## 性能参考

测试环境：2× Intel Xeon Gold 5218R（80 vCPU / 40 物理核），纯 CPU，容器内实测：

| 用例 | 字符数 | 音频时长 | 墙钟 | RTF | 实时倍数 |
| --- | --- | --- | --- | --- | --- |
| 英文句子 | 82 | 5.5 s | 0.22 s | 0.040 | 25× |
| 中英混排 | 118 | 20.6 s | 0.60 s | 0.029 | 34× |
| 电力告警（编号/缩写/单位） | 90 | 20.3 s | 0.64 s | 0.031 | 32× |
| **长文本（300 字级）** | 261 | 57.0 s | **1.09 s** | 0.019 | **52×** |

复现：

```bash
docker run --rm --network host -v $PWD/bench:/bench <image> python /bench/sherpa_route_check.py
```

---

## 配置项

全部通过 `docker-compose.yaml` 环境变量配置，改完 `docker compose up -d` 生效：

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MAX_CONCURRENT_WORKERS` / `ONNX_NUM_THREADS` | 10 / 10 | 合成并发数 / 每会话 ONNX 线程数 |
| `DEFAULT_VOICE` | female | 不传 `voice` 时的默认音色 |
| `DOMAIN_NORMALIZE` | true | 领域文本规范化总开关 |
| `LETTER_NAMES_ZH` | true | 大写字母/缩写是否转中文近似读法 |
| `AUTO_MIN_SPAN_CHARS` | 4 | 语种片段合并阈值（`auto` 生效） |
| `CHUNK_MAX_CHARS` | 80 | 每块字符上限（越大越连贯、并行度略降） |
| `SENTENCE_GAP_MS` / `CLAUSE_GAP_MS` / `DEFAULT_GAP_MS` | 120 / 50 / 20 | 句末 / 逗号 / 句中停顿 |
| `TRIM_SILENCE` / `SILENCE_KEEP_MS` / `FADE_MS` | true / 25 / 8 | 静音裁剪与边缘淡化 |

---

## 目录结构

```
.
├── docker-compose.yaml          # 服务编排（CPU、host 网络、健康检查）
├── Dockerfile                   # 镜像构建（依赖见 requirements-sherpa.txt）
├── model_app/                   # 服务代码（挂载到容器 /app）
│   ├── vits_tts_server.py       # FastAPI：/api/tts、/api/tts_sse、/health
│   ├── text_routing.py          # 文本规范化 + 语种路由
│   ├── power_normalizer.py      # 通用中文 + 电力领域规范化规则
│   └── letter_names.json        # 26 字母中文近似读法表
├── model_weight/                # 模型权重（不入库）
├── bench/                       # 验证 / 压测脚本与样音
├── docs/                        # 接口文档与方案评估
├── tests/                       # 前处理回归测试
├── releases/                    # 发布基线与回滚归档
├── fast-kokoro/                 # Kokoro v1.1-zh（PyTorch）对比实现与数据
└── test.py                      # 接口冒烟测试
```

---

## 模型权重

模型来自 [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) 的 `tts-models` releases，
解压到 `model_weight/`，目录名需与 `model_app/vits_tts_server.py` 的 `VOICE_MODELS` 一致：

| 音色 | 目录 | 大小 |
| --- | --- | --- |
| female（中） | `vits-piper-zh_CN-xiao_ya-medium` | 63 MB |
| male（中） | `vits-piper-zh_CN-chaowen-medium` | 63 MB |
| en_female（英） | `vits-piper-en_US-amy-medium` | 79 MB |
| en_male（英） | `vits-piper-en_US-ryan-medium` | 79 MB |

```bash
BASE=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models
for m in vits-piper-zh_CN-xiao_ya-medium vits-piper-zh_CN-chaowen-medium \
         vits-piper-en_US-amy-medium vits-piper-en_US-ryan-medium; do
  curl -LO $BASE/$m.tar.bz2 && tar xf $m.tar.bz2 -C model_weight/
done
```

> 权重体积较大，已在 `.gitignore` 中排除，请勿提交到 Git。

---

## 文档

| 文档 | 内容 |
| --- | --- |
| [docs/sherpa-tts-api.md](docs/sherpa-tts-api.md) | **接口文档**：字段、错误码、SSE 解析、配置项、限制 |
| [docs/sherpa-dual-model-routing.md](docs/sherpa-dual-model-routing.md) | 双模型路由方案、规范化规则、试听问题修复记录 |
| [docs/text-normalization-v3.md](docs/text-normalization-v3.md) | 前处理规则、RTF 评估与基线回滚 |
| [docs/kokoro-multilang-feasibility.md](docs/kokoro-multilang-feasibility.md) | Kokoro 多语言模型可行性评估与实测 |
| [docs/kokoro-v1.1-zh-cpu-notes.md](docs/kokoro-v1.1-zh-cpu-notes.md) | Kokoro v1.1-zh 跑通、音色修复、ONNX / 调度效率对比 |

---

## 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) — 推理框架与模型导出
- [piper](https://github.com/rhasspy/piper) — 中英文 VITS 音色
- [Kokoro](https://github.com/hexgrad/kokoro) — `fast-kokoro/` 中的对比方案
