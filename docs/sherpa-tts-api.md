# TTS 服务 API 文档（sherpa + piper 中英双模型）

> 版本：R01.00.02（2026-11-12）
> 服务：`tts-cpu-only/docker-compose.yaml` → 容器 `sherpa-tts`
> 默认地址：`http://<host>:8000`

---

## 1. 概览

| 项 | 值 |
| --- | --- |
| 推理框架 | sherpa-onnx (ONNX Runtime, CPU) |
| 模型 | piper VITS：中文 xiao_ya / chaowen，英文 amy / ryan |
| 采样率 | **22050 Hz**，单声道，16-bit PCM |
| 输出格式 | `POST /api/tts` → WAV 二进制；`POST /api/tts_sse` → SSE + Base64 WAV |
| 并发 | 10 个合成 worker × 10 个 ONNX 线程（`MAX_CONCURRENT_WORKERS` / `ONNX_NUM_THREADS`） |
| 语种 | 中文、英文、中英混排（`voice=auto` 自动路由） |
| 认证 | 无（内网服务） |

接口一览：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/tts` | 整段合成，返回一个完整 WAV |
| POST | `/api/tts_sse` | 逐块流式推送（SSE），可边合成边播 |
| GET | `/health` | 健康检查 + 当前配置 + 音色状态 |

---

## 2. 快速开始

```bash
# 中文
curl -X POST http://127.0.0.1:8000/api/tts -H 'Content-Type: application/json' \
  -d '{"text":"红岭变 1198F6 红荷线开关跳闸，请派人检查。","voice":"auto"}' -o out.wav

# 中英混排
curl -X POST http://127.0.0.1:8000/api/tts -H 'Content-Type: application/json' \
  -d '{"text":"一次设备 QF、QS 位置异常。Please check immediately.","voice":"auto","speed":1.0}' -o mix.wav

# 健康检查
curl http://127.0.0.1:8000/health
```

```python
import requests

r = requests.post(
    "http://127.0.0.1:8000/api/tts",
    json={"text": "乐园变11F5园开线相间距离I段保护动作。", "voice": "auto", "speed": 1.0},
    timeout=60,
)
r.raise_for_status()
open("out.wav", "wb").write(r.content)      # 22050 Hz 单声道 WAV
```

---

## 3. 接口详情

### 3.1 `POST /api/tts` — 整段合成

**请求体（JSON）**

| 字段 | 类型 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- | --- |
| `text` | string | 是 | — | 待合成文本，支持中文、英文、中英混排 |
| `voice` | string | 否 | `female` | 音色，见 [§4](#4-音色与语种路由) |
| `sid` | int | 否 | `0` | 说话人 ID（当前所有模型均为单说话人，保持 0） |
| `speed` | float | 否 | `1.0` | 语速系数，建议 0.5~2.0，越大越快 |

**响应**

| 项 | 值 |
| --- | --- |
| HTTP 状态 | `200` |
| Content-Type | `audio/wav` |
| 内容 | WAV 文件二进制（22050 Hz / 单声道 / 16-bit PCM） |

**示例**

```bash
curl -X POST http://127.0.0.1:8000/api/tts -H 'Content-Type: application/json' \
  -d '{"text":"开关跳闸，重合成功。","voice":"male","speed":1.1}' -o out.wav
```

**出错**

| 场景 | 状态码 | 响应体 |
| --- | --- | --- |
| `voice` 不在支持列表 | 400 | `{"detail":"Invalid voice. Supported: [...] + auto/auto:male/auto:female"}` |
| `text` 为空或全是标点 | 400 | `{"detail":"Text cannot be empty."}` |
| 所有分块都没合成出音频 | 500 | `{"detail":"Failed to generate any audio."}` |

### 3.2 `POST /api/tts_sse` — 流式合成（SSE）

请求体与 `/api/tts` 完全一致。响应为 `text/event-stream`，每个事件形如：

```
data: {"chunk_index":0,"text":"红岭变 一一九八 艾弗 六 红荷线开关跳闸，","voice_used":"female","audio_base64":"<WAV base64>","is_final":false}

data: {"chunk_index":1,"text":"请派人检查。","voice_used":"female","audio_base64":"...","is_final":true}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `chunk_index` | int | 分块序号，从 0 开始，顺序递增 |
| `text` | string | 该块文本（规范化之后） |
| `voice_used` | string | 该块实际使用的音色（`female`/`male`/`en_female`/`en_male`） |
| `audio_base64` | string | 该块的 WAV（22050 Hz 单声道）Base64 |
| `is_final` | bool | 是否最后一块 |

客户端按 `chunk_index` 顺序解码拼接即可边收边播：

```python
import base64, io, json, requests, soundfile as sf, numpy as np

chunks = []
with requests.post("http://127.0.0.1:8000/api/tts_sse",
                   json={"text": "……", "voice": "auto"}, stream=True, timeout=60) as r:
    for line in r.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        payload = json.loads(line[6:])
        audio, sr = sf.read(io.BytesIO(base64.b64decode(payload["audio_base64"])))
        chunks.append(audio)
        print(f"[{payload['chunk_index']}] {payload['voice_used']} {payload['text'][:20]}")
sf.write("out.wav", np.concatenate(chunks), sr)
```

> 说明：SSE 每一块是**独立完整的小 WAV**，便于边收边播；拼接后与本接口整体输出的停顿规则一致。

仓库内提供了 PyAudio 实时播放示例：

```bash
python bench/local_realtime_tts.py \
  --base-url http://127.0.0.1:18000 \
  --text "现在的时间是3：55，请问您需要我为您播报什么内容呢？" \
  --voice female
```

使用前执行 `pip install -r requirements-local-playback.txt`。为兼容远端部署，示例使用
`ssh -N -L 18000:127.0.0.1:8000 root@172.18.190.100` 建立隧道。

### 3.3 `GET /health` — 健康检查

```json
{
  "status": "ok",
  "timestamp": 1790574074.89,
  "config": {
    "max_workers": 10,
    "onnx_num_threads": 10,
    "architecture": "global_shared_memory",
    "default_voice": "female",
    "auto_min_span_chars": 4,
    "language_routing": "zh -> male/female, en -> en_male/en_female (voice=auto)",
    "text_processing": {
      "domain_normalize": true,
      "letter_names_zh": true,
      "chunk_max_chars": 80,
      "sentence_gap_ms": 120,
      "clause_gap_ms": 50,
      "default_gap_ms": 20,
      "silence_keep_ms": 25,
      "fade_ms": 8,
      "trim_silence": true
    }
  },
  "voices": {
    "male":       {"status": "ready", "model_path": "zh_CN-chaowen-medium.onnx", "shared_instance": 1},
    "female":     {"status": "ready", "model_path": "zh_CN-xiao_ya-medium.onnx", "shared_instance": 1},
    "en_female":  {"status": "ready", "model_path": "en_US-amy-medium.onnx",      "shared_instance": 1},
    "en_male":    {"status": "ready", "model_path": "en_US-ryan-medium.onnx",     "shared_instance": 1}
  }
}
```

`status` 为 `ok` 表示服务已就绪（模型已加载并预热完成）。可用作容器健康检查。

---

## 4. 音色与语种路由

| `voice` 取值 | 语言 | 模型 | 说明 |
| --- | --- | --- | --- |
| `female` | 中 | piper zh xiao_ya | 默认值 |
| `male` | 中 | piper zh chaowen | |
| `en_female` | 英 | piper en amy | |
| `en_male` | 英 | piper en ryan | |
| **`auto`** | 中+英 | 自动 | 按语种切分后分别选音色（默认女声） |
| `auto:female` | 中+英 | 自动 | 同 `auto` |
| `auto:male` | 中+英 | 自动 | 中文用 `male`，英文用 `en_male` |

### `auto` 的路由规则

1. 文本按语种切成片段：中文片段 → 中文音色，英文片段 → 英文音色；
2. 数字/标点/空格跟随前一个片段（`2026年` 不会被切开）；
3. 含**小写字母的英文单词/句子**才走英文音色；
4. **大写字母与缩写**（A 相、AC、QF、QS、SCADA、GPU、11F5 里的 F…）统一转成
   **中文近似读法**由中文音色朗读（见 §5.3），因此电力告警文本通常是**单一音色、0 次切换**；
5. 片段全部落在同一音色时，按原文整段合成（不做多余切分）。

---

## 5. 文本规范化规则

服务在合成前会做一层**保守的领域规范化**（总开关 `DOMAIN_NORMALIZE`）。
规则只对中文语境生效，**纯英文句子不会被改动**。

### 5.1 数字与编号

| 输入 | 输出 | 说明 |
| --- | --- | --- |
| `1198F6` | `一一九八 艾弗 六` | 编号类 token：数字逐位读 + 字母中文近似音 |
| `11F5` | `一一 艾弗 五` | 同上 |
| `1198号`（≥4 位数字 + 号/线/开关/刀闸/杆/塔…） | `一一九八号` | 长设备号逐位读 |
| `35号主变` | `35号主变`（读「三十五号」） | 短数字仍按数量读 |
| `1.359km` | `一点三五九千米` | 小数转中文读法 |
| `220.5kV` | `二百二十点五千伏` | 数值 + 单位 |
| `2026年9月28日` | `二零二六年九月二十八日` | 年份逐位、月日按数量读 |
| `8：35` / `08：05` | `八点三十五分` / `八点零五分` | 支持全角和半角冒号 |
| `8:35:20` | `八点三十五分二十秒` | 支持时分秒 |
| `8：35至12：30` | `八点三十五分至十二点三十分` | 时刻范围 |
| `2026年11月12日8：35` | `二零二六年十一月十二日八点三十五分` | 日期与时刻组合 |
| `1:2` / `变比10：1` | `一比二` / `变比十比一` | 非时刻冒号按比例读 |

### 5.2 电力术语与单位

| 类别 | 规则 |
| --- | --- |
| 保护段号 / 罗马数字 | `I段`→`一段`、`II段`→`二段`、`III母`→`三母`、`第III部分`→`第三部分`、全角 `Ⅰ/Ⅱ/Ⅲ` 同样处理 |
| 单位 | `kV→千伏`、`kA→千安`、`MW→兆瓦`、`MVA→兆伏安`、`kWh→千瓦时`、`km→千米`、`Hz→赫兹`、`ms→毫秒`、`s→秒`、`V→伏`、`A→安`、`Ω→欧姆`（仅数值后） |
| HTML 实体 | `&#x20;`、`&nbsp;` 等清理为空格；零宽字符删除；空白收敛 |
| 全角字符 | 全角数字/字母/罗马数字转半角 |

### 5.3 大写字母 / 缩写 → 中文近似读法

映射表在 **`model_app/letter_names.json`**（改完重启生效，无需改代码）：

| 字母 | 读法 | 字母 | 读法 | 字母 | 读法 | 字母 | 读法 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | 诶 | H | 艾尺 | O | 欧 | V | 维 |
| B | 必 | I | 爱 | P | 皮 | W | 达布留 |
| C | 西 | J | 杰 | Q | 丘 | X | 艾克斯 |
| D | 迪 | K | 开 | R | 阿尔 | Y | 歪 |
| E | 伊 | L | 艾勒 | S | 艾丝 | Z | 兹德 |
| F | 艾弗 | M | 艾姆 | T | 提 | | |
| G | 记 | N | 恩 | U | 优 | | |

例：`AC相` → `诶西相`；`SCADA` → `艾丝西诶迪诶`；`GPU` → `记皮优`。

### 5.4 符号 → 中文

| 符号 | 读法 | 示例 |
| --- | --- | --- |
| `℃` `°C` / `℉` `°F` | 摄氏度 / 华氏度 | `25.3℃` → 二十五点三摄氏度 |
| `%` `‰` | 百分之 / 千分之（前置于数字） | `30%` → 百分之三十 |
| `±` | 正负 | `±1.5%` → 正负百分之一点五 |
| `°` | 度 | `45°` → 45度 |
| `×` `÷` `·` | 乘 / 除 / 点 | `2×3` → 2乘3 |
| `≤` `≥` `≈` `≠` | 小于等于 / 大于等于 / 约等于 / 不等于 | |
| `~` `～` `→` `←` | 到 | `5~10米` → 5到10米 |
| `#` `№` | 号（后置于数字） | `#2主变` → 2号主变 |
| `&` `@` `μ` `²` `³` `√` `∞` | 和 / 艾特 / 微 / 平方 / 立方 / 根号 / 无穷 | |

### 5.5 拼接与停顿

每个合成块会**裁掉首尾自带静音**，再按标点插入受控停顿，并做 8 ms 边缘淡化：

| 边界 | 停顿 |
| --- | --- |
| 句末 `。！？.!?` | 120 ms |
| 逗号等 `，、；：,;:` | 50 ms |
| 句中（多为语种切换） | 20 ms |

---

## 6. 配置项（环境变量，`docker-compose.yaml`）

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `MAX_CONCURRENT_WORKERS` | 10 | 合成并发 worker 数 |
| `ONNX_NUM_THREADS` | 10 | 每个 ONNX 会话的线程数 |
| `DEFAULT_VOICE` | female | 不传 `voice` 时的默认音色 |
| `AUTO_MIN_SPAN_CHARS` | 4 | 语种片段合并阈值（`auto` 时生效） |
| `AUTO_VOICE_ZH_FEMALE/MALE` | female / male | `auto` 的中文音色 |
| `AUTO_VOICE_EN_FEMALE/MALE` | en_female / en_male | `auto` 的英文音色 |
| `DOMAIN_NORMALIZE` | true | 领域文本规范化总开关 |
| `LETTER_NAMES_ZH` | true | 大写字母/缩写是否转中文近似读法（false=交英文音色） |
| `LETTER_NAMES_FILE` | /app/letter_names.json | 字母映射表路径 |
| `CHUNK_MAX_CHARS` | 80 | 每个合成块的字符上限（越大拼接点越少、越连贯，并行度略降） |
| `SENTENCE_GAP_MS` / `CLAUSE_GAP_MS` / `DEFAULT_GAP_MS` | 120 / 50 / 20 | 句末 / 逗号 / 句中停顿 |
| `TRIM_SILENCE` / `SILENCE_KEEP_MS` / `FADE_MS` | true / 25 / 8 | 是否裁静音 / 保留边距 / 边缘淡化 |

---

## 7. 性能参考（本机 2×Xeon Gold 5218R，80 vCPU，纯 CPU）

| 用例 | 字符数 | 音频时长 | 墙钟 | RTF | 实时倍数 |
| --- | --- | --- | --- | --- | --- |
| 中文短句 | 32 | 6.5 s | 0.33 s | 0.051 | 20× |
| 英文句子 | 82 | 5.6 s | 0.22 s | 0.040 | 25× |
| 中英混排 | 61 | 6.3 s | 0.19 s | 0.030 | 33× |
| 电力告警（编号/缩写/小数/单位） | 96 | 20.0 s | 0.65 s | 0.032 | 31× |
| **长文本（≈300 字）** | 231 | 50.6 s | **1.03 s** | 0.020 | **49×** |

**300 字目标（≤2 s）**：实测 1.0~1.3 s，满足。

---

## 8. 限制与注意事项

1. 输出固定 **22050 Hz 单声道**；如需 16 k/48 k 或立体声请客户端重采样。
2. 单请求文本建议 ≤ 1000 字；超长文本请客户端分段（每段 ≤ 300 字可保持并行收益）。
3. 语速建议 0.5~2.0；`speed` 会同时影响韵律自然度。
4. 若一条告警要求**绝对单一音色**：用 `voice=female`（或 `male`）+ 领域规范化即可，
   字母与缩写都会由中文音色读；`auto` 只在出现**成句英文**时才会切换音色。
5. 并发过高时请求会排队（`MAX_CONCURRENT_WORKERS` 之外的请求等待），
   压测/容量核算请用 `bench/sherpa_route_check.py`。
6. 站点名、设备专名读音（如「乐园变」）目前交给模型；需要固定读法时可扩展
   `model_app/text_routing.py` 的词表或改用 `letter_names.json` 同类的映射文件。

---

## 9. 相关文件

| 文件 | 说明 |
| --- | --- |
| `model_app/vits_tts_server.py` | 服务实现（FastAPI） |
| `model_app/text_routing.py` | 中英文语境选择 + 语种路由 |
| `model_app/power_normalizer.py` | 通用中文 + 电力领域规范化规则 |
| `model_app/letter_names.json` | 字母中文近似读法表（可改） |
| `tests/test_power_normalizer.py` | 前处理回归测试 |
| `bench/normalizer_bench.py` | 前处理耗时与 API RTF 基准 |
| `bench/sherpa_route_check.py` | 端到端 RTF 验证脚本 |
| `docs/sherpa-dual-model-routing.md` | 方案与实测记录 |
