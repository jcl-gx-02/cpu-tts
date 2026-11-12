# 中英文 TTS 方案可行性评估（纯 CPU 环境）

评估日期：2026-09-24　评估环境：本机（2× Intel Xeon Gold 5218R，80 vCPU / 40 物理核，251 GB RAM，
支持 AVX-512 + VNNI），Docker 镜像 `tts-sherpa-x86-image:R01.00.01`，`sherpa_onnx 1.13.1`。
所有数据均由本仓库 `bench/` 下脚本在该镜像内实测得到，复现命令见附录。

---

## 1. 现状架构梳理

**部署**：`docker-compose.yaml` 单服务 `sherpa-tts`，`network_mode: host`，镜像
`dockhub.ghtchina.com:6060/model-images/tts-sherpa-x86-image:R01.00.01`（Python 3.10 + sherpa-onnx 1.13.1，662 MB）。
`model_app/` 挂到 `/app`，`model_weight/` 只读挂到 `/models`，`logs/` 挂到 `/logs`。

**服务端**（`model_app/vits_tts_server.py`，334 行）：

- 启动时把两个 piper 音色（`male`=chaowen、`female`=xiao_ya）一次性加载进进程级全局字典
  `GLOBAL_TTS_ENGINES`，所有线程共享同一份 ORT session（省内存、ORT session 本身线程安全）。
- `ThreadPoolExecutor(MAX_CONCURRENT_WORKERS=10)`，每个 ORT session `num_threads=10`。
- 请求进来先按标点切句（`split_text_by_punctuation`），再把每个句子作为独立推理任务并发提交，
  最后拼接音频（`/api/tts`）或按句 SSE 推流（`/api/tts_sse`）。
- 端点：`POST /api/tts`、`POST /api/tts_sse`、`GET /health`。
- 预热：启动时对每个音色跑 `MAX_CONCURRENT_WORKERS` 次空转推理。

**当前实测基线**（生产日志与本次复现一致）：

| 场景 | 音频时长 | 墙钟时间 | RTF | 实时倍数 |
| --- | --- | --- | --- | --- |
| 单句 21 字（xiao_ya，T=10） | 4.53 s | 0.179 s | 0.0395 | 25× |
| 生产长文本 135 字 / 11 句（10 worker） | 30.79 s | 0.436 s | **0.0142** | 70× |
| 生产日志（17–19 句） | — | — | 0.012–0.022 | — |

**当前模型的根本限制（已实测确认）**：piper zh 模型是「字符级中文词表」，
喂英文时全部 OOV、直接丢弃：

```
character-lexicon.cc: Ignore OOV 'The' / 'rescue' / 'team' ...
→ 输出 0.24 s 近似静音（peak=0.000）
```

即：现有两个模型**完全不能生成英文**，这不是配置问题，是模型词表里没有英文音素。

---

## 2. 结论速览

1. **`kokoro-multi-lang-v1_0` 在现有镜像上开箱可用**：`sherpa_onnx 1.13.1` 已内置
   `OfflineTtsKokoroModelConfig`，现有模型目录（`model_weight/kokoro-multi-lang-v1_0`）文件齐全，
   **不需要换镜像、不需要新增 pip 依赖、不需要联网**。
2. 中英混读、纯中、纯英**均可用同一个引擎实例完成**，无需按语言切换引擎。
3. **但速度达不到当前的 0.0127**：单请求 RTF 约 **0.093–0.10**（≈10× 实时），
   比 piper 慢约 **6.5–7 倍**；单句首包 TTFB 从 ~0.2 s 升到 ~0.5–1.2 s（取决于首句长度）。
4. 该差距是模型的算力需求决定的（Kokoro 每生成 1 s 音频需 **2.4 核·秒**，piper 只需 0.3–0.4 核·秒），
   在本机 40 物理核上理论上限也只有约 16× 实时，**不是调参能补回来的**。
5. 官方 **int8 量化版反而更慢**（每 1 s 音频需 6.3 核·秒，RTF≈1.0），**不可用于提速**。
6. 若「速度优先于音色统一」，还有一条能保住当前效率的路线：**双语分别路由到 piper**
   （zh 用现有模型 70×、en 用 piper 英文音色 64×，实测）。
7. 折中选项 **MeloTTS zh_en（单模型双语 VITS）**：RTF 0.076（13× 实时），比 Kokoro 略快但音色单一、质量一般。

**推荐**：先把 Kokoro 作为**新增音色族**接入（不动现有链路，可灰度/AB），
再按需要加「语言路由」把纯中文请求继续留在 piper 上。这样在拿到中英能力的同时，
把速度损失限制在英文/混读请求上。

---

## 3. 实测数据

### 3.1 测试方法

- 容器内运行（同一镜像），文本与生产 `test.py` 同源（135 字 / 11 句）。
- 单句基准：同一文本重复 3 次取中位数；并发基准：`ThreadPoolExecutor` 按句切分并发生成后取中位数。
- `RTF = 墙钟时间 / 音频时长`；`cpu/audio = 进程 CPU 时间 / 音频时长`（反映算力需求，与并发度无关）。

### 3.2 对比结果

| 引擎 / 模型 | 语言 | 单流 RTF | 请求级 RTF（并发后） | 实时倍数 | 算力需求（核·秒/音频秒） | 模型体积 | 采样率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| piper zh（现状 xiao_ya，VITS 22.05k） | 中 | 0.0395 | **0.0142** | 70× | 0.34 | 63 MB | 22.05 kHz |
| piper en（amy-medium，VITS 22.05k） | 英 | 0.0360 | 0.0156 | 64× | 0.28 | 67 MB | 22.05 kHz |
| **MeloTTS zh_en**（VITS 双语单模型） | 中+英 | 0.148 | 0.0764 | 13× | 1.47 | 170 MB | 44.1 kHz |
| **Kokoro multi-lang v1_0（fp32）** | 中+英(+日/法/西/意/葡/印地) | 0.22–0.28 | **0.093**（最好 0.0931） | 11× | 2.40 | 325 MB + 27 MB voices | 24 kHz |
| Kokoro multi-lang int8（官方） | 同上 | 0.93–1.02 | 0.21 | 5× | 6.30 | 114 MB | 24 kHz |

并发/线程调参实测（生产文本 11 句，Kokoro fp32）：

| ONNX 线程 T | worker W | 墙钟 | 请求级 RTF |
| --- | --- | --- | --- |
| 1 | 10 | 5.93 s | 0.191 |
| 4 | 10 | 4.09 s | 0.132 |
| 8 | 10 | 3.60 s | 0.116 |
| 10 | 10 | 3.25 s | 0.105 |
| 20 | 10 | 3.23 s | 0.104 |
| 40 | 10 | 2.89 s | 0.093 |
| 8 | 24 | 3.35 s | 0.099 |

结论：T 在 10–40、W 在 10–24 之间即进入平台期，~10× 实时是本机（含其他容器负载，
loadavg≈11）的实际吞吐上限。

细分句长度（8/12/20 字切分）实验：RTF 仍在 0.10–0.13，**没有收益**，
说明瓶颈是总算力而非「最长句决定墙钟」，因此**不需要改现有切句策略**。

### 3.3 关键工程发现

1. **`lang` 必须填合法的 espeak-ng voice**。`lang="zh"` 时英文文本直接抛
   `Failed to set eSpeak-ng voice`；`lang="en"` 时中文走 `lexicon-zh.txt`（与 lang 无关，
   波形逐样本一致，相关系数 1.0000），英文走 espeak-ng，混读正常。→ **统一用 `lang="en"`**。
2. **`dict_dir` 不再需要**（≥1.12.15 会打印 `you don't need to provide dict_dir` 并忽略）；
   `data_dir=espeak-ng-data` 必填（英文 OOV 走 espeak）。
3. **官方 int8 版本不可用于加速**：其图是 ORT 动态量化（90×ConvInteger + 83×MatMulInteger +
   98×DynamicQuantizeLinear，Gemm 未量化），在本 runtime 上每音频秒算力从 2.4 涨到 6.3 核·秒。
   本机虽是 AVX-512 VNNI（Cascade Lake），仍然更慢。
4. **`max_num_sentences>1` 对 Kokoro 无效**：会打印
   `max_num_sentences (N) != 1 is ignored for Kokoro TTS models`。
   即 sherpa-onnx 不提供 Kokoro 批量推理，**现有「应用层切句 + 线程池」就是最优结构**。
5. **说话人映射（取自模型 metadata，权威）**：

   | sid | 音色 | 语言 |
   | --- | --- | --- |
   | 0–44 | af/am/bf/bm/ef… 等 | 英、西、法、印地、意、日、葡 |
   | **45** | zf_xiaobei | 中 |
   | **46** | zf_xiaoni | 中 |
   | **47** | zf_xiaoxiao | 中 |
   | **48** | zf_xiaoyi | 中 |
   | **49** | zm_yunjian | 中 |
   | **50** | zm_yunxi | 中 |
   | **51** | zm_yunxia | 中 |
   | **52** | zm_yunyang | 中 |

   新版 int8 包（2025-09 之后）为 54 音色，仅在 sid=53 追加西语 `em_santa`，0–52 不变；
   现有 fp32 包是 53 音色版本。
6. **内存**：Kokoro fp32 加载后 RSS 457 MB；高并发跑完后进程 RSS 可达 ~1.3 GB（ORT arena），
   需在 `mem_limit`/监控里留余量。

---

## 4. 备选方案对比

| 方案 | 中英能力 | 请求级 RTF | 音色 | 实现成本 | 风险 |
| --- | --- | --- | --- | --- | --- |
| **A. 全量 Kokoro**（替换/并存） | 单模型，纯中/纯英/混读均好 | 0.093 | 53 音色，中英各 8/45 个 | 低（加一个 config 分支 + 音色映射） | 比现状慢 7×，算力翻 6× |
| **B. 混合路由**（zh→piper，en/混读→Kokoro） | 需按语言切句后分别调用 | 中文 0.014 / 英文 0.093 | 跨语言切换时音色不连续 | 中（语言检测 + 分段 + 拼接） | 混读句中切换处韵律不连续 |
| **C. 全量 MeloTTS zh_en** | 单模型双语，单音色 | 0.076 | 仅 1 个音色，44.1 kHz | 低 | 音色不可选、质量一般、44.1k 冗余 |
| **D. piper zh + piper en 双模型路由** | 需按语言切句 | 中文 0.014 / 英文 0.0156 | 每语言独立音色 | 中 | 同上音色切换；英文音色需另选/下载 |

速度硬指标视角：**只有 B / D 能保住「当前效率」**（因为 piper 级模型已经是这个量级的算力下限）；
**A / C 无法达到 0.0127**，任何线程/量化调参都无法弥补 6–7 倍差距。

---

## 5. 推荐落地路径

### 阶段 1：Kokoro 作为新增音色族接入（建议先做，风险最低）

在 `VOICE_MODELS` 之外增加 Kokoro 引擎注册，保留 piper 原样，API 用 `voice` 区分，例如
`kokoro_zh_xiaoxiao` / `kokoro_en_af_heart`。核心配置（已实测可用）：

```python
kokoro = sherpa_onnx.OfflineTtsKokoroModelConfig(
    model=f"{MODEL_ROOT}/kokoro-multi-lang-v1_0/model.onnx",
    voices=f"{MODEL_ROOT}/kokoro-multi-lang-v1_0/voices.bin",
    tokens=f"{MODEL_ROOT}/kokoro-multi-lang-v1_0/tokens.txt",
    lexicon=f"{MODEL_ROOT}/kokoro-multi-lang-v1_0/lexicon-us-en.txt,"
            f"{MODEL_ROOT}/kokoro-multi-lang-v1_0/lexicon-zh.txt",
    data_dir=f"{MODEL_ROOT}/kokoro-multi-lang-v1_0/espeak-ng-data",
    lang="en",          # 关键：不能填 "zh"
)
cfg = sherpa_onnx.OfflineTtsConfig(
    model=sherpa_onnx.OfflineTtsModelConfig(kokoro=kokoro, num_threads=ONNX_NUM_THREADS),
    rule_fsts=f"{MODEL_ROOT}/kokoro-multi-lang-v1_0/phone-zh.fst,"
              f"{MODEL_ROOT}/kokoro-multi-lang-v1_0/date-zh.fst,"
              f"{MODEL_ROOT}/kokoro-multi-lang-v1_0/number-zh.fst",
)
```

不需要的：`dict/`、int8 权重、`max_num_sentences`。

建议参数：`ONNX_NUM_THREADS=8~10`、`MAX_CONCURRENT_WORKERS=10~24`（保持 T×W ≤ 物理核数量级），
其余切句/SSE 逻辑保持不变。

### 阶段 2：语言路由（若中文请求必须保持 0.0127）

按「句」判定语言（纯 CJK → piper 中文；含拉丁词或混排 → Kokoro），同一请求内混排时
在语言边界切分并分别合成后拼接；在 `TTSRequest` 增加 `engine: auto|piper|kokoro` 供灰度与回退。
切分处可加 10–20 ms 交叉淡入淡出以减轻音色切换感。

### 阶段 3（可选）

- 英文单语言吞吐要求高时，加 piper 英文音色（实测 64× 实时，与中文同量级）。
- 若后续允许 GPU，Kokoro 在 GPU 上可回到「比实时快数十倍」，是目前质量/速度最优解。

---

## 6. 三个参考项目评估

| 项目 | 对本项目的价值 | 结论 |
| --- | --- | --- |
| **k2-fsa/sherpa-onnx** | 就是当前引擎。1.13.1 已原生支持 `kokoro-multi-lang-v1_0`（含 `lang`/lexicon/espeak 细节），无需新增依赖即可中英混读。 | **直接采用**，是本方案的落地路径 |
| **remsky/Kokoro-FastAPI** | 基于 PyTorch + `kokoro`/`misaki` 的 FastAPI 封装。CPU 镜像存在，但 torch 版 Kokoro 在 CPU 上比 sherpa-onnx 的 ONNX 路径更慢、镜像更大（GB 级）。真正值得借鉴的是**接口与产品能力**：OpenAI 兼容 `/v1/audio/speech`、多说话人/音色混合、逐词时间戳、SSE 分块。 | **参考设计，不引入运行时** |
| **nari-labs/nari-qwen3-tts** | 单卡 H100 上 10 RPS、p95 TTFA <50 ms 的 Qwen3-TTS 1.7B 服务（CUDA Graph + 编译内核，官方只测试英文）。属于 GPU-only、LLM 自回归路线。 | **不适用于纯 CPU 环境**，无收益 |

补充：`sherpa-onnx` 1.13.1 还支持 Matcha / Kitten / ZipVoice / Pocket 等 TTS。
Kitten 仅英文、ZipVoice 是流匹配音色克隆（CPU 上更慢），都不比 Kokoro/Melo 更合适，
因此未纳入本次实测。

---

## 7. 风险与待确认

1. **音色主观质量**：Kokoro 中文 8 个音色、英文 45 个音色的实际听感需业务确认。
   本次已生成样音（`bench/out/`），请重点听 `zh_46_zf_xiaoni.wav`、`zh_47_zf_xiaoxiao.wav`、
   `en_0_af_alloy.wav`、`mixed_46.wav` 与 `melo_*`、`piper_en_*` 做横向对比。
2. **吞吐容量**：Kokoro 在本机（含其他容器负载）约 **10–13× 实时**总吞吐，
   即每小时约 600–780 分钟音频；piper 约 4200 分钟/小时。请根据实际并发与日均文本量核算需要几台/几个副本。
3. **首包时延**：Kokoro 首句 TTFB ≈ 0.5–1.2 s（实测：21 字中文句 1.19 s、74 字符英文句 0.91 s），
   piper 约 0.18 s。若前端有 300 ms 级首包要求，需要走方案 B/D（或把首句切得更短，但短句会牺牲韵律）。
4. **长稳测试**：Kokoro 并发后 RSS 增长到 ~1.3 GB，需要压测观察是否存在持续增长（ORT arena 通常收敛）。
5. **混合语言句**：已验证同一引擎可处理混读，但跨语言切换的自然度、以及数字/日期归一化（`rule_fsts`）需要用真实业务文本回归。
6. 本次为评估，**未改动生产 `docker-compose.yaml` 与 `model_app/vits_tts_server.py`**；
   下载到 `/tmp/kokoro-models/` 的 int8 / Melo / piper-en 模型仅为测试，可随时删除。

---

## 附录

### A. 复现命令

```bash
IMG=dockhub.ghtchina.com:6060/model-images/tts-sherpa-x86-image:R01.00.01

# Kokoro fp32：单句 + 生产文本并发
docker run --rm -v $PWD/model_weight:/models:ro -v $PWD/bench:/bench -w /bench \
  --entrypoint python $IMG /bench/kokoro_bench.py --mode both --threads 10 --workers 10

# 线程/worker 扫描 & 细分句实验
docker run ... $IMG /bench/scaling.py
docker run ... $IMG /bench/fine_chunk.py

# 语言/词表行为验证
docker run ... $IMG /bench/lang_effect.py

# 算力需求（CPU 秒/音频秒）与 int8 对比、piper 英文行为
docker run ... -e PIPER=1 $IMG /bench/diag.py

# MeloTTS zh_en 与 piper 英文
docker run ... -e MELO_DIR=/melo/vits-melo-tts-zh_en $IMG /bench/melo_bench.py
docker run ... $IMG /bench/piper_en_bench.py

# 生成试听样音（写入 bench/out/）
docker run ... $IMG /bench/samples.py
```

### B. 本次新增文件

- `bench/`：`kokoro_bench.py`、`scaling.py`、`fine_chunk.py`、`lang_effect.py`、`diag.py`、
  `batching.py`、`int8_singles.py`、`melo_bench.py`、`piper_en_bench.py`、`samples.py`（均为只读测试脚本）
- `bench/out/`：中/英/混读样音、Melo 样音、piper 英文样音
- 本文档
