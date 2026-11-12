# Kokoro v1.1-zh（PyTorch 权重）CPU 跑通实测 + C++ 加速可行性

日期：2026-09-24　环境：本机 2×Xeon Gold 5218R（80 vCPU / 40 物理核），无 GPU，
`tts-kokoro-cpu:dev`（python 3.10 + torch 2.8.0+cpu）。权重：`fast-kokoro/weights/v1_1-zh`。

---

## 1. 结论

1. **`v1_1-zh` 权重已在 CPU 上跑通**：单模型支持中文、英文、中英混读，103 个音色包，
   API 与 Kokoro-FastAPI 一致（`POST /v1/audio/speech`）。样音见 `fast-kokoro/bench/out/`。
2. **纯中文单请求 RTF ≈ 0.18–0.20**（约 5× 实时），流式首包 TTFB ≈ 1.6–1.9 s（首句较长时）。
3. **当前代码把推理跑在 event loop 上，同一进程内请求是串行的**：无论并发多少，
   聚合吞吐都停在 ~5.4× 实时。改成 **4 个 uvicorn worker 进程**后，聚合吞吐升到 **~12.6× 实时**，
   并发 4 时的单请求 RTF 从 0.74 降到 0.32。**这才是 CPU 上最有效的加速手段**。
4. **用 C++/ONNX Runtime 重写推理并不能加速**：把同一份权重导出成 ONNX 后，
   ONNX Runtime(C++ 内核) 在本机 **比 PyTorch 慢 1.3–2.6 倍**（见 §4）。
   `sherpa-onnx` 本身就是 C++（ONNX Runtime C++ API），C++ 只是语言，不是性能来源。

---

## 2. 部署方式（新增文件）

| 文件 | 作用 |
| --- | --- |
| `fast-kokoro/Dockerfile.cpu` | x86_64 CPU 镜像，torch 走 `download.pytorch.org/whl/cpu` 索引 |
| `fast-kokoro/requirements-cpu.txt` | 按 `api/src/kokoro_fastapi.egg-info/requires.txt` 固定版本，去掉 GPU 依赖 |
| `fast-kokoro/docker-compose.cpu.yml` | CPU 部署（4 workers，挂载 code/weights/logs） |

启动：

```bash
docker build -f fast-kokoro/Dockerfile.cpu -t tts-kokoro-cpu:dev fast-kokoro
docker compose -f fast-kokoro/docker-compose.cpu.yml up -d --build   # 或 --no-build 复用镜像
curl -X POST http://127.0.0.1:8880/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"kokoro","input":"今天天气不错。","voice":"zf_001","response_format":"wav"}' \
  -o out.wav
```

### 跑通过程中必须处理的问题

1. `api/src/core/model_config.py` 里写的是 `from core.config import settings`（绝对导入）。
   容器里必须加 `PYTHONPATH=/app/api/src:/app`，否则 `ModuleNotFoundError: No module named 'core'`。
2. 原 `requires.txt` 的 CPU 组合缺两个包：`av`、`gputil`（`services/streaming_audio_writer.py`、
   debug 路由会 import），已补进 `requirements-cpu.txt`。
3. torch 必须用 CPU 索引安装，否则 PyPI 默认会拉 CUDA 版（数 GB 依赖）。
4. 配置项按现有默认值工作：`model_dir=/app/weights`、`voices_dir=/app/weights/v1_1-zh/voices`、
   `kokoro_v1_file=v1_1-zh/kokoro-v1_1-zh.pth`、`use_gpu=false`、`default_voice=zf_094`。
5. 启动日志出现 `上报数据失败: All connection attempts failed` 属正常（往 172.17.0.1:15006 上报）。

---

## 3. 实测数据（本机）

### 3.1 单请求（文本 66 字，中文）

| 配置 | 音频时长 | 墙钟 | RTF | 流式 TTFB |
| --- | --- | --- | --- | --- |
| OMP=8，1 worker | 15.44 s | 3.11 s | 0.201 | 1.57 s |
| OMP=32，1 worker | 15.44 s | 2.71 s | 0.176 | 1.72 s |
| OMP=8，4 workers | 15.44 s | 2.78 s | 0.180 | 1.95 s |

### 3.2 并发/吞吐（同一段文本 ×N 同时请求）

| worker 数 | 并发 | 单请求 RTF | 聚合吞吐 |
| --- | --- | --- | --- |
| 1 | 1 / 2 / 4 / 8 / 16 | 0.20 / 0.39 / 0.80 / 1.57 / 3.06 | 5.1–5.6× 实时 |
| **4** | 2 / 4 / 8 / 16 | 0.25 / 0.32 / 0.87 / 1.32 | **8.1–12.6× 实时** |

要点：

- 单 worker 时**并发不提升吞吐**（墙钟随并发线性变长）→ 推理在 event loop 上串行执行；
  提高 `OMP_NUM_THREADS` 几乎无效（5.1× → 5.6×）。
- 增加 **worker 进程数**才有效（4 workers ≈ 2.4× 吞吐）。本机 40 物理核，
  继续加 worker（8 个 × 4–8 线程）预计还能提升，需在目标机上调参。
- 音色包 103 个可切换；`zf_*` 为女声、`zm_*` 为男声、`af_*/bf_*` 为英文音色。

---

## 4. C++ / ONNX Runtime 加速验证（关键结论）

把 `kokoro-v1_1-zh.pth` 导出为 ONNX（脚本 `fast-kokoro/bench/export_onnx.py`，产物 325 MB），
用 ONNX Runtime CPU EP 与 PyTorch 在同一份权重、同一段文本上对比：

| 线程数 | PyTorch | ONNX Runtime | 结论 |
| --- | --- | --- | --- |
| 4 | 2.14 s (RTF 0.230) | 2.77 s (RTF 0.297) | ORT 慢 1.3× |
| 8 | 1.41 s (RTF 0.151) | 1.92 s (RTF 0.206) | ORT 慢 1.4× |
| 16 | 1.05 s (RTF 0.113) | 1.67 s (RTF 0.179) | ORT 慢 1.6× |
| 32 | 1.04 s (RTF 0.111) | 1.88 s (RTF 0.202) | ORT 慢 1.8× |
| 64 | 0.90 s (RTF 0.097) | 2.37 s (RTF 0.255) | ORT 慢 2.6× |

即：**同样的权重，换成 ONNX Runtime（C++ 内核）反而更慢**；而且 PyTorch 的线程扩展性更好。
这与上一轮结论一致：此前的 `sherpa-onnx`（C++）跑 Kokoro multi-lang ONNX 时，
单流 RTF ≈0.22–0.28，也不比 PyTorch 快。

### 导出过程中踩到的坑（都已解决，脚本可直接复用）

| 问题 | 处理 |
| --- | --- |
| `torch.istft` 产生 complex 张量，TorchScript 导出器报 `Unknown number type: complex` | 用实数实现重写 `TorchSTFT.inverse`：onesided→共轭对称全谱、逆 DFT 矩阵乘、窗函数 + `conv_transpose1d` 重叠相加、除以窗平方包络、`center` 裁剪。与 `torch.istft` 逐样本误差 1e-6 |
| `torch.stft`（谐波源）同样受 complex 限制 | 用实数 DFT 矩阵重写 `TorchSTFT.transform`，幅度误差 4e-6 |
| `pred_aln_trg[indices, arange] = 1`（index_put 双 1-D 索引） | 改成 `one_hot(indices).T` |
| `TextEncoder` / `DurationEncoder` 的 `x_pad[:, :, :L] = x` | 改成 `F.pad` / 切片 |
| `SineGen._f02sine` 里 `rand_ini[:, 0] = 0` 等原地写 | 改成 `torch.cat` 拼接 |
| dynamo 导出器（`dynamo=True`）在 SDPA 注意力 + data-dependent shape 上失败 | 放弃 dynamo，改用 TorchScript 导出器 + 上述补丁 |

### 结论性建议

- **不要为了“加速”去写 C++ 推理框架**：收益为负，成本很高（还要把中文 G2P 也搬到 C++，
  否则 token 对不上）。若目标是“去掉 Python/减小内存/便于嵌入”，那才值得评估。
- 真正有效的加速手段（按性价比排序）：
  1. **多 worker 进程**（本机 1→4 workers：吞吐 5.4×→12.6× 实时）；
  2. 每进程线程数与 worker 数按物理核配平（避免超订）；
  3. 量化（int8）——注意上一轮已实测：官方 `kokoro-int8` 在该 runtime 上**更慢**
     （每音频秒算力 2.4→6.3 核·秒），要用需自己重新做 QDQ/校准并实测；
  4. 减少文本切分粒度/合批以减少每句固定开销（该服务按 `target_max_tokens=250` 切块）；
  5. 有 GPU 时（当前环境没有）再考虑 CUDA/TensorRT。

---

## 5. 与上一轮 multi-lang v1_0 对比

| 项 | `kokoro-multi-lang-v1_0`（ONNX/sherpa-onnx） | `v1_1-zh`（PyTorch，本次） |
| --- | --- | --- |
| 中文质量 | 外国口音，业务不可接受 | 中文母语音色（业务认可） |
| 中文音色 | 8 个（zf/zm 混排） | 103 个音色包，男女声可选 |
| 英文 | 支持 | 支持（`af_*`/`bf_*` 及混读） |
| 单请求 RTF | 0.093–0.10（10 worker 并行分句） | 0.18–0.20（单进程串行，可通过多 worker 提升吞吐） |
| 中英混读 | 支持 | 支持 |

---

## 6. 待办 / 建议

1. 在目标设备（aarch64 `tts-imt900`）上按本文方法复测 worker/线程配比。
2. 生产部署建议 `--workers N`（N≈物理核/每 worker 线程数）并加健康检查、`mem_limit`
   （每 worker 各持一份模型，实测单进程 RSS 约 1.5 GB）。
3. 若要上 int8/ONNX，需要重新做量化并用本文脚本实测，不能默认“int8 一定更快”。
4. 中英混读文本建议按语言/标点切分后分别合成再拼接，减少语种切换处的韵律突变。

---

## 7. 【重要】发音不对的根因与修复（2026-09-24 追加）

### 现象

切换任何男/女中文音色（zf_001 / zf_094 / zm_010 …）发音都不对，不是正常普通话；
与音色无关，属于系统性问题。

### 根因：中文前端版本与模型词表不匹配

`kokoro` 的 `KPipeline` 用 **repo_id** 决定中文前端（G2P）版本：

```python
# kokoro/__init__.py, KPipeline.__init__
elif lang_code == 'z':
    self.g2p = zh.ZHG2P(
        version=None if repo_id.endswith('/Kokoro-82M') else '1.1',
        en_callable=en_callable,
    )
```

项目里 `settings.repo_id = "hexgrad/Kokoro-82M"`，`.endswith('/Kokoro-82M')` 为真，
于是走 **version=None（v1.0 的 IPA 前端）**。该前端输出的音素含
`↓ ↗ ↘ → ɤ ɥ ꭧ`，而 `v1_1-zh` 的词表里**没有这些符号**（词表 171 个符号，无调号箭头）。
`KModel.forward` 会把词表里查不到的音素静默丢弃：

```python
input_ids = list(filter(lambda i: i is not None, map(lambda p: self.vocab.get(p), phonemes)))
```

结果：**所有声调（↓↗↘→）和部分元音被丢掉**，剩下的音素拼出来的自然就不是普通话，
而且与音色无关——完全符合观测到的现象。

### 修复（一行配置）

```yaml
# fast-kokoro/docker-compose.cpu.yml
environment:
  REPO_ID: "hexgrad/Kokoro-82M-v1.1-zh"   # 只要不以 /Kokoro-82M 结尾即可走 v1.1 前端
```

权重全部是本地文件，repo_id 只影响前端版本选择，不会触发下载。
这与权重作者自带脚本 `weights/v1_1-zh/samples/make_zh.py` 中的
`REPO_ID = 'hexgrad/Kokoro-82M-v1.1-zh'` 一致。

### 验证（本地 Paraformer-zh ASR 回读，`sherpa-onnx-paraformer-zh-small-2024-03-09`）

输入文本：「据中央广播电视总台报道，救援队已抵达核心位置。」

| 版本 | ASR 回读结果 |
| --- | --- |
| 修复前（v1.0 前端） | 徐荣阳狂破天是总裁泡涛九恩退一体的心为 ← 完全是乱码 |
| 修复后（v1.1 前端） | **据中央广播电视总台报道救援队已抵达核心位置** ← 完全正确 |

输入文本：「今天是2026年9月24日，气温零下5度，进度百分之三十，电话13800138000。」

| 版本 | ASR 回读结果 |
| --- | --- |
| 修复前 | 今天是零六年九十四曲文零下午图金土排分三日天花要三八零零要三怕零零零 |
| 修复后 | **今天是二零二六年九月二十四日气温零下五度进度百分之三十电话幺三八零零幺三八零零零** |

长文本（135 字）修复后 ASR 回读与原文基本一致（仅个别字识别偏差），
音素覆盖检查 `missing-from-vocab = []`。

### 附带结论：语言与音色必须配对

同一模型下，中文音色读英文会明显劣化（ASR 回读「The service supports both Chinese and English.」）：

| 音色 | ASR 回读 |
| --- | --- |
| zf_001（中文女声） | 德妙 ervis 秒 apport 秒 boss china 秒 ind engli 秒 a ← 明显扭曲 |
| zf_094（中文女声） | the 妙 ervis 秒 appor 秒 boss 秒 indiind engli 秒 age ← 明显扭曲 |
| af_maple（英文女声） | **the service supports both chinese and english** ← 正确 |

所以：

- 中文文本 → `zf_*` / `zm_*`；英文文本 → `af_maple` / `af_sol` / `bf_vale`；
- 中英混读建议**按语种切分后分别选音色合成再拼接**（代价是切换处音色变化），
  否则总有一侧语言发音劣化。
- 权重作者脚本里还有两个可借鉴点：
  1. `en_callable` 对专有名词做特例映射（如 `Kokoro → kˈOkəɹO`、`Sol → sˈOl`）；
  2. 超过约 100 个音素会“赶读”，他们用 `speed_callable` 随长度降速
     ——本服务 `target_max_tokens=250 / absolute_max_tokens=450` 偏长，长句建议下调。

---

## 8. `voice=auto` 语言路由 + 100 音素分块（已实现）

### 代码改动（4 处，均在 `fast-kokoro/code/api/src/`）

| 文件 | 改动 |
| --- | --- |
| `services/language_split.py`（新增） | 按中/英切分文本、短片段合并、按语言选音色 |
| `core/config.py` | 新增 `auto_voice_zh_female=zf_001`、`auto_voice_zh_male=zm_010`、`auto_voice_en=af_maple`、`auto_min_span_chars=4`；分块阈值改为 `target_min=50 / target_max=100 / absolute_max=110` |
| `services/tts_service.py` | 新增 `_iter_voice_spans()`，`generate_audio_stream` 改为按「语种片段 × 音色」迭代（单音色时行为不变） |
| `routers/openai_compatible.py` | 允许 `auto` / `auto:male` / `auto:female` 直接透传 |

用法：

```bash
# 中文自动选女声 zf_001，英文自动选 af_maple
curl ... -d '{"model":"kokoro","input":"你好，这里是智能语音合成服务。The service supports both Chinese and English.","voice":"auto"}'
# 中文男声
curl ... -d '{"model":"kokoro","input":"收到指令，请把 GPU 监控数据导出来。","voice":"auto:male"}'
```

规则：

- 数字/标点/空格跟随前一个语种（`2026年` 不会被切成两段）；
- 短于 `AUTO_MIN_SPAN_CHARS`（默认 4）的语种片段并入相邻片段，避免 `GPU`／`AI`
  这类短词导致音色来回切换；
- 英文只有 3 个音色（`af_maple`/`af_sol`/`bf_vale`，均为女声），
  `auto:male` 的英文部分同样用 `auto_voice_en`；
- 可用环境变量覆盖四个配置项。

### 验证（ASR 回读，Paraformer-zh）

| 输入 | voice | 回读 |
| --- | --- | --- |
| 你好，这里是智能语音合成服务。 | auto | 你好这里是智能语音合成服务（zf_001） |
| The service supports both Chinese and English. | auto | the service supports both chinese and english（af_maple） |
| 中文句 + 英文句混排 | auto | **你好这里是智能语音合成服务 the service supports both chinese and english** |
| 收到指令，请把 GPU 监控数据导出来，然后 tell me the status of the system. | auto:male | **收到指令请把 gpu 监控数据导出来然后 tell me the status of the system** |

### 分块到 ~100 音素的代价与收益

| 指标 | 旧（≈450 音素/块） | 新（≈100 音素/块） |
| --- | --- | --- |
| 首包 TTFB（66 字中文） | 1.57–1.95 s | **1.35 s** |
| 135 字长文本总时长 | 30.67 s | 31.58 s（+3%，块间停顿） |
| 66 字文本总时长 | 15.44 s | 19.12 s（+24%，多了一个块边界） |
| 单块赶读现象 | 130+ 音素时明显 | 消除 |

即：**换来了首包更快、无赶读；代价是块边界会多出约 0.4 s 停顿**。若觉得停顿多，
把 `TARGET_MIN_TOKENS/TARGET_MAX_TOKENS` 调到 150/180 仍是安全区间。

---

## 9. 那套 GPU 优化经验在 CPU 上逐条验证

用 `fast-kokoro/bench/opt_probe.py` 在单块（129 音素、9.32 s 音频、8 线程）上实测：

| 经验 | CPU 上是否成立 | 实测结果 |
| --- | --- | --- |
| 固定数据形状 / 预分配内存 | 基本不成立（PyTorch CPU 侧） | 模型前向 1505 ms，整条调用链 1550 ms，**非模型开销仅 45 ms（3%）**；MKL 已复用内存池 |
| CUDA Graph（把重复计算打包） | 等价物是 `torch.compile` | 1513 ms → **1386 ms（1.09×，+9%）**，但首次编译 43 s、镜像需装 g++（也可挂 `TORCHINDUCTOR_CACHE_DIR` 复用编译产物） |
| 攒批处理多个请求 | 该模型**不支持**直接批处理 | `forward_with_tokens` 的 duration→帧对齐（`pred_aln_trg`）是按单条序列构造的，批量需改模型（逐项对齐 + padding）；未验证收益，改造量中等 |
| 缓存参考音频 / 文本预处理 | 收益极小 | `torch.load` 0.3 ms、`torch.save` 1.1 ms、G2P 4.4 ms，合计 **~6 ms/请求（0.4%）** |
| 删除清显存 / 冗余日志 | 无显存可清，日志开销也小 | 已含在上面 45 ms 里；再压 1–2% |
| **分层限流、控制并发** | **成立且最重要** | 单 worker 并发 1→16 吞吐恒为 **5.4×实时**（推理在 event loop 上串行）；4 workers → **12.6×**；8 workers → 9–14×（噪声内，内存翻倍到 14.4 GB）。超过「worker 数」的并发只会排队，单请求 RTF 线性变差 |

结论：**CPU 上的瓶颈不是「启动计算任务/数据搬运」，而是「一个进程一次只能算一条请求」**。
所以三件事按性价比排序：① 多 worker 进程（实测 2.3×）；② `torch.compile`（+9%，需编译器）；
③ 其余（缓存、去日志）合计 <5%。批量在 CPU 上不是免费午餐——模型本身需要先改造。

---

## 10. 接口压测脚本 `bench/tts_benchmark.py`（新增）

只依赖 HTTP 接口，不 import 服务端代码，可在任意机器（建议服务同机 `--network host`）运行：

```bash
# 默认：5 组内置文本（中/英/混读/长文/数字） + 流式 TTFB + 非流式 RTF + 并发阶梯
docker run --rm --network host -v $PWD/fast-kokoro/bench:/bench tts-kokoro-cpu:dev \
  python /bench/tts_benchmark.py --voice auto --repeat 3 --concurrency 1,2,4,8 \
  --save-dir /bench/out_bench --json /bench/out_bench/result.json

# 自定义文本 / 音色
python tts_benchmark.py --text-file long.txt --voice auto:male --format mp3
```

指标口径：

| 指标 | 含义 |
| --- | --- |
| TTFB | 首个响应字节到达时间（流式接口 = 首块音频合成完成时间） |
| RTF | 墙钟 / 音频时长（<1 即比实时快） |
| xRT | 实时倍数 = 1/RTF |
| 聚合吞吐 | 并发请求音频总时长 / 总墙钟（秒音频/秒） |
| audio arrival | 第 1/25%/50%/75%/100% 音频字节到达时间，用于判断流式是否真的边合成边下发 |

实测结果（`voice=auto`，4 workers，本机）：

| 文本 | 首包 TTFB p50 | 单请求 RTF | xRT |
| --- | --- | --- | --- |
| 中文 42 字 | 0.598 s | 0.188 | 5.3× |
| 中文 135 字 | 0.981 s | 0.176 | 5.7× |
| 中文含数字 43 字 | 0.655 s | 0.180 | 5.6× |
| 英文 97 字 | 0.989 s | 0.180 | 5.6× |
| 中英混读 61 字 | 0.636 s | 0.200 | 5.0× |

并发（同一段中文长文本）：n=1 → 5.1× 实时；**n=4 → 13.8× 实时**（单请求 RTF 0.289，延迟 p50 8.8 s）。

### 压测过程中发现并修复的两个并发缺陷

1. **`AudioChunk.combine([])` 越界** → 某语种片段没产出音频时抛 `list index out of range`，
   接口返回 500（非流式）或静默丢字（流式）。已在 `inference/base.py` 增加空列表兜底。
2. **音色张量走临时文件 → 并发写坏**：原代码每个请求 `torch.save` 一份
   `/tmp/temp_voice_<name>.pt` 再把路径传回 pipeline，中间有 `await`，并发请求会同时读写同一文件，
   偶发读到坏音色（表现为英文段丢失、响应体 158 KB 而不是 287 KB）。
   改为在 `KokoroV1` 里**按路径缓存音色张量**、直接把张量交给 pipeline
   （`KPipeline.load_voice` 本就支持传张量），同时省掉每请求的 save+load。

修复后 8 并发 × 3 组文本共 24 次请求：**0 失败，响应长度一致**。

---

## 11. 横向对比：sherpa（ONNX） vs fast-kokoro（PyTorch），以及「要不要搬到 sherpa」

### 11.1 三条路线的请求级 RTF（同一段 135 字中文，约 30.5 s 音频）

| 方案 | 调度方式 | 请求级 RTF | 实时倍数 | 备注 |
| --- | --- | --- | --- | --- |
| sherpa + **piper VITS**（现生产线） | 按句切分 + 10 线程池 | **0.0142** | 70× | 模型不同（VITS 63 MB），中文可用但**不能中英混读** |
| sherpa + **Kokoro multi-lang v1_0 ONNX** | 按句切分 + T10/W10 | **0.0926** | 10.8× | 中文有外国口音（业务已否决） |
| fast-kokoro + **v1.1-zh PyTorch**，改前 | 分块**串行** | 0.173–0.195 | 5–6× | 音色正确，但调度吃亏 |
| fast-kokoro + **v1.1-zh PyTorch**，改后 | **首块优先 + 其余并行** | **0.120**（1 worker）/ 0.140（4 workers） | 7–8× | 音色正确 |
| 同权重 v1.1-zh → **ONNX/ORT**，同调度 | 分块并行 | 0.085–0.093 | 11–12× | 比 PyTorch 快 **7–9%** |
| 同权重 v1.1-zh → ONNX/ORT，**单流** | 不分块 | 比 PyTorch **慢 1.3–2.6×** | — | ORT 并不总是更快 |

### 11.2 结论：sherpa 快在「调度」，不在「ONNX」

1. **改前 fast-kokoro 慢一倍，原因是分块串行**（推理跑在 event loop 上，一次只算一块），
   而不是 PyTorch 比 ONNX 慢——用同一份权重实测，把「按句切分 + 线程池」补上后，
   PyTorch 从 RTF 0.162 → 0.083（探针）/ 0.176 → 0.120（线上接口），已接近 sherpa+ONNX 的 0.093。
2. **同一份权重、同一调度下，ONNX 只快 7–9%**（T=16/W=4：torch 0.0933 vs ORT 0.0848），
   而单流下 ORT 反而慢 1.3–2.6×。sherpa 跑 Kokoro 的 0.0926 与我们的 ONNX 0.0925 几乎一致，
   说明**框架差异很小，调度与模型差异才是主因**。
3. 因此「把 v1.1-zh 搬到 sherpa-onnx」收益有限（≤10%），代价却很大：
   sherpa 的 Kokoro 前端走自己的 lexicon/jieba 路线，misaki 的 v1.1 前端（拼音→注音/音素、
   多音字与数字规则）没有 C++ 版，需要离线生成等价 lexicon 并回归验证，
   等于是重做刚修好的音色那一层，风险高。

### 11.3 在 fast-kokoro 中落地的调度优化

- `core/config.py` 新增 `chunk_parallelism`（env `CHUNK_PARALLELISM`），
  `services/tts_service.py` 改为「分块计划 → 线程池并行 → 按序取结果」：
  - 文本块数 > 2：**首块独占算力先跑**（保 TTFB），首块产出后再并行其余块；
  - 文本块数 ≤ 2：直接全部并行（竞争小，整体最快）。
- 实测（同一条 135 字中文，线上接口）：

  | 部署配置 | 单请求 RTF | TTFB | 聚合吞吐 |
  | --- | --- | --- | --- |
  | 4 workers / OMP=8 / 串行（改前） | 0.173–0.195 | 0.98 s | 13.8× |
  | 1 worker / OMP=16 / 并行 4 块 | **0.120** | 1.12 s | ~10× |
  | **4 workers / OMP=8 / 并行 2 块（当前默认）** | 0.140 | 1.13 s | **13.0×** |

  单请求延迟改善约 20–30%，聚合吞吐不掉；要极致低延迟可切到「1 worker + 并行 4 块」。
- 若还想再快 7–9%：**只把声学模型换成 ONNX Runtime**（前端仍用 Python 的 misaki G2P +
  音色张量，导出脚本见 `bench/export_onnx.py`），无需迁到 sherpa-onnx。

---

## 12. 「300 字 / 2 秒」目标的物理账

### 12.1 目标换算

300 字中文 ≈ **87 s 音频**（实测 231 字 → 67.1 s）。
若要求在 **2 s 内整段生成完**，需要 **RTF ≤ 0.023（≈43× 实时）**；
即每秒墙钟要产出 43 秒音频。

### 12.2 同机同文本实测（231 字 → 67 s 音频）

| 引擎 | 并行度 | 墙钟 | RTF | 实时倍数 | 算力需求 |
| --- | --- | --- | --- | --- | --- |
| piper VITS（sherpa 现生产线） | 10 路分块 | **0.74 s** | **0.0139** | **72×** | 0.24 核·秒/音频秒 |
| piper VITS | 串行 | 2.12 s | 0.0430 | 23× | 0.43 核·秒/音频秒 |
| **Kokoro v1.1-zh**（线上现配置） | 4 worker × 2 块 | 7.65 s | 0.1422 | 7× | — |
| **Kokoro v1.1-zh**（探针极限） | T=4/W=24 | **5.11 s** | **0.0761** | 13.1× | 1.21 核·秒/音频秒（纯净） |

### 12.3 结论

1. **Kokoro 在本机已基本到顶**：从线上 0.142 优化到探针极限 0.076 后，
   再提升只能靠更多核；本机共享（loadavg 19/80 vCPU），实际可用约 20–25 个核。
2. **理论核数需求**：Kokoro 每音频秒需 1.21 核·秒 →
   87 s 音频需约 **105 核·秒**；要 2 s 完成，需要约 **50 个能效打满的物理核**。
   本机 40 物理核且与他人共享，算下来是 4–5 s，**达不到 2 s**。
3. **回 sherpa 架构解决不了**：同一份权重同调度下 ONNX 只快 7–9%；
   sherpa+piper 快是因为 **VITS 模型本身便宜 5 倍**（0.24 vs 1.21 核·秒/音频秒），
   不是框架差异。把 Kokoro 放进 sherpa 仍受同样的算力约束。
4. **达标路线**：用 piper 类 VITS 做主力（中/英各一套 + 语种路由），
   实测 300 字 ≈ 1.2 s，**留有 1.6 倍余量**；Kokoro 作为「质量档/中英混读档」，
   其 231 字耗时 5–8 s。
5. 若「2 秒」指的是**首包 TTFB**（而非整段生成完），Kokoro **已经达标**：
   实测 TTFB 1.16 s（231 字，`voice=auto`）。两条口径需要在需求里明确。

---

## 13. 中英混排的真实场景：路由策略与领域文本规范化

样例（电力告警）：

```
乐园变11F5园开线相间距离I段、零序过流段保护动作，开关跳闸，重合成功，故障相别AC相，
故障测距1.359km(距离乐园变侧)，故障电流二次值A相&#x20;7.594A,C相9.313A.
```

### 13.1 路由会不会频繁切换音色？——实测：清洗后 0 次切换

切分结果（`min_span_chars=4`）：`乐园变11F5` / `园开线相间距离I…` / … 全部落在 `z`，
**音色切换次数 0**；`voice=auto` 与 `voice=zf_001` 输出完全一致（19.56 s，938 KB ± 几十字节）。

原因：`F5`(2) / `I`(1) / `AC`(2) / `km`(2) / `A`(1) 这些片段都短于阈值，会并入中文片段，
由中文音色的英文前端读字母/数字。

未清洗前的 `&#x20;` 是罪魁祸首：它导致 Latin 片段 `x20;7.594A,C`（12 字符）≥ 阈值，
于是切到英文音色 → 这才是"音色不对劲"的来源。

### 13.2 所以要不要用「中英双音色」？

| 场景 | 建议 |
| --- | --- |
| 电力告警这类「中文 + 短英文 token」 | **用单音色**（`zf_001`/`zm_010`）即可，路由是 no-op；字母数字可懂 |
| 出现成句英文（≥ `AUTO_MIN_SPAN_CHARS`） | 会切到英文音色，音色会变（同性别不同说话人） |
| 希望更保守 | 把 `AUTO_MIN_SPAN_CHARS` 调到 8~10；或主链路直接固定单音色，`auto` 只留给明确双语场景 |

### 13.3 真正要修的是领域文本规范化（已实现 `services/domain_normalizer.py`）

| 输入 | 修复前 ASR | 修复后 ASR |
| --- | --- | --- |
| `&#x20;` | 「x 二十」（且触发英文音色） | 作为空格处理，无异常 |
| `距离I段` | 「距离 i 段」 | **「距离一段」** |
| `7.594A` / `9.313A` | 「七点五九四了」 | **「七点五九四安」/「九点三一三安」** |
| `1.359km` | 「一点三五九千米」（内置归一化已对） | 同上 |

规则集（保守替换，可用 `DOMAIN_NORMALIZE=false` 关闭）：

1. HTML 实体（`&#x20;`/`&nbsp;`…）与零宽字符清理、空白收敛；
2. 全角数字/字母 → 半角；
3. 继电保护段号 `I/II/III 段`（含全角罗马数字）→ 一/二/三段；
4. 数值单位：`kV→千伏`、`kA→千安`、`MW→兆瓦`、`MVA→兆伏安`、`kWh→千瓦时`、
   `km→千米`、`Hz→赫兹`、`ms→毫秒`、`Ω→欧姆`、`V→伏`、**`A→安`**（仅当紧跟数字且后面
   不是字母/汉字，避免把 `A相` 的相别改掉）；
5. 不触碰正常中英混排与纯英文文本（已验证）。

### 13.4 仍待业务确认的读法

- 站点/设备专名（如「乐园变」）与开关编号读法：`11F5` 现在读「十一 F 五」，
  若业务习惯是「一一 F 五」需要加词典；
- `AC相` 建议按字母读（当前中文音色读作 A、C 两个字母，英文音色更清晰）；
- 建议把这类专名/缩写做成**可配置词表**（启动时加载），而不是硬编码在代码里。
