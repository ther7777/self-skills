---
name: vllm-optimization
description: vLLM 推理优化审计
user-invocable: false
---

# Skill: vLLM 推理优化审计

## 描述
针对使用 vLLM 推理框架的项目（包括独立推理部署和 RLHF/GRPO 训练中的推理组件），快速识别已采用和未采用的优化手段，提供针对性的吞吐、延迟和显存优化建议。

## 触发条件
当识别到用户项目满足以下任一条件时自动触发：
- 代码中 import 了 `vllm`（如 `from vllm import LLM`、`from vllm import SamplingParams`、`from vllm.engine`）
- 配置文件中包含 vLLM 特有参数：`tensor_parallel_size`（与 vllm 上下文共现）、`gpu_memory_utilization`、`max_model_len`、`enforce_eager`、`enable_prefix_caching`、`enable_chunked_prefill`
- 依赖文件中包含 `vllm`
- 启动命令使用 `vllm serve`、`python -m vllm.entrypoints`
- 项目中使用 TRL/OpenRLHF/verl 等 RL 框架并配置了 vLLM 作为推理后端（如 `use_vllm=True`、`vllm_mode`、`infer_backend: vllm`）
- 配置中出现 `vllm_gpu_util`、`vllm_maxlen`、`vllm_mode`、`sleep_level`、`enable_sleep_mode`

## 执行指令

你是 vLLM 推理优化专家。被调用时，请按照以下清单对目标项目进行全面审计。对每个检查项，判定其状态：**已启用** / **未启用** / **建议启用** / **不适用**，并给出具体的配置修改建议。

### 第一步：识别使用模式与基础信息

扫描项目的 Python 代码、配置文件和启动脚本，确定：

1. **使用模式**：
   - `standalone_serve`：独立部署推理服务（`vllm serve` / OpenAI 兼容 API）
   - `standalone_offline`：离线批量推理（`LLM()` + `generate()`）
   - `rlhf_server`：RLHF/GRPO 训练中的独立推理服务（如 TRL `vllm_mode="server"`）
   - `rlhf_colocate`：RLHF/GRPO 训练中与训练器共享 GPU（如 TRL `vllm_mode="colocate"`、verl colocate）
   - `framework_integrated`：通过训练框架集成（LlamaFactory `infer_backend: vllm`、ms-swift `use_vllm: true`）
2. **模型与规模**：`model` 参数值，识别模型系列和参数量
3. **硬件环境**：GPU 型号/数量、显存大小（结合 `/system-resources` 结果）
4. **GPU 架构**：Volta(V100) / Turing(T4/2080) / Ampere(A100/A10/3090/4090) / Ada(L40/4090) / Hopper(H100/H200/H20) / Blackwell(B200)
5. **并行配置**：`tensor_parallel_size`、`pipeline_parallel_size`、`data_parallel_size`

---

### 第二步：推理引擎优化审计

#### A. 引擎与编译优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **优化级别** | `-O0` / `-O1` / `-O2`（默认）/ `-O3` | O2 是默认级别（完整编译+CUDA Graph）。O0 适合调试，O1 适合快速启动，O2/O3 适合生产 |
| **CUDA Graph** | `enforce_eager: false`（默认）或通过 `-O` 级别控制 | CUDA Graph 减少 kernel launch 开销。`enforce_eager=True` 禁用 CUDA Graph，调试或显存不足时使用 |
| **torch.compile** | 通过 `-O1`/`-O2` 自动启用<br>`compilation_config` 高级配置 | V1 引擎默认启用。可自定义 `cudagraph_capture_sizes` 减少 CUDA Graph 显存占用 |
| **V1 引擎** | vLLM ≥0.8 默认使用 V1 引擎 | V1 引擎优化：多进程架构、decode 优先调度、RECOMPUTE 抢占模式。确认版本 ≥0.8 |
| **CUDA Graph 显存控制** | `cudagraph_capture_sizes=[1,2,4,8,16]` | 减少默认 capture sizes 可降低 CUDA Graph 显存占用，代价是更多 padding |

**建议**：
- 生产环境使用 `-O2`（默认）或 `-O3`
- 显存极紧张时设 `enforce_eager=True` 释放 CUDA Graph 显存
- 调试时使用 `-O0`（禁用所有编译和 CUDA Graph）

---

#### B. 显存管理

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **GPU 显存利用率** | `gpu_memory_utilization: 0.92`（默认） | 控制分配给模型+KV Cache 的显存比例。OOM 可降低，显存充裕可提至 0.95 |
| **最大序列长度** | `max_model_len`（默认从 model config 读取） | 限制最大序列长度可大幅减少 KV Cache 显存。设为实际最大需求而非模型上限 |
| **KV Cache 量化** | `kv_cache_dtype: "fp8"` / `"fp8_e4m3"` / `"fp8_e5m2"`<br>`calculate_kv_scales: true` | FP8 KV Cache 减少约 50% 缓存显存，轻微精度损失。H100/Ada+ 推荐 |
| **Swap Space** | `swap_space`（GiB） | CPU 内存用于 KV Cache 交换。增大可处理更多并发请求 |
| **KV Cache Offload** | `kv_transfer_config: {"kv_connector": "OffloadingConnector", ...}` | 将完成的 KV blocks 异步 DMA 到 CPU。多轮对话/长上下文场景 |
| **Block Size** | `block_size: 16`（默认） | KV Cache 块大小（token 数）。通常不需要调整 |

**建议**：
- 长上下文不需要时设置 `max_model_len` 为实际最大值（如 4096 而非模型支持的 128K）
- Hopper/Ada GPU 启用 `kv_cache_dtype: "fp8"` + `calculate_kv_scales: true`
- 并发请求多时适当提高 `gpu_memory_utilization`（0.93~0.95）
- 多轮长对话启用 KV Cache Offloading

---

#### C. 并行策略

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **张量并行（TP）** | `tensor_parallel_size: N` | 单节点内多 GPU 最常用方案。模型层内拆分。N = 最少满足显存需求的 GPU 数 |
| **流水线并行（PP）** | `pipeline_parallel_size: N` | 层间拆分。可与 TP 组合（TP×PP = 总 GPU 数）。适合超大模型 |
| **数据并行（DP）** | `data_parallel_size: N`<br>`data_parallel_backend: ray/mp` | 多副本负载均衡。内部 DP 提供单端点多副本。TP×PP×DP = 总 GPU 数 |
| **Expert 并行（EP）** | `enable_expert_parallel: true` | MoE 模型专用（DeepSeek-V3、Qwen3MoE、Llama-4）。将专家分配到不同 GPU |
| **DP + TP 组合** | `--data-parallel-size 4 --tensor-parallel-size 2` | 8 GPU: 4 个 DP 副本 × 每副本 2 卡 TP |

**建议**：
- 单模型装不下单卡 → TP=装下模型的最少 GPU 数（通常 2/4/8）
- 追求高吞吐+多副本 → DP × TP（如 8 卡 = DP4 × TP2）
- 超大模型跨节点 → PP + TP
- MoE 模型 → EP + TP

---

#### D. 量化策略

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **在线 FP8 量化** | `quantization: "fp8_per_tensor"` / `"fp8_per_block"` / `"mxfp8"` | 无需预量化模型，直接加载 BF16 权重在线量化。Hopper+ 推荐 `fp8_per_block` |
| **AWQ 量化** | `quantization: "awq"` | 4-bit 权重量化。需预量化模型。Turing+ 支持 |
| **GPTQ 量化** | `quantization: "gptq"` | 4-bit 权重量化。需预量化模型。Volta+ 支持 |
| **BitsAndBytes** | `quantization: "bitsandbytes"` | 4/8-bit 量化。灵活但性能略低。Volta+ 支持 |
| **GGUF** | `quantization: "gguf"` | llama.cpp 格式。支持多种量化位宽。全 GPU 支持 |
| **Marlin Kernel** | 自动选择（AWQ/GPTQ 预量化模型） | 高性能 4-bit kernel。Turing+ 自动启用 |
| **混合量化** | `quantization_config: {"linear": "fp8_per_block", "moe": "fp8_per_tensor"}` | 不同层类型用不同量化精度。MoE 模型特别有用 |

**建议**：
- H100/H200/Ada → `quantization: "fp8_per_block"`（在线量化，无需预处理）
- A100/V100 → AWQ 或 GPTQ 预量化模型
- 追求最大压缩 → `quantization: "bitsandbytes"` 4-bit
- MoE 模型 → 混合量化（Dense 层 FP8 per-block + MoE 层 FP8 per-tensor）

---

#### E. 调度与批处理优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **分块预填充** | `enable_chunked_prefill: true`（V1 默认开启） | 将大 prefill 拆块与 decode 交错，平衡 TTFT 和 ITL |
| **最大批处理 Token** | `max_num_batched_tokens: 2048`（默认） | 每次迭代最大 token 数。增大→TTFT↓ 吞吐↑；减小→ITL↓ 延迟优先 |
| **最大并发序列** | `max_num_seqs: 128`（默认） | 每次迭代最大并发请求数。减小可降低 ITL |
| **前缀缓存** | `enable_prefix_caching: true`（V1 默认开启） | 自动缓存共享前缀的 KV Cache。多轮对话/RAG 场景大幅加速 |
| **最大部分预填充** | `max_num_partial_prefills` | 控制并发部分 prefill 操作数量 |

**建议**：
- **吞吐优先**：`max_num_batched_tokens=8192+`、`max_num_seqs=256`
- **延迟优先**（对话场景）：`max_num_batched_tokens=2048`、`max_num_seqs=64`
- 多轮对话/RAG 确保 `enable_prefix_caching=true`
- 大模型+小 GPU 时减小 `max_num_seqs` 避免抢占

---

#### F. 注意力后端

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Flash Attention 版本** | `--attention-config.flash_attn_version=2/3/4` | FA2 通用；FA3 Hopper+；FA4 Blackwell(SM100+) |
| **注意力后端选择** | `--attention-backend FLASH_ATTN / FLASHINFER / TRITON_ATTN` | Ampere/Hopper: FLASH_ATTN 优先；Blackwell: FLASHINFER 优先 |
| **MLA 后端** | `FLASHMLA` / `FLASH_ATTN_MLA` / `TRITON_MLA` / `CUTLASS_MLA` | DeepSeek 系列 MLA 架构专用 |
| **FP8 注意力** | FA3 + `kv_cache_dtype="fp8"` 自动启用 | Hopper+FA3 可在 FP8 域执行 attention |

**建议**：
- Ampere(A100)/Hopper(H100) → `FLASH_ATTN` + FA2/FA3
- Blackwell(B200) → `FLASHINFER` + FA4
- DeepSeek-V2/V3 → 使用 MLA 专用后端（FLASHMLA 或 FLASH_ATTN_MLA）
- 通常使用默认 auto 选择即可，仅在性能敏感时手动指定

---

#### G. 推测解码

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **EAGLE/EAGLE3** | `speculative_config: {"method": "eagle", "model": "...", "num_speculative_tokens": 5}` | 高收益推测解码。需下载对应 EAGLE 模型 |
| **MTP（多 Token 预测）** | `speculative_config: {"method": "mtp", "num_speculative_tokens": 1}` | 利用模型��生 MTP head（如 DeepSeek-V3）。无需额外模型 |
| **Draft Model** | `speculative_config: {"method": "draft_model", "model": "...", "num_speculative_tokens": 5}` | 使用同系列小模型。需额外显存 |
| **N-gram** | `speculative_config: {"method": "ngram", "num_speculative_tokens": 5}` | 轻量级，无需额外模型。重复性文本场景效果好 |
| **Draft TP** | `speculative_config: {..., "draft_tensor_parallel_size": N}` | Draft model 独立 TP 配置 |

**建议**：
- 有对应 EAGLE 模型 → 优先 EAGLE（加速 2-3x）
- DeepSeek-V3 等原生 MTP → 使用 MTP（免费加速，无需额外模型）
- 无对应模型但需加速 → N-gram（零成本尝试）
- 高 QPS 场景推测解码收益降低，需权衡

---

#### H. RLHF/训练集成优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Sleep Mode** | `enable_sleep_mode: true`<br>`llm.sleep(level=1/2)` / `llm.wake_up()` | Level 1: offload 权重到 CPU；Level 2: 释放所有显存。RLHF colocate 必用 |
| **Weight Transfer** | `weight_transfer_config: {"backend": "nccl"/"ipc"}` | NCCL: 跨 GPU 传权重；IPC: 同 GPU CUDA IPC。训练后同步权重到推理引擎 |
| **Colocate 模式** | `vllm_mode: "colocate"`（TRL）<br>verl colocate | 推理与训练共享 GPU。必须配合 sleep mode 切换显存 |
| **Server 模式** | `vllm_mode: "server"`（TRL）<br>���立 vLLM 进程 | 推理独占 GPU，通过 HTTP 通信。更稳定但需额外 GPU |
| **Async RL** | `POST /pause` + `POST /resume`<br>`engine.pause_generation()` / `engine.resume_generation()` | 异步 RL：生成暂停→同步权重→恢复生成。避免阻塞 |
| **动态 LoRA** | `POST /v1/load_lora_adapter {"load_inplace": true}` | RLHF 训练完一轮后热更新 LoRA 权重，无需重启 |
| **Layerwise Reloading** | QeRL 分层重载 | 量化权重热更新，不需重新编译 |

**建议**：
- **GRPO/PPO + colocate**：必须启用 `enable_sleep_mode=true`，训练时 `sleep(level=2)`，推理时 `wake_up()`
- **GRPO/PPO + server**：独立 GPU 跑 vLLM，用 weight transfer（NCCL）同步权重
- **LoRA RL**：训练完一轮 → `load_lora_adapter(load_inplace=True)` 热更新
- 大规模 RL → 使用 async RL（pause/resume）避免全量阻塞

---

#### I. 生产部署优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **LoRA 服务** | `--enable-lora --lora-modules sql=path/`<br>动态加载 API | 多 LoRA 并发服务。运行时动态加载/卸载 |
| **API Server 多进程** | `--api-server-count 4` | 多个 API 处理进程，提升输入处理吞吐 |
| **Disaggregated Prefill** | `--kv-transfer-config {"kv_connector": "NixlConnector"}` | Prefill 和 Decode 分离到不同实例。独立扩缩容 |
| **YAML 配置文件** | `vllm serve --config config.yaml` | 生产环境推荐使用 YAML 配置文件统一管理 |
| **多模态限制** | `limit_mm_per_prompt: {"image": 3}`<br>`mm_processor_kwargs: {"max_pixels": 768*768}` | 控制多模态输入资源，防止 OOM |

**建议**：
- 生产环境用 YAML 配置文件管理所有参数
- 高 QPS → `--api-server-count` 匹配 CPU 核数
- 多模型服务 → LoRA 热加载（共享基座模型）
- 超低延迟需求 → Disaggregated Prefill（独立扩 prefill 实例）

---

#### J. 环境与调优

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **快速 Tokenizer** | `VLLM_USE_FASTOKENS=1` | Rust 后端 BPE tokenizer，高 QPS 场景加速 |
| **CPU 核心数** | 物理核心 ≥ 2 + N（N = GPU worker 数） | CPU 不足会成为隐性瓶颈。DP 场景需更多核 |
| **Worker 进程方式** | `VLLM_WORKER_MULTIPROC_METHOD=spawn` | 默认 fork。某些环境需 spawn（如 CUDA MPS） |
| **Dev Mode** | `VLLM_SERVER_DEV_MODE=1` | 启用 sleep/wake、weight transfer 等开发接口。RLHF 场景需要 |
| **长序列允许** | `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` | 允许 max_model_len 超过模型默认值 |
| **预抢占处理** | 监控 preemption 计数 | 频繁抢占 = KV Cache 不足。增大 `gpu_memory_utilization` 或减小并发 |

**建议**：
- 高 QPS 生产 → `VLLM_USE_FASTOKENS=1`
- RLHF 场景 → `VLLM_SERVER_DEV_MODE=1`
- 确保 CPU 核心数满足 `2 + tensor_parallel_size + data_parallel_size`
- 监控 preemption 日志，频繁抢占说明需要增加显存/减少并发

---

### 第三步：场景化配置模板

根据使用模式和硬件条件，推荐最优配置组合：

#### 场景 1：独立推理服务 — 8B 模型 + 单卡 A100 80GB

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    dtype="auto",                        # BF16 on A100
    gpu_memory_utilization=0.92,
    max_model_len=8192,                  # 限制为实际需求
    enable_prefix_caching=True,          # V1 默认
    enable_chunked_prefill=True,         # V1 默认
    max_num_seqs=128,
)
```

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.92 \
  --max-model-len 8192 \
  -O2
```

#### 场景 2：70B 模型 + 4xA100 80GB — 高吞吐服务

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.93 \
  --max-model-len 16384 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --kv-cache-dtype fp8 \
  -O2
```

#### 场景 3：70B 模型 + 8xH100 — DP×TP 最大吞吐

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --data-parallel-size 4 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8 \
  --max-num-batched-tokens 8192 \
  --api-server-count 4 \
  -O2
```

#### 场景 4：GRPO 训练 + vLLM Colocate（TRL）

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="colocate",
    vllm_gpu_memory_utilization=0.4,   # colocate 模式需分配较少显存
)

# vLLM 侧配置（自动生成）
# enable_sleep_mode=True
# 训练时自动 sleep(level=2)，推理时自动 wake_up()
```

#### 场景 5：GRPO 训练 + vLLM Server 模式（TRL）

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    use_vllm=True,
    vllm_mode="server",
    # vLLM 独立进程在单独 GPU 上运行
    # 通过 HTTP API 通信，NCCL 同步权重
)
```

```bash
# 单独启动 vLLM server（RLHF 场景需 dev mode）
VLLM_SERVER_DEV_MODE=1 vllm serve model \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --enable-sleep-mode \
  -O2
```

#### 场景 6：8B 模型 + 单卡 24GB（RTX 4090）— 量化推理

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    quantization="fp8_per_block",         # 在线 FP8 量化，无需预处理
    gpu_memory_utilization=0.92,
    max_model_len=4096,
    max_num_seqs=64,
)
```

#### 场景 7：DeepSeek-V3 671B MoE + 8xH100 — Expert 并行

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --attention-backend FLASHMLA \
  -O2
```

#### 场景 8：推测解码 — EAGLE 加速

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --speculative-config '{
    "method": "eagle",
    "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
    "num_speculative_tokens": 5
  }' \
  --gpu-memory-utilization 0.92 \
  -O2
```

---

### 第四步：输出审计报告

按以下格式输出审计结果：

```markdown
# vLLM 推理优化审计报告

## 基本信息
- 模型：{model}（{参数量}）
- 使用模式：{standalone_serve / rlhf_colocate / rlhf_server / ...}
- 硬件：{GPU型号} x {数量}（{显存}GB）
- vLLM 版本：{如可检测}
- GPU 架构：{Ampere / Hopper / ...}

## 审计结果总览

| 类别 | 得分 | 已启用 | 建议启用 |
|------|------|--------|----------|
| A. 引擎与编译优化 | x/5 | ... | ... |
| B. 显存管理 | x/6 | ... | ... |
| C. 并行策略 | x/5 | ... | ... |
| D. 量化策略 | x/6 | ... | ... |
| E. 调度与批处理 | x/6 | ... | ... |
| F. 注意力后端 | x/4 | ... | ... |
| G. 推测解码 | x/4 | ... | ... |
| H. RLHF/训练集成 | x/7 | ... | ... |
| I. 生产部署 | x/5 | ... | ... |
| J. 环境与调优 | x/4 | ... | ... |
| **总计** | **x/52** | | |

## 优先优化建议（按影响排序）

### P0 - 立即执行���显著收益，零风险）
1. ...

### P1 - 强烈推荐（明显收益，低风险）
1. ...

### P2 - 建议尝试（中等收益，需测试）
1. ...

## 推荐配置修改

（给出具体的代码/配置/启动命令修改）
```

---

## 常见优化决策树

```
开始
├── 显存不足（OOM）？
│   ├── 是 → 已量化？
│   │   ├── 否 → Hopper/Ada? → quantization: "fp8_per_block"
│   │   │   └── 其他 → 使用 AWQ/GPTQ 预量化模型
│   │   └── 是（已量化）
│   │       ├── 减小 max_model_len（限制为实际需求）
│   │       ├── 减小 max_num_seqs（如 64→32）
│   │       ├── enforce_eager=True（释放 CUDA Graph 显存）
│   │       ├── 增加 tensor_parallel_size（需更多 GPU）
│   │       └── 启用 KV Cache offloading
│   └── 否 → 继续性能优化
├── 吞吐不足？
│   ├── GPU 利用率低 → 增大 max_num_seqs / max_num_batched_tokens
│   ├── 单实例已满 → 启用 data_parallel_size（多副本）
│   ├── Tokenization 慢 → VLLM_USE_FASTOKENS=1 + --api-server-count
│   ├── 推测解码 → 启用 EAGLE/MTP/N-gram
│   └── 多模型共享 → LoRA 热加载（共享基座模型）
├── 延迟过高（TTFT）？
│   ├── 增大 max_num_batched_tokens（允许更大 prefill batch）
│   ├── 启用 prefix_caching（共享前缀复用）
│   └── Disaggregated Prefill（独立 prefill 实例）
├── 延迟过高（ITL）？
│   ├── 减小 max_num_batched_tokens（decode 优先）
│   ├── 减小 max_num_seqs（减少并发）
│   └── 推测解码（减少 decode 步数）
└── RLHF 场景？
    ├── Colocate 模式
    │   ├── 必须 enable_sleep_mode=True
    │   ├── 训练时 sleep(level=2)
    │   ├── 推理时 wake_up()
    │   └── 降低 gpu_memory_utilization（给训练留空间）
    └── Server 模式
        ├── VLLM_SERVER_DEV_MODE=1
        ├── Weight Transfer (NCCL backend)
        └── 或 Dynamic LoRA 热更新
```

## 性能参考

### 单卡 A100 80GB 可服务模型规模

| 模型规模 | 量化方式 | max_model_len | 并发能力 |
|---------|---------|---------------|---------|
| 7-8B | BF16 | 32K | ~256 并发 |
| 7-8B | FP8 | 32K | ~384 并发 |
| 13-14B | BF16 | 16K | ~128 并发 |
| 30-34B | FP8 | 8K | ~64 并发 |
| 70B | 需 2+ GPU TP | - | - |

### RLHF 模式显存分配参考

| 模式 | vLLM gpu_memory_utilization | 训练侧可用 | 说明 |
|------|---------------------------|-----------|------|
| Colocate（交替） | 0.4~0.5 | sleep 时释放全部 | sleep/wake 切换，总利用率高 |
| Colocate（并行） | 0.3~0.4 | 0.5~0.6 | 推理+训练同时占用，需精确分配 |
| Server（独立 GPU） | 0.92 | 100%（独立 GPU） | 各自独立，资源隔离 |

## vLLM 特有关键词速查

用于识别 vLLM 项目的关键词和文件模式：

| 类别 | 关键词/模式 |
|------|------------|
| CLI | `vllm serve`、`python -m vllm.entrypoints` |
| 配置 | `gpu_memory_utilization`、`tensor_parallel_size`（vllm 上下文）、`enforce_eager`、`enable_prefix_caching`、`enable_chunked_prefill`、`max_num_seqs`、`max_num_batched_tokens`、`kv_cache_dtype`、`speculative_config` |
| import | `from vllm import LLM`、`from vllm import SamplingParams`、`from vllm.engine`、`from vllm.config`、`import vllm` |
| 依赖 | `vllm`（pip/conda） |
| RLHF 集成 | `use_vllm=True`（TRL）、`vllm_mode`（TRL）、`infer_backend: vllm`（LlamaFactory）、`use_vllm: true`（ms-swift）、`enable_sleep_mode` |
| 环境变量 | `VLLM_USE_FASTOKENS`、`VLLM_SERVER_DEV_MODE`、`VLLM_WORKER_MULTIPROC_METHOD`、`VLLM_ALLOW_LONG_MAX_MODEL_LEN` |
