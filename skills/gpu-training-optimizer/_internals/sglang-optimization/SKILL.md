---
name: sglang-optimization
description: SGLang 推理优化审计
user-invocable: false
---

# Skill: SGLang 推理优化审计

## 描述
针对使用 SGLang 推理框架的项目（包括独立推理部署和 RLHF/GRPO 训练中的推理组件），快速识别已采用和未采用的优化手段，提供针对性的吞吐、延迟和显存优化建议。SGLang 以 RadixAttention（自动前缀缓存）、DP Attention、Piecewise CUDA Graph 等创新特性著称，是 DeepSeek 系列模型的推荐推理引擎。

## 触发条件
当识别到用户项目满足以下任一条件时自动触发：
- 代码中 import 了 `sglang`（如 `from sglang import`、`import sglang`、`from sglang.srt`、`from sglang.launch_server`）
- 配置文件中包含 SGLang 特有参数：`mem_fraction_static`、`schedule_policy`、`chunked_prefill_size`、`enable_dp_attention`、`radix_cache`
- 依赖文件中包含 `sglang` 或 `sgl-kernel`
- 启动命令使用 `python -m sglang.launch_server`、`python -m sglang_router.launch_server`、`sglang.launch_server`
- 项目中使用 RL 框架并配置了 SGLang 作为推理后端（如 `infer_backend: sglang`、`sglang_maxlen`、`sglang_mem_fraction`）
- 配置中出现 `enable_memory_saver`、`update_weights_from_distributed`、`pause_generation`

## 执行指令

你是 SGLang 推理优化专家。被调用时，请按照以下清单对目标项目进行全面审计。对每个检查项，判定其状态：**已启用** / **未启用** / **建议启用** / **不适用**，并给出具体的配置修改建议。

### 第一步：识别使用模式与基础信息

扫描项目的 Python 代码、配置文件和启动脚本，确定：

1. **使用模式**：
   - `standalone_serve`：独立部署推理服务（`python -m sglang.launch_server` / OpenAI 兼容 API）
   - `standalone_offline`：离线批量推理（`sgl.Engine` + `generate()`）
   - `rlhf_colocate`：RLHF/GRPO 训练中与训练器共享 GPU（`enable_memory_saver`、sleep/wake）
   - `rlhf_server`：RLHF/GRPO 训练中的独立推理服务
   - `framework_integrated`：通过训练框架集成（LlamaFactory `infer_backend: sglang`、ms-swift `use_sglang`）
   - `router_gateway`：使用 SGLang Model Gateway（`sglang_router`）多副本部署
2. **模型与规模**：`--model-path` 参数值，识别模型系列（DeepSeek/Llama/Qwen/MiniMax 等）和参数量
3. **模型架构类型**：Dense / MoE / MLA（Multi-head Latent Attention）/ DSA（DeepSeek Sparse Attention）
4. **硬件环境**：GPU 型号/数量、显存大小（结合 `/system-resources` 结果）
5. **GPU 架构**：Ampere(A100/A10) / Ada(L40/4090) / Hopper(H100/H200/H20) / Blackwell(B200)
6. **并行配置**：`--tp-size`、`--dp-size`、`--ep-size`、`--pp-size`

---

### 第二步：推理引擎优化审计

#### A. 引擎与计算优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Piecewise CUDA Graph** | 默认启用<br>`--piecewise-cuda-graph-max-tokens`<br>`--piecewise-cuda-graph-compiler: eager/inductor` | SGLang 独有。逐层捕获 CUDA Graph，支持动态 token 数。prefill + decode 均可 Graph 化。非 MLA 默认 = chunked_prefill_size，MLA 默认 = 2048 |
| **CUDA Graph Max BS** | `--cuda-graph-max-bs`（默认自动） | 增大可支持更大 decode batch。TP 大时建议 512-768 |
| **torch.compile** | `--enable-torch-compile`<br>`--torch-compile-max-bs 32` | 小模型 + 小 batch 可加速。注意：当前标记为"out of maintenance"，可能有兼容性问题 |
| **Overlap Scheduling** | 默认启用（`--disable-overlap-schedule` 关闭） | CPU 调度与 GPU 计算重叠。不建议关闭 |
| **Two-Batch Overlap** | `--enable-two-batch-overlap` | MoE/EP 场景：将 batch 拆为 micro-batch，attention 与 dispatch/combine 交错。吞吐最高 2x |
| **Single-Batch Overlap** | `--enable-single-batch-overlap` | 共享专家计算与通信重叠 |
| **连续解码步数** | `--num-continuous-decode-steps 1`（默认） | 增大可减少调度开销、提升吞吐，但增大 TTFT |
| **FP8 GEMM 后端** | `--fp8-gemm-backend auto` | 选项：`deep_gemm`（Hopper 推荐）、`flashinfer_trtllm`、`cutlass`、`triton` |
| **DeepGEMM JIT** | `SGLANG_ENABLE_JIT_DEEPGEMM=true`（SM90/SM100 默认） | Hopper/Blackwell 自动 JIT 编译高性能 GEMM kernel |

**建议**：
- 生产环境保持 Piecewise CUDA Graph 默认开启
- MoE 模型启用 `--enable-two-batch-overlap`
- Hopper GPU 确认 DeepGEMM JIT 已启用（默认开启）
- 高 QPS 吞吐场景可适当增大 `--num-continuous-decode-steps`（如 2-4）

---

#### B. 显存管理

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **静态显存比例** | `--mem-fraction-static`（默认自动） | 控制模型权重 + KV Cache 池占 GPU 显存比例。默认自动留 5-8GB 给激活。可按 0.01 递增至接近 OOM |
| **最大 Token 总数** | `--max-total-tokens`（默认自动） | KV Cache 池大小。增大可处理更多并发请求 |
| **上下文长度** | `--context-length`（默认从 model config） | 限制为实际需求可节省 KV Cache 分配 |
| **KV Cache 量化** | `--kv-cache-dtype fp8_e4m3` / `fp8_e5m2` / `fp4_e2m1` | FP8 KV Cache 减少 ~50% 缓存显存；FP4 减少 ~75%（实验性）。精度损失极小 |
| **分块预填充大小** | `--chunked-prefill-size`（默认自动） | 减小可降低 prefill 显存峰值。OOM 时设 4096 或 2048 |
| **最大运行请求数** | `--max-running-requests`（默认自动） | 减小可降低显存占用和抢占 |
| **HiCache 分层缓存** | `--enable-hierarchical-cache`<br>`--hicache-ratio 2.0` | 三层缓存：GPU → CPU → 外部存储。极大提升长上下文缓存命中率 |
| **CPU Offload** | `--cpu-offload-gb` | 预留 CPU 内存用于模型参数卸载 |

**建议**：
- 逐步提高 `--mem-fraction-static`（0.01 递增）直到接近 OOM 以最大化 KV Cache
- 长上下文不需要时设置 `--context-length` 为实际最大值
- Hopper/Ada GPU 启用 `--kv-cache-dtype fp8_e4m3`
- 多轮长对话/RAG 场景启用 HiCache（`--enable-hierarchical-cache --hicache-ratio 2`）
- OOM 时先减 `--chunked-prefill-size` 再减 `--mem-fraction-static`

---

#### C. 前缀缓存（RadixAttention）

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **RadixAttention** | 默认启用（`--disable-radix-cache` 关闭） | SGLang 核心特性。通过 radix tree 自动检测和复用共享前缀的 KV Cache。不建议关闭 |
| **驱逐策略** | `--radix-eviction-policy lru`（默认） | 可选 `lfu`。LRU 通用性好，LFU 适合热点前缀固定的场景 |
| **调度策略配合** | `--schedule-policy lpm`（最长前缀匹配） | 共享前缀多时用 `lpm` 替代默认 `fcfs`，提升 RadixAttention 缓存命中率 |
| **Chunked Prefix Cache** | `SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD=8192` | MLA 模型长序列优化：将前缀切块处理再合并。超过阈值才启用 |
| **Page Size** | `--page-size 1`（默认） | 影响前缀匹配粒度。1 = token 级（最精确），FlashMLA 强制 64，CutlassMLA 强制 128 |

**建议**：
- **确保 RadixAttention 未被关闭**（默认开启）
- 多轮对话/RAG/共享 system prompt → `--schedule-policy lpm` 大幅提升缓存命中
- MLA 模型 + 长序列 → 确保 Chunked Prefix Cache 阈值合理
- 如需完全确定性结果 → `--disable-radix-cache`（牺牲性能）

---

#### D. 并行策略

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **张量并行（TP）** | `--tp-size N` | 单节点内多 GPU。N = 满足显存需求的最少 GPU 数 |
| **数据并行（DP）** | `--dp-size N` | 多副本。**强烈建议通过 SGLang Model Gateway（SMG）部署** |
| **DP Attention（DPA）** | `--enable-dp-attention` | **SGLang 独有**。MLA 模型必用。Attention 层走 DP + FFN/MoE 层走 TP/EP。消除 KV Cache 复制。约束：`tp_size % dp_size == 0` |
| **Expert 并行（EP）** | `--ep-size N` | MoE 模型。配合 DeepEP（`--moe-a2a-backend deepep`）和 DeepGEMM（`--moe-runner-backend deep_gemm`） |
| **流水线并行（PP）** | `--pp-size N` | 超大模型跨节点 |
| **SGLang Model Gateway** | `python -m sglang_router.launch_server --dp-size N` | Rust 实现的生产级路由器。cache-aware 路由策略提升 92% 吞吐、275% 缓存命中率 |
| **多节点** | `--nnodes N --node-rank R --dist-init-addr host:port` | 多机部署 |
| **NCCL NVLS** | `--enable-nccl-nvls` | Prefill-heavy 请求加速 |
| **Symmetric Memory** | `--enable-symm-mem` | SM90+ 快速 collective |

**建议**：
- MLA 模型（DeepSeek/MiniMax/Kimi-K2）→ 必须 `--enable-dp-attention`
- MoE 模型 → `--ep-size` + `--moe-a2a-backend deepep` + `--moe-runner-backend deep_gemm`
- 生产多副本 → 用 SMG（`sglang_router`）而非原生 DP
- 追求吞吐 → DP > TP（显存允许时优先增大 DP）
- DeepSeek-V3 推荐：`--tp 8 --dp-size 8 --ep 8 --enable-dp-attention`

---

#### E. 量化策略

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **在线 FP8 量化** | `--quantization fp8` | 动态 FP8 量化，无需预处理。Hopper+ 推荐 |
| **离线预量化** | 加载 AWQ/GPTQ/FP8/NVFP4 预量化模型（**不要加 --quantization**） | 推荐方式：精度更可控 |
| **AWQ/GPTQ/Marlin** | `--quantization awq` / `gptq` / `marlin` / `awq_marlin` / `gptq_marlin` | 4-bit 量化。Marlin kernel 高性能 |
| **BitsAndBytes** | `--quantization bitsandbytes` | 灵活的 4/8-bit 量化 |
| **NVIDIA ModelOpt** | `--quantization modelopt_fp8` / `modelopt_fp4` | Hopper/Blackwell NVIDIA 官方量化 |
| **TorchAO** | `--torchao-config fp8wo` / `int4wo-128` / `fp8dq-per_row` | PyTorch 原生量化方案 |
| **MoE 专用** | `--quantization moe_wna16` | MoE 模型 W4A16 量化 |
| **混合量化** | 不同层用不同精度 | 通过预量化模型实现 |

**建议**：
- Hopper/Ada → `--quantization fp8` 或预量化 FP8 模型
- A100/V100 → AWQ/GPTQ 预量化模型
- 追求最大压缩 → NVFP4 或 `--torchao-config int4wo-128`
- MoE 模型 → `--quantization moe_wna16`
- **预量化模型效果更好，优先于在线量化**

---

#### F. 注意力后端

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **MHA 后端** | `--attention-backend flashinfer/fa3/fa4/triton/trtllm_mha` | Ampere: FlashInfer 默认；Hopper: FA3 默认；Blackwell: TRTLLM MHA 默认 |
| **MLA 后端** | `--attention-backend flashinfer/flashmla/cutlass_mla/trtllm_mla/fa3/triton` | MLA 模型（DeepSeek）专用。FlashMLA page_size=64，CutlassMLA page_size=128 |
| **DSA 后端** | `--attention-backend dsa`<br>`--dsa-prefill-backend` / `--dsa-decode-backend` | DeepSeek V3.2 Sparse Attention。可分别配置 prefill 和 decode 的子后端 |
| **混合后端** | `--prefill-attention-backend fa4 --decode-attention-backend trtllm_mla` | 实验性：prefill 和 decode 用不同后端 |
| **FA3 Kernel** | `SGLANG_USE_SGL_FA3_KERNEL=true`（默认） | SGLang 自研 FA3 实现 |

**建议**：
- 通常使用默认 auto 选择即可
- DeepSeek V2/V3 → MLA 后端（默认自动选择）
- DeepSeek V3.2 → DSA 后端（`--attention-backend dsa`）
- Blackwell → 确认 TRTLLM 后端启用
- 性能敏感时可测试不同后端组合

---

#### G. 推测解码

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **EAGLE-3** | `--speculative-algorithm EAGLE3`<br>`--speculative-draft-model-path ...`<br>`--speculative-num-draft-tokens 5` | 推荐方案。H100 上可达 2.35x 加速（373 vs 158 tokens/s） |
| **EAGLE-2** | `--speculative-algorithm EAGLE` | 前代方案，兼容性好 |
| **MTP** | 使用 EAGLE workflow + 模型原生 MTP head | DeepSeek-V3 等原生 MTP 模型，无需额外模型 |
| **STANDALONE** | `--speculative-algorithm STANDALONE`<br>`--speculative-draft-model-path small_model` | 独立小模型作为 draft |
| **N-gram** | `--speculative-algorithm NGRAM`<br>`--speculative-num-draft-tokens 5` | 无需额外模型，重复性文本场景效果好 |
| **Spec V2** | `SGLANG_ENABLE_SPEC_V2=true` | 推测解码 V2，支持 overlap scheduler 集成 |

**建议**：
- 有对应 EAGLE-3 模型 → 优先 EAGLE-3（加速 2-3x）
- DeepSeek-V3 等原生 MTP → 使用 MTP（免费加速）
- 无对应模型但需加速 → N-gram（零成本尝试）
- 高 QPS 场景推测解码收益降低

---

#### H. RLHF/训练集成优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Memory Saver（Sleep/Wake）** | `--enable-memory-saver`<br>`POST /release_memory_occupation`<br>`POST /resume_memory_occupation` | RLHF colocate 必用。可选择释放 KV Cache 和/或权重 |
| **权重更新 — 从磁盘** | `POST /update_weights_from_disk` | 最简单。训练保存 checkpoint → SGLang 重载 |
| **权重更新 — 从 Tensor** | `POST /update_weights_from_tensor` | Co-located 场景最快。直接传 tensor，无需磁盘 IO |
| **权重更新 — 分布式** | `POST /init_weights_update_group`<br>`POST /update_weights_from_distributed`<br>`POST /destroy_weights_update_group` | Disaggregated 场景。通过 NCCL/IB 跨 GPU 传输 |
| **暂停/继续生成** | `POST /pause_generation`（mode: abort/retract/in_place）<br>`POST /continue_generation` | 异步 RL：暂停 → 更新权重 → 继续 |
| **确定性推理** | `--enable-deterministic-inference`<br>`--rl-on-policy-target fsdp` | 减少跨 batch 不确定性。可匹配特定训练系统（如 FSDP） |
| **CPU 权重备份** | `--enable-weights-cpu-backup` | 权重更新后保留 CPU 备份，加速恢复 |
| **R-Fork 快速启动** | `--load-format remote_instance`<br>`--remote-instance-weight-loader-backend nccl` | 从运行中的实例 GPU-to-GPU 传权重（秒级 vs 分钟级） |

**建议**：
- **GRPO/PPO + colocate**：`--enable-memory-saver` 必须。训练时 `release_memory_occupation`，推理时 `resume_memory_occupation`
- **权重同步首选**：colocate → `update_weights_from_tensor`；disaggregated → `update_weights_from_distributed`
- **确定性 RL**：`--enable-deterministic-inference` 确保 on-policy 训练对齐
- **多实例 RL**：用 R-Fork 秒级拉起新推理实例

---

#### I. 生产部署优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **SGLang Model Gateway** | `python -m sglang_router.launch_server --dp-size N --router-policy cache_aware` | **强烈推荐**。Rust 实现，cache-aware 路由提升 92% 吞吐 |
| **PD Disaggregation** | `--disaggregation-mode prefill/decode` + `sglang_router --pd-disaggregation` | Prefill 和 Decode 分离。独立扩缩容，降低互相干扰 |
| **LoRA 服务** | `--enable-lora --lora-paths name=path`<br>`--max-loras-per-batch 8`<br>`--enable-lora-overlap-loading` | 多 LoRA 并发，异步加载 |
| **结构化输出** | `--grammar-backend xgrammar`（默认）/ `outlines` / `llguidance` | JSON Schema、正则、grammar 约束解码 |
| **YAML 配置** | `python -m sglang.launch_server --config config.yaml` | 生产环境推荐 |
| **监控** | `--enable-metrics`（Prometheus）<br>`--enable-cache-report`<br>`--enable-mfu-metrics` | 40+ Prometheus 指标 |
| **Watchdog** | `--watchdog-timeout 300` | 挂起自动崩溃重启 |
| **HiSparse** | `--enable-hisparse --hisparse-config '{...}'` | DSA 模型（DeepSeek V3.2）：GPU 仅保留 top-k KV，大幅提升并发 |

**建议**：
- 生产多副本必须用 SGLang Model Gateway + cache-aware 路由
- 大规模服务 → PD Disaggregation（独立扩 prefill 实例）
- DeepSeek V3.2 → 启用 HiSparse 提升 decode 并发
- 生产环境开启 `--enable-metrics` + `--watchdog-timeout`

---

#### J. 环境与调优

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **调度保守度** | `--schedule-conservativeness 1.0`（默认） | token usage < 0.9 且有排队 → 降低到 0.3；频繁 retraction → 提高到 1.3 |
| **Docker 优化** | `SGLANG_SET_CPU_AFFINITY=1`<br>`SGLANG_MOE_PADDING=1` | Docker/K8s 环境必设 |
| **共享内存** | Docker: `--shm-size`<br>K8s: `/dev/shm` 挂载 | 多 GPU 通信需要足够共享内存 |
| **CPU 核心数** | 物理核心 ≥ 2 + TP + DP | CPU 不足是隐性瓶颈 |
| **Token Usage 监控** | 日志中 `token usage` 指标 | 目标 > 0.9，排队 100-2000 |
| **Max New Tokens 裁剪** | `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=4096` | 防止用户设过大 max_new_tokens 导致显存预留过多 |
| **DeepEP 参数** | `SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128` | EP 场景每 GPU 最大分发 token 数 |

**建议**：
- **关键调优**：观察 `token usage` 日志，目标 > 0.9
- Docker 环境 → `SGLANG_SET_CPU_AFFINITY=1` + `SGLANG_MOE_PADDING=1`
- 频繁 retraction → 提高 `--schedule-conservativeness`
- 用户 max_new_tokens 设置过大 → `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION`

---

### 第三步：场景化配置模板

根据使用模式和硬件条件，推荐最优配置组合：

#### 场景 1：独立推理服务 — 8B 模型 + 单卡 A100 80GB

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --port 30000 \
  --mem-fraction-static 0.88 \
  --context-length 8192
```

#### 场景 2：70B 模型 + 4xA100 — 高吞吐服务（SMG）

```bash
python -m sglang_router.launch_server \
  --model-path meta-llama/Llama-3.3-70B-Instruct \
  --tp 4 --dp-size 2 \
  --router-policy cache_aware \
  --mem-fraction-static 0.85 \
  --kv-cache-dtype fp8_e4m3 \
  --cuda-graph-max-bs 256
```

#### 场景 3：DeepSeek-V3 + 8xH100 — DP Attention + EP（推荐配置）

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3 \
  --tp 8 --dp-size 8 --ep 8 \
  --enable-dp-attention \
  --moe-a2a-backend deepep \
  --moe-runner-backend deep_gemm \
  --enable-two-batch-overlap \
  --mem-fraction-static 0.85 \
  --trust-remote-code
```

#### 场景 4：GRPO 训练 + SGLang Colocate（RL 场景）

```bash
# SGLang 推理引擎（colocate 模式，与训练共享 GPU）
python -m sglang.launch_server \
  --model-path model \
  --enable-memory-saver \
  --enable-weights-cpu-backup \
  --enable-deterministic-inference \
  --mem-fraction-static 0.4 \
  --port 30000
```

```python
# RL 训练循环中的 SGLang 集成
import requests

# 1. 推理：生成 rollout
responses = requests.post("http://localhost:30000/generate", json={...})

# 2. 训练前：释放推理引擎显存
requests.post("http://localhost:30000/release_memory_occupation",
              json={"tags": ["kv_cache", "weights"]})

# 3. 执行训练更新...

# 4. 训练后：更新权重并恢复
requests.post("http://localhost:30000/update_weights_from_tensor",
              json={"serialized_named_tensors": ..., "flush_cache": True})
requests.post("http://localhost:30000/resume_memory_occupation")
```

#### 场景 5：GRPO + SGLang Server 模式（分离 GPU）

```bash
# 独立 SGLang 推理服务
python -m sglang.launch_server \
  --model-path model \
  --tp 2 \
  --mem-fraction-static 0.92 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-deterministic-inference \
  --port 30000
```

```python
# RL 训练循环中通过分布式更新权重
import requests

# 初始化权重同步组
requests.post("http://localhost:30000/init_weights_update_group",
              json={"master_address": "...", "master_port": ..., "rank_offset": ..., "world_size": ...})

# 训练一轮后同步权重
requests.post("http://localhost:30000/update_weights_from_distributed",
              json={"flush_cache": True, "recapture_cuda_graph": False})
```

#### 场景 6：8B 模型 + 单卡 24GB（RTX 4090）— 量化推理

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --quantization fp8 \
  --mem-fraction-static 0.88 \
  --context-length 4096 \
  --port 30000
```

#### 场景 7：多轮对话 / RAG — 前缀缓存最大化

```bash
python -m sglang_router.launch_server \
  --model-path model \
  --dp-size 4 \
  --router-policy cache_aware \
  --schedule-policy lpm \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --port 30000
```

#### 场景 8：推测解码 — EAGLE-3 加速

```bash
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path path/to/eagle3-model \
  --speculative-num-draft-tokens 5 \
  --mem-fraction-static 0.88 \
  --port 30000
```

---

### 第四步：输出审计报告

按以下格式输出审计结果：

```markdown
# SGLang 推理优化审计报告

## 基本信息
- 模型：{model_path}（{参数量}）
- 模型架构：{Dense / MoE / MLA / DSA}
- 使用模式：{standalone_serve / rlhf_colocate / router_gateway / ...}
- 硬件：{GPU型号} x {数量}（{显存}GB）
- SGLang 版本：{如可检测}
- GPU 架构：{Ampere / Hopper / Blackwell}

## 审计结果总览

| 类别 | 得分 | 已启用 | 建议启用 |
|------|------|--------|----------|
| A. 引擎与计算优化 | x/6 | ... | ... |
| B. 显存管理 | x/6 | ... | ... |
| C. 前缀缓存 | x/5 | ... | ... |
| D. 并行策略 | x/6 | ... | ... |
| E. 量化策略 | x/5 | ... | ... |
| F. 注意力后端 | x/4 | ... | ... |
| G. 推测解码 | x/4 | ... | ... |
| H. RLHF/训练集成 | x/7 | ... | ... |
| I. 生产部署 | x/6 | ... | ... |
| J. 环境与调优 | x/5 | ... | ... |
| **总计** | **x/54** | | |

## 优先优化建议（按影响排序）

### P0 - 立即执行（显著收益，零风险）
1. ...

### P1 - 强烈推荐（明显收益，低风险）
1. ...

### P2 - 建议尝试（中等收益，需测试）
1. ...

## 推荐配置修改

（给出具体的启动命令/配置/代码修改）
```

---

## 常见优化决策树

```
开始
├── 显存不足（OOM）？
│   ├── 是 → 已量化？
│   │   ├── 否 → Hopper/Ada? → --quantization fp8
│   │   │   └── 其他 → 使用 AWQ/GPTQ 预量化模型
│   │   └── 是（已量化）
│   │       ├── 减小 --context-length（限制为实际需求）
│   │       ├── 减小 --chunked-prefill-size（4096/2048）
│   │       ├── 减小 --max-running-requests
│   │       ├── 减小 --cuda-graph-max-bs
│   │       ├── 降低 --mem-fraction-static（留更多给激活）
│   │       └── 增加 --tp-size（需更多 GPU）
│   └── 否 → 继续性能优化
├── 吞吐不足？
│   ├── token usage < 0.9 → 降低 --schedule-conservativeness 至 0.3
│   ├── 提高 --mem-fraction-static（增大 KV Cache 池）
│   ├── MLA 模型 → --enable-dp-attention（最高 1.9x decode 吞吐）
│   ├── MoE 模型 → --enable-two-batch-overlap（最高 2x 吞吐）
│   ├── 增大 --cuda-graph-max-bs（512-768）
│   ├── 多副本 → SGLang Model Gateway + cache_aware 路由
│   ├── 推测解码 → EAGLE-3 / MTP / N-gram
│   └── 增大 --num-continuous-decode-steps（2-4）
├── 延迟过高（TTFT）？
│   ├── 减小 --chunked-prefill-size
│   ├── --schedule-policy lpm（前缀缓存加速）
│   ├── PD Disaggregation（独立 prefill 实例）
│   └── 启用 HiCache（长上下文缓存命中）
├── 频繁 retraction / 抢占？
│   ├── 提高 --schedule-conservativeness 至 1.3-1.5
│   ├── 减小 --max-running-requests
│   └── 增大 KV Cache（提高 --mem-fraction-static）
├── DeepSeek 系列？
│   ├── --enable-dp-attention（MLA 必须）
│   ├── --ep-size + --moe-a2a-backend deepep + --moe-runner-backend deep_gemm
│   ├── --enable-two-batch-overlap
│   ├── V3.2 → --attention-backend dsa + --enable-hisparse
│   └── --kv-cache-dtype fp8_e4m3
└── RLHF 场景？
    ├── Colocate 模式
    │   ├── --enable-memory-saver（必须）
    │   ├── release/resume_memory_occupation 切换显存
    │   ├── update_weights_from_tensor（最快同步）
    │   └── --enable-deterministic-inference
    └── Server 模式
        ├── 独立 GPU 跑 SGLang
        ├── init_weights_update_group + update_weights_from_distributed
        └── 或 update_weights_from_disk（简单场景）
```

## 性能参考

### SGLang vs vLLM 优势场景

| 场景 | SGLang 优势 | 关键特性 |
|------|------------|---------|
| DeepSeek MLA 模型 | DP Attention 提升 decode 1.9x | `--enable-dp-attention` |
| 多轮对话/RAG | RadixAttention 自动前缀缓存 | 默认开启 + `--schedule-policy lpm` |
| MoE 高吞吐 | Two-Batch Overlap 提升 2x | `--enable-two-batch-overlap` |
| 大规模多副本 | SMG cache-aware 路由 +92% 吞吐 | `sglang_router` |
| RLHF 集成 | 三种权重更新策略 + 确定性推理 | `--enable-memory-saver` |
| 长上下文缓存 | HiCache 三层缓存 | `--enable-hierarchical-cache` |

### 单卡 A100 80GB 可服务模型规模

| 模型规模 | 量化方式 | context_length | 并发能力 |
|---------|---------|---------------|---------|
| 7-8B | BF16 | 32K | ~256 并发 |
| 7-8B | FP8 | 32K | ~384 并发 |
| 13-14B | BF16 | 16K | ~128 并发 |
| 30-34B | FP8 | 8K | ~64 并发 |
| 70B | 需 2+ GPU TP | - | - |

### RLHF 模式显存分配参考

| 模式 | SGLang mem_fraction_static | 训练侧可用 | 说明 |
|------|--------------------------|-----------|------|
| Colocate（交替） | 0.3~0.5 | release 时释放全部 | release/resume 切换 |
| Server（独立 GPU） | 0.85~0.92 | 100%（独立 GPU） | 资源隔离 |

## SGLang 特有关键词速查

用于识别 SGLang 项目的关键词和文件模式：

| 类别 | 关键词/模式 |
|------|------------|
| CLI | `python -m sglang.launch_server`、`python -m sglang_router.launch_server`、`sglang.launch_server` |
| 配置 | `mem_fraction_static`、`schedule_policy`、`chunked_prefill_size`、`enable_dp_attention`、`radix_cache`、`piecewise_cuda_graph`、`cuda_graph_max_bs`、`schedule_conservativeness` |
| import | `from sglang import`、`import sglang`、`from sglang.srt`、`from sglang.launch_server`、`from sglang_router` |
| 依赖 | `sglang`、`sgl-kernel`、`sglang-router` |
| RLHF 集成 | `enable_memory_saver`、`release_memory_occupation`、`resume_memory_occupation`、`update_weights_from_distributed`、`update_weights_from_tensor`、`pause_generation`、`continue_generation`、`infer_backend.*sglang`、`sglang_maxlen`、`sglang_mem_fraction` |
| 环境变量 | `SGLANG_ENABLE_TORCH_COMPILE`、`SGLANG_SET_CPU_AFFINITY`、`SGLANG_MOE_PADDING`、`SGLANG_ENABLE_JIT_DEEPGEMM`、`SGLANG_CHUNKED_PREFIX_CACHE_THRESHOLD`、`SGLANG_ENABLE_SPEC_V2` |
