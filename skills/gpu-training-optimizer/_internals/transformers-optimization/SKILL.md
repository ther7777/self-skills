---
name: transformers-optimization
description: HuggingFace Transformers Trainer 训练优化审计
user-invocable: false
---

## 描述
针对直接使用 HuggingFace Transformers `Trainer` + `TrainingArguments` 的训练项目，快速识别已采用和未采用的优化手段，提供针对性的显存优化与性能加速建议。适用于使用 Transformers 原生训练流程（非 LlamaFactory / ms-swift 等上层封装框架）的项目。

## 触发条件
当识别到用户项目满足以下条件时触发：
- 代码中 `from transformers import Trainer` / `from transformers import TrainingArguments` / `from transformers import Seq2SeqTrainer`
- 代码中使用 `Trainer(model=..., args=...).train()`
- 依赖文件中包含 `transformers`（且不匹配 LlamaFactory / ms-swift / VideoX-Fun 等上层框架）
- 配置文件或代码中包含 `TrainingArguments` 特有参数：`per_device_train_batch_size`、`gradient_accumulation_steps`、`fp16`/`bf16`、`torch_compile`、`use_liger_kernel`、`fsdp`、`deepspeed`

**注意**：此 skill 仅在未检测到 LlamaFactory / ms-swift / VideoX-Fun 等上层框架时触发。上层框架有自己的专项 skill，能提供更精准的优化建议。

## 执行指令

你是 HuggingFace Transformers 训练优化专家。被调用时，请按照以下清单对目标项目进行全面审计。对每个检查项，判定其状态：**已启用** / **未启用** / **建议启用** / **不适用**，并给出具体的配置修改建议。

### 第一步：识别训练模式与基础信息

扫描项目的 Python 训练脚本和配置文件，确定：

1. **训练任务类型**：因果语言模型（CausalLM）/ 序列分类 / Token 分类 / 问答 / Seq2Seq / 自定义任务
2. **Trainer 类型**：`Trainer` / `Seq2SeqTrainer` / 自定义 `Trainer` 子类
3. **模型与规模**：`model_name_or_path` 或 `AutoModelFor*` 调用，识别模型系列和参数量
4. **PEFT 使用**：是否使用 `peft` 库的 LoRA / QLoRA 等适配器
5. **硬件环境**：GPU 型号/数量、显存大小（结合 `/system-resources` 结果）
6. **数据规模**：数据集大小、`max_length` / `max_seq_length`

---

### 第二步：显存优化审计

#### A. 混合精度训练

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **BF16 混合精度** | `TrainingArguments(bf16=True)` | Ampere+(A100/H100/H20) 首选。无需 loss scaling，数值稳定 |
| **FP16 混合精度** | `TrainingArguments(fp16=True)` | V100/T4 使用。需配合 loss scaling |
| **TF32 模式** | `TrainingArguments(tf32=True)` | Ampere+ GPU，matmul 最高 8x 加速。`torch_compile=True` 时自动启用 |
| **BF16 全量 Eval** | `TrainingArguments(bf16_full_eval=True)` | Eval 阶段全 BF16，节省显存加速推理 |
| **FP16 全量 Eval** | `TrainingArguments(fp16_full_eval=True)` | Eval 阶段全 FP16 |

**评分 /5**

**建议**：
- A100/H100/H20 必须 `bf16=True`
- V100/T4 使用 `fp16=True`
- Ampere+ GPU 建议同时 `tf32=True`
- `bf16` 和 `fp16` 互斥，不可同时启用

---

#### B. 梯度检查点与显存管理

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **梯度检查点** | `TrainingArguments(gradient_checkpointing=True)` | 以 ~20% 计算换 ~60% 激活显存。长序列/大 batch 必开 |
| **GC Kwargs** | `gradient_checkpointing_kwargs={"use_reentrant": False}` | `use_reentrant=False` 兼容性更好，**推荐**。FSDP 场景强制 False |
| **显存定期清理** | `TrainingArguments(torch_empty_cache_steps=100)` | 定期调用 `torch.cuda.empty_cache()`，~10% 性能换 OOM 缓解 |
| **自动 Batch Size** | `TrainingArguments(auto_find_batch_size=True)` | OOM 时自动减小 batch size 重试。不兼容 DeepSpeed ZeRO-3 |
| **Eval 累积** | `TrainingArguments(eval_accumulation_steps=N)` | 评估时在 GPU 上累积 N 步后再移到 CPU，防止 eval OOM |

**评分 /5**

**建议**：
- 序列长度 >2048 或 batch size >4 时务必启用 `gradient_checkpointing=True`
- 建议设置 `gradient_checkpointing_kwargs={"use_reentrant": False}`
- 显存极紧张时设置 `torch_empty_cache_steps`（如每 100 步）
- 不确定最大 batch size 时启用 `auto_find_batch_size=True`

---

#### C. PEFT 与量化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **LoRA** | `peft.LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"])` | 参数高效微调，7B 模型 ~16GB。`target_modules="all-linear"` 覆盖更广 |
| **QLoRA 4-bit** | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)` | 7B→~6GB。NF4 + double_quant 最优 |
| **QLoRA 8-bit** | `BitsAndBytesConfig(load_in_8bit=True)` | LLM.int8()，7B→~10GB |
| **GPTQ** | `GPTQConfig(bits=4, group_size=128)` | 训练后量化，需校准数据集 |
| **AWQ** | `AwqConfig(bits=4)` | 激活感知量化 |
| **TorchAO** | `TorchAoConfig(quant_type=...)` | PyTorch 原生量化，兼容 torch.compile |
| **FP8** | `FbgemmFp8Config()` / `FineGrainedFP8Config()` | Hopper+ GPU，~50% 显存节省 |
| **DoRA** | `LoraConfig(use_dora=True)` | 权重分解 LoRA，效果更好但略增显存 |

**评分 /8**

**建议**：
- 显存紧张首选 QLoRA 4bit（`load_in_4bit=True` + `bnb_4bit_quant_type="nf4"` + `bnb_4bit_use_double_quant=True`）
- 效果优先选 LoRA rank=64~128 + `use_dora=True`
- PEFT 适配器热交换：`model.load_adapter(path, hotswap=True)` 避免内存累积
- H100/H200 可尝试 FP8 量化

---

#### D. 优化器选择

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Fused AdamW** | `optim="adamw_torch_fused"` | PyTorch >= 2.8 默认。比标准 AdamW 更快（融合 CUDA kernel） |
| **8-bit AdamW** | `optim="adamw_bnb_8bit"` | 优化器状态显存 -75%，需 bitsandbytes |
| **4-bit AdamW** | `optim="adamw_torch_4bit"` | 优化器状态显存 -87%（PyTorch >= 2.3） |
| **Paged AdamW** | `optim="paged_adamw_8bit"` | OOM 时自动将优化器状态卸载到 CPU |
| **Adafactor** | `optim="adafactor"` | 存储行/列均值代替完整状态，大幅降低优化器显存 |
| **GaLore** | `optim="galore_adamw"` / `"galore_adamw_layerwise"` | 梯度低秩投影，全参训练显存接近 LoRA 水平 |
| **APOLLO** | `optim="apollo_adamw"` / `"apollo_adamw_layerwise"` | 自适应梯度缩放，显存高效 |
| **LOMO** | `optim="lomo"` / `"adalomo"` | 近零优化器显存，适合全参微调大模型 |
| **Schedule-Free** | `optim="schedule_free_adamw"` | 无需 LR scheduler，自适应学习率 |

**评分 /7**

**建议**：
- 默认使用 `adamw_torch_fused`（PyTorch 2.8+ 自动启用）
- 显存紧张 → `adamw_bnb_8bit`（优化器状态 -75%）
- 全参训练内存受限 → `galore_adamw_layerwise`（全参效果 + LoRA 级别显存）
- 极端受限 → `lomo`（近零优化器显存，仅全参微调）

---

#### E. 分布式显存优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **DDP 基本设置** | `ddp_backend="nccl"`<br>`ddp_find_unused_parameters=False`<br>`ddp_bucket_cap_mb=25` | 多卡基本分布式。GC 开启时自动 `find_unused_parameters=False` |
| **FSDP Full Shard** | `fsdp="full_shard auto_wrap"` | 分片参数+梯度+优化器（ZeRO-3 等效），最大显存节省 |
| **FSDP Shard Grad Op** | `fsdp="shard_grad_op auto_wrap"` | 分片梯度+优化器（ZeRO-2 等效），更快 |
| **FSDP Hybrid Shard** | `fsdp="hybrid_shard auto_wrap"` | 节点内全分片，跨节点复制。多机首选 |
| **FSDP CPU Offload** | `fsdp="full_shard offload auto_wrap"` | CPU 卸载，显存极不足时使用 |
| **FSDP2** | `fsdp_config={"fsdp_version": 2}` | FSDP v2，Per-parameter sharding，支持 DTensor |
| **FSDP 激活检查点** | `fsdp_config={"activation_checkpointing": True}` | FSDP 原生 GC，优于 `gradient_checkpointing`（避免冗余 AllGather） |
| **FSDP 预取** | `fsdp_config={"backward_prefetch": "backward_pre", "forward_prefetch": True}` | 预取下一分片，隐藏通信延迟 |
| **FSDP 高效加载** | `fsdp_config={"cpu_ram_efficient_loading": True, "sync_module_states": True}` | 仅 rank 0 加载权重，广播到其他 rank |
| **DeepSpeed ZeRO-2** | `deepspeed="ds_z2_config.json"` | 分片梯度+优化器。LoRA 多卡首选 |
| **DeepSpeed ZeRO-3** | `deepspeed="ds_z3_config.json"` | 分片参数+梯度+优化器。全参大模型必用 |
| **DeepSpeed CPU Offload** | ZeRO-3 + `offload_optimizer.device: "cpu"` | 优化器卸载到 CPU |
| **Parallelism Config** | `parallelism_config=ParallelismConfig(tp_size=N, cp_size=N, sp_size=N)` | 张量并行/上下文并行/序列并行（Accelerate 1.10.1+） |

**评分 /10**

**建议**：
- LoRA 多卡 → DeepSpeed ZeRO-2 或 FSDP `shard_grad_op`
- 全参微调 → DeepSpeed ZeRO-3 或 FSDP `full_shard`
- 多机训练 → FSDP `hybrid_shard`（减少跨节点通信）
- 显存极不足 → DeepSpeed ZeRO-3 + CPU Offload
- FSDP 场景用 `activation_checkpointing` 而非 `gradient_checkpointing`（避免冗余 AllGather）
- 启用 `cpu_ram_efficient_loading=True` 节省模型加载时 CPU 内存

---

### 第三步：计算性能优化审计

#### F. 注意力机制优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **SDPA** | `model = AutoModel.from_pretrained(..., attn_implementation="sdpa")` | PyTorch 2.1+ 默认。自动调度 Flash/Memory-Efficient/Math kernel |
| **Flash Attention 2** | `attn_implementation="flash_attention_2"` | **高优先级**。需安装 `flash-attn`。显存 O(N²)→O(N)，速度 2-4x |
| **Flash Attention 3** | `attn_implementation="flash_attention_3"` | Hopper GPU 专属 |
| **Flex Attention** | `attn_implementation="flex_attention"` | PyTorch 2.5+，支持自定义注意力模式（block-sparse 等） |
| **注意力动态切换** | `model.set_attention_implementation("flash_attention_2")` | 加载后动态切换注意力实现 |

**评分 /5**

**建议**：
- Ampere+ GPU 必须设置 `attn_implementation="flash_attention_2"`
- 安装 `flash-attn` 失败时退回 `attn_implementation="sdpa"`（PyTorch 2.1+ 默认）
- Hopper GPU 可尝试 `flash_attention_3`

---

#### G. 计算加速引擎

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **torch.compile** | `TrainingArguments(torch_compile=True)` | **高优先级**。PyTorch 2.0+，吞吐 +20~50%，零代码修改 |
| **compile 后端** | `torch_compile_backend="inductor"` | `inductor`（默认推荐）/ `cudagraphs` / `aot_eager` |
| **compile 模式** | `torch_compile_mode="max-autotune"` | `default` / `reduce-overhead` / `max-autotune`（最激进，编译慢但运行快） |
| **Liger Kernel** | `TrainingArguments(use_liger_kernel=True)` | **高优先级**。Triton 融合算子，吞吐 +20%，显存 -60% |
| **Liger 细粒度** | `liger_kernel_config={"rope": True, "swiglu": True, "cross_entropy": True, "fused_linear_cross_entropy": True, "rms_norm": True}` | 按需选择融合算子 |

**评分 /5**

**建议**：
- 全参训练 → `torch_compile=True` + `use_liger_kernel=True`（可同时使用）
- `max-autotune` 模式编译较慢但运行最快，适合长训练任务
- 设置 `torch_compile_backend` 或 `torch_compile_mode` 会自动启用 `torch_compile`
- Liger Kernel 支持 Llama/Mistral/Gemma/Mixtral 等模型

---

#### H. 数据处理优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **DataLoader Workers** | `TrainingArguments(dataloader_num_workers=4)` | 多进程数据加载。默认 0（主进程加载）。建议 4~8 |
| **Pin Memory** | `dataloader_pin_memory=True` | 默认开启。加速 CPU→GPU 数据传输 |
| **Persistent Workers** | `dataloader_persistent_workers=True` | 跨 epoch 保持 worker，避免重启开销 |
| **Prefetch Factor** | `dataloader_prefetch_factor=2` | 每 worker 预取 batch 数。需 `num_workers>0` |
| **Group By Length** | `train_sampling_strategy="group_by_length"` | 按序列长度分组，最小化 padding 浪费 |
| **Length Column** | `length_column_name="length"` | 预计算长度列，加速 `group_by_length` 排序 |
| **移除未用列** | `remove_unused_columns=True` | 默认开启。减少数据传输开销 |

**评分 /7**

**建议**：
- 必须设置 `dataloader_num_workers=4`（或更高）+ `dataloader_persistent_workers=True`
- 变长序列场景启用 `train_sampling_strategy="group_by_length"` 减少 padding
- 设置 `dataloader_prefetch_factor=2` 预取数据
- `accelerator_config={"non_blocking": True}` 启用异步数据传输

---

#### I. 训练策略优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **NEFTune** | `neftune_noise_alpha=5.0` | 嵌入层注入噪声，显著提升 SFT 效果。推荐范围 5~15 |
| **梯度累积** | `gradient_accumulation_steps=N` | 等效增大 batch size，不增加显存。有效 batch = per_device × devices × accum |
| **梯度裁剪** | `max_grad_norm=1.0` | 默认开启。设 0 禁用 |
| **Label Smoothing** | `label_smoothing_factor=0.1` | 软标签防止过拟合，典型 0.0~0.1 |
| **仅保存模型** | `save_only_model=True` | 跳过保存优化器/调度器状态，大幅减小 checkpoint 体积 |
| **JIT Checkpoint** | `enable_jit_checkpoint=True` | SIGTERM 信号时立即保存，适合抢占式集群 |

**评分 /5**

**建议**：
- SFT 训练建议 `neftune_noise_alpha=5`（显著提升效果，零性能成本）
- 显存不足时增大 `gradient_accumulation_steps` 等效增大 batch
- 磁盘空间紧张或 checkpoint 过大时启用 `save_only_model=True`

---

#### J. 推理优化（如适用）

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Static KV Cache** | `model.generation_config.cache_implementation="static"` | 固定大小缓存，支持 torch.compile，推理加速 2-4x |
| **Quantized KV Cache** | `cache_implementation="quantized"`<br>`cache_config={"nbits": 4, "backend": "quanto"}` | KV Cache 量化到 4/2-bit，支持超长序列 |
| **Offloaded KV Cache** | `cache_implementation="offloaded"` | KV Cache 卸载到 CPU，序列超长时使用 |
| **推测解码** | `model.generate(assistant_model=draft_model)` | 小模型草稿 + 大模型验证，推理速度 2-3x |
| **Prompt Lookup** | `model.generate(prompt_lookup_num_tokens=3)` | N-gram 匹配推测，无需额外模型 |
| **torch.compile 推理** | `model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)` | 配合 Static Cache，推理加速最高 4x |
| **Prefill Chunking** | `generation_config.prefill_chunk_size=2048` | 分块预填充，降低长 prompt 峰值显存 |
| **推理量化 (INT4/INT8)** | `AutoModelForCausalLM.from_pretrained(..., quantization_config=AwqConfig(...))`<br>`AutoGPTQForCausalLM.from_quantized(...)` | AWQ (INT4, 激活感知) 和 GPTQ (INT4, 梯度感知) 在显存减少 3-4x 的同时精度损失极小。AWQ 推理更快（kernel 更优），GPTQ 量化精度略高 |
| **FP8 推理量化** | `FbgemmFp8Config()`、`FineGrainedFP8Config()` | H100+ GPU 可用 FP8 量化 GEMM，接近 BF16 精度但吞吐提升 ~2x。无需校准（post-training 即可） |
| **推理引擎集成** | `vllm.LLM(model=...)`、`tensorrt_llm.Builder(...)`<br>`onnxruntime.InferenceSession(...)` | 生产推理优先使用成熟引擎。vLLM（PagedAttention + Continuous Batching）、TensorRT-LLM（图优化 + kernel 融合）、ONNX Runtime（跨平台 + 图优化）通常比裸 HF generate 快 3-10x |
| **Continuous Batching** | vLLM/TensorRT-LLM/SGLang 默认支持 | 动态迭代级调度，新请求不等当前 batch 完成就可插入。静态 batch 中短序列完成后 GPU 空闲等长序列，浪费算力。HF Transformers 原生 generate 不支持 |
| **Prefill/Decode 分离** | vLLM `chunked_prefill`、TRT-LLM `paged_kv_cache` | Prefill 是 compute-bound（大矩阵乘），Decode 是 memory-bound（逐 token 取 KV Cache）。两者用不同 kernel 策略。Chunked prefill 避免长 prompt 阻塞 decode 请求 |
| **Tensor Parallel 推理** | `tensor_parallel_size=N`（vLLM）<br>TP mapping（TRT-LLM） | 大模型推理跨多卡部署时，TP 比 PP 延迟低（无 pipeline bubble）。优先用 NVLink 连接的卡做 TP |

**评分 /12**

**建议**：
- 推理场景必须启用 Static KV Cache + torch.compile
- 长序列推理启用 Quantized KV Cache（4-bit）
- 批量推理场景使用推测解码或 Prompt Lookup
- 超长 prompt 使用 `prefill_chunk_size` 分块预填充
- 生产部署优先使用 vLLM/TensorRT-LLM/SGLang 推理引擎
- INT4 量化（AWQ/GPTQ）可在显存减少 75% 的同时保持接近原始精度
- H100+ 优先用 FP8，无需校准且精度损失极小
- 多卡推理优先 Tensor Parallel，延迟比 Pipeline Parallel 更低

#### 推理场景配置模板：7B 模型高吞吐服务

```python
# 方案 A：HF Transformers 原生推理（简单场景）
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
model.generation_config.cache_implementation = "static"
model.generation_config.max_new_tokens = 256

# 方案 B：vLLM（生产推荐）
from vllm import LLM, SamplingParams

llm = LLM(
    model="model_path",
    dtype="bfloat16",
    tensor_parallel_size=1,        # 单卡
    gpu_memory_utilization=0.9,    # KV Cache 使用 90% 空闲显存
    max_model_len=4096,
    # enforce_eager=False,         # 默认开启 CUDA Graph
)
params = SamplingParams(temperature=0.7, max_tokens=256)

# 方案 C：AWQ INT4 量化推理（显存受限）
from transformers import AutoModelForCausalLM, AwqConfig

quantization_config = AwqConfig(bits=4, fuse_max_seq_len=4096)
model = AutoModelForCausalLM.from_pretrained(
    "model_path_awq",
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
)
```

---

### 第四步：场景化配置模板

#### 场景 1：7B 模型 + 单卡 A100/H100 80GB — 全参 SFT

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="output/7b-full-sft",
    # 混合精度
    bf16=True,
    tf32=True,
    # 计算加速
    torch_compile=True,
    torch_compile_mode="max-autotune",
    use_liger_kernel=True,
    # 显存优化
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    # 数据加载
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    dataloader_prefetch_factor=2,
    train_sampling_strategy="group_by_length",
    # 训练策略
    neftune_noise_alpha=5,
    learning_rate=2e-5,
    num_train_epochs=3,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
)

# 模型加载
model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

#### 场景 2：7B 模型 + 单卡 24GB — QLoRA SFT

```python
from transformers import TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# 4bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "model_path",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",
)

# LoRA
lora_config = LoraConfig(
    r=32, lora_alpha=64,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="output/7b-qlora",
    bf16=True,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="paged_adamw_8bit",           # Paged 8-bit，OOM 自动卸载
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
    neftune_noise_alpha=5,
)
```

#### 场景 3：7B 模型 + 8xA100 80GB — 全参 SFT + FSDP

```python
training_args = TrainingArguments(
    output_dir="output/7b-full-fsdp",
    bf16=True,
    # FSDP
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_version": 2,
        "backward_prefetch": "backward_pre",
        "forward_prefetch": True,
        "activation_checkpointing": True,       # FSDP 原生 GC
        "cpu_ram_efficient_loading": True,
        "sync_module_states": True,
    },
    # 计算加速
    torch_compile=True,
    use_liger_kernel=True,
    # 训练配置
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    optim="adamw_torch_fused",
    learning_rate=2e-5,
    num_train_epochs=3,
    # 数据
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
    train_sampling_strategy="group_by_length",
    neftune_noise_alpha=5,
)

# 启动: torchrun --nproc_per_node=8 train.py
```

#### 场景 4：70B 模型 + 8xA100 80GB — LoRA + DeepSpeed ZeRO-2

```python
training_args = TrainingArguments(
    output_dir="output/70b-lora-ds",
    bf16=True,
    deepspeed="ds_z2_config.json",
    # 计算加速
    use_liger_kernel=True,
    # 训练配置
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    learning_rate=5e-5,
    num_train_epochs=3,
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
    save_only_model=True,
)

# ds_z2_config.json
# {
#   "zero_optimization": {"stage": 2, "overlap_comm": true},
#   "bf16": {"enabled": "auto"},
#   "train_batch_size": "auto",
#   "train_micro_batch_size_per_gpu": "auto",
#   "gradient_accumulation_steps": "auto"
# }
```

#### 场景 5：70B 全参 + 8xA100 — DeepSpeed ZeRO-3

```python
training_args = TrainingArguments(
    output_dir="output/70b-full-ds",
    bf16=True,
    deepspeed="ds_z3_config.json",
    torch_compile=True,
    use_liger_kernel=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch_fused",
    learning_rate=1e-5,
    num_train_epochs=3,
    dataloader_num_workers=4,
    dataloader_persistent_workers=True,
    save_only_model=True,
)

# ds_z3_config.json
# {
#   "zero_optimization": {
#     "stage": 3,
#     "overlap_comm": true,
#     "contiguous_gradients": true,
#     "stage3_gather_16bit_weights_on_model_save": true
#   },
#   "bf16": {"enabled": "auto"},
#   "train_batch_size": "auto",
#   "train_micro_batch_size_per_gpu": "auto",
#   "gradient_accumulation_steps": "auto",
#   "gradient_clipping": "auto"
# }
```

---

### 第五步：输出审计报告

按以下格式输出审计结果：

```markdown
# HuggingFace Transformers Trainer 训练优化审计报告

## 基本信息
- 模型：{model_name_or_path}（{参数量}）
- Trainer 类型：{Trainer / Seq2SeqTrainer / 自定义}
- 训练任务：{CausalLM / 分类 / Seq2Seq / ...}
- PEFT：{LoRA / QLoRA / 无}
- 硬件：{GPU型号} x {数量}（{显存}GB）
- 序列长度：{max_length}

## 审计结果总览

| 类别 | 得分 | 已启用 | 建议启用 |
|------|------|--------|----------|
| A. 混合精度 | x/5 | ... | ... |
| B. 梯度检查点与显存 | x/5 | ... | ... |
| C. PEFT 与量化 | x/8 | ... | ... |
| D. 优化器 | x/7 | ... | ... |
| E. 分布式优化 | x/10 | ... | ... |
| F. 注意力优化 | x/5 | ... | ... |
| G. 计算加速 | x/5 | ... | ... |
| H. 数据处理 | x/7 | ... | ... |
| I. 训练策略 | x/5 | ... | ... |
| J. 推理优化 | x/12 | ... | ... |
| **总计** | **x/69** | | |

## 优先优化建议（按影响排序）

### P0 — 立即执行（显著收益，零风险）
1. ...

### P1 — 强烈推荐（明显收益，低风险）
1. ...

### P2 — 建议尝试（中等收益，需测试）
1. ...

## 推荐配置修改

（给出具体的 TrainingArguments 参数修改和模型加载参数修改）
```

---

## 常见优化决策树

```
开始
├── 显存不足（OOM）？
│   ├── 是 → 当前用全参训练？
│   │   ├── 是 → 切换为 LoRA（peft）
│   │   │   └── 仍然 OOM？ → QLoRA 4-bit（BitsAndBytesConfig）
│   │   └── 否（已用 LoRA/QLoRA）
│   │       ├── 确认 gradient_checkpointing 已开启
│   │       ├── 减小 batch size + 增大 gradient_accumulation_steps
│   │       ├── 切换 8-bit 优化器（optim="adamw_bnb_8bit"）
│   │       ├── 尝试 auto_find_batch_size=True
│   │       └── 仍然 OOM？ → torch_empty_cache_steps / 降低 max_length / DeepSpeed Offload
│   └── 否 → 继续性能优化
├── 训练速度慢？
│   ├── attn_implementation 已设 flash_attention_2？
│   │   ├── 否 → 设置 attn_implementation="flash_attention_2"
│   │   └── 是 → 继续
│   ├── torch_compile 已启用？
│   │   ├── 否 → 设置 torch_compile=True
│   │   └── 是 → 继续
│   ├── use_liger_kernel 已启用？
│   │   ├── 否 → 设置 use_liger_kernel=True
│   │   └── 是 → 继续
│   ├── bf16/tf32 已启用？
│   │   ├── 否 → 设置 bf16=True, tf32=True
│   │   └── 是 → 继续
│   ├── 数据加载是瓶颈？
│   │   ├── 是 → dataloader_num_workers=4~8, persistent_workers=True, prefetch_factor=2
│   │   └── 否 → 继续
│   ├── 变长序列 padding 浪费？
│   │   ├── 是 → train_sampling_strategy="group_by_length"
│   │   └── 否 → 继续
│   ├── 多卡训练？
│   │   ├── LoRA → DeepSpeed ZeRO-2 或 FSDP shard_grad_op
│   │   └── 全参 → DeepSpeed ZeRO-3 或 FSDP full_shard
│   └── 优化器慢？ → optim="adamw_torch_fused"
└── 训练效果差？
    ├── 启用 neftune_noise_alpha=5
    ├── 调整 LoRA rank（增大到 64~128）+ use_dora=True
    ├── 使用 label_smoothing_factor=0.1
    └── 调整 lr_scheduler_type="cosine"
```

## 显存估算参考

| 方法 | 精度 | 7B | 14B | 30B | 70B |
|------|------|-----|-----|-----|------|
| 全参（bf16 混合） | 混合精度 | 120GB | 240GB | 600GB | 1200GB |
| LoRA | 16-bit | 16GB | 32GB | 64GB | 160GB |
| QLoRA | 8-bit | 10GB | 20GB | 40GB | 80GB |
| QLoRA | 4-bit | 6GB | 12GB | 24GB | 48GB |
| GaLore（全参） | 16-bit | ~16GB | ~32GB | ~64GB | ~160GB |
| LOMO（全参） | 16-bit | ~16GB | ~30GB | ~60GB | ~140GB |

## Transformers Trainer 特有关键词速查

用于识别直接使用 HuggingFace Transformers Trainer 的项目：

| 类别 | 关键词/模式 |
|------|------------|
| import | `from transformers import Trainer`、`from transformers import TrainingArguments`、`from transformers import Seq2SeqTrainer`、`from transformers import Seq2SeqTrainingArguments` |
| 实例化 | `Trainer(model=`、`TrainingArguments(`、`trainer.train()` |
| 模型加载 | `AutoModelForCausalLM.from_pretrained`、`AutoModelForSequenceClassification.from_pretrained`、`attn_implementation=` |
| PEFT | `from peft import`、`LoraConfig`、`get_peft_model`、`PeftModel` |
| 量化 | `BitsAndBytesConfig`、`GPTQConfig`、`AwqConfig`、`load_in_4bit`、`load_in_8bit` |
| 分布式 | `deepspeed=`（指向 JSON 配置文件路径）、`fsdp=`、`fsdp_config=` |
| 启动方式 | `torchrun`、`accelerate launch`、`deepspeed` |
| 依赖 | `transformers`、`accelerate`、`peft`、`bitsandbytes`、`flash-attn`、`liger-kernel` |
