---
name: training-acceleration-audit
description: LLM/MLLM 训练加速审计，扫描项目并输出评分报告
user-invocable: false
---

## 描述
分析用户提供的 LLM/MLLM 训练项目，判断哪些加速工具和策略已在使用、哪些尚未采用，并提供可落地的优化建议。

## 触发条件
当用户要求分析、审计或评估其项目的训练加速能力时触发，或者当用户提供一个训练代码库并希望了解如何提速时触发。

## 执行指令

你是 LLM/MLLM 分布式训练优化专家。被调用时，请按照以下清单对目标项目进行全面审计。对每个检查项，判定其状态：**已使用** / **部分使用** / **未使用** / **不适用**。

### 第零步：瓶颈类型预判

**在逐项审计之前，先判断项目主要受哪类瓶颈约束。** 这是 GPU 优化的首要原则——不分类就优化，很容易"做了很多优化，结果没收益"。

根据项目代码、配置、启动命令和已有的 Profiling 数据（如有），将主要瓶颈归类到以下类型：

| 瓶颈类型 | 典型表现 | 诊断信号 | 优化方向 |
|----------|---------|---------|---------|
| **Compute-bound** | SM 利用率高但吞吐未达理论峰值 | GPU 利用率 > 80%、Tensor Core 未命中、FP32 计算 | 混合精度、Tensor Core 对齐、算子融合、torch.compile |
| **Memory-bound** | 显存带宽成瓶颈，算力吃不满 | 大量 element-wise 操作、频繁显存分配/释放、HBM 带宽饱和 | 算子融合减少中间张量、Flash Attention、Liger Kernel、内存复用 |
| **Latency-bound** | Kernel 粒度小、launch 开销占比大 | 大量 < 10μs 的小 kernel、CPU-GPU 同步点多 | CUDA Graph、torch.compile、persistent kernel、减少同步 |
| **Launch-bound** | Kernel 数量多但每个都很短 | nsys 时间线中 kernel 间有明显 gap | 算子融合、CUDA Graph、减少 Python 层调用开销 |
| **Communication-bound** | 多卡通信等待时间长 | AllReduce/AllGather 占比 > 30%、GPU 空闲等通信完成 | 通信计算重叠、梯度压缩、ZeRO++、NCCL 调优、更高效并行策略 |
| **Data I/O-bound** | 数据加载跟不上 GPU 消耗速度 | DataLoader 占比 > 20%、GPU 周期性空闲 | num_workers、prefetch、pin_memory、数据格式优化、本地 SSD |

**操作**：在审计报告的"项目概览"中标注初步判断的瓶颈类型，并在优化建议中优先推荐针对该类瓶颈的优化项。如无 Profiling 数据，可根据代码特征（如模型规模、序列长度、数据来源）做初步推断。

### 第一步：扫描项目结构

在目标项目中搜索配置文件、训练脚本和依赖声明：

1. **依赖文件**：`requirements.txt`、`setup.py`、`pyproject.toml`、`environment.yml`、`Dockerfile`
2. **训练入口**：包含 `Trainer`、`train`、`main`、`run_`、`pretrain`、`finetune` 的文件
3. **配置文件**：`*.yaml`、`*.yml`、`*.json`、`deepspeed_config*`、`megatron*`、`accelerate*`
4. **启动脚本**：`*.sh`、`Makefile`、包含 `torchrun`、`deepspeed`、`accelerate launch` 的文件

### 第二步：逐项检查各加速类别

扫描代码和配置，查找以下能力的使用证据：

---

#### A. 并行策略

| 检查项 | 搜索关键词 / 模式 | 自查要点 |
|---|---|---|
| **数据并行 (DDP)** | `DistributedDataParallel`、`torch.nn.parallel`、`torchrun`、`nproc_per_node` | 多卡训练的基础，确认是否正确使用 DDP 且所有 GPU 均参与训练 |
| **全分片数据并行 (FSDP)** | `FullyShardedDataParallel`、`fsdp`、`ShardingStrategy`、`FSDP` | 模型参数量超过单卡显存时，应启用 FSDP 分片参数和优化器状态 |
| **DeepSpeed ZeRO** | `deepspeed`、`zero_optimization`、`stage`、`ZeRO` | 与 FSDP 类似，通过 ZeRO Stage 1/2/3 逐级减少显存冗余 |
| **张量并行 (TP)** | `tensor_model_parallel`、`megatron`、`ColumnParallelLinear`、`RowParallelLinear`、`tp_size` | 超大模型单层参数超过单卡显存时需要 TP 切分 |
| **流水线并行 (PP)** | `pipeline_model_parallel`、`PipelineModule`、`num_stages`、`pp_size` | 模型层数极多时可通过 PP 分段放置，注意 bubble 开销 |
| **序列并行 (SP)** | `sequence_parallel`、`ring_attention`、`ulysses`、`sp_size` | 长序列场景下可切分序列维度降低单卡 Attention 显存 |
| **专家并行** | `expert_parallel`、`moe`、`num_experts`、`MixtureOfExperts`、`ep_size` | MoE 模型应将 Expert 分布到不同 GPU 以均衡负载 |

---

#### B. 训练框架

| 检查项 | 搜索关键词 / 模式 | 自查要点 |
|---|---|---|
| **DeepSpeed** | `deepspeed`、`ds_config`、`DeepSpeedEngine` | 生态最全，支持 ZeRO、Offload、MoE 等 |
| **Megatron-LM** | `megatron`、`mcore`、`megatron.core` | 大规模预训练首选，支持 3D 并行 |
| **HuggingFace Accelerate** | `accelerate`、`Accelerator`、`accelerate launch` | 易用性好，可快速集成 FSDP/DeepSpeed |
| **ColossalAI** | `colossalai`、`ColosalAI`、`Gemini` | 异构内存管理，多种并行策略一键配置 |
| **PyTorch FSDP** | `FullyShardedDataParallel`、`torch.distributed.fsdp` | PyTorch 原生集成，无额外依赖 |
| **veScale** | `vescale`、`DTensor` | 基于 DTensor 的自动并行框架 |

---

#### C. 显存优化

| 检查项 | 搜索关键词 / 模式 | 自查要点 |
|---|---|---|
| **混合精度训练 (FP16/BF16)** | `fp16`、`bf16`、`float16`、`bfloat16`、`amp`、`GradScaler`、`mixed_precision`、`half()` | **高优先级**：未使用混合精度是 GPU 利用率低的常见原因。FP32 训练计算量翻倍且显存占用大，应优先启用 BF16（A100/H100）或 FP16+GradScaler |
| **FP8 训练** | `fp8`、`float8`、`TransformerEngine`、`msamp` | H100/H200 可使用 FP8 进一步加速 GEMM 运算 |
| **激活重计算 (Activation Checkpointing)** | `gradient_checkpointing`、`activation_checkpointing`、`checkpoint_activations`、`torch.utils.checkpoint` | 长序列或大 batch 训练时，以重计算换显存 |
| **CPU/NVMe 卸载** | `offload_optimizer`、`offload_param`、`cpu_offload`、`nvme_offload`、`pin_memory` | 显存极度紧张时将优化器状态或参数卸载到 CPU/NVMe |
| **梯度累积** | `gradient_accumulation_steps`、`accumulate_grad_batches`、`accumulation` | 显存不足以支撑大 batch 时，通过多步累积梯度等效增大 batch size |
| **显存池化与 Buffer 复用** | `memory_pool`、`CachingAllocator`、`empty_cache`、`inplace`、`inplace=True`、`out=` | 避免反复 allocate/free 造成碎片化和额外开销。使用 inplace 操作（如 `F.relu(x, inplace=True)`）减少中间张量分配。对重复运算预分配 buffer 并用 `out=` 参数复用 |

---

#### D. 计算优化

| 检查项 | 搜索关键词 / 模式 | 自查要点 |
|---|---|---|
| **Flash Attention** | `flash_attn`、`flash_attention`、`FlashAttention`、`attn_implementation.*flash` | 显著减少 HBM 读写，已成为训练标配 |
| **torch.compile** | `torch.compile`、`compiled_model`、`_dynamo`、`inductor` | PyTorch 2.x 图编译，自动算子融合 |
| **Triton 自定义算子** | `triton`、`@triton.jit`、`tl.load`、`tl.store`、`triton.autotune`、`tl.program_id` | 用 Python 编写高性能 GPU kernel。参考 `/triton-optimization` skill 的决策树判断是否需要编写自定义 Triton kernel，或使用 Liger Kernel/FlagGems 等生态工具 |
| **CUDA Graph** | `cuda_graph`、`CUDAGraph`、`make_graphed_callables` | 减少 kernel launch 开销，适用于固定形状输入 |
| **Liger Kernel** | `liger_kernel`、`AutoLigerKernelForCausalLM`、`apply_liger_kernel_to_`、`LigerRMSNorm`、`LigerSwiGLUMLP`、`LigerCrossEntropyLoss`、`LigerFusedLinearCrossEntropyLoss`、`liger_rotary_pos_emb` | **高优先级**：LinkedIn 开源的 Triton 算子融合库，对 RMSNorm、RoPE、SwiGLU、CrossEntropy 等层做 kernel fusion 和 in-place 替换。可提升训练吞吐 20%、降低显存 60%，仅需一行代码接入。与 Flash Attention、FSDP、DeepSpeed 兼容。支持 LLaMA、Qwen、Gemma、Mistral、Phi、GLM、InternVL 等主流模型。`FusedLinearCrossEntropy` 将最后的 linear 层与 cross entropy 融合，对大词表场景显存节省尤为显著 |
| **算子融合** | `fused_adam`、`FusedAdam`、`fused_layer_norm`、`apex.fused`、`xformers` | 减少 kernel 调用次数和中间显存分配 |
| **Tensor Core 维度对齐** | 矩阵维度检查：hidden_dim、vocab_size、attention head 数 | 矩阵乘维度应为 8（FP16/BF16）或 16（INT8/FP8）的倍数以命中 Tensor Core。不对齐会导致 padding 或回退到普通 CUDA Core |
| **Persistent Kernel** | `persistent_kernel`、`grid_persistent`、长驻 kernel 设计 | 减少 kernel launch 开销，适用于 decode 等短 kernel 高频场景 |

---

#### E. 通信优化

| 检查项 | 搜索关键词 / 模式 | 自查要点 |
|---|---|---|
| **GDRDMA 启用（多机场景）** | `NCCL_NET_GDR_LEVEL`、`NCCL_IB_GID_INDEX`、`NCCL_IB_HCA`、`GDR`、`gdrcopy`、`ib_write` | 多机多卡训练务必确认已开启 GPUDirect RDMA，否则多机间通信耗时将显著增大。检查环境变量和 NCCL 日志中是否出现 `NET/IB` 和 `GPU Direct` 相关信息 |
| **通信计算重叠** | `overlap_comm`、`communication_overlap`、`overlap_grad_reduce`、`backward_prefetch` | 梯度同步应与反向计算重叠执行，避免 GPU 空等通信完成 |
| **梯度压缩** | `gradient_compression`、`PowerSGD`、`fp16_allreduce`、`communication_data_type` | 跨节点带宽受限时，启用梯度压缩或低精度 AllReduce 可减少通信量 |
| **NCCL 调优** | `NCCL_`、`nccl`、`NCCL_IB_DISABLE`、`NCCL_SOCKET_IFNAME`、`NCCL_P2P`、`NCCL_ALGO`、`NCCL_PROTO` | 检查是否根据集群拓扑设置了合理的 NCCL 环境变量（如 IB 网卡绑定、P2P 模式、通信算法选择等） |

---

#### F. 数据与 I/O 优化

| 检查项 | 搜索关键词 / 模式 | 自查要点 |
|---|---|---|
| **多进程并行加载** | `num_workers`、`persistent_workers`、`DataLoader`、`num_parallel_reads`、`num_parallel_calls` | 是否启用多进程并行读取数据？PyTorch 检查 `num_workers` 是否 > 0（建议 4~8）；TensorFlow 检查 `num_parallel_reads` 和 `num_parallel_calls` |
| **数据预取 (Prefetch)** | `prefetch_factor`、`prefetch`、`tf.data.Dataset.prefetch` | 是否启用提前加载机制来实现 CPU 与 GPU 并行？PyTorch 检查 `prefetch_factor`（建议 2~4）；TensorFlow 检查 `.prefetch(tf.data.AUTOTUNE)` |
| **共享内存 Pin Memory** | `pin_memory`、`pin_memory=True` | 是否设置 `pin_memory=True`？可加速 CPU→GPU 数据传输，避免额外的内存拷贝 |
| **小文件合并** | `TFRecord`、`WebDataset`、`tar`、`MMapIndexedDataset`、`concat`、`pack` | 输入数据是否存在大量小文件？小文件过多会导致寻址读取效率极低，应提前合并为大文件（如 TFRecord、WebDataset tar 包、二进制索引格式等） |
| **Minibatch 分批处理** | `batch_size`、`micro_batch`、`DataLoader`、`batch` | 数据集过大时是否按 minibatch 进行分批处理？避免一次性加载全部数据导致内存溢出或 I/O 阻塞 |
| **数据预处理离线化** | `preprocess`、`tokenize`、`pre_tokenize`、`save_to_disk`、`map(.*batched)` | 数据预处理部分（如分词、特征提取）是否已提前离线处理好？训练时做在线预处理会严重拖慢数据供给速度 |
| **数据打包 (Packing)** | `packing`、`concat_tokens`、`group_texts`、`ConstantLengthDataset` | 是否将短样本拼接填满 max_seq_len 以避免 padding 浪费算力 |
| **内存映射数据** | `mmap`、`memmap`、`MMapIndexedDataset`、`numpy.memmap` | 是否使用内存映射方式读取大文件，避免全量加载到内存 |
| **流式数据集** | `IterableDataset`、`streaming`、`load_dataset.*streaming` | TB 级数据是否使用流式加载，避免全量加载 |
| **预分词二进制格式** | `.bin`、`.idx`、`MMapIndexedDataset`、`indexed_dataset` | 是否将文本提前 tokenize 并存储为二进制格式，减少训练时的 I/O 和计算开销 |
| **H2D/D2H 异步流水线** | `non_blocking=True`、`cudaMemcpyAsync`、`cuda.Stream`、`torch.cuda.stream` | 是否使用多 CUDA stream 实现 Host-to-Device 传输、GPU 计算、Device-to-Host 回传的三级流水线重叠？单 stream 串行执行会浪费 PCIe/NVLink 带宽。在 `to(device, non_blocking=True)` 的基础上，可通过创建独立 stream 实现 prefetch 下一批数据与当前批计算的重叠 |

---

#### G. 训练策略优化

| 检查项 | 搜索关键词 / 模式 | 自查要点 |
|---|---|---|
| **学习率调度** | `warmup`、`cosine`、`lr_scheduler`、`get_scheduler`、`OneCycleLR` | 合理的 warmup + cosine/linear 调度是大 batch 训练的基础 |
| **大 Batch 优化器** | `LAMB`、`LARS`、`FusedLAMB` | 超大 batch size 训练时应使用 LAMB/LARS 保证收敛 |
| **序列长度渐进 (Curriculum)** | `curriculum`、`seqlen_warmup`、`variable_seq_lengths`、`increase_seq_length` | 先短序列后长序列，前期节省算力 |
| **LoRA / QLoRA** | `lora`、`LoraConfig`、`peft`、`qlora`、`BitsAndBytesConfig`、`4bit` | **重点自查**：模型参数量过多、训练计算量过大时，应评估是否可用 LoRA/QLoRA 等高效微调方式替代全量微调，大幅降低显存占用和计算量 |
| **层冻结** | `freeze`、`requires_grad.*False`、`frozen_layers` | 微调场景下冻结底层参数，减少需更新的参数量 |
| **对齐训练 Loss 优化** | `LigerFusedLinearDPOLoss`、`LigerFusedLinearORPOLoss`、`LigerFusedLinearSimPOLoss`、`LigerFusedLinearCPOLoss`、`LigerFusedLinearKTOLoss`、`chunked_loss`、`LigerFusedLinearJSD`、`LigerKLDIVLoss` | 对齐训练（DPO/ORPO/SimPO/CPO/KTO）和知识蒸馏（JSD/KLDiv）场景下，使用 Liger Kernel 提供的 Fused Linear Loss 可将 loss 计算与 linear 层融合并分块计算，显存降低最高 80%。若项目涉及 RLHF / 偏好对齐 / 蒸馏且未使用融合 loss，属于高价值优化点 |

---

#### H. 多模态专项（如适用）

| 检查项 | 搜索关键词 / 模式 |
|---|---|
| **视觉编码器冻结** | `freeze.*vision`、`vision_tower.*requires_grad`、`freeze_vision_encoder` |
| **动态分辨率** | `anyres`、`dynamic_resolution`、`variable_resolution`、`image_aspect_ratio` |
| **视觉 Token 裁剪** | `token_pruning`、`token_merge`、`visual_token_select` |

---

#### I. 基础设施与运维

| 检查项 | 搜索关键词 / 模式 | 自查要点 |
|---|---|---|
| **存储与计算同城/同集群** | 数据路径中的挂载点、`/mnt/`、`hdfs://`、`s3://`、`cfs://`、`ceph`、`nfs` | 确认数据存储和 GPU 计算节点是否在同一城市/同一集群内。跨城读取数据会引入巨大的网络延迟，严重降低数据供给速度 |
| **高性能存储介质** | 数据路径中是否包含 `ssd`、`nvme`、`ceph`、`cfs`、`hdfs`、`mdfs` | 优先使用本机 SSD/NVMe 或高性能 Ceph 存储。CFS-1.5、HDFS、MDFS 等分布式文件系统读取速度较慢，不适合作为训练数据的直接读取源。如必须使用慢速存储，应先将数据拷贝到本地 SSD |
| **Checkpoint 保存频率** | `save_steps`、`save_interval`、`checkpoint_interval`、`save_every`、`ModelCheckpoint`、`save_on_steps` | 模型保存（checkpoint）是否过于频繁？过频的保存操作会阻塞训练进程，尤其在大模型场景下单次保存耗时较长。建议根据训练总步数合理设置保存间隔 |
| **日志打印频率** | `logging_steps`、`log_interval`、`print_freq`、`log_every`、`report_to` | 日志打印、指标上报是否过于频繁？每步都打印日志会增加不必要的 I/O 开销和 GPU 同步等待。建议 logging_steps >= 10 |
| **进度上报频率** | `progress_bar`、`tqdm`、`callback`、`on_step_end`、`WandbCallback`、`TensorBoardCallback` | 进度上报（如 WandB、TensorBoard）是否每步都触发？高频上报会引入额外的网络和 I/O 开销，建议适当降低上报频率 |

---

#### J. 推理优化（如适用）

| 检查项 | 搜索关键词 / 模式 | 自查要点 |
|---|---|---|
| **KV Cache 管理** | `kv_cache`、`past_key_values`、`PagedAttention`、`paged_kv`、`block_table`、`cache_engine` | KV Cache 是 LLM 推理的显存核心。是否使用分页管理（PagedAttention）避免碎片？是否有 cache 淘汰/压缩策略？连续 KV Cache 在长序列下会导致大量显存浪费 |
| **Continuous Batching** | `continuous_batch`、`dynamic_batch`、`iteration_level_scheduling`、`inflight_batching` | 是否使用迭代级调度而非静态 batch？静态 batch 中短序列完成后 GPU 空闲等待长序列，造成吞吐浪费。vLLM/TensorRT-LLM/SGLang 默认支持 |
| **Prefill/Decode 分离** | `prefill`、`decode`、`chunked_prefill`、`disaggregated`、`prefix_caching` | Prefill（计算密集、可并行）和 Decode（访存密集、串行）的瓶颈完全不同。是否分别优化？是否使用 chunked prefill 避免长 prompt 阻塞 decode？ |
| **推理量化** | `int8`、`int4`、`awq`、`gptq`、`fp8`、`bitsandbytes`、`auto_gptq`、`autoawq`、`quanto`、`torchao` | 推理场景下是否使用 INT8/INT4/FP8 量化减少显存和加速计算？AWQ（激活感知）和 GPTQ（梯度感知）对精度友好。FP8 在 H100+ 上无损加速 |
| **推理引擎** | `vllm`、`tensorrt_llm`、`sglang`、`onnxruntime`、`triton_inference_server`、`text-generation-inference`、`FasterTransformer` | 是否使用成熟推理引擎而非裸 PyTorch eager 模式？推理引擎做了图优化、kernel 融合、动态 batch、内存规划等大量工程优化，通常比手写 PyTorch 推理快 2-5x |
| **推理 CUDA Graph** | `cuda_graph`、`CUDAGraph`、`enforce_eager=False`、`graph_mode` | Decode 阶段 shape 稳定，是否用 CUDA Graph 消除 kernel launch 开销？vLLM 默认启用。注意 Graph 不适用于 shape 变化的 prefill 阶段 |
| **Speculative Decoding** | `speculative`、`draft_model`、`assistant_model`、`spec_decode`、`ngram_prompt_lookup` | 是否通过小模型草稿+大模型验证的方式加速自回归生成？可在不损失精度的前提下提升 2-3x 生成速度。适用于大模型+可用小同系模型的场景 |

---

### 第三步：生成审计报告

按以下格式输出结构化报告：

```markdown
# LLM/MLLM 训练加速审计报告

## 项目概览
- 项目路径：<路径>
- 检测到的框架：<列表>
- 模型类型：<LLM / MLLM / 未知>
- 预估规模：<如可检测则显示参数量>
- 初步瓶颈类型判断：<Compute-bound / Memory-bound / Latency-bound / Launch-bound / Communication-bound / Data I/O-bound / 待 Profiling 确认>

## 审计总览

| # | 类别 | 得分 | 详情 |
|---|------|------|------|
| A | 并行策略 | X/7 | ... |
| B | 训练框架 | X/6 | ... |
| C | 显存优化 | X/6 | ... |
| D | 计算优化 | X/8 | ... |
| E | 通信优化 | X/4 | ... |
| F | 数据与 I/O 优化 | X/11 | ... |
| G | 训练策略 | X/6 | ... |
| H | 多模态专项 | X/3 或 不适用 | ... |
| I | 基础设施与运维 | X/5 | ... |
| J | 推理优化 | X/7 或 不适用 | ... |

**综合得分：X / Y (Z%)**

## 详细审查结果

### A. 并行策略
（每项格式：状态图标 + 检查项名称 + 证据来源或优化建议）
- [x] DDP — 在 `train.sh:L12` 中发现 `torchrun --nproc_per_node=8`
- [ ] FSDP — 未检测到。建议在模型参数量超过 100 亿时启用 FSDP。
...

### B-I.（各类别采用相同格式）
...

## 优先级排序的优化建议

1. **[高]** <建议内容> — 预期收益：<说明>
2. **[中]** <建议内容> — 预期收益：<说明>
3. **[低]** <建议内容> — 预期收益：<说明>

## 推荐配置代码片段

（为排名前 3 的建议提供可直接使用的配置代码片段）
```

### 第四步：建议优先级规则

按预期收益排序建议：

1. **高优先级** — 缺失的能力预计可带来 > 30% 的加速，或是规模扩展的前提条件
   - 存储与计算跨城部署 → 迁移数据到同城/同集群，或拷贝到本地 SSD
   - 使用慢速存储介质（CFS-1.5/HDFS/MDFS）→ 切换到本机 SSD 或高性能 Ceph
   - 未启用混合精度（FP16/BF16）→ 启用 BF16 或 FP16+GradScaler
   - 未使用 Flash Attention → 添加 flash_attn
   - 使用 HuggingFace 模型训练但未启用 Liger Kernel → 一行代码接入（`AutoLigerKernelForCausalLM` 或 `apply_liger_kernel_to_*`），预期提升吞吐 20%、降低显存 60%
   - 大模型仅使用 DDP 而无其他并行策略 → 添加 FSDP/ZeRO
   - 多机训练未开启 GDRDMA → 确认并启用 GPUDirect RDMA
   - 长序列场景未启用激活重计算 → 启用激活重计算
   - 模型参数量过大却做全量微调 → 评估 LoRA/QLoRA 等高效微调方式

2. **中优先级** — 有实质意义的改进（10-30% 加速）
   - 未启用多进程数据加载（num_workers=0）→ 设置 num_workers=4~8
   - 未启用数据预取（prefetch）→ 设置 prefetch_factor=2~4
   - 未设置 pin_memory → 启用 pin_memory=True
   - 输入数据小文件过多 → 合并为 TFRecord/WebDataset/二进制格式
   - 数据预处理未离线化 → 提前处理好分词/特征提取
   - 未使用梯度累积优化
   - 未启用通信计算重叠
   - 未使用数据打包 / padding 浪费严重
   - 未使用 torch.compile
   - 对齐训练（DPO/ORPO 等）未使用 Liger Fused Linear Loss → 接入 `LigerFusedLinearDPOLoss` 等，显存降低 80%

3. **低优先级** — 锦上添花的优化（< 10% 加速）
   - Checkpoint 保存 / 日志打印 / 进度上报过于频繁 → 降低保存和日志频率
   - NCCL 参数调优
   - 序列长度渐进策略
   - 高级算子融合

---

### 附录 A：场景化优化优先级速查

以下按常见场景给出 **"先做什么、再做什么"** 的优先顺序列表。优化时从第 1 项开始，逐项排查，跳过已完成的项。

#### 场景 A：PyTorch 训练慢

1. 开启 AMP / BF16 混合精度
2. 检查 DataLoader 和 H2D pipeline（num_workers、pin_memory、prefetch）
3. 用 PyTorch Profiler 定位热点算子
4. 对热点算子使用 fused op（FusedAdam、Liger Kernel）
5. 使用 `torch.compile`
6. Attention 层用 Flash Attention
7. 分布式训练做通信计算重叠（FSDP backward_prefetch / DeepSpeed overlap_comm）
8. 减少不必要的 CPU-GPU 同步点（`.item()`、`print(cuda_tensor)`）
9. 启用激活重计算（gradient checkpointing）
10. 数据打包（packing）减少 padding 浪费

#### 场景 B：LLM 推理慢

1. 上成熟推理引擎：vLLM / TensorRT-LLM / SGLang
2. KV Cache 优化（PagedAttention、分页管理）
3. Continuous Batching
4. Flash Attention / Paged Attention
5. INT8 / FP8 / INT4 量化（AWQ/GPTQ）
6. Decode 阶段启用 CUDA Graph
7. Prefill / Decode 分离优化
8. 检查多卡通信是否成瓶颈（TP 切分策略）
9. Speculative Decoding（小模型草稿 + 大模型验证）

#### 场景 C：自定义 CUDA Kernel 慢

1. 确认是否值得自己写——能否用 cuBLAS / cuDNN / Flash Attention / Triton 库替代
2. 用 Nsight Compute (ncu) 判断是 memory-bound 还是 compute-bound
3. 优化 coalesced memory access
4. 做 tiling / blocking
5. 合理使用 shared memory（注意 bank conflict）
6. 控制 register pressure（避免 spill 到 local memory）
7. 减少 warp divergence
8. Auto-tune block size / tile size 参数
9. 用 ncu 的 roofline model 验证优化效果

#### 场景 D：推理服务延迟高

1. 降低 kernel launch 开销（CUDA Graph）
2. Request batching / Continuous Batching
3. 内存池预分配（避免运行时 malloc）
4. 模型量化（INT8/FP8/INT4）
5. CPU 预处理与 GPU 计算重叠（多 stream）
6. 减少 CPU-GPU 同步
7. 用 TensorRT / ONNX Runtime 做图优化
8. KV Cache 压缩与高效管理
9. Speculative Decoding

---

### 附录 B：GPU 优化 Checklist 速查表

#### 通用检查项

| # | 检查项 | 快速判断方法 |
|---|--------|-------------|
| 1 | 是否开启 BF16 / FP16 / INT8 / FP8？ | 搜索 `bf16`、`fp16`、`amp`、`mixed_precision` |
| 2 | 是否命中 Tensor Core？ | 矩阵维度是否为 8（FP16）/16（INT8）的倍数 |
| 3 | 是否存在大量小 kernel？ | nsys 时间线或 profiler 中 < 10μs kernel 占比 |
| 4 | 是否能做算子融合？ | 连续的 pointwise ops、未使用 torch.compile/Liger Kernel |
| 5 | 是否存在不必要的 CPU-GPU 同步？ | 搜索 `.item()`、`print(cuda_tensor)`、`.cpu()` |
| 6 | H2D / D2H 是否异步？ | 检查 `non_blocking=True`、`pin_memory` |
| 7 | kernel 是否 memory-bound？ | ncu roofline 或 DRAM throughput vs 峰值带宽比 |
| 8 | global memory 访问是否 coalesced？ | ncu memory workload analysis |
| 9 | shared memory 是否有 bank conflict？ | ncu shared memory bank conflict 指标 |
| 10 | register 是否过高导致 spill？ | ncu register file usage / spill loads |
| 11 | occupancy 是否过低？ | ncu achieved occupancy vs theoretical |
| 12 | warp 是否严重 divergence？ | ncu branch efficiency / warp execution efficiency |
| 13 | 是否可以用更成熟的库替代？ | 自定义 kernel vs cuBLAS/Flash Attention/Liger Kernel |
| 14 | 是否可以用 TorchCompile / TensorRT / Triton？ | 搜索 `torch.compile`、`tensorrt` |
| 15 | 多卡时通信是否成瓶颈？ | AllReduce 占比 > 30%、GPU idle 与通信交替 |
| 16 | H2D/计算/D2H 是否流水线化？ | 是否使用多 CUDA stream 做三级 overlap |

#### LLM / Transformer 专项检查

| # | 检查项 | 快速判断方法 |
|---|--------|-------------|
| 1 | 是否用了 Flash Attention？ | 搜索 `flash_attn`、`attn_implementation` |
| 2 | KV Cache 是否高效管理？ | PagedAttention、cache 压缩、cache 量化 |
| 3 | 是否做了 Continuous Batching？ | vLLM/TRT-LLM 默认支持，裸 PyTorch 需手动实现 |
| 4 | Prefill / Decode 是否分开优化？ | 两者瓶颈不同：prefill 算力密集、decode 访存密集 |
| 5 | 是否能量化？ | INT8/FP8（推理）、QLoRA（训练） |
| 6 | Decode 阶段是否适合 CUDA Graph？ | shape 稳定的 decode 可用 Graph 消除 launch 开销 |
| 7 | 多卡切分策略是否合理？ | TP vs PP vs DP 选择、NVLink 拓扑匹配 |
