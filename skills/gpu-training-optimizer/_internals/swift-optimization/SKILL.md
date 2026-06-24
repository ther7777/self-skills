---
name: swift-optimization
description: ms-swift 训练优化审计
user-invocable: false
---

# Skill: ms-swift 训练优化审计

## 描述
针对基于 ms-swift（ModelScope SWIFT）框架的训练项目，快速识别已采用和未采用的优化手段，提供针对性的显存优化与性能加速建议。ms-swift 支持 600+ 文本模型和 400+ 多模态模型的微调、RLHF、预训练、推理和部署。

## 触发条件
当识别到用户项目满足以下任一条件时自动触发：
- 代码中 `import swift` / `from swift import` / `from swift.llm` / `from swift.trainers`
- 启动命令使用 `swift sft` / `swift pt` / `swift rlhf` / `swift infer` / `swift deploy` / `megatron sft` / `megatron pt`
- 配置文件（YAML/JSON）中包含 ms-swift 特有参数：`tuner_type`、`tuner_backend`、`padding_free`、`loss_scale`、`attn_impl`、`sequence_parallel_size`
- 依赖文件中包含 `ms-swift` 或 `swift`（ModelScope SWIFT 包名）
- 检查点目录中的 `args.json` 包含 `swift_version` 字段
- DeepSpeed 通过字符串快捷方式引用：`--deepspeed zero2` / `zero3` / `zero2_offload` / `zero3_offload`

## 执行指令

你是 ms-swift 训练优化专家。被调用时，请按照以下清单对目标项目进行全面审计。对每个检查项，判定其状态：**已启用** / **未启用** / **建议启用** / **不适用**，并给出具体的配置修改建议。

### 第一步：识别训练模式与基础信息

扫描项目的 YAML 配置文件和启动命令，确定：

1. **训练模式**：`swift sft` / `swift pt` / `swift rlhf --rlhf_type <type>` / `megatron sft`
2. **RLHF 算法**（如适用）：`dpo` / `kto` / `cpo` / `simpo` / `orpo` / `grpo` / `ppo` / `gkd` / `rm`
3. **微调方式**（`tuner_type`）：`lora` / `full` / `adalora` / `longlora` / `llamapro` / `adapter` / `vera` / `boft` / `fourierft` / `reft` / `bone`
4. **模型与规模**：`--model` 参数，识别模型系列（Qwen/Llama/DeepSeek/InternLM/GLM 等）和参数量
5. **硬件环境**：GPU 型号/数量、显存大小（结合 `/system-resources` 结果）
6. **数据规模**：数据集大小、`max_length`（序列长度）

---

### 第二步：显存优化审计

#### A. 参数高效微调（PEFT）

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **LoRA** | `--tuner_type lora`<br>`--lora_rank 8 --lora_alpha 32`<br>`--target_modules all-linear` | `all-linear` 是 ms-swift 快捷方式，作用于所有线性层。默认 rank=8，效果优先可增至 64~128 |
| **QLoRA（BnB 4bit）** | `--quant_method bnb --quant_bits 4`<br>`--bnb_4bit_quant_type nf4`<br>`--bnb_4bit_use_double_quant true` | 7B→~6GB。NF4 优于 FP4。double_quant 额外节省 ~0.4GB/B |
| **QLoRA（HQQ）** | `--quant_method hqq --quant_bits 4` | Half-Quadratic 量化，支持 1/2/3/4/8 bit |
| **QLoRA（EETQ）** | `--quant_method eetq --quant_bits 8` | 仅 8bit，速度快 |
| **FP8 量化** | `--quant_method fp8` | Hopper+ GPU，FineGrainedFP8Config |
| **DoRA** | `--use_dora true` | 权重分解 LoRA，效果更好但略增显存 |
| **LoRA+** | `--lorap_lr_ratio <float>` | B 矩阵使用更大学习率，自动设置 lorap 优化器 |
| **RS-LoRA** | `--use_rslora true` | 大 rank 时收敛更稳定 |
| **PiSSA** | `--init_weights pissa` | 主奇异值初始化，收敛更快 |
| **LoRA-GA** | `--init_weights lora-ga`<br>`--lora_ga_batch_size 2 --lora_ga_iters 2` | 梯度感知初始化 |
| **AdaLoRA** | `--tuner_type adalora`<br>`--adalora_target_r 8 --adalora_init_r 12` | 自适应秩分配 |
| **LISA** | `--lisa_activated_layers N`<br>`--lisa_step_interval 20` | 层采样激活，需配合 `--tuner_type full` |
| **UnSloth** | `--tuner_backend unsloth` | LoRA 训练速度提升 ~70%，显存降低 ~50% |

**评分 /12**

**建议**：
- 显存紧张首选 QLoRA 4bit（`--quant_method bnb --quant_bits 4`）
- 效果优先选 LoRA rank=64~128 + `--use_dora true`
- LoRA 训练用 `--tuner_backend unsloth` 获得额外加速
- 全参训练显存不足时考虑 LISA（`--lisa_activated_layers 4`）

---

#### B. 混合精度

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **BF16 混合精度** | `--torch_dtype bfloat16` | Ampere+(A100/H100/H20) 首选。ms-swift 自动设置 `--bf16 true` |
| **FP16 混合精度** | `--torch_dtype float16` | V100/T4 使用 |
| **FP8 训练** | Megatron 模式下 `--fp8_format e4m3`<br>`--fp8_recipe tensorwise/delayed/blockwise` | H100/H200 专属，GEMM 速度翻倍 |

**评分 /4**

**建议**：
- A100/H100/H20 必须 `--torch_dtype bfloat16`
- V100 使用 `--torch_dtype float16`
- H100 Megatron 模式可用 FP8

---

#### C. 激活重计算与梯度优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **梯度检查点** | `--gradient_checkpointing true` | ms-swift **默认开启**（HF 默认关闭）。以 ~30% 计算换 ~60% 激活显存 |
| **ViT 梯度检查点** | `--vit_gradient_checkpointing true` | 多模态模型单独控制 vision tower 的 GC |
| **梯度累积** | `--gradient_accumulation_steps N` | ms-swift 默认自动计算：`max(1, ceil(16 / bs / world_size))`，等效 batch size ~16 |
| **use_logits_to_keep** | `--use_logits_to_keep true` | 仅计算有 label 位置的 logits，减少显存（RLHF 场景效果明显） |
| **激活 CPU 卸载** | `--callbacks activation_cpu_offload` | 激活卸载到 CPU RAM，FSDP2 场景使用 |

**评分 /5**

**建议**：
- ms-swift 默认已开启 gradient_checkpointing，确认未被关闭
- 多模态模型同时开启 `--vit_gradient_checkpointing true`
- RLHF 场景启用 `--use_logits_to_keep true` 节省显存

---

#### D. 分布式显存优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **DeepSpeed ZeRO-2** | `--deepspeed zero2` | ms-swift 字符串快捷方式。LoRA 多卡首选 |
| **DeepSpeed ZeRO-3** | `--deepspeed zero3` | 全参微调大模型必用 |
| **ZeRO-2 + CPU Offload** | `--deepspeed zero2_offload` | 优化器卸载到 CPU |
| **ZeRO-3 + CPU Offload** | `--deepspeed zero3_offload` | 参数+优化器卸载到 CPU |
| **ZeRO++** | `--zero_hpz_partition_size N` | 节点内高精度 + 跨节点量化通信。N=每节点 GPU 数 |
| **DeepSpeed AutoTP** | `--deepspeed_autotp_size N` | ZeRO-0/1/2 + 自动张量并行，全参训练专用 |
| **FSDP2** | `--fsdp fsdp2` | PyTorch 原生分片（FSDP v2），支持 DTensor、per-parameter sharding |
| **device_map 模型并行** | `--device_map auto` | 简单模型并行（不与 DDP 兼容） |
| **Megatron 并行** | `megatron sft --tensor_model_parallel_size N`<br>`--pipeline_model_parallel_size N`<br>`--context_parallel_size N`<br>`--expert_model_parallel_size N` | TP+PP+CP+EP 四维并行，大规模预训练/SFT |

**评分 /8**

**建议**：
- LoRA 多卡 → `--deepspeed zero2`
- 全参微调 → `--deepspeed zero3`
- 显存极不足 → `--deepspeed zero3_offload`
- 多机训练 → 加 `--zero_hpz_partition_size <gpus_per_node>`（ZeRO++）
- 大规模预训练 → `megatron sft` 使用 TP+PP+SP

---

#### E. 高效优化器

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **GaLore** | `--use_galore true`<br>`--galore_rank 128 --galore_update_proj_gap 50` | 全参训练显存接近 LoRA 水平 |
| **Q-GaLore** | `--use_galore true --galore_quantization true` | 量化版 GaLore，显存更低 |

**评分 /3**

---

### 第三步：计算性能优化审计

#### F. 注意力机制优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Flash Attention 2** | `--attn_impl flash_attn` | **高优先级**。ms-swift 使用 `attn_impl` 而非 HF 的 `attn_implementation` |
| **Flash Attention 3** | `--attn_impl flash_attention_3` | Hopper GPU，实验性 |
| **SDPA** | `--attn_impl sdpa` | PyTorch 原生 SDPA，兼容性好 |
| **FlexAttention** | `--attn_impl flex_attention` | PyTorch 2.5+ 灵活注意力 |
| **序列并行** | `--sequence_parallel_size N` | Ulysses + Ring Attention，超长序列（512K+）训练必用 |

**评分 /5**

**建议**：
- Ampere+ GPU 必须 `--attn_impl flash_attn`
- 超长序列（>128K） → `--sequence_parallel_size 4~8` + `--attn_impl flash_attn`
- FA 安装失败退回 `--attn_impl sdpa`

---

#### G. 训练加速引擎

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Liger Kernel** | `--use_liger_kernel true` | Triton 融合算子（RMSNorm/RoPE/SwiGLU/CrossEntropy），吞吐 +20%、显存 -60%。不兼容 device_map |
| **torch.compile** | `--torch_compile true` | HF Trainer 原生支持，图编译优化 |
| **UnSloth** | `--tuner_backend unsloth` | LoRA 专用加速后端 |

**评分 /4**

**建议**：
- 全参训练 → `--use_liger_kernel true`
- LoRA 训练 → `--tuner_backend unsloth`（与 Liger Kernel 二选一）
- 长期训练可尝试 `--torch_compile true`

---

#### H. 数据处理优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **序列打包（Packing）** | `--packing true`<br>`--packing_length N` | 多条短样本拼接，减少 padding 浪费。需 `--attn_impl flash_attn` |
| **Padding-Free** | `--padding_free true` | 展平 batch 消除 padding，零预处理开销。需 `--attn_impl flash_attn`。比 packing 更灵活 |
| **流式数据集** | `--streaming true` | 超大数据集流式加载，需配合 `--max_steps` |
| **延迟 Tokenize** | `--lazy_tokenize true` | 推迟 tokenization 到取数据时执行，多模态自动启用 |
| **预处理并行** | `--dataset_num_proc N` | 多进程数据预处理 |
| **DataLoader Workers** | `--dataloader_num_workers N` | ms-swift 默认 Linux=1。建议设 4~8 |
| **Persistent Workers** | `--dataloader_persistent_workers true` | 跨 epoch 保持 worker |
| **Prefetch Factor** | `--dataloader_prefetch_factor 2` | 数据预取 |

**评分 /7**

**建议**：
- SFT/DPO 训练建议启用 `--packing true` 或 `--padding_free true`（二选一）
- padding_free 零预处理开销，更适合多模态和变长场景
- 大数据集用 `--streaming true`
- 设 `--dataloader_num_workers 4` + `--dataloader_persistent_workers true`

---

#### I. 训练策略优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Loss Scale** | `--loss_scale default` | ms-swift 独有。支持 `default`（仅 assistant）/ `last_round`（仅最后轮）/ `all`（全部）/ `ignore_empty_think`（CoT）等，可组合使用 |
| **NEFTune** | `--neftune_noise_alpha 5` | Embedding 噪声微调，提升 SFT 效果 |
| **RoPE Scaling** | `--rope_scaling linear/dynamic/yarn` | 扩展上下文长度。YaRN 适合超长序列 |
| **DFT Loss** | `--enable_dft_loss true` | 动态焦点训练 |
| **多模态冻结** | `--freeze_llm false/true`<br>`--freeze_vit true/false`<br>`--freeze_aligner true/false` | 细粒度控制多模态各模块是否冻结 |
| **分层学习率** | `--vit_lr X --aligner_lr Y` | 多模态各模块独立学习率 |

**评分 /6**

---

#### J. RLHF / GRPO 优化（如适用）

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **vLLM 推理加速** | `--use_vllm true`<br>`--vllm_mode colocate/server` | GRPO/GKD rollout 推理加速。`colocate`=同进程，`server`=独立服务 |
| **异步生成** | `--async_generate true` | 使用旧权重异步 rollout，提升训练流水线效率 |
| **优化器 CPU 卸载** | `--offload_optimizer true` | vLLM 推理期间将优化器卸载到 CPU |
| **模型 CPU 卸载** | `--offload_model true` | vLLM 推理期间将模型卸载到 CPU |
| **Sleep Level** | `--sleep_level 1/2` | 训练期间释放 GPU 显存级别（给 vLLM 腾空间） |
| **GRPO 采样** | `--num_generations 8`<br>`--advantage_estimator grpo/rloo/reinforce_plus_plus` | 组内相对策略优化配置 |
| **动态采样** | `--dynamic_sample true` | DAPO：过滤全对/全错组 |
| **Prefix Caching** | `--vllm_enable_prefix_caching true` | vLLM prefix 缓存，重复 prompt 加速 |

**评分 /6**（仅 RLHF 场景评分）

---

### 第四步：场景化配置模板

#### 场景 1：7B 模型 + 单卡 A100/H100 80GB — LoRA SFT

```yaml
# sft_lora.yaml
model: Qwen/Qwen2.5-7B-Instruct
tuner_type: lora
tuner_backend: unsloth            # UnSloth 加速
lora_rank: 64
lora_alpha: 128
target_modules: all-linear
torch_dtype: bfloat16
attn_impl: flash_attn             # Flash Attention
use_liger_kernel: false           # UnSloth 和 Liger 二选一
dataset: your_dataset
packing: true                     # 序列打包
max_length: 4096
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
learning_rate: 1e-4
num_train_epochs: 3
neftune_noise_alpha: 5            # 提升效果
dataloader_num_workers: 4
dataloader_persistent_workers: true
output_dir: output/qwen2.5-7b-lora
```

启动：`swift sft --config sft_lora.yaml`

#### 场景 2：7B 模型 + 单卡 24GB — QLoRA SFT

```yaml
# sft_qlora.yaml
model: Qwen/Qwen2.5-7B-Instruct
tuner_type: lora
tuner_backend: unsloth
quant_method: bnb
quant_bits: 4
bnb_4bit_quant_type: nf4
bnb_4bit_use_double_quant: true
lora_rank: 32
lora_alpha: 64
target_modules: all-linear
torch_dtype: bfloat16
attn_impl: flash_attn
dataset: your_dataset
packing: true
max_length: 2048
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2e-4
num_train_epochs: 3
output_dir: output/qwen2.5-7b-qlora
```

启动：`swift sft --config sft_qlora.yaml`

#### 场景 3：70B 模型 + 8xA100 80GB — LoRA SFT + DeepSpeed ZeRO-2

```yaml
# sft_lora_ds.yaml
model: Qwen/Qwen2.5-72B-Instruct
tuner_type: lora
lora_rank: 64
lora_alpha: 128
target_modules: all-linear
torch_dtype: bfloat16
attn_impl: flash_attn
use_liger_kernel: true            # 大模型用 Liger Kernel
deepspeed: zero2
dataset: your_dataset
packing: true
max_length: 4096
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5e-5
num_train_epochs: 3
output_dir: output/qwen2.5-72b-lora
```

启动：`NPROC_PER_NODE=8 swift sft --config sft_lora_ds.yaml`

#### 场景 4：7B 全参微调 + 8xA100 — DeepSpeed ZeRO-3 + Liger Kernel

```yaml
# sft_full.yaml
model: Qwen/Qwen2.5-7B-Instruct
tuner_type: full
torch_dtype: bfloat16
attn_impl: flash_attn
use_liger_kernel: true
deepspeed: zero3
dataset: your_dataset
packing: true
max_length: 4096
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 1e-5
num_train_epochs: 3
save_only_model: true
output_dir: output/qwen2.5-7b-full
```

启动：`NPROC_PER_NODE=8 swift sft --config sft_full.yaml`

#### 场景 5：GRPO + vLLM 推理加速

```yaml
# grpo.yaml
model: Qwen/Qwen2.5-7B-Instruct
tuner_type: lora
lora_rank: 16
target_modules: all-linear
torch_dtype: bfloat16
attn_impl: flash_attn
rlhf_type: grpo
use_vllm: true
vllm_mode: colocate
num_generations: 8
reward_funcs: accuracy
deepspeed: zero2
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 1e-6
beta: 0.04
max_completion_length: 512
output_dir: output/qwen2.5-7b-grpo
```

启动：`NPROC_PER_NODE=4 swift rlhf --config grpo.yaml`

#### 场景 6：超长序列训练（512K）— 序列并行

```yaml
# long_seq.yaml
model: Qwen/Qwen2.5-7B-Instruct
tuner_type: lora
lora_rank: 16
target_modules: all-linear
torch_dtype: bfloat16
attn_impl: flash_attn
rope_scaling: yarn
max_length: 524288
sequence_parallel_size: 8
use_liger_kernel: true
deepspeed: zero3_offload
packing: true
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
output_dir: output/qwen2.5-7b-long
```

启动：`NPROC_PER_NODE=8 swift sft --config long_seq.yaml`

---

### 第五步：输出审计报告

按以下格式输出审计结果：

```markdown
# ms-swift 训练优化审计报告

## 基本信息
- 模型：{model}（{参数量}）
- 训练模式：{swift sft/pt/rlhf}
- RLHF 算法：{rlhf_type}（如适用）
- 微调方式：{tuner_type}
- 硬件：{GPU型号} x {数量}（{显存}GB）
- 序列长度：{max_length}

## 审计结果总览

| 类别 | 得分 | 已启用 | 建议启用 |
|------|------|--------|----------|
| A. PEFT | x/12 | ... | ... |
| B. 混合精度 | x/4 | ... | ... |
| C. 激活重计算 | x/5 | ... | ... |
| D. 分布式优化 | x/8 | ... | ... |
| E. 高效优化器 | x/3 | ... | ... |
| F. 注意力优化 | x/5 | ... | ... |
| G. 加速引擎 | x/4 | ... | ... |
| H. 数据处理 | x/7 | ... | ... |
| I. 训练策略 | x/6 | ... | ... |
| J. RLHF/GRPO | x/6 | ... | ... |
| **总计** | **x/60** | | |

## 优先优化建议（按影响排序）

### P0 — 立即执行（显著收益，零风险）
1. ...

### P1 — 强烈推荐（明显收益，低风险）
1. ...

### P2 — 建议尝试（中等收益，需测试）
1. ...

## 推荐配置修改

（给出具体的 YAML/CLI 参数修改 diff）
```

---

## 常见优化决策树

```
开始
├── 显存不足（OOM）？
│   ├── 是 → 当前用全参训练？
│   │   ├── 是 → 切换为 LoRA（--tuner_type lora）
│   │   │   └── 仍然 OOM？ → QLoRA（--quant_method bnb --quant_bits 4）
│   │   └── 否（已用 LoRA）
│   │       ├── 确认 gradient_checkpointing 已开启（ms-swift 默认开启）
│   │       ├── 减小 batch size + 自动 gradient_accumulation_steps
│   │       ├── 启用 --use_logits_to_keep true（RLHF 场景）
│   │       └── 仍然 OOM？ → 降低 quant_bits / 减小 max_length / FSDP2+QLoRA
│   └── 否 → 继续性能优化
├── 训练速度慢？
│   ├── attn_impl 已启用 flash_attn？
│   │   ├── 否 → 设置 --attn_impl flash_attn
│   │   └── 是 → 继续
│   ├── LoRA 训练？ → --tuner_backend unsloth
│   ├── 全参训练？ → --use_liger_kernel true
│   ├── 数据已打包？
│   │   ├── 否 → --packing true 或 --padding_free true
│   │   └── 是 → 继续
│   ├── 多卡训练？
│   │   ├── LoRA → --deepspeed zero2
│   │   └── 全参 → --deepspeed zero3
│   ├── GRPO 训练慢？ → --use_vllm true --vllm_mode colocate
│   └── 检查 dataloader_num_workers（建议 4~8）
└── 超长序列（>128K）？
    ├── 启用 --sequence_parallel_size N
    ├── 配合 --rope_scaling yarn
    └── --deepspeed zero3_offload + --packing true
```

## 显存估算参考

| 方法 | 精度 | 7B | 14B | 30B | 70B |
|------|------|-----|-----|-----|------|
| 全参（bf16） | 16-bit 混合 | 120GB | 240GB | 600GB | 1200GB |
| LoRA | 16-bit | 16GB | 32GB | 64GB | 160GB |
| QLoRA | 8-bit | 10GB | 20GB | 40GB | 80GB |
| QLoRA | 4-bit | 6GB | 12GB | 24GB | 48GB |
| GaLore（全参） | 16-bit | ~16GB | ~32GB | ~64GB | ~160GB |

## ms-swift 特有关键词速查

用于识别 ms-swift 项目的关键词和文件模式：

| 类别 | 关键词/模式 |
|------|------------|
| CLI | `swift sft`、`swift pt`、`swift rlhf`、`swift infer`、`swift deploy`、`swift export`、`megatron sft`、`megatron pt` |
| 配置 | `tuner_type`、`tuner_backend`、`attn_impl`、`padding_free`、`loss_scale`、`sequence_parallel_size`、`packing_length`、`model_author`、`model_name`、`use_hf`、`lisa_activated_layers` |
| import | `from swift`、`import swift`、`from swift.llm`、`from swift.trainers` |
| 依赖 | `ms-swift`、`swift`（ModelScope） |
| DeepSpeed | `--deepspeed zero0/zero1/zero2/zero3/zero2_offload/zero3_offload`（字符串快捷方式） |
| RLHF | `--rlhf_type grpo/dpo/kto/cpo/simpo/orpo/ppo/gkd/rm` |
| Megatron | `megatron sft`、`--tensor_model_parallel_size`、`--pipeline_model_parallel_size` |
| 环境变量 | `NPROC_PER_NODE`（ms-swift 多卡启动方式）、`NNODES`、`NODE_RANK`、`MASTER_ADDR` |
| 检查点 | `args.json` 含 `swift_version` 字段 |
