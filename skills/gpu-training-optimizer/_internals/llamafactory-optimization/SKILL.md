---
name: llamafactory-optimization
description: LlamaFactory 训练优化审计
user-invocable: false
---

# Skill: LlamaFactory 训练优化审计

## 描述
针对基于 LlamaFactory（hiyouga/LLaMA-Factory）框架的训练项目，快速识别已采用和未采用的优化手段，提供针对性的显存优化与性能加速建议。

## 触发条件
当识别到用户项目满足以下任一条件时自动触发：
- 代码中 import 了 `llamafactory` 或使用 `llamafactory-cli` / `llamafactory.cli`
- 配置文件（YAML/JSON）中包含 LlamaFactory 特有参数：`neat_packing`、`use_unsloth`、`enable_liger_kernel`、`flash_attn: fa2`、`finetuning_type`
- 依赖文件中包含 `llamafactory` 或 `llama-factory`
- 启动命令使用 `llamafactory-cli train` 或通过 LlamaBoard WebUI 发起训练
- 项目结构包含 LlamaFactory 典型文件：`data/dataset_info.json`、`examples/` 目录下有 `train_lora/`、`train_qlora/`、`train_full/` 等

## 执行指令

你是 LlamaFactory 训练优化专家。被调用时，请按照以下清单对目标项目进行全面审计。对每个检查项，判定其状态：**已启用** / **未启用** / **建议启用** / **不适用**，并给出具体的配置修改建议。

### 第一步：识别训练模式与基础信息

扫描项目的 YAML 配置文件和启动脚本，确定：

1. **训练阶段**（`stage`）：`pt`(预训练) / `sft`(微调) / `rm`(奖励建模) / `ppo` / `dpo` / `kto` / `orpo` / `simpo`
2. **微调方式**（`finetuning_type`）：`full`(全参) / `freeze`(冻结) / `lora`
3. **模型与规模**：`model_name_or_path`，识别模型系列（Qwen/Llama/DeepSeek/GLM 等）和参数量
4. **硬件环境**：GPU 型号/数量、显存大小（结合 `/system-resources` 结果）
5. **数据规模**：数据集大小、`cutoff_len`（序列长度）

---

### 第二步：显存优化审计

#### A. 参数高效微调（PEFT）

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **LoRA** | `finetuning_type: lora`<br>`lora_rank`、`lora_alpha`、`lora_target` | 默认 rank=8。7B 模型 LoRA 仅需 ~16GB 显存（16bit）vs 全参 ~120GB（32bit）。`lora_target: all` 可作用于所有线性层以提升效果 |
| **QLoRA（量化 LoRA）** | `quantization_bit: 4`（或 2/3/5/6/8）<br>`quantization_method: bitsandbytes`（默认）<br>`quantization_type: nf4`（默认）<br>`double_quantization: true`（默认） | 4bit QLoRA：7B→~6GB、70B→~48GB。NF4 比 FP4 更适合正态分布权重。双重量化可进一步节省约 0.4GB/B。Ascend NPU 需设 `double_quantization: false` |
| **DoRA** | `use_dora: true` | 权重分解 LoRA，比标准 LoRA 效果更好但略增显存。适合追求效果时使用 |
| **rsLoRA** | `use_rslora: true` | 秩稳定 LoRA，大 rank 时收敛更稳定 |
| **PiSSA** | `pissa_init: true`、`pissa_iter` | 主奇异值初始化，比随机初始化收敛更快 |
| **LoftQ** | 需安装 LoftQ，通过量化感知初始化 | LoRA 微调感知的量化初始化策略 |
| **LoRA+** | `loraplus_lr_ratio` | 增强 LoRA，B 矩阵使用更大学习率 |
| **OFT/QOFT** | `finetuning_type: lora` + PEFT OFT 适配器 | 正交微调，保持预训练表示结构 |

**建议**：
- 显存紧张首选 QLoRA 4bit (`quantization_bit: 4`)
- 效果优先选 LoRA rank=64~128 + `use_dora: true`
- 70B 模型 2 卡 24GB → 使用 FSDP+QLoRA（`finetuning_type: lora` + `quantization_bit: 4` + FSDP 配置）

---

#### B. 混合精度

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **BF16 混合精度** | `bf16: true` | Ampere+(A100/H100/H20) 首选。loss scale 免调，数值稳定 |
| **FP16 混合精度** | `fp16: true` | V100/T4 使用。需配合 loss scale，注意溢出风险 |
| **Pure BF16** | `pure_bf16: true` | 16bit 纯精度训练，显存减半（7B: 120GB→60GB），但精度略降 |
| **FP8 训练** | `fp8: true`<br>`fp8_backend: auto/torchao/te/msamp`<br>`fp8_enable_fsdp_float8_all_gather: true` | H100/H200 专属（Hopper+）。GEMM 计算速度翻倍。仅量化 Linear 层（skip embedding/lm_head），输入输出维度需被 16 整除。FSDP2 可开启 FP8 all-gather 通信 |
| **Upcast LayerNorm** | `upcast_layernorm: true` | 将 LayerNorm 权重上转为 FP32，提升混合精度训练稳定性 |
| **Upcast LM Head** | `upcast_lmhead_output: true` | LM Head 输出转 FP32，稳定 loss 计算 |

**建议**：
- A100/H100/H20 必须启用 `bf16: true`
- V100 使用 `fp16: true`
- H100 可尝试 `fp8: true` 获得额外加速
- 训练不稳定（loss spike）时启用 `upcast_layernorm: true` + `upcast_lmhead_output: true`

---

#### C. 激活重计算与梯度优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **梯度检查点** | `gradient_checkpointing: true`（HF TrainingArguments 中设置）<br>`disable_gradient_checkpointing: false`（默认） | 以 ~30% 计算换 ~60% 激活显存。长序列/大 batch 必开 |
| **Unsloth 梯度检查点** | `use_unsloth_gc: true` | 智能将激活卸载到 CPU RAM，比标准 GC 更省显存。无需安装 unsloth 即可使用 |
| **重入模式** | `use_reentrant_gc: true`（默认）<br>FSDP2 时强制 `false` | 非重入模式（`false`）兼容性更好但稍慢。FSDP2 自动处理 |
| **梯度累积** | `gradient_accumulation_steps` | 等效增大 batch size，不增加显存。设置过大注意 BN 影响 |

**建议**：
- 序列长度 >2048 或 batch size >4 时，务必启用 `gradient_checkpointing`
- 显存极紧张（如 24GB 卡训 7B）启用 `use_unsloth_gc: true`
- 全参微调大模型必须结合梯度累积（建议 `gradient_accumulation_steps: 4~16`）

---

#### D. 分布式显存优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **DeepSpeed ZeRO-2** | `deepspeed: examples/deepspeed/ds_z2_config.json` | 分片梯度+优化器状态。LoRA 微调多卡首选。`overlap_comm: true` 可隐藏通信 |
| **DeepSpeed ZeRO-3** | `deepspeed: examples/deepspeed/ds_z3_config.json` | 分片参数+梯度+优化器。全参微调大模型必用。`stage3_gather_16bit_weights_on_model_save: true` 保存可用权重 |
| **ZeRO-3 + CPU Offload** | `deepspeed: examples/deepspeed/ds_z3_offload_config.json` | 优化器/参数卸载到 CPU。极低显存场景（如 2x24GB 训 70B） |
| **DeepSpeed AutoTP** | `ds_z2_autotp_config.json`<br>`autotp_size: 2` | ZeRO-2 + 自动张量并行，适合单层参数超大的模型 |
| **FSDP** | `accelerate` YAML 配置<br>`fsdp_sharding_strategy: FULL_SHARD` | PyTorch 原生分片。支持 FSDP1 和 FSDP2（`fsdp_version: 2`） |
| **FSDP + CPU Offload** | `fsdp_offload_params: true` | 参数卸载到 CPU，注意会影响训练速度 |
| **FSDP + QLoRA** | FSDP 配置 + `quantization_bit: 4` | 70B 模型可在 2x24GB GPU 上微调 |
| **Megatron-Core** | `use_mca: true`<br>`tensor_model_parallel_size`、`pipeline_model_parallel_size`<br>`expert_model_parallel_size`、`sequence_parallel: true` | 大规模预训练/SFT/DPO，支持 TP+PP+EP+SP 四维并行。MoE 模型用 `moe_grouped_gemm: true` + `moe_token_dispatcher_type: alltoall` |
| **Ray 分布式** | `USE_RAY=1`<br>`ray_num_workers`、`resources_per_worker` | 弹性分布式训练，支持多机多卡自动调度，PACK 策略优先同节点放置 |

**建议**：
- LoRA 多卡 → DeepSpeed ZeRO-2（通信开销低）
- 全参微调 → DeepSpeed ZeRO-3 或 FSDP FULL_SHARD
- 70B 模型 2x24GB → FSDP+QLoRA
- 显存极端不足 → ZeRO-3 + CPU Offload

---

#### E. 高效优化器

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **GaLore** | `use_galore: true`<br>`galore_rank`、`galore_scale`、`galore_target` | 梯度低秩投影，全参训练显存接近 LoRA 水平（7B ~16GB），但训练全部参数 |
| **BAdam** | `use_badam: true` | 块状 Adam，分块更新参数，全参训练的显存高效方案 |
| **APOLLO** | `use_apollo: true` | 自适应梯度缩放优化器，显存效率高 |
| **Adam-mini** | `optim: adam_mini` | 轻量 Adam 变体，减少优化器状态显存 |
| **Muon** | 通过 `optim` 参数配置 | 高效训练优化器 |
| **FusedAdam** | DeepSpeed 配置中 `type: "Adam"` | 融合 Adam 减少 kernel 调用开销 |

**建议**：
- 想要全参训练效果但只有 LoRA 级别显存 → GaLore
- BAdam 适合全参微调的显存优化
- DeepSpeed 场景使用 FusedAdam 替代原生 Adam

---

### 第三步：计算性能优化审计

#### F. 注意力机制优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Flash Attention 2** | `flash_attn: fa2` | **高优先级**。RTX4090/A100/H100/H20 必开。显著减少 HBM 读写，加速训练 |
| **SDPA** | `flash_attn: sdpa` | PyTorch 原生 Scaled Dot Product Attention。兼容性好但比 FA2 稍慢 |
| **Auto** | `flash_attn: auto`（默认） | 自动选择最佳注意力实现 |
| **Shift Short Attention** | `shift_attn: true` | LongLoRA 的 S²-Attn，长序列训练高效替代方案。需配合 LoRA 使用 |

**建议**：
- Ampere+ GPU 必须设置 `flash_attn: fa2`
- 长序列场景（>8K）结合 `shift_attn: true` + LoRA
- 安装 flash-attn 失败时退回 `flash_attn: sdpa`

---

#### G. 训练加速引擎

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Unsloth** | `use_unsloth: true` | **高优先级**。LoRA 训练速度提升 **170%**，长序列训练速度提升 **117%**、显存降低 **50%**（对比 FA2）。仅支持 LoRA 模式。自带优化无需额外配置 |
| **Liger Kernel** | `enable_liger_kernel: true` | **高优先级**。LinkedIn 开源 Triton 融合算子（RMSNorm/RoPE/SwiGLU/CrossEntropy），吞吐提升 ~20%、显存降低 ~60%。与 LoRA/全参/DeepSpeed/FSDP 均兼容 |
| **KTransformers** | `use_kt: true`<br>`kt_optimize_rule`、`cpu_infer`、`chunk_size` | CPU+GPU 异构训练。可用 2x4090+CPU 微调 1000B 模型。设置 `cpu_infer: 32`（CPU 核数） |
| **V1 Kernels** | `use_v1_kernels: true` | LlamaFactory 高性能训练 kernel，实验性功能 |

**建议**：
- LoRA 训练 → 优先 `use_unsloth: true`（效果最好的单一加速开关）
- 全参训练或 Unsloth 不支持的模型 → `enable_liger_kernel: true`
- Unsloth 和 Liger Kernel 可分别使用，选一个即可
- 极大模型受限硬件 → KTransformers 异构方案

---

#### H. 数据处理优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **序列打包（Packing）** | `packing: true` | 将多条短样本拼接到 `cutoff_len` 长度，减少 padding 浪费，提升 GPU 利用率 |
| **Neat Packing** | `neat_packing: true` | **推荐**。无交叉注意力的打包（block-diagonal attention），避免不同样本间信息泄露。自动启用 `packing: true` |
| **数据集流式加载** | `streaming: true` + `max_steps` | 超大数据集无需全量加载到内存，适合 TB 级数据 |
| **预处理并行** | `preprocessing_num_workers: N` | 多进程数据预处理。建议设为 CPU 核数的一半 |
| **预处理批大小** | `preprocessing_batch_size: 1000`（默认） | 适当增大可加速 tokenization |
| **预 tokenize** | `tokenized_path: /path/to/cache` | 预处理后缓存 tokenized 数据，重复训练免二次处理 |
| **DataLoader Workers** | `dataloader_num_workers: N` | 数据加载并行度。建议 4~8，Windows 设 0 |

**建议**：
- SFT 训练**必须**启用 `neat_packing: true`（避免 padding 浪费+防止交叉污染）
- 预训练自动启用 packing
- 大数据集使用 `streaming: true` 避免 OOM
- 设置 `preprocessing_num_workers` 和 `dataloader_num_workers` 加速数据管道

---

#### I. 模型与训练策略优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **NEFTune** | `neftune_noise_alpha: 5` | 噪声嵌入微调。在 embedding 层加入均匀噪声，提升 SFT 效果（来自 HF Trainer 原生支持） |
| **RoPE 缩放** | `rope_scaling: linear`（训练）<br>`rope_scaling: dynamic`（推理） | 扩展上下文长度。训练时用 linear，推理时用 dynamic |
| **Mixture-of-Depths** | `mixture_of_depths: convert` 或 `load` | 选择性计算，部分 token 跳过某些层，加速推理 |
| **LLaMA Pro** | 通过 block expansion 实现 | 块扩展方法，增加模型容量而不从头训练 |
| **Freeze Tuning** | `finetuning_type: freeze`<br>`freeze_trainable_layers`、`freeze_trainable_modules` | 冻结大部分层，仅训练顶层。显存略低于 LoRA 但效果一般 |
| **DFT Loss** | `use_dft_loss: true` | 动态焦点训练：按 `exp(-loss)` 加权每个 token 的 CE，聚焦模型预测差的 token。LlamaFactory 独有 |
| **ASFT Loss** | `use_asft_loss: true`<br>`asft_alpha: 0.1` | 自适应 SFT：DFT + 与参考模型的 KL 散度，平衡困难 token 聚焦与分布保持 |
| **EAFT Loss** | `use_eaft_loss: true`<br>`eaft_alpha: 1.0` | 熵自适应微调：用 top-20 logit 熵加权 token，高不确定性 token 获得更多训练信号 |
| **Mask History** | `mask_history: true` | 仅在最后一轮对话上计算 loss，多轮对话场景减少无效梯度 |
| **Train on Prompt** | `train_on_prompt: false`（默认） | 不在 prompt 部分计算 loss，节省计算量 |
| **Auto Batch Size** | `auto_find_batch_size: true` | OOM 时自动减小 batch size 重试 |

---

#### J. 推理加速

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **vLLM** | `infer_backend: vllm`<br>`vllm_maxlen`、`vllm_gpu_util` | 推理速度提升 **270%**。支持 LoRA 热加载 |
| **SGLang** | `infer_backend: sglang`<br>`sglang_maxlen`、`sglang_mem_fraction` | 结构化生成优化，高吞吐推理 |
| **KV Cache** | `use_kv_cache: true`（默认） | 推理时复用 KV Cache，关闭梯度检查点时自动启用 |

---

### 第四步：场景化配置模板

根据模型规模和硬件条件，推荐最优配置组合：

#### 场景 1：7B 模型 + 单卡 A100/H100 80GB — LoRA SFT

```yaml
### model
model_name_or_path: Qwen/Qwen3-8B
flash_attn: fa2
use_unsloth: true          # 170% 加速

### method
stage: sft
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_target: all

### dataset
dataset: your_dataset
neat_packing: true          # 高效打包
cutoff_len: 4096

### output
output_dir: output/qwen3-8b-lora

### train
per_device_train_batch_size: 8
gradient_accumulation_steps: 2
learning_rate: 1.0e-4
num_train_epochs: 3.0
bf16: true
neftune_noise_alpha: 5      # 提升 SFT 效果
```

#### 场景 2：7B 模型 + 单卡 24GB (RTX 4090) — QLoRA SFT

```yaml
### model
model_name_or_path: Qwen/Qwen3-8B
flash_attn: fa2
use_unsloth: true
quantization_bit: 4          # QLoRA 4bit，~6GB 模型显存
quantization_type: nf4

### method
stage: sft
finetuning_type: lora
lora_rank: 32
lora_alpha: 64
lora_target: all

### dataset
dataset: your_dataset
neat_packing: true
cutoff_len: 2048

### output
output_dir: output/qwen3-8b-qlora

### train
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
num_train_epochs: 3.0
bf16: true
```

#### 场景 3：70B 模型 + 8xA100 80GB — LoRA SFT + DeepSpeed ZeRO-2

```yaml
### model
model_name_or_path: Qwen/Qwen3-72B
flash_attn: fa2
enable_liger_kernel: true    # Liger Kernel 加速

### method
stage: sft
finetuning_type: lora
lora_rank: 64
lora_alpha: 128
lora_target: all

### dataset
dataset: your_dataset
neat_packing: true
cutoff_len: 4096

### output
output_dir: output/qwen3-72b-lora
deepspeed: examples/deepspeed/ds_z2_config.json

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 5.0e-5
num_train_epochs: 3.0
bf16: true
```

#### 场景 4：70B 模型 + 2x24GB — FSDP+QLoRA

```yaml
### model
model_name_or_path: Qwen/Qwen3-72B
flash_attn: fa2
quantization_bit: 4
quantization_type: nf4

### method
stage: sft
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all

### dataset
dataset: your_dataset
neat_packing: true
cutoff_len: 2048

### output
output_dir: output/qwen3-72b-fsdp-qlora

### train
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2.0e-4
num_train_epochs: 3.0
bf16: true

# 需配合 accelerate 启动：
# accelerate launch --config_file examples/accelerate/fsdp_config.yaml \
#   llamafactory-cli train config.yaml
```

#### 场景 5：7B 模型全参微调 + 8xA100 — DeepSpeed ZeRO-3

```yaml
### model
model_name_or_path: Qwen/Qwen3-8B
flash_attn: fa2
enable_liger_kernel: true

### method
stage: sft
finetuning_type: full

### dataset
dataset: your_dataset
neat_packing: true
cutoff_len: 4096

### output
output_dir: output/qwen3-8b-full
deepspeed: examples/deepspeed/ds_z3_config.json

### train
per_device_train_batch_size: 2
gradient_accumulation_steps: 4
learning_rate: 2.0e-5
num_train_epochs: 3.0
pure_bf16: true              # 全参训练用 pure_bf16 节省显存
```

---

### 第五步：输出审计报告

按以下格式输出审计结果：

```markdown
# LlamaFactory 训练优化审计报告

## 基本信息
- 模型：{model_name_or_path}（{参数量}）
- 训练阶段：{stage}
- 微调方式：{finetuning_type}
- 硬件：{GPU型号} x {数量}（{显存}GB）
- 序列长度：{cutoff_len}

## 审计结果总览

| 类别 | 得分 | 已启用 | 建议启用 |
|------|------|--------|----------|
| A. PEFT | x/10 | ... | ... |
| B. 混合精度 | x/6 | ... | ... |
| C. 激活重计算 | x/4 | ... | ... |
| D. 分布式优化 | x/7 | ... | ... |
| E. 高效优化器 | x/6 | ... | ... |
| F. 注意力优化 | x/4 | ... | ... |
| G. 加速引擎 | x/4 | ... | ... |
| H. 数据处理 | x/7 | ... | ... |
| I. 训练策略 | x/7 | ... | ... |
| J. 推理加速 | x/3 | ... | ... |
| **总计** | **x/58** | | |

## 优先优化建议（按影响排序）

### P0 - 立即执行（显著收益，零风险）
1. ...

### P1 - 强烈推荐（明显收益，低风险）
1. ...

### P2 - 建议尝试（中等收益，需测试）
1. ...

## 推荐配置修改

（给出具体的 YAML 配置修改 diff）
```

---

## 常见优化决策树

```
开始
├── 显存不足（OOM）？
│   ├── 是 → 当前用全参训练？
│   │   ├── 是 → 切换为 LoRA（finetuning_type: lora）
│   │   │   └── 仍然 OOM？ → 启用 QLoRA（quantization_bit: 4）
│   │   └── 否（已用 LoRA）
│   │       ├── 启用梯度检查点（gradient_checkpointing: true）
│   │       ├── 减小 batch size + 增大 gradient_accumulation_steps
│   │       ├── 启用 use_unsloth_gc: true
│   │       └── 仍然 OOM？ → 降低 quantization_bit / 减小 cutoff_len / 使用 FSDP+QLoRA
│   └── 否 → 继续性能优化
├── 训练速度慢？
│   ├── flash_attn 已启用？
│   │   ├── 否 → 设置 flash_attn: fa2
│   │   └── 是 → 继续
│   ├── LoRA 训练？ → 启用 use_unsloth: true
│   ├── 全参训练？ → 启用 enable_liger_kernel: true
│   ├── 数据已打包？
│   │   ├── 否 → 设置 neat_packing: true
│   │   └── 是 → 继续
│   ├── 多卡训练？ → 检查 DeepSpeed/FSDP 配置
│   │   ├── LoRA → ZeRO-2（开销最低）
│   │   └── 全参 → ZeRO-3 或 FSDP FULL_SHARD
│   └── 检查 DataLoader workers（dataloader_num_workers: 4~8）
└── 训练效果差？
    ├── 启用 neftune_noise_alpha: 5
    ├── 调整 LoRA rank（增大到 64~128）
    ├── 使用 use_dora: true
    ├── 多轮对话设 mask_history: true
    └── 确保 train_on_prompt: false
```

## 显存估算参考

| 方法 | 精度 | 7B | 14B | 30B | 70B |
|------|------|-----|-----|-----|------|
| 全参（bf16/fp16） | 32-bit 有效 | 120GB | 240GB | 600GB | 1200GB |
| 全参（pure_bf16） | 16-bit | 60GB | 120GB | 300GB | 600GB |
| LoRA/Freeze/GaLore/BAdam | 16-bit | 16GB | 32GB | 64GB | 160GB |
| QLoRA | 8-bit | 10GB | 20GB | 40GB | 80GB |
| QLoRA | 4-bit | 6GB | 12GB | 24GB | 48GB |
| QLoRA | 2-bit | 4GB | 8GB | 16GB | 24GB |

## LlamaFactory 特有关键词速查

用于识别 LlamaFactory 项目的关键词和文件模式：

| 类别 | 关键词/模式 |
|------|------------|
| CLI | `llamafactory-cli`、`llamafactory.cli` |
| 配置 | `finetuning_type`、`lora_target`、`lora_rank`、`neat_packing`、`use_unsloth`、`enable_liger_kernel`、`flash_attn: fa2`、`quantization_bit`、`neftune_noise_alpha`、`shift_attn`、`use_dora` |
| 文件 | `data/dataset_info.json`、`examples/train_lora/`、`examples/deepspeed/`、`examples/accelerate/` |
| import | `from llamafactory`、`import llamafactory`、`llamafactory-cli train` |
| 依赖 | `llamafactory`、`llama-factory`、`LLaMA-Factory` |
| DeepSpeed | `ds_z0_config.json`、`ds_z2_config.json`、`ds_z3_config.json`、`ds_z3_offload_config.json`、`ds_z2_autotp_config.json`、`ds_z3_fp8_config.json` |
| Accelerate | `fsdp_config.yaml`、`fsdp_config_offload.yaml`、`fsdp2_config.yaml`、`fsdp_multi_node_config.yaml` |
