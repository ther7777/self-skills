---
name: flow-factory-optimization
description: Flow-Factory 训练优化审计
user-invocable: false
---

## 描述
针对基于 Flow-Factory（X-GenGroup/Flow-Factory）框架的视觉生成模型 RL 训练项目，快速识别已采用和未采用的优化手段，提供针对性的显存优化与性能加速建议。Flow-Factory 是专注于 Diffusion/Flow-Matching 模型 RL 微调的统一框架，支持 GRPO、DPO、NFT 等 8 种 RL 算法和 20+ 模型变体（Wan、Flux、SD3.5、LTX-2 等）。

## 触发条件
当识别到用户项目满足以下任一条件时自动触发：
- 代码中 import 了 `flow_factory`（如 `from flow_factory`、`import flow_factory`）
- 启动命令使用 `ff-train` 或 `flow-factory-train`
- 配置文件（YAML）中包含 Flow-Factory 特有参数：`trainer_type`（值为 `grpo`/`nft`/`awm`/`dpo`/`dgpo`/`crd`/`diffusion-opd`）、`model_type`（值为 `wan2_t2v`/`flux1`/`sd3-5` 等）、`dynamics_type`（值为 `Flow-SDE`/`Dance-SDE`/`CPS`/`ODE`）
- 配置文件中包含 `rewards:` 列表且包含 `reward_model:` 条目
- 依赖文件中包含 `flow-factory` 或 `flow_factory`
- 项目结构包含 `src/flow_factory/` 目录，或 `config/accelerate_configs/`、`config/deepspeed/` 目录
- Git remote URL 包含 `Flow-Factory`

## 执行指令

你是 Flow-Factory 视觉生成模型 RL 训练优化专家。被调用时，请按照以下清单对目标项目进行全面审计。对每个检查项，判定其状态：**已启用** / **未启用** / **建议启用** / **不适用**，并给出具体的配置修改建议。

### 第一步：识别训练模式与基础信息

扫描项目的 YAML 配置文件和启动脚本，确定：

1. **RL 算法**（`trainer_type`）：`grpo` / `nft` / `awm` / `dpo` / `dgpo` / `crd` / `diffusion-opd` / `grpo-guard`
2. **模型类型**（`model.model_type`）：`wan2_t2v` / `wan2_i2v` / `flux1` / `flux1-kontext` / `flux2` / `flux2-klein` / `sd3-5` / `ltx2_t2av` / `bagel` / `qwen-image` / `z-image` 等
3. **微调方式**（`model.finetune_type`）：`full`（全参）/ `lora`
4. **模型规模**：从 `model.model_path` 推断（如 Wan2.1-T2V-1.3B/14B、Flux.1-dev 13B、LTX-2 19B 等）
5. **硬件环境**：GPU 型号/数量、显存大小（结合 `/system-resources` 结果）
6. **分布式后端**：从 `config_file` 推断（FSDP2/FSDP/DeepSpeed ZeRO-1/2/3）
7. **生成任务类型**：文生图 / 图生图 / 文生视频 / 图生视频 / 音视频生成
8. **分辨率与帧数**：`train.resolution` 和 `train.num_frames`

---

### 第二步：优化审计

#### A. 混合精度与数据类型

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **BF16 混合精度** | `mixed_precision: "bf16"` | Ampere+(A100/H100/H20) 首选。数值稳定，loss scale 免调 |
| **FP16 混合精度** | `mixed_precision: "fp16"` | V100/T4 使用。注意溢出风险 |
| **Master Weight Dtype** | `model.master_weight_dtype: "bfloat16"/"fp32"` | 可训练参数的存储精度。BF16 可节省一半参数显存 |
| **Latent 存储精度** | `train.latent_storage_dtype: "fp16"` | 轨迹 latent 存储精度。FP16 比 FP32 省 50% 显存 |
| **推理 Dtype** | `model.infer_dtype: "bfloat16"` | 模型推理时的精度（采样阶段） |

**建议**：
- A100/H100/H20 必须设置 `mixed_precision: "bf16"`
- V100 使用 `mixed_precision: "fp16"`
- 确保 `latent_storage_dtype: "fp16"` 以减少轨迹存储显存
- 大模型可考虑 `master_weight_dtype: "bfloat16"`（牺牲少量精度换显存）

---

#### B. 梯度检查点与显存管理

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **梯度检查点** | `train.enable_gradient_checkpointing: true` | 以 ~30% 计算换 ~60% 激活显存。视频模型必开 |
| **CPU Sample Offload** | `train.offload_samples_to_cpu: true` | 将 sample tensors 在 sample() 和 optimize() 之间卸载到 CPU。**视频模型必须启用** |
| **EMA 设备** | `train.ema_device: "cpu"` | 将 EMA 参数存放 CPU 节省 GPU 显存。大模型推荐 |
| **参考模型设备** | `train.ref_param_device: "cpu"` | 将参考模型参数放 CPU。DPO/CRD 等需要参考模型的算法适用 |
| **梯度累积** | `train.gradient_accumulation_steps` / `train.gradient_step_per_epoch` | `"auto"` 根据 epoch 设置自动推导。大 batch 减少通信频率 |

**建议**：
- 视频模型训练**必须**同时启用 `enable_gradient_checkpointing: true` 和 `offload_samples_to_cpu: true`
- 使用 EMA 时设 `ema_device: "cpu"` 节省显存（EMA 参数等同模型大小）
- DPO/CRD 等双模型算法设 `ref_param_device: "cpu"`

---

#### C. LoRA 配置

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **LoRA 微调** | `model.finetune_type: "lora"` | 显存大幅降低。RL 微调推荐 LoRA |
| **LoRA Rank** | `model.lora_rank: 8`（默认）<br>示例中常用 32~128 | Rank 越大效果越好但显存越多。RL 微调建议 32~128 |
| **LoRA Alpha** | `model.lora_alpha`（默认 `2 * rank`） | 通常设为 `2 * rank` 或等于 rank |
| **Target Modules** | `model.target_modules: "all"/"default"/[list]` | `all` 覆盖全部线性层；`default` 使用模型预设 |
| **Target Components** | `model.target_components: "transformer"` 或 `["transformer", "transformer_2"]` | Wan2.2 双 transformer 模型需指定两个组件 |

**建议**：
- 显存有限首选 LoRA（`finetune_type: "lora"`）
- RL 微调推荐 `lora_rank: 64~128`（比 SFT 需更大 rank 以保证探索能力）
- Wan2.2 模型确保 `target_components` 包含所有需训练的 transformer

---

#### D. 分布式训练

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **FSDP2** | `config_file: config/accelerate_configs/fsdp2.yaml` | **推荐方案**。PyTorch 原生，性能好，多节点友好 |
| **FSDP Full Shard** | `config_file: config/accelerate_configs/fsdp_full_shard.yaml` | FSDP v1，可同时 shard 两个 transformer |
| **DeepSpeed ZeRO-2** | `config_file: config/deepspeed/deepspeed_zero2.yaml` | LoRA 多卡默认选择。梯度+优化器分片 |
| **DeepSpeed ZeRO-3** | `config_file: config/deepspeed/deepspeed_zero3.yaml` | 全参微调大模型。参数+梯度+优化器全分片 |
| **ZeRO-2 + CPU Offload** | `config_file: config/deepspeed/deepspeed_zero2_offparam.yaml` | 参数卸载到 CPU，极低显存场景 |
| **DeepSpeed ZeRO-1** | `config_file: config/deepspeed/deepspeed_zero1.yaml` | 仅优化器分片 |
| **多节点** | `multinode_examples/` 启动脚本<br>环境变量 `MASTER_ADDR`、`WORLD_SIZE` | 多机多卡训练 |
| **GPU 数量** | `num_processes: N` | 参与训练的 GPU 数 |

**建议**：
- 单节点多卡 LoRA → DeepSpeed ZeRO-2 或 FSDP2
- 全参微调大模型（14B+）→ FSDP2 或 DeepSpeed ZeRO-3
- 多节点 → FSDP2（推荐）
- 极低显存 → ZeRO-2 + CPU Offload

---

#### E. 注意力后端

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Flash Attention 2** | `model.attn_backend: "flash"` 或 `"flash_hub"` | **高优先级**。通过 `kernels` 包安装，避免编译 flash-attn |
| **Flash Attention 3** | `model.attn_backend: "_flash_3"` 或 `"_flash_3_hub"` | Hopper GPU（H100/H200）专用。性能更佳 |
| **Flash Varlen** | `model.attn_backend: "flash_varlen_hub"` / `"_flash_3_varlen_hub"` | 变长序列场景 |
| **SageAttention** | `model.attn_backend: "sage"` | SageAttention 包。8-bit 注意力 |
| **xformers** | `model.attn_backend: "xformers"` | xformers 包。兼容性好 |
| **SDPA（默认）** | `model.attn_backend: null`（默认） | PyTorch 原生 SDPA。不设置 attn_backend 时使用 |

**建议**：
- Ampere+ GPU 必须设置 `attn_backend: "flash_hub"`（推荐通过 `pip install kernels` 安装）
- Hopper GPU 使用 `attn_backend: "_flash_3_hub"` 获得最佳性能
- 未设置 attn_backend 是常见遗漏，**默认 SDPA 性能不如 Flash Attention**

---

#### F. 优化器与训练策略

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **学习率** | `train.learning_rate: 1e-5`（默认） | RL 微调建议 1e-6 ~ 1e-5 |
| **Adam 参数** | `train.adam_betas: [0.9, 0.999]`<br>`train.adam_weight_decay: 1e-4`<br>`train.adam_epsilon: 1e-8` | 标准配置。weight_decay 过大可能影响 RL 探索 |
| **梯度裁剪** | `train.max_grad_norm: 1.0`（默认） | RL 训练梯度波动大，建议保留 |
| **Epochs** | `train.num_epochs: 500` | RL 训练的 epoch 数（每 epoch = 一次 sample + optimize） |
| **Sample Group Size** | `train.sample_group_size` | GRPO 等算法每 prompt 生成的样本数 |
| **LR Scheduler** | **Flow-Factory 无内置 LR 调度器** | 仅支持常量学习率。这是一个优化缺口 |

**建议**：
- 保持 `max_grad_norm: 1.0`（RL 训练梯度不稳定时可降至 0.5）
- Flow-Factory 当前仅支持常量学习率。如需 warmup/cosine，需在 fork 中自行实现
- GRPO 推荐 `sample_group_size: 4~8`（更多样本 = 更稳定的 advantage 估计 = 更多显存）

---

#### G. 数据管道

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **DataLoader Workers** | `train.dataloader_num_workers: 16`（默认） | 数据加载并行度。根据 CPU 核数调整 |
| **预处理 Batch Size** | `train.preprocessing_batch_size: 8`（默认） | 预处理时的 batch 大小 |
| **采样器策略** | `train.sampler: "auto"`（默认） | `auto` 优先 `group_contiguous`（同 rank 内聚合），减少跨 rank 通信 |
| **数据缓存** | `data.cache_dir: "~/.cache/flow_factory/datasets"` | 预处理结果缓存。避免重复 tokenize/编码 |
| **强制重新处理** | `data.force_reprocess: false`（默认） | 数据变更后需设为 `true` |
| **预处理并行模式** | `data.preprocess_parallelism: "global"/"local"` | `local` 无需共享文件系统。多节点推荐 `local` |
| **多源数据集** | `data.datasets: [...]` + `weight` | 多数据源加权混合 |

**建议**：
- 确保 `dataloader_num_workers` 与 CPU 核数匹配（建议 8~16）
- 视频数据预处理慢时增大 `preprocessing_batch_size`
- 多节点无共享存储 → `preprocess_parallelism: "local"`
- 利用缓存避免每次重新编码（确认 `force_reprocess: false`）

---

#### H. 奖励系统优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **异步奖励** | `rewards[*].async_reward: true`<br>`rewards[*].num_workers: 4` | API-based 奖励模型（如 VLM-as-judge）使用异步计算 |
| **奖励 Batch Size** | `rewards[*].batch_size` | 减小奖励计算的 batch size 可避免 OOM |
| **奖励设备** | `rewards[*].device: "cuda"/"cpu"` | 大奖励模型可放 CPU（牺牲速度换显存） |
| **奖励 Dtype** | `rewards[*].dtype: "bfloat16"` | BF16 计算奖励节省显存 |
| **多源路由** | `rewards[*].applicable_datasets` | 不同数据集使用不同奖励模型 |
| **Advantage 处理** | `train.advantage_type: "sum"/"gdpo"` | `gdpo` 使用 GDPO 风格的 advantage 计算 |

**建议**：
- VLM-as-judge 等慢奖励 → `async_reward: true` + `num_workers: 4~8`
- 奖励模型 OOM → 减小 `batch_size` 或设 `device: "cpu"`
- 多奖励模型确保使用 BF16 精度（`dtype: "bfloat16"`）

---

#### I. EMA 与调度器

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **EMA 衰减** | `train.ema_decay: 0.995`（默认） | 0 = 禁用。RL 训练推荐启用 EMA |
| **EMA 更新间隔** | `train.ema_update_interval: 10`（默认） | 每 N epoch 更新一次 EMA |
| **EMA 设备** | `train.ema_device: "cpu"/"cuda"` | CPU 节省显存但增加通信开销 |
| **EMA 衰减调度** | `train.ema_decay_schedule: "power"/"constant"/"cosine"/"warmup_cosine"` | `power` 是默认，`warmup_cosine` 适合长训练 |
| **Dynamics 类型** | `scheduler.dynamics_type: "Flow-SDE"/"Dance-SDE"/"CPS"/"ODE"` | 控制 RL 轨迹采样的动力学类型 |
| **Flow Shift** | `scheduler.flow_shift: 3.0` | Wan 模型特有。480P 用 3.0，720P 用 5.0 |
| **SDE Steps** | `scheduler.sde_steps: [1, 2, 3]`<br>`scheduler.num_sde_steps: 1` | 哪些步骤注入 SDE 噪声、每次 rollout 的 SDE 步数 |

**建议**：
- RL 训练建议启用 EMA（`ema_decay: 0.995`），用于评估和保存最终模型
- 显存紧张时 `ema_device: "cpu"`
- Wan 720P 训练确保 `flow_shift: 5.0`（默认 3.0 适合 480P）

---

#### J. 深度优化机会

以下是 Flow-Factory 当前未内置但可通过代码修改实现的高收益优化：

| 检查项 | 当前状态 | 优化方式 | 预期收益 |
|---|---|---|---|
| **torch.compile** | 未集成 | 在 fork 中对 transformer 模型添加 `torch.compile(model, mode="reduce-overhead")` | 算子融合，吞吐 +10~30% |
| **FusedAdam** | 仅标准 AdamW | 替换为 `torch.optim.AdamW(fused=True)` 或 DeepSpeed FusedAdam | Kernel 调用减少，速度 +5~10% |
| **8-bit Adam** | 未支持 | 使用 `bitsandbytes.optim.AdamW8bit` | 优化器显存 -75% |
| **Liger Kernel** | 未集成 | 针对模型中的 RMSNorm/SwiGLU 等层应用 Liger Kernel | 吞吐 +20%，显存 -60% |
| **QLoRA** | 未支持 | 使用 `bitsandbytes` 4-bit 量化基座模型 + LoRA | 模型显存 -75% |
| **LR Scheduler** | 仅常量 LR | 在 fork 中添加 cosine/warmup scheduler | 训练稳定性和收敛质量提升 |
| **Gradient Accumulation 优化** | 基础支持 | 结合 `torch.no_sync()` 延迟 allreduce | 多卡通信开销降低 |
| **DataLoader 优化** | 基础配置 | 添加 `pin_memory=True`、`prefetch_factor=2`、`persistent_workers=True` | 数据加载速度提升 |
| **NCCL 环境变量** | 未调优 | 添加 `NCCL_IB_DISABLE=0`、`NCCL_BUFFSIZE=16777216` 等 | 多机通信速度 +20~40% |
| **ZeRO++ 量化通信** | 未启用 | DeepSpeed 配置 `zero_quantized_gradients` / `zero_quantized_weights` | 跨节点通信量 -50~75% |

**建议**：
- **P0**（零风险，立即可做）：`attn_backend: "flash_hub"`、`mixed_precision: "bf16"`、`offload_samples_to_cpu: true`、`enable_gradient_checkpointing: true`
- **P1**（低风险，需测试）：`torch.optim.AdamW(fused=True)`、DataLoader 优化、NCCL 调优
- **P2**（需改代码）：torch.compile、Liger Kernel、8-bit Adam、QLoRA、LR scheduler

---

### 第三步：场景化配置模板

#### 场景 1：Wan2.1-T2V-1.3B + 8xA100 — GRPO LoRA（入门配置）

```yaml
launcher: "accelerate"
config_file: config/accelerate_configs/fsdp2.yaml
num_processes: 8
mixed_precision: "bf16"

model:
  model_path: Wan-AI/Wan2.1-T2V-1.3B
  model_type: "wan2_t2v"
  finetune_type: "lora"
  lora_rank: 64
  target_modules: "all"
  attn_backend: "flash_hub"          # 必须启用
  master_weight_dtype: "bfloat16"

train:
  trainer_type: "grpo"
  num_epochs: 500
  learning_rate: 1e-5
  resolution: [384, 720]
  num_frames: 17
  enable_gradient_checkpointing: true  # 必须启用
  offload_samples_to_cpu: true         # 视频模型必须
  latent_storage_dtype: "fp16"
  sample_group_size: 4
  max_grad_norm: 1.0
  ema_decay: 0.995
  ema_device: "cpu"                    # 节省显存
  dataloader_num_workers: 8

scheduler:
  dynamics_type: "Flow-SDE"
  flow_shift: 3.0                      # 480P 用 3.0
```

#### 场景 2：Wan2.1-T2V-14B + 8xH100 — GRPO LoRA（大模型）

```yaml
launcher: "accelerate"
config_file: config/deepspeed/deepspeed_zero2.yaml
num_processes: 8
mixed_precision: "bf16"

model:
  model_path: Wan-AI/Wan2.1-T2V-14B
  model_type: "wan2_t2v"
  finetune_type: "lora"
  lora_rank: 128
  target_modules: "all"
  attn_backend: "_flash_3_hub"         # H100 用 FA3
  master_weight_dtype: "bfloat16"

train:
  trainer_type: "grpo"
  num_epochs: 300
  learning_rate: 5e-6
  resolution: [384, 720]
  num_frames: 33
  enable_gradient_checkpointing: true
  offload_samples_to_cpu: true
  latent_storage_dtype: "fp16"
  sample_group_size: 4
  max_grad_norm: 1.0
  ema_decay: 0.995
  ema_device: "cpu"
  ref_param_device: "cpu"
  dataloader_num_workers: 8
```

#### 场景 3：Flux.1-dev 13B + 4xA100 — GRPO LoRA

```yaml
launcher: "accelerate"
config_file: config/accelerate_configs/fsdp2.yaml
num_processes: 4
mixed_precision: "bf16"

model:
  model_path: black-forest-labs/FLUX.1-dev
  model_type: "flux1"
  finetune_type: "lora"
  lora_rank: 64
  target_modules: "all"
  attn_backend: "flash_hub"

train:
  trainer_type: "grpo"
  num_epochs: 500
  learning_rate: 1e-5
  resolution: 512
  enable_gradient_checkpointing: true
  offload_samples_to_cpu: true
  sample_group_size: 4
  ema_decay: 0.995
  ema_device: "cpu"
```

#### 场景 4：Wan2.1-T2V-14B + 8xA100 — 全参微调 + ZeRO-3

```yaml
launcher: "accelerate"
config_file: config/deepspeed/deepspeed_zero3.yaml
num_processes: 8
mixed_precision: "bf16"

model:
  model_path: Wan-AI/Wan2.1-T2V-14B
  model_type: "wan2_t2v"
  finetune_type: "full"
  target_modules: "all"
  attn_backend: "flash_hub"
  master_weight_dtype: "bfloat16"

train:
  trainer_type: "grpo"
  num_epochs: 200
  learning_rate: 1e-6
  resolution: [384, 720]
  num_frames: 17
  enable_gradient_checkpointing: true
  offload_samples_to_cpu: true
  latent_storage_dtype: "fp16"
  max_grad_norm: 0.5
  ema_decay: 0.995
  ema_device: "cpu"
  gradient_accumulation_steps: 4
```

#### 场景 5：LTX-2 音视频 + 8xH100 — NFT LoRA

```yaml
launcher: "accelerate"
config_file: config/accelerate_configs/fsdp2.yaml
num_processes: 8
mixed_precision: "bf16"

model:
  model_path: Lightricks/LTX-Video-0.9.7-dev
  model_type: "ltx2_t2av"
  finetune_type: "lora"
  lora_rank: 64
  target_modules: "all"
  attn_backend: "_flash_3_hub"

train:
  trainer_type: "nft"
  num_epochs: 300
  learning_rate: 1e-5
  resolution: 512
  num_frames: 33
  enable_gradient_checkpointing: true
  offload_samples_to_cpu: true
  ema_decay: 0.995
  ema_device: "cpu"
```

---

### 第四步：输出审计报告

按以下格式输出审计结果：

```markdown
# Flow-Factory 训练优化审计报告

## 基本信息
- 模型：{model_path}（{参数量}）
- 模型类型：{model_type}
- RL 算法：{trainer_type}
- 微调方式：{finetune_type}
- 硬件：{GPU型号} x {数量}（{显存}GB）
- 分辨率/帧数：{resolution} / {num_frames}
- 分布式后端：{FSDP2/DeepSpeed ZeRO-N}

## 审计结果总览

| 类别 | 得分 | 已启用 | 建议启用 |
|------|------|--------|----------|
| A. 混合精度与数据类型 | x/5 | ... | ... |
| B. 梯度检查点与显存管理 | x/5 | ... | ... |
| C. LoRA 配置 | x/5 | ... | ... |
| D. 分布式训练 | x/6 | ... | ... |
| E. 注意力后端 | x/4 | ... | ... |
| F. 优化器与训练策略 | x/5 | ... | ... |
| G. 数据管道 | x/5 | ... | ... |
| H. 奖励系统优化 | x/4 | ... | ... |
| I. EMA 与调度器 | x/4 | ... | ... |
| J. 深度优化机会 | x/6 | ... | ... |
| **总计** | **x/49** | | |

## 优先优化建议（按影响排序）

### P0 - 立即执行（显著收益，零风险）
1. ...

### P1 - 强烈推荐（明显收益，低风险）
1. ...

### P2 - 建议尝试（中等收益，需测试）
1. ...

## 推荐配置修改

（给出具体的 YAML 配置修改 diff 或代码修改）
```

---

## 常见优化决策树

```
开始
├── 显存不足（OOM）？
│   ├── 是 → 当前用全参训练？
│   │   ├── 是 → 切换为 LoRA（finetune_type: "lora"）
│   │   └── 否（已用 LoRA）
│   │       ├── enable_gradient_checkpointing: true
│   │       ├── offload_samples_to_cpu: true
│   │       ├── ema_device: "cpu"
│   │       ├── ref_param_device: "cpu"（如有参考模型）
│   │       ├── 降低 lora_rank（128→64→32）
│   │       ├── 减小 resolution 或 num_frames
│   │       ├── 减小 sample_group_size
│   │       └── 仍然 OOM? → DeepSpeed ZeRO-2 + CPU Offload
│   └── 否 → 继续性能优化
├── 训练速度慢？
│   ├── attn_backend 已设置？
│   │   ├── 否 → 设置 attn_backend: "flash_hub"
│   │   └── 是 → 继续
│   ├── mixed_precision 已启用？
│   │   ├── 否 → 设置 mixed_precision: "bf16"
│   │   └── 是 → 继续
│   ├── 多卡训练？ → 检查 FSDP/DeepSpeed 配置
│   │   ├── LoRA → DeepSpeed ZeRO-2
│   │   └── 全参 → FSDP2 或 ZeRO-3
│   ├── 检查 dataloader_num_workers（建议 8~16）
│   └── 代码级优化：torch.compile、FusedAdam
├── RL 训练不稳定（reward 不升 / loss 震荡）？
│   ├── 降低 learning_rate（1e-5 → 5e-6 → 1e-6）
│   ├── 降低 max_grad_norm（1.0 → 0.5）
│   ├── 增大 sample_group_size（更稳定的 advantage）
│   ├── 启用 EMA（ema_decay: 0.995）
│   └── 检查 reward 模型是否正常工作
└── 奖励计算慢？
    ├── 大 VLM 奖励模型 → async_reward: true + num_workers: 4~8
    ├── 奖励模型 OOM → 减小 batch_size 或 device: "cpu"
    └── 多奖励模型 → 确保 dtype: "bfloat16"
```

## 显存估算参考

| 模型 | 方法 | 精度 | 分辨率×帧数 | 预估显存/GPU |
|------|------|------|-----------|-------------|
| Wan2.1-T2V-1.3B | LoRA r=64 | BF16 | 384×720, 17帧 | ~30GB |
| Wan2.1-T2V-14B | LoRA r=64 | BF16 | 384×720, 17帧 | ~60GB |
| Wan2.1-T2V-14B | Full | BF16+ZeRO-3 | 384×720, 17帧 | ~70GB |
| Flux.1-dev 13B | LoRA r=64 | BF16 | 512 | ~40GB |
| LTX-2 19B | LoRA r=64 | BF16 | 512, 33帧 | ~60GB |

> 以上为 GRPO + sample_group_size=4 + gradient_checkpointing + CPU offload 的估算值，实际显存取决于配置。

## Flow-Factory 特有关键词速查

用于识别 Flow-Factory 项目的关键词和文件模式：

| 类别 | 关键词/模式 |
|------|------------|
| CLI | `ff-train`、`flow-factory-train` |
| 配置 | `trainer_type`（grpo/nft/awm/dpo/dgpo/crd/diffusion-opd）、`model_type`（wan2_t2v/flux1/sd3-5/ltx2_t2av 等）、`dynamics_type`（Flow-SDE/Dance-SDE/CPS/ODE）、`finetune_type`、`sample_group_size`、`offload_samples_to_cpu` |
| import | `from flow_factory`、`import flow_factory` |
| 依赖 | `flow-factory`、`flow_factory` |
| 项目结构 | `src/flow_factory/`、`config/accelerate_configs/`、`config/deepspeed/` |
| YAML 结构 | 顶层含 `data:`、`model:`、`train:`、`eval:`、`scheduler:`、`rewards:` 嵌套块 |
| 奖励系统 | `rewards:` 列表、`reward_model:`、`async_reward`、`advantage_type` |
| RL 特有 | `sample_group_size`、`ema_decay`、`ref_param_device`、`latent_storage_dtype` |
