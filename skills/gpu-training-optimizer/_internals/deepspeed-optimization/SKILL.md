---
name: deepspeed-optimization
description: DeepSpeed ZeRO 显存优化与训练加速
---

# Skill: DeepSpeed 显存优化与训练加速

## 描述
指导用户使用 DeepSpeed 进行 LLM/MLLM 训练的显存优化和加速，涵盖 ZeRO Stage 选型、混合精度、CPU/NVMe 卸载、通信优化、激活重计算等全链路配置，目标是根据用户的硬件条件和模型规模选择最优的 DeepSpeed 配置方案。

## 触发条件
当用户需要使用 DeepSpeed 进行训练加速或显存优化时触发。适用场景包括：选择 ZeRO Stage、配置 CPU/NVMe Offload、启用混合精度训练、优化多卡通信效率、解决 OOM 问题、为大模型训练选择最优配置方案等。

## 执行指令

你是 DeepSpeed 分布式训练优化专家。根据用户的模型规模、GPU 硬件和训练需求，指导用户选择并配置最优的 DeepSpeed 方案。

---

### 第一步：评估场景与选型

#### 1.1 关键输入信息

在给出建议前，先确认以下信息：

| 信息 | 作用 | 示例 |
|------|------|------|
| 模型参数量 | 决定 ZeRO Stage | 7B / 13B / 70B |
| GPU 型号与数量 | 决定显存预算和通信方案 | 8x A100-80G / 4x V100-32G |
| 单机 vs 多机 | 决定通信优化策略 | 1 node / 4 nodes |
| 训练类型 | 决定是否需要全量参数更新 | 预训练 / 全量微调 / LoRA |
| 序列长度 | 影响激活显存 | 2048 / 8192 / 128K |
| 目标 batch size | 影响梯度累积配置 | 256 / 1024 |

#### 1.2 ZeRO Stage 选型决策树

```
模型能否放进单卡显存？
├── 能（含优化器状态）→ ZeRO Stage 0（或不用 DeepSpeed）
├── 模型能放但优化器放不下 → ZeRO Stage 1（分片优化器状态）
├── 仍不够 → ZeRO Stage 2（分片优化器 + 梯度）
├── 仍 OOM → ZeRO Stage 2 + CPU Offload
├── 仍不够 → ZeRO Stage 3（分片优化器 + 梯度 + 参数）
├── 仍 OOM → ZeRO Stage 3 + CPU Offload
└── CPU 内存也不够 → ZeRO Stage 3 + NVMe Offload（ZeRO-Infinity）
```

**各 Stage 显存节省对比（以 Adam 优化器、FP16 训练为例）：**

| Stage | 分片内容 | 每 GPU 显存（相对 DDP） | 通信量（相对 DDP） |
|-------|---------|----------------------|-------------------|
| **0** | 无 | 1x（等同 DDP） | 1x |
| **1** | 优化器状态 | ~0.25x（显著减少） | 1x |
| **2** | 优化器 + 梯度 | ~0.125x | 1x |
| **3** | 优化器 + 梯度 + 参数 | 线性随 GPU 数降低 | ~1.5x（额外参数收集） |

> **经验法则**：优先使用低 Stage（性能更好），只在显存不够时升级 Stage。Stage 1→2 通信量不变，通常优先尝试 Stage 2。Stage 3 有额外通信开销，但可训练最大模型。

#### 1.3 显存估算速查表

对于 FP16/BF16 混合精度 + Adam 优化器：

| 组件 | 每参数显存 | 10B 模型 |
|------|----------|---------|
| FP16 参数 | 2 bytes | 20 GB |
| FP16 梯度 | 2 bytes | 20 GB |
| FP32 优化器（Adam：权重 + 动量 + 方差） | 12 bytes | 120 GB |
| **总计** | **16 bytes** | **160 GB** |
| **+ 激活显存（取决于序列长度和 batch）** | 变化大 | 10-100+ GB |

ZeRO Stage 分片后：
- Stage 1（N GPU）：参数 20G + 梯度 20G + 优化器 120G/N
- Stage 2（N GPU）：参数 20G + (梯度 20G + 优化器 120G)/N
- Stage 3（N GPU）：(参数 20G + 梯度 20G + 优化器 120G)/N

---

### 第二步：DeepSpeed 配置模板

#### 模板 A：ZeRO Stage 2（最常用，推荐起步）

适用：模型参数量 < 单卡显存 × GPU 数量 ÷ 2（如 8x A100-80G 训练 13B 模型）

```json
{
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
    "gradient_accumulation_steps": "auto",

    "bf16": {
        "enabled": true
    },

    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },

    "gradient_clipping": 1.0,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "betas": "auto",
            "eps": "auto",
            "weight_decay": "auto"
        }
    },

    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto"
        }
    }
}
```

> `"auto"` 值在 HuggingFace Trainer 中会自动从 `TrainingArguments` 中读取，非 HF 场景需填具体值。

#### 模板 B：ZeRO Stage 2 + CPU Offload

适用：Stage 2 仍 OOM，但希望保持较好的训练速度

```json
{
    "bf16": { "enabled": true },

    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "round_robin_gradients": true
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    "gradient_clipping": 1.0
}
```

**关键点：**
- `offload_optimizer.pin_memory: true` — 使用 page-locked CPU 内存，提高 CPU↔GPU 传输速度
- `round_robin_gradients: true` — 多 rank 并行拷贝梯度到 CPU，加速 offload

#### 模板 C：ZeRO Stage 3（大模型必选）

适用：模型参数量很大，单卡放不下完整模型（如 70B 模型）

```json
{
    "bf16": { "enabled": true },

    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_bucket_size": 1e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    "gradient_clipping": 1.0
}
```

**Stage 3 关键参数调优：**

| 参数 | 作用 | 调优建议 |
|------|------|---------|
| `stage3_prefetch_bucket_size` | 预取参数的 buffer 大小 | 减小可降低显存，但可能降低吞吐 |
| `stage3_param_persistence_threshold` | 小于此值的参数常驻 GPU 不分片 | 默认 1e5 适合大多数场景 |
| `stage3_max_live_parameters` | GPU 上最大常驻参数量 | OOM 时减小 |
| `stage3_max_reuse_distance` | 参数复用距离阈值 | OOM 时减小 |
| `stage3_gather_16bit_weights_on_model_save` | 保存时聚合完整权重 | 需要保存 checkpoint 时开启 |

#### 模板 D：ZeRO Stage 3 + CPU Offload

适用：GPU 显存严重不足，愿意牺牲速度换取可训练性

```json
{
    "bf16": { "enabled": true },

    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_bucket_size": 1e7,
        "stage3_prefetch_bucket_size": 1e7,
        "stage3_param_persistence_threshold": 1e5,
        "sub_group_size": 1e9,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "stage3_gather_16bit_weights_on_model_save": true
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    "gradient_clipping": 1.0
}
```

#### 模板 E：ZeRO Stage 3 + NVMe Offload（ZeRO-Infinity，极限场景）

适用：GPU + CPU 内存都不够，需要利用 NVMe SSD

```json
{
    "bf16": { "enabled": true },

    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_bucket_size": 1e7,
        "stage3_prefetch_bucket_size": 1e7,
        "stage3_param_persistence_threshold": 1e5,
        "sub_group_size": 1e9,
        "offload_optimizer": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 4,
            "pipeline_read": true,
            "pipeline_write": true,
            "fast_init": true
        },
        "offload_param": {
            "device": "nvme",
            "nvme_path": "/local_nvme",
            "pin_memory": true,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9
        }
    },

    "aio": {
        "block_size": 1048576,
        "queue_depth": 8,
        "thread_count": 1,
        "single_submit": false,
        "overlap_events": true
    },

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    "gradient_clipping": 1.0
}
```

**NVMe Offload 关键参数：**

| 参数 | 作用 | 调优建议 |
|------|------|---------|
| `nvme_path` | NVMe 挂载路径 | 必须指向高速 NVMe SSD |
| `pipeline_read` / `pipeline_write` | I/O 与计算重叠 | 建议开启 |
| `fast_init` | 快速 NVMe 初始化 | 建议开启 |
| `buffer_count` | I/O buffer 数量 | optimizer 至少 4（Adam 有 4 个状态） |
| `max_in_cpu` | NVMe offload 时 CPU 中保留的元素数 | 增大可提速但占 CPU 内存 |
| `aio.queue_depth` | 异步 I/O 队列深度 | 增大可提高并行 I/O |
| `aio.thread_count` | I/O 线程数 | 多核 CPU 可增大 |

#### 模板 F：ZeRO++ 多机通信优化

适用：多机多卡训练，跨节点通信是瓶颈

在任意 ZeRO Stage 配置基础上添加：

```json
{
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "zero_hpz_partition_size": 8,
        "zero_quantized_weights": true,
        "zero_quantized_gradients": true
    }
}
```

| 参数 | 作用 | 建议值 |
|------|------|--------|
| `zero_hpz_partition_size` | 层级分区组大小（优先节点内通信） | 设为每节点 GPU 数（如 8） |
| `zero_quantized_weights` | 权重通信量化压缩 | 跨节点带宽受限时开启 |
| `zero_quantized_gradients` | 梯度通信量化压缩 | 跨节点带宽受限时开启 |

---

### 第三步：核心配置项详解

#### 3.1 Batch Size 三元关系

```
train_batch_size = train_micro_batch_size_per_gpu × gradient_accumulation_steps × GPU 数量
```

只需指定其中 2 个，DeepSpeed 自动推算第 3 个：

```json
{
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": "auto"
}
```

| 参数 | 说明 | 调优 |
|------|------|------|
| `train_micro_batch_size_per_gpu` | 单卡单步实际 batch | 在显存允许范围内尽量大 |
| `gradient_accumulation_steps` | 梯度累积步数 | 越大 = 越大等效 batch，但延迟更高 |
| `train_batch_size` | 全局有效 batch | 通常由研究需求决定 |

#### 3.2 混合精度配置

**BF16（推荐，A100/H100）：**
```json
{
    "bf16": {
        "enabled": true
    }
}
```
- 不需要 loss scaling，训练更稳定
- 要求 GPU 支持 BF16（Ampere 及以上）

**FP16（V100 等旧卡）：**
```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
```

| 参数 | 说明 |
|------|------|
| `loss_scale: 0` | 动态 loss scaling（推荐） |
| `initial_scale_power` | 初始 scale = 2^16 = 65536 |
| `loss_scale_window` | 每 1000 步尝试上调 scale |
| `hysteresis` | 连续 2 次无溢出才上调 |
| `min_loss_scale` | scale 下限，防止降到过低 |

**梯度累积精度控制：**
```json
{
    "data_types": {
        "grad_accum_dtype": "fp32"
    }
}
```
梯度累积建议使用 FP32 避免精度损失。

#### 3.3 优化器

**FusedAdam（DeepSpeed 默认，推荐）：**
```json
{
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "torch_adam": false,
            "adam_w_mode": true
        }
    }
}
```
- `torch_adam: false` → 使用 DeepSpeed FusedAdam（比 PyTorch Adam 快）
- `adam_w_mode: true` → AdamW 权重衰减方式

**通信高效优化器（多机场景）：**

| 优化器 | 通信压缩比 | 适用场景 | 关键参数 |
|--------|-----------|---------|---------|
| `OneBitAdam` | ~5x | 多机训练、带宽受限 | `freeze_step`（warmup 步数，总步数的 15-25%） |
| `ZeroOneAdam` | 最高 26x | 推荐替代 1-bit Adam | `var_freeze_step`, `var_update_scaler` |
| `OneBitLamb` | ~5x | 超大 batch 训练 | `freeze_step`, `max_coeff`, `min_coeff` |

```json
{
    "optimizer": {
        "type": "ZeroOneAdam",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "var_freeze_step": 1000000,
            "var_update_scaler": 16,
            "local_step_scaler": 1000,
            "local_step_clipper": 16,
            "cuda_aware": false,
            "comm_backend_name": "nccl"
        }
    }
}
```

#### 3.4 激活重计算

```json
{
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false,
        "contiguous_memory_optimization": false,
        "number_checkpoints": null,
        "synchronize_checkpoint_boundary": false,
        "profile": false
    }
}
```

| 参数 | 作用 | 使用场景 |
|------|------|---------|
| `partition_activations` | 跨 GPU 分片激活 | 模型并行场景 |
| `cpu_checkpointing` | 激活卸载到 CPU | 极度显存不足 |
| `contiguous_memory_optimization` | 连续内存分配 | 减少碎片化 |
| `profile` | 打印 checkpoint 耗时 | 性能分析 |

> **注意**：DeepSpeed 的 activation checkpointing 配置通常需要配合代码中的 `deepspeed.checkpointing.checkpoint()` 使用。HuggingFace Trainer 用户可直接在 `TrainingArguments` 中设置 `gradient_checkpointing=True`。

#### 3.5 张量并行（AutoTP）

DeepSpeed 支持自动张量并行，无需手动切分模型：

```json
{
    "tensor_parallel": {
        "autotp_size": 4,
        "preset_model": "llama"
    }
}
```

支持的预设模型：`llama`, `bloom`, `chatglm`, `mixtral`, `deepseek_v2`, `qwen2`, `phi3`。

> AutoTP 可与 ZeRO Stage 0/1/2 混合使用（不支持 Stage 3）。

---

### 第四步：启动方式

#### 4.1 deepspeed launcher

```bash
# 单机多卡
deepspeed --num_gpus=8 train.py --deepspeed ds_config.json [args]

# 多机多卡
deepspeed --num_nodes=4 --num_gpus=8 \
  --hostfile=hostfile \
  train.py --deepspeed ds_config.json [args]

# 绑定 CPU 核心（CPU Offload 时推荐）
deepspeed --bind_cores_to_rank train.py --deepspeed ds_config.json [args]
```

**hostfile 格式：**
```
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
```

#### 4.2 torchrun 兼容启动

```bash
torchrun --nnodes=1 --nproc-per-node=8 train.py --deepspeed ds_config.json [args]
```

#### 4.3 HuggingFace Trainer 集成

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    bf16=True,
    deepspeed="ds_config.json",  # 关键：指定 DeepSpeed 配置文件
    gradient_checkpointing=True,
    # ... 其他参数
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()
```

> HuggingFace Trainer 会自动处理 `"auto"` 配置值，从 `TrainingArguments` 中读取对应参数。

#### 4.4 代码中初始化 DeepSpeed

```python
import deepspeed

# 标准初始化
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config="ds_config.json",
)

# 训练循环
for batch in dataloader:
    loss = model_engine(batch)
    model_engine.backward(loss)
    model_engine.step()
```

**ZeRO Stage 3 模型初始化（超大模型）：**

```python
# 模型参数直接在各 GPU 上分片创建，避免 CPU 内存不足
with deepspeed.zero.Init(config_dict_or_path="ds_config.json"):
    model = MyLargeModel()
```

---

### 第五步：场景化配置推荐

#### 场景 1：单机 8x A100-80G 训练 7B 模型

```json
{
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    },
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 8,
    "gradient_accumulation_steps": 4
}
```

预期显存：参数 14G + 梯度 14G + 优化器 ~10.5G/GPU ≈ 39G/GPU，80G 卡充裕。

#### 场景 2：单机 8x A100-80G 训练 70B 模型

```json
{
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 1e7,
        "stage3_prefetch_bucket_size": 5e7,
        "stage3_param_persistence_threshold": 1e5,
        "stage3_gather_16bit_weights_on_model_save": true,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "activation_checkpointing": {
        "partition_activations": false,
        "cpu_checkpointing": false
    },
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 32
}
```

#### 场景 3：4 机 32x A100-80G 训练 70B 模型（追求速度）

```json
{
    "bf16": { "enabled": true },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e5,
        "zero_hpz_partition_size": 8,
        "zero_quantized_weights": true,
        "zero_quantized_gradients": true,
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2
}
```

- `zero_hpz_partition_size: 8` — 节点内优先通信
- `zero_quantized_weights/gradients` — 跨节点量化压缩

#### 场景 4：2x V100-32G 微调 7B 模型（显存极度受限）

```json
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 1e7,
        "stage3_prefetch_bucket_size": 1e7,
        "stage3_param_persistence_threshold": 1e5,
        "sub_group_size": 1e9,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": true
        },
        "stage3_gather_16bit_weights_on_model_save": true
    },
    "gradient_clipping": 1.0,
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 16
}
```

---

### 第六步：性能调优 Checklist

按优先级逐项检查：

| # | 检查项 | 操作 | 优先级 |
|---|--------|------|--------|
| 1 | **启用混合精度** | BF16（A100+）或 FP16（V100） | 高 |
| 2 | **选择合适的 ZeRO Stage** | 从低 Stage 开始，OOM 再升级 | 高 |
| 3 | **最大化 micro_batch_size** | 在不 OOM 的前提下尽量大 | 高 |
| 4 | **启用 overlap_comm** | `"overlap_comm": true` | 高 |
| 5 | **启用梯度累积** | 通信量与累积步数成反比 | 中 |
| 6 | **开启 activation checkpointing** | 长序列场景必选 | 中 |
| 7 | **CPU Offload 时绑核** | `--bind_cores_to_rank` | 中 |
| 8 | **CPU Offload 开 pin_memory** | `"pin_memory": true` | 中 |
| 9 | **多机启用 ZeRO++** | `zero_hpz_partition_size` + 量化通信 | 中 |
| 10 | **使用通信高效优化器** | 带宽受限时用 ZeroOneAdam | 低 |
| 11 | **NVMe Offload 开 pipeline** | `pipeline_read/write: true` | 低 |
| 12 | **使用 Autotuning** | `deepspeed --autotuning run` | 低 |

---

### 第七步：诊断与排错

#### 7.1 常见 OOM 解决路径

```
OOM!
├── 1. 减小 train_micro_batch_size_per_gpu（最直接）
├── 2. 启用 gradient_checkpointing（减少激活显存）
├── 3. 提升 ZeRO Stage（1 → 2 → 3）
├── 4. 开启 CPU Offload（offload_optimizer → offload_param）
├── 5. 减小 Stage 3 buffer（prefetch_bucket_size, max_live_parameters）
├── 6. 开启 NVMe Offload（极限方案）
└── 7. 考虑 LoRA 等参数高效微调
```

#### 7.2 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 训练速度极慢 | CPU Offload 瓶颈 | 启用 `pin_memory`、`--bind_cores_to_rank`，检查 CPU 核心数和内存带宽 |
| FP16 loss 变 NaN | loss scale 降到 0 | 增大 `min_loss_scale`，或切换到 BF16 |
| Stage 3 保存 checkpoint 失败 | 权重分散在多卡 | 开启 `stage3_gather_16bit_weights_on_model_save` |
| Stage 3 比 Stage 2 慢很多 | 参数收集通信开销 | 调大 `prefetch_bucket_size`，或增大 `param_persistence_threshold` |
| 多机训练速度不如预期 | 跨节点通信瓶颈 | 启用 ZeRO++（`zero_hpz_partition_size` + 量化通信） |
| `DeepSpeedCPUAdam` 编译失败 | 缺少编译工具 | `apt install build-essential`，或 `DS_BUILD_CPU_ADAM=1 pip install deepspeed` |
| OOM 在 `zero.Init` 阶段 | 模型初始化未分片 | 确保用 `deepspeed.zero.Init()` 上下文管理器包裹模型创建 |
| Autotuning 不生效 | 缺少启动参数 | 必须同时在 config 和命令行 (`--autotuning run`) 中启用 |

#### 7.3 有用的诊断工具

```bash
# 检查 DeepSpeed 安装状态和兼容的 ops
ds_report

# 启用 FLOPS Profiler 分析吞吐
# 在 ds_config.json 中添加：
{
    "flops_profiler": {
        "enabled": true,
        "profile_step": 1,
        "detailed": true
    }
}

# 启用通信日志分析通信开销
{
    "comms_logger": {
        "enabled": true,
        "verbose": false,
        "prof_all": true
    }
}
```

---

### 附录：ZeRO 全配置参数速查

| 参数 | Stage | 默认值 | 说明 |
|------|-------|--------|------|
| `stage` | all | `0` | 0/1/2/3 |
| `contiguous_gradients` | 1,2,3 | `true` | 梯度连续存储，减少碎片 |
| `reduce_scatter` | 1,2,3 | `true` | 用 reduce_scatter 替代 allreduce |
| `reduce_bucket_size` | 1,2,3 | `5e8` | 每次 reduce 的元素数 |
| `allgather_partitions` | 1,2,3 | `true` | 用 allgather 收集更新后的参数 |
| `allgather_bucket_size` | 1,2,3 | `5e8` | 每次 allgather 的元素数 |
| `overlap_comm` | 1,2,3 | `false` | 通信与计算重叠 |
| `round_robin_gradients` | 1,2 | `false` | CPU offload 时并行梯度拷贝 |
| `offload_optimizer.device` | 1,2,3 | — | `"cpu"` 或 `"nvme"` |
| `offload_param.device` | 3 | — | `"cpu"` 或 `"nvme"` |
| `stage3_prefetch_bucket_size` | 3 | `5e7` | 参数预取 buffer |
| `stage3_param_persistence_threshold` | 3 | `1e5` | 小参数常驻阈值 |
| `stage3_max_live_parameters` | 3 | `1e9` | GPU 上最大常驻参数 |
| `stage3_max_reuse_distance` | 3 | `1e9` | 参数复用距离 |
| `sub_group_size` | 3 | `1e9` | 万亿参数分片粒度 |
| `zero_hpz_partition_size` | 1,2,3 | `1` | ZeRO++ 层级分区组大小 |
| `zero_quantized_weights` | 1,2,3 | `false` | ZeRO++ 权重量化通信 |
| `zero_quantized_gradients` | 1,2,3 | `false` | ZeRO++ 梯度量化通信 |

### 附录：DeepSpeed vs PyTorch FSDP 对比

| 维度 | DeepSpeed | PyTorch FSDP |
|------|-----------|-------------|
| **显存优化** | ZeRO Stage 0/1/2/3 + Offload + NVMe | 类似 ZeRO Stage 3 |
| **CPU/NVMe 卸载** | 全面支持（Stage 1-3 + NVMe） | 有限支持 |
| **通信优化** | ZeRO++、1-bit/0-1 Adam、量化通信 | 基本通信原语 |
| **混合精度** | FP16 + 动态 loss scaling、BF16 | 依赖 PyTorch AMP |
| **张量并行** | AutoTP（自动切分） | 需手动配合 TP API |
| **易用性** | JSON 配置驱动，生态丰富 | PyTorch 原生，API 更 Pythonic |
| **HuggingFace 集成** | 深度集成 | 通过 Accelerate 集成 |
| **生态成熟度** | ⭐⭐⭐（生产验证最广） | ⭐⭐（快速追赶中） |
| **推荐场景** | 大规模预训练、显存极端紧张、多机优化 | 中小规模训练、纯 PyTorch 栈 |

### 附录：与其他工具的配合

| 工具 | 配合方式 |
|------|---------|
| **Flash Attention** | DeepSpeed 自动检测并使用，也可通过 HuggingFace `attn_implementation` 配合 |
| **Liger Kernel** | 与 DeepSpeed 完全兼容，可同时启用 |
| **HuggingFace Trainer** | `TrainingArguments(deepspeed="ds_config.json")` 一行集成 |
| **HuggingFace Accelerate** | `accelerate launch --use_deepspeed` 启动 |
| **Megatron-LM** | DeepSpeed 可与 Megatron 3D 并行结合使用 |
| **PEFT/LoRA** | ZeRO Stage 3 + `zero_quantized_nontrainable_weights` 对 LoRA 特别优化 |

### 附录：DeepSpeed-Inference — 推理加速

DeepSpeed 不仅用于训练，也提供推理优化能力。当项目包含推理代码或需要从训练直接转推理时可参考。

#### 核心特性

| 特性 | 说明 | 适用场景 |
|------|------|---------|
| **Tensor Parallel 推理** | 自动将模型切分到多卡推理 | 模型超过单卡显存 |
| **Kernel Injection** | 将 Transformer 层替换为 DeepSpeed 高性能 kernel | HuggingFace 模型加速 |
| **Dynamic Quantization** | 推理时动态 INT8 量化 | 显存受限或追求吞吐 |
| **ZeRO-Inference** | ZeRO Stage 3 的推理模式（参数分片加载） | 超大模型推理 |

```python
import deepspeed

# 基本推理加速（Kernel Injection + Tensor Parallel）
model = deepspeed.init_inference(
    model,
    tensor_parallel={"tp_size": 4},  # 4 卡 TP
    dtype=torch.float16,
    replace_with_kernel_inject=True,  # 替换为 DeepSpeed 高性能 kernel
)

# ZeRO-Inference（超大模型分片加载）
ds_config = {
    "zero": {
        "stage": 3,
        "offload_param": {"device": "cpu"},  # 参数放 CPU，按需搬 GPU
    },
    "dtype": "fp16",
}
model = deepspeed.init_inference(model, config=ds_config)
```

**注意**：对于生产推理服务，优先考虑 vLLM / TensorRT-LLM（更成熟的 Continuous Batching + PagedAttention）。DeepSpeed-Inference 更适合**研究场景**或从训练代码直接转推理的过渡方案。

### 附录：通信重叠 (overlap_comm) 深入配置

`overlap_comm` 是 DeepSpeed 中收益最明显但配置最容易出错的优化之一。

#### 工作原理

```
标准模式（overlap_comm=false）:
  backward(layer_N) → allreduce(grad_N) → backward(layer_{N-1}) → allreduce(grad_{N-1})
  ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^   串行等待通信完成

重叠模式（overlap_comm=true）:
  backward(layer_N) → backward(layer_{N-1}) → backward(layer_{N-2})
  allreduce(grad_N)    allreduce(grad_{N-1})   ← 通信与下一层计算并行
```

#### 关键参数调优

| 参数 | 默认值 | 说明 | 调优建议 |
|------|--------|------|---------|
| `overlap_comm` | `false` | 主开关 | 多卡训练时必须开启 |
| `reduce_bucket_size` | `5e8` (500M elements) | 每次 AllReduce 的梯度元素数 | 太大 → 无法及时开始通信；太小 → 通信碎片化。推荐 `5e7` ~ `5e8` |
| `allgather_bucket_size` | `5e8` | 每次 AllGather 的元素数 | 与 reduce_bucket_size 类似，按实际 GPU 显存调整 |
| `contiguous_gradients` | `true` | 梯度连续存储 | 必须 true 以配合 overlap |

#### 与梯度累积的交互

当 `gradient_accumulation_steps > 1` 时，只有最后一步需要做 AllReduce：

```json
{
    "zero_optimization": {
        "stage": 2,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e7
    },
    "gradient_accumulation_steps": 4
}
```

DeepSpeed 自动处理：前 3 步只做本地梯度累加，第 4 步触发 AllReduce + overlap。无需手动管理 `model.no_sync()`。

### 附录：NCCL 调优速查

多机多卡训练中，NCCL 参数直接影响通信效率。以下是关键环境变量及其调优建议：

#### 网络与传输

| 环境变量 | 默认值 | 说明 | 推荐配置 |
|---------|--------|------|---------|
| `NCCL_IB_DISABLE` | `0` | 是否禁用 InfiniBand | 有 IB 网卡时保持 `0`；无 IB 时设 `1` 强制用 TCP |
| `NCCL_NET_GDR_LEVEL` | `0` | GPUDirect RDMA 等级 | 有 IB + GPUDirect 时设 `5`（LOC: GPU-local RDMA） |
| `NCCL_IB_HCA` | auto | IB 网卡列表 | 多网卡时指定高速网卡，如 `mlx5_0,mlx5_1` |
| `NCCL_IB_GID_INDEX` | `0` | IB GID 索引 | RoCE v2 通常设 `3` |
| `NCCL_SOCKET_IFNAME` | auto | TCP 网络接口名 | 指定高速网口如 `eth0`、`bond0` |
| `NCCL_P2P_LEVEL` | auto | P2P 通信等级 | NVLink 可用时自动 P2P；无 NVLink 时可尝试 `NVL` 或 `PIX` |

#### 通信算法与协议

| 环境变量 | 默认值 | 说明 | 推荐配置 |
|---------|--------|------|---------|
| `NCCL_ALGO` | auto | 通信算法 | `Ring`（延迟低）/ `Tree`（大消息吞吐高）/ auto（NCCL 自动选） |
| `NCCL_PROTO` | auto | 传输协议 | `Simple`（小消息）/ `LL`（低延迟）/ `LL128`（NVLink 优化） |
| `NCCL_MIN_NCHANNELS` | auto | 最小通信通道数 | 增大可提高带宽利用率，但增加显存占用 |
| `NCCL_MAX_NCHANNELS` | auto | 最大通信通道数 | 显存紧张时可限制 |

#### 调试与诊断

| 环境变量 | 值 | 说明 |
|---------|-----|------|
| `NCCL_DEBUG` | `INFO` | 打印 NCCL 初始化信息（网络拓扑、选择的算法、P2P 状态） |
| `NCCL_DEBUG_SUBSYS` | `INIT,GRAPH` | 仅打印特定子系统的 debug 信息 |
| `NCCL_TOPO_DUMP_FILE` | `/path/file` | 导出 NCCL 检测到的拓扑结构 |

#### 常见多机场景配置

```bash
# 场景 1：InfiniBand + GPUDirect RDMA（最优）
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_DEBUG=INFO  # 首次运行开启，确认 GDR 生效后关闭

# 场景 2：RoCE v2（以太网 RDMA）
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_NET_GDR_LEVEL=5
export NCCL_SOCKET_IFNAME=eth0

# 场景 3：纯 TCP（无 RDMA）
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
# TCP 模式下带宽较低，优先考虑梯度压缩（ZeRO++ / PowerSGD）

# 场景 4：NVLink 节点内 + IB 跨节点
export NCCL_P2P_LEVEL=NVL       # 节点内走 NVLink
export NCCL_IB_DISABLE=0         # 跨节点走 IB
export NCCL_NET_GDR_LEVEL=5
```

**诊断流程**：
1. 设 `NCCL_DEBUG=INFO`，检查日志中是否出现 `NET/IB`（IB）或 `NET/Socket`（TCP）
2. 检查是否看到 `GPU Direct RDMA`（GDR 生效）
3. 如通信慢，对比 `Tree` vs `Ring` 算法（设 `NCCL_ALGO=Tree` 或 `Ring`）
4. 用 nsys 看 NCCL kernel 与计算 kernel 是否重叠
