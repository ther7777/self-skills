---
name: videox-fun-optimization
description: VideoX-Fun 训练优化审计
user-invocable: false
---

# Skill: VideoX-Fun 训练优化审计

## 描述
针对基于 VideoX-Fun（aigc-apps/VideoX-Fun）框架的视频生成训练项目，快速识别已采用和未采用的优化手段，提供针对性的显存优化与性能加速建议。适用于 Wan 系列（Wan2.1/2.2/Fun/VACE）、CogVideoX-Fun、HunyuanVideo、Flux、LongCatVideo 等模型的微调场景。

## 触发条件
当识别到用户项目满足以下任一条件时自动触发：
- 代码中 import 了 `videox_fun` 或使用 `videox_fun` 包中的模块
- 依赖文件中包含 `videox-fun` 或 `videox_fun`
- 项目结构包含 `videox_fun/` 目录（含 `models/`、`pipeline/`、`data/` 子目录）
- 配置文件中包含 VideoX-Fun 特有参数：`vae_mini_batch`、`train_mode`（值为 `normal`/`inpaint`/`control_ref`）、`video_sample_n_frames`、`enable_bucket`、`random_hw_adapt`
- 启动命令包含 `scripts/wan2.1_fun/`、`scripts/wan2.2_fun/`、`scripts/cogvideox_fun/` 等 VideoX-Fun 脚本路径
- 存在 `config/wan2.1/wan_civitai.yaml`、`config/wan2.2/` 等 VideoX-Fun 配置目录
- 启动命令中的训练脚本接受 `--pretrained_model_name_or_path` + `--train_mode` 参数组合
- 项目为 VideoX-Fun 的 fork（检查 `.git/config` 中的 remote URL 包含 `VideoX-Fun`）

## 执行指令

你是视频生成模型训练优化专家，精通 VideoX-Fun 框架。被调用时，请按照以下清单对目标项目进行全面审计。对每个检查项，判定其状态：**已启用** / **未启用** / **建议启用** / **不适用**，并给出具体的配置修改建议。

### 第一步：识别训练模式与基础信息

扫描项目的训练脚本（`.sh`/`.py`）和配置文件（YAML/JSON），确定：

1. **模型系列**：Wan2.1-Fun / Wan2.1 / Wan2.2-Fun / Wan2.2-VACE / CogVideoX-Fun / HunyuanVideo / Flux / LongCatVideo / FantasyTalking / TurboDiffusion
2. **模型规模**：1.3B / 5B / 14B（从 `pretrained_model_name_or_path` 推断）
3. **训练模式**（`train_mode`）：`normal`（T2V）/ `inpaint`（I2V）/ `control_ref`（控制生成）
4. **微调方式**：全参微调（`train.py`）/ LoRA（`train_lora.py`）/ Control（`train_control.py`）/ Control LoRA / Reward LoRA / Distillation
5. **硬件环境**：GPU 型号/数量、显存大小
6. **数据规模**：视频数量、分辨率（`video_sample_size`）、帧数（`video_sample_n_frames`）
7. **分布式配置**：accelerate / DeepSpeed ZeRO-2/3 / FSDP / 多机

---

### 第二步：显存优化审计

#### A. 混合精度与量化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **BF16 混合精度** | `--mixed_precision="bf16"` | Ampere+(A100/H100/H20) 必开。VideoX-Fun 默认 shell 脚本已启用 bf16 |
| **FP16 混合精度** | `--mixed_precision="fp16"` | V100/T4 使用。注意 loss scale 问题 |
| **TF32** | `--allow_tf32` | Ampere+ 额外加速，仅影响 matmul 精度（FP32 输入 TF32 计算）。无显存节省但有计算加速 |
| **FP8 推理量化** | `convert_model_weight_to_float8()` | 仅用于推理/ComfyUI，训练不支持 |
| **8-bit Adam** | `--use_8bit_adam` | 使用 bitsandbytes 8-bit Adam，优化器状态显存减半 |
| **CAME 优化器** | `--use_came` | 显存高效优化器替代方案 |

**建议**：
- A100/H100/H20 必须 `--mixed_precision="bf16"` + `--allow_tf32`
- V100 使用 `--mixed_precision="fp16"`
- 显存紧张时启用 `--use_8bit_adam`

---

#### B. 激活重计算

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **标准梯度检查点** | `--gradient_checkpointing` | 以 ~30% 计算换 ~60% 激活显存。长视频训练必开 |
| **分数梯度检查点** | `apply_checkpointing(model, block, p)` | VideoX-Fun 独有。`p=1/3` 仅检查点 1/3 的 block，平衡速度和显存 |
| **梯度累积** | `--gradient_accumulation_steps N` | 等效增大 batch size 而不增加显存 |

**建议**：
- 14B 模型或视频帧数 >=49 时，`--gradient_checkpointing` 必开
- 显存紧张但需要速度时，可用分数检查点（如 `p=0.5`）替代全量检查点

---

#### C. 模型卸载与低显存

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Low VRAM 模式** | `--low_vram` | **高优先级**。将 VAE 和 Text Encoder 在 CPU/GPU 间穿梭，仅训练 DiT 时占 GPU。训练脚本默认启用 |
| **VAE Mini-Batch** | `--vae_mini_batch N`（默认 32） | 分批编码视频帧到 latent，避免一次编码全部帧导致 OOM |
| **Multi-Stream VAE** | `--multi_stream` | 使用多 CUDA Stream 并行编码 clean latent 和 mask latent，加速 I2V 场景 |

**建议**：
- 14B 模型训练时 `--low_vram` 必开
- 高分辨率（>=720P）或长视频（>=81帧）时减小 `--vae_mini_batch`（如 16 或 8）

---

#### D. 分布式显存优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **DeepSpeed ZeRO-2** | `--use_deepspeed --deepspeed_config_file config/zero_stage2_config.json` | 分片梯度+优化器状态。LoRA 多卡首选 |
| **DeepSpeed ZeRO-3** | `--use_deepspeed --deepspeed_config_file config/zero_stage3_config.json` | 分片参数+梯度+优化器。14B 全参微调必用。需配合 `zero_to_bf16.py` 转换权重 |
| **ZeRO-3 + CPU Offload** | `config/zero_stage3_config_cpu_offload.json` | 优化器/参数卸载到 CPU。极端显存不足时使用（14B@24GB） |
| **FSDP Full Shard** | `--use_fsdp --fsdp_sharding_strategy "FULL_SHARD" --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" --fsdp_transformer_layer_cls_to_wrap "WanAttentionBlock"` | **14B 训练推荐**。README 明确指出 FSDP 比 ZeRO-3 更稳定。注意 wrap class 因模型不同 |
| **FSDP Backward Prefetch** | `--fsdp_backward_prefetch "BACKWARD_PRE"` | 预取下一层参数，与反向计算重叠 |

**FSDP wrap class 速查**：

| 模型系列 | `--fsdp_transformer_layer_cls_to_wrap` |
|---------|---------------------------------------|
| Wan2.1/2.2 | `WanAttentionBlock` |
| CogVideoX | `CogVideoXBlock` |
| HunyuanVideo | 需查看具体模型的 transformer block class |
| Flux | `FluxTransformerBlock` / `FluxSingleTransformerBlock` |

**建议**：
- 1.3B LoRA → 无需 DeepSpeed/FSDP，plain accelerate 即可
- 14B LoRA → DeepSpeed ZeRO-2
- 14B 全参 → FSDP Full Shard（首选）或 DeepSpeed ZeRO-3
- 14B 全参 + 显存极紧张 → ZeRO-3 + CPU Offload

---

#### E. LoRA 配置优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **PEFT LoRA** | `--use_peft_lora --rank N --network_alpha M` | HuggingFace PEFT 库实现。推荐 rank=64, alpha=32 |
| **自定义 LoRA** | `--rank N --network_alpha M --target_name "q,k,v,ffn.0,ffn.2"` | Kohya 风格实现。支持 rank dropout、module dropout |
| **LoRA 目标模块** | `--target_name` | 默认 `"q,k,v,ffn.0,ffn.2"` 覆盖注意力和 FFN |
| **Rank Dropout** | `--rank_drop_out` | LoRA rank 维度随机 dropout，正则化效果 |
| **Module Dropout** | `--module_drop_out` | 随机跳过整个 LoRA 模块，防过拟合 |

**建议**：
- 入门：`--rank 64 --network_alpha 32 --use_peft_lora`
- 效果优先：`--rank 128 --network_alpha 64 --target_name "q,k,v,ffn.0,ffn.2"`
- 防过拟合：添加 `--rank_drop_out 0.1 --module_drop_out 0.1`

---

### 第三步：计算性能优化审计

#### F. 注意力机制优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **Flash Attention 2** | 自动检测 `flash_attn` 包 | **高优先级**。VideoX-Fun 自动优先使用 FA2（安装即启用）。检查是否已安装 `flash-attn` |
| **Flash Attention 3** | 自动检测 `flash_attn_interface` 包 | Hopper (H100/H200) 上优先于 FA2。检查是否已安装 `flash-attn>=3.0` |
| **Variable-Length FA** | `flash_attn_varlen_func` | 用于不等长序列的 padding-free 注意力。Bucket 训练时自动使用 |
| **SageAttention** | 自动检测 `sageattention` 包 | 推理时优先于 FA（按 SM 架构分发），训练时自动回退到 FA |
| **PyTorch SDPA** | 兜底方案 | FA 不可用时的 fallback。性能低于 FA |
| **Sparse Linear Attention** | TurboDiffusion 模型专用 | Triton 实现的稀疏注意力，`topk=0.1` 仅关注 10% 的 key block |

**建议**：
- **必须安装 `flash-attn`**。未安装 FA 会退回 SDPA，注意力计算慢 2-3 倍
- H100/H200 安装 `flash-attn>=3.0` 使用 FA3
- 可通过环境变量 `VIDEOX_ATTENTION_TYPE` 强制指定注意力类型

---

#### G. 数据管道优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **DataLoader Workers** | `--dataloader_num_workers N` | 多进程加载视频数据。建议 4~8 |
| **Persistent Workers** | 代码中 `persistent_workers=True` when workers>0 | 复用 worker 进程，避免每 epoch 重启 |
| **Bucket 采样** | `--enable_bucket` | **推荐**。按宽高比分桶，减少 padding 浪费 |
| **随机分辨率适配** | `--random_hw_adapt` | 训练时随机缩放分辨率，数据增强 + 减少计算 |
| **视频 Token 长度训练** | `--training_with_video_token_length` | 在固定 token 预算内动态调整分辨率和帧数 |
| **自动 Tile Batch** | `--auto_tile_batch_size` | 小序列自动增大 batch（<=1/16 token budget -> 4x tiling），提高 GPU 利用率 |
| **保持节点同 Token 长度** | `--keep_all_node_same_token_length` | 截断帧数使各节点处理相近的 token 数，防止 straggler |
| **视频采样步长** | `--video_sample_stride N` | 跳帧采样（stride=2 隔帧取），减少帧数 |
| **随机帧裁剪** | `--random_frame_crop` | 90% 概率取最大帧数，10% 随机少取，数据增强 |
| **预编码 Safetensors 数据集** | `ImageVideoSafetensorsDataset` | 预编码 latent 存储为 `.safetensors`，跳过 VAE+TextEncoder 前向 |
| **文本 Dropout** | `--text_drop_ratio 0.1` | 10% 概率丢弃文本，训练 classifier-free guidance |

**建议**：
- 混合分辨率数据集必须启用 `--enable_bucket`
- 大规模训练启用 `--auto_tile_batch_size` 提高小分辨率样本的 GPU 利用率
- 多机训练启用 `--keep_all_node_same_token_length` 避免节点间等待
- **高优先级**：预编码数据集（`ImageVideoSafetensorsDataset`）跳过 VAE 和 Text Encoder 编码，训练速度大幅提升

---

#### H. 训练策略优化

| 检查项 | 配置参数 | 审计要点 |
|---|---|---|
| **学习率调度** | `--lr_scheduler` | 支持 `constant`/`constant_with_warmup`/`cosine`/`linear`/`polynomial`/`cosine_with_restarts` |
| **Warmup** | `--lr_warmup_steps N` | 大 batch 训练需要充分 warmup |
| **梯度裁剪** | `--max_grad_norm 1.0` | 默认 1.0。视频训练梯度波动大，建议保持 |
| **异常梯度范数裁剪** | `--abnormal_norm_clip_start N` | VideoX-Fun 独有。第 N 步后开始用历史移动平均检测异常梯度，自动裁剪尖峰 |
| **EMA** | `--use_ema` | 指数移动平均，稳定训练质量。增加约 1x 模型参数的显存 |
| **均匀时间步采样** | `--uniform_sampling` | 分布式训练时各 worker 均匀覆盖 noise schedule，避免重复采样 |
| **运动子损失** | `--motion_sub_loss` + `--motion_sub_loss_ratio 0.25` | 对帧间差分额外计算 MSE loss，鼓励时间一致性 |
| **时间步加权** | `--weighting_scheme` | 支持 `sigma_sqrt`/`logit_normal`/`mode`/`cosmap`/`none` |
| **双学习率** | `--trainable_modules_low_learning_rate` + `--learning_rate_low` | 不同模块使用不同学习率 |
| **Reward LoRA** | `--reward_fn "HPSReward"` + `--backprop_strategy "tail"` | 基于人类偏好的无数据训练，支持 HPS/MPS/Aesthetic/PickScore 奖励模型 |

**建议**：
- `--lr_scheduler cosine_with_restarts` + 充分 warmup 适合长训练
- 14B 模型建议启用 `--abnormal_norm_clip_start 100`
- 多卡训练启用 `--uniform_sampling`

---

### 第四步：深度优化机会（框架代码级）

以下优化项 VideoX-Fun 框架当前**未实现**，但可在 fork 副本中通过代码修改获得显著收益：

#### I. 未实现的高收益优化

| 优化项 | 当前状态 | 实施方式 | 预期收益 |
|--------|---------|---------|---------|
| **torch.compile** | 未使用 | 在训练入口添加 `transformer3d = torch.compile(transformer3d, mode="reduce-overhead")` | 算子融合，吞吐 +10~30% |
| **Liger Kernel** | 未使用 | 对 RMSNorm/SwiGLU 等层应用 Liger Triton 融合算子 | 吞吐 +20%，显存 -60% |
| **CUDA Graph** | 未使用 | 对固定输入形状的训练步骤启用 CUDA Graph 捕获 | Kernel launch 开销 -90%（小 batch 时效果显著） |
| **QLoRA (4-bit)** | 仅 FP8 推理 | 使用 bitsandbytes NF4 量化 + LoRA | 14B 模型显存从 ~28GB 降至 ~10GB |
| **Wan VAE 空间 Tiling** | 仅 CogVideoX 有 | 参考 CogVideoX VAE 的 `tiled_encode`/`tiled_decode` 实现 Wan VAE 空间 tiling | 高分辨率（>=720P）VAE 显存 -75% |
| **预编码 Text Embeddings** | 部分（Safetensors 数据集支持但非默认） | 预跑 Text Encoder 保存 embeddings，训练时直接加载 | 跳过 Text Encoder 前向，节省 ~5GB 显存 + 加速 |
| **FP8 训练** | 仅推理 | H100/H200 使用 TransformerEngine 或 torchao FP8 训练 | GEMM 计算速度翻倍 |
| **ZeRO++ 量化通信** | 未配置 | DeepSpeed 配置中启用 `zero_quantized_weights`/`zero_quantized_gradients` | 多机通信量 -50~75% |
| **NVMe Offload（ZeRO-Infinity）** | 未配置 | DeepSpeed ZeRO-3 配置中添加 NVMe offload | 突破 CPU 内存限制 |
| **Fused Optimizer** | 使用原生 AdamW | DeepSpeed FusedAdam 或 apex FusedAdam | Kernel 调用次数减少，速度 +5~10% |
| **DataLoader prefetch_factor** | 未设置 | `prefetch_factor=2~4` | CPU->GPU 数据传输与训练重叠 |
| **序列并行训练** | 仅推理 | 将推理时的 Ulysses 序列并行扩展到训练 | 超长视频（>=81帧高分辨率）训练的显存突破 |
| **训练时 TeaCache** | 仅推理 | Reward LoRA 训练中复用时间步间的中间表示 | Reward LoRA 训练加速 |

---

### 第五步：场景化配置模板

#### 场景 1：Wan2.1-Fun 1.3B LoRA — 单卡 24GB (RTX 4090)

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-1.3B-InP"
export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="datasets/internal_datasets/metadata.json"

accelerate launch --mixed_precision="bf16" scripts/wan2.1_fun/train_lora.py \
  --config_path "config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $DATASET_NAME \
  --train_data_meta $DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --num_train_epochs=100 \
  --learning_rate=1e-4 \
  --rank=64 \
  --network_alpha=32 \
  --train_mode="inpaint" \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --mixed_precision="bf16" \
  --allow_tf32 \
  --gradient_checkpointing \
  --low_vram \
  --enable_bucket \
  --use_8bit_adam \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=100
```

#### 场景 2：Wan2.1-Fun 14B LoRA — 8xA100 80GB + DeepSpeed ZeRO-2

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-InP"

accelerate launch --mixed_precision="bf16" \
  --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json \
  scripts/wan2.1_fun/train_lora.py \
  --config_path "config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $DATASET_NAME \
  --train_data_meta $DATASET_META_NAME \
  --image_sample_size=1024 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_n_frames=81 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --learning_rate=1e-4 \
  --rank=128 \
  --network_alpha=64 \
  --train_mode="inpaint" \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --mixed_precision="bf16" \
  --allow_tf32 \
  --gradient_checkpointing \
  --low_vram \
  --enable_bucket \
  --auto_tile_batch_size \
  --uniform_sampling \
  --abnormal_norm_clip_start=100 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=200
```

#### 场景 3：Wan2.1-Fun 14B 全参微调 — 8xA100 80GB + FSDP

```bash
export MODEL_NAME="models/Diffusion_Transformer/Wan2.1-Fun-V1.1-14B-InP"

accelerate launch --mixed_precision="bf16" \
  --use_fsdp \
  --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" \
  --fsdp_transformer_layer_cls_to_wrap "WanAttentionBlock" \
  --fsdp_sharding_strategy "FULL_SHARD" \
  --fsdp_state_dict_type "SHARDED_STATE_DICT" \
  --fsdp_backward_prefetch "BACKWARD_PRE" \
  scripts/wan2.1_fun/train.py \
  --config_path "config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $DATASET_NAME \
  --train_data_meta $DATASET_META_NAME \
  --trainable_modules "." \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=512 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=8 \
  --num_train_epochs=100 \
  --learning_rate=2e-5 \
  --mixed_precision="bf16" \
  --allow_tf32 \
  --gradient_checkpointing \
  --low_vram \
  --enable_bucket \
  --uniform_sampling \
  --abnormal_norm_clip_start=100 \
  --lr_scheduler="cosine_with_restarts" \
  --lr_warmup_steps=500
```

#### 场景 4：Wan2.1-Fun 14B 全参 — 2x24GB (RTX 4090) + ZeRO-3 CPU Offload

```bash
accelerate launch --mixed_precision="bf16" \
  --use_deepspeed --deepspeed_config_file config/zero_stage3_config_cpu_offload.json \
  --zero_stage 3 --zero3_save_16bit_model true --zero3_init_flag true \
  scripts/wan2.1_fun/train.py \
  --config_path "config/wan2.1/wan_civitai.yaml" \
  --pretrained_model_name_or_path $MODEL_NAME \
  --trainable_modules "." \
  --image_sample_size=320 \
  --video_sample_size=320 \
  --token_sample_size=320 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --dataloader_num_workers=4 \
  --learning_rate=2e-5 \
  --mixed_precision="bf16" \
  --gradient_checkpointing \
  --low_vram \
  --vae_mini_batch=8 \
  --use_8bit_adam \
  --enable_bucket
```

#### 场景 5：CogVideoX-Fun 5B LoRA — 单卡 A100 80GB

```bash
accelerate launch --mixed_precision="bf16" scripts/cogvideox_fun/train_lora.py \
  --config_path "config/cogvideox/civitai.yaml" \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $DATASET_NAME \
  --train_data_meta $DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-4 \
  --rank=64 \
  --network_alpha=32 \
  --mixed_precision="bf16" \
  --allow_tf32 \
  --gradient_checkpointing \
  --low_vram \
  --enable_bucket \
  --use_8bit_adam
```

---

### 第六步：输出审计报告

按以下格式输出审计结果：

```markdown
# VideoX-Fun 训练优化审计报告

## 基本信息
- 模型：{model_name}（{参数量}）
- 模型系列：{Wan2.1-Fun / CogVideoX-Fun / ...}
- 训练模式：{train_mode}（T2V/I2V/Control）
- 微调方式：{全参 / LoRA / Control / Reward LoRA}
- 硬件：{GPU型号} x {数量}（{显存}GB）
- 视频参数：{video_sample_size}P, {video_sample_n_frames} 帧

## 审计结果总览

| 类别 | 得分 | 已启用 | 建议启用 |
|------|------|--------|----------|
| A. 混合精度与量化 | x/6 | ... | ... |
| B. 激活重计算 | x/3 | ... | ... |
| C. 模型卸载与低显存 | x/3 | ... | ... |
| D. 分布式显存优化 | x/5 | ... | ... |
| E. LoRA 配置 | x/5 | ... | ... |
| F. 注意力优化 | x/6 | ... | ... |
| G. 数据管道 | x/11 | ... | ... |
| H. 训练策略 | x/10 | ... | ... |
| I. 深度优化机会 | x/12 | ... | ... |
| **总计** | **x/61** | | |

## 优先优化建议（按影响排序）

### P0 - 立即执行（显著收益，零风险）
1. ...

### P1 - 强烈推荐（明显收益，低风险）
1. ...

### P2 - 建议尝试（中等收益，需测试）
1. ...

## 推荐配置修改

（给出具体的启动命令修改 diff）
```

---

## 常见优化决策树

```
开始
├── 显存不足（OOM）？
│   ├── 是 → 当前用全参训练？
│   │   ├── 是 → 切换为 LoRA（train_lora.py, --rank 64）
│   │   │   └── 仍然 OOM？
│   │   │       ├── 降低分辨率（video_sample_size: 512->320）
│   │   │       ├── 减少帧数（video_sample_n_frames: 81->49->25）
│   │   │       └── 启用 ZeRO-3 + CPU Offload
│   │   └── 否（已用 LoRA）
│   │       ├── 启用 --gradient_checkpointing
│   │       ├── 启用 --low_vram
│   │       ├── 启用 --use_8bit_adam
│   │       ├── 减小 --vae_mini_batch（32->16->8）
│   │       ├── 减小 batch_size=1 + 增大 gradient_accumulation_steps
│   │       └── 仍然 OOM？ → 降低分辨率 / 减少帧数
│   └── 否 → 继续性能优化
├── 训练速度慢？
│   ├── flash-attn 已安装？
│   │   ├── 否 → pip install flash-attn（最重要的单一加速项）
│   │   └── 是 → 继续
│   ├── 混合精度已启用？
│   │   ├── 否 → --mixed_precision="bf16"
│   │   └── 是 → 继续
│   ├── GPU 利用率低？
│   │   ├── DataLoader 瓶颈 → --dataloader_num_workers=8
│   │   ├── 小 batch 浪费 → --auto_tile_batch_size + --enable_bucket
│   │   └── VAE 编码慢 → 预编码数据集（ImageVideoSafetensorsDataset）
│   ├── 多卡训练？
│   │   ├── LoRA → DeepSpeed ZeRO-2
│   │   └── 全参 → FSDP Full Shard（首选）
│   ├── 多机训练？
│   │   ├── --uniform_sampling + --keep_all_node_same_token_length
│   │   └── NCCL 调优 + ZeRO++ 量化通信
│   └── 检查未实现优化（torch.compile / Liger Kernel / FusedAdam）
└── 训练效果差？
    ├── 启用 --motion_sub_loss（时间一致性）
    ├── 启用 --abnormal_norm_clip_start（梯度稳定）
    ├── LoRA rank 增大（64->128）
    ├── 添加 --rank_drop_out 0.1（正则化）
    └── 使用 --lr_scheduler cosine_with_restarts
```

## 显存估算参考（BF16 混合精度，单样本 49帧 512P）

| 方法 | 1.3B | 5B | 14B |
|------|------|-----|------|
| 全参（BF16 + GC） | ~24GB | ~60GB | ~160GB |
| LoRA rank=64（BF16 + GC + low_vram） | ~12GB | ~24GB | ~48GB |
| LoRA rank=64 + ZeRO-2 | ~10GB/卡 | ~18GB/卡 | ~35GB/卡 |
| 全参 + FSDP Full Shard（8卡） | ~8GB/卡 | ~16GB/卡 | ~40GB/卡 |
| 全参 + ZeRO-3 + CPU Offload（2卡） | ~16GB/卡 | ~40GB/卡 | ~48GB/卡 |

*注：实际显存与分辨率、帧数、batch_size 强相关。81 帧约为 49 帧的 1.7x 显存，720P 约为 480P 的 2.25x 显存。*

## VideoX-Fun 特有关键词速查

| 类别 | 关键词/模式 |
|------|------------|
| 包名 | `videox-fun`、`videox_fun` |
| import | `from videox_fun`、`import videox_fun` |
| 模型 | `WanTransformer3DModel`、`CogVideoXTransformer3DModel`、`WanAttentionBlock` |
| VAE | `AutoencoderKLWan`、`AutoencoderKLCogVideoX`、`vae_mini_batch` |
| 训练参数 | `train_mode`、`video_sample_n_frames`、`video_sample_size`、`enable_bucket`、`random_hw_adapt`、`training_with_video_token_length`、`auto_tile_batch_size`、`low_vram` |
| 脚本路径 | `scripts/wan2.1_fun/`、`scripts/wan2.2_fun/`、`scripts/cogvideox_fun/`、`scripts/hunyuanvideo/` |
| 配置文件 | `config/wan2.1/wan_civitai.yaml`、`config/wan2.2/`、`zero_stage2_config.json`、`zero_stage3_config.json` |
| LoRA | `--rank`、`--network_alpha`、`--target_name`、`--use_peft_lora`、`rank_drop_out`、`module_drop_out` |
| 独有功能 | `abnormal_norm_clip_start`、`motion_sub_loss`、`uniform_sampling`、`auto_tile_batch_size`、`keep_all_node_same_token_length`、`multi_stream` |
| 分布式 | `--use_deepspeed`、`--use_fsdp`、`--fsdp_transformer_layer_cls_to_wrap "WanAttentionBlock"` |
| 数据 | `ImageVideoDataset`、`ImageVideoControlDataset`、`ImageVideoSafetensorsDataset`、`AspectRatioBatchImageVideoSampler` |
