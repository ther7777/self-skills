---
name: gpu-training-optimizer
description: 端到端 GPU 训练智能调优 — 静态分析 → 硬件采集 → Profiling → 瓶颈诊断 → 代码优化 → 验证迭代
---

## 描述

端到端的 GPU 训练项目智能分析与优化工具。接收用户的项目路径和启动命令，通过「代码静态分析 → 硬件资源采集 → 性能 Profiling → 瓶颈诊断 → 代码优化 → 验证迭代」的闭环流程，自动完成 GPU 训练项目的智能调优，最终输出优化后的代码和完整的优化报告。

## 触发条件

当用户提供一个 GPU 训练项目的路径和启动命令，要求进行智能调优、性能优化或加速分析时触发。

## 核心约束

**⚠️ 绝对不修改用户的原始项目。** 所有代码修改、配置变更、profiling 插桩等操作仅在 fork 的副本目录中进行。原始项目目录在整个流程中视为**只读**。

## 核心保障：最终报告必须生成

**⚠️ 无论流程中发生任何情况，第九步（生成 `final_optimization-summary.md`）和第十步（完整性检查 + 交付）必须执行。** 这是用户最终交付物，缺失则整个优化流程等于无效。

**Turn 预算管理规则：**

1. **迭代轮数控制**：最多执行 **5 轮**迭代优化（v1 → v2 → v3 → v4 → v5），之后无论是否收敛都必须进入第九步
2. **提前收敛优先**：每轮迭代后评估是否收敛，一旦收敛立即进入第九步，不要浪费 Turn 在收益甚微的额外迭代上
3. **异常降级**：如果某个步骤反复失败（如 Profiling 运行失败、优化验证失败），跳过该步骤继续后续流程，在最终报告中标注跳过原因
4. **部分报告兜底**：即使只完成了静态审计（第二步），也必须生成最终报告，在报告中说明哪些步骤已完成、哪些被跳过及原因

**流程保障检查点：**

- 完成第五步（瓶颈诊断）后：检查已消耗的工作量，如果已经很多，减少迭代轮数
- 完成每轮第八步（收敛判断）后：如果已完成 4 轮迭代，强制进入第九步
- 任何步骤失败 3 次后：跳过该步骤，继续后续流程

## 执行指令

你是 GPU 训练优化专家。按以下 10 个步骤完成端到端的智能调优流程。

---

### 第一步：确认输入与初始化工作目录

#### 1.1 收集必要信息

从用户的 prompt 中提取以下信息。**不要向用户反问**，直接从 prompt 和项目代码中获取。如果某项信息未在 prompt 中提供，按照默认策略处理：

| 信息                         | 必填 | 获取方式                                                       | 默认策略                                 |
| ---------------------------- | ---- | -------------------------------------------------------------- | ---------------------------------------- |
| **项目路径**           | 是   | 从 prompt 提取                                                 | —                                       |
| **启动命令**           | 是   | 从 prompt 提取                                                 | —                                       |
| **模型参数量**         | 推荐 | 从项目代码/配置推断（搜索 config.json、model_name_or_path 等） | 如无法推断则跳过，在第五步瓶颈诊断时补充 |
| **优化目标**           | 推荐 | 从 prompt 提取                                                 | 默认"综合优化（吞吐优先）"               |
| **Profiling 运行步数** | 可选 | 从 prompt 提取                                                 | 默认 10                                  |
| **多机模式**           | 自动 | 从启动命令/prompt 检测                                         | 默认否                                   |
| **训练框架**           | 自动 | 从项目代码/配置/依赖自动检测                                   | 默认"通用"                               |

#### 1.2 训练框架检测

扫描项目文件，识别是否使用特定训练框架。框架检测影响后续审计和优化策略的选择。

**LlamaFactory 检测规则**（满足任一即判定为 LlamaFactory 项目）：

| 检测方式              | 匹配模式                                                                                                                                                      |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **依赖文件**    | `requirements.txt`、`pyproject.toml`、`setup.py` 中包含 `llamafactory` 或 `llama-factory`                                                           |
| **import 语句** | `from llamafactory`、`import llamafactory`                                                                                                                |
| **CLI 命令**    | 启动命令包含 `llamafactory-cli`、`llamafactory.cli`                                                                                                       |
| **配置参数**    | YAML/JSON 中包含 LlamaFactory 特有参数：`finetuning_type`、`neat_packing`、`use_unsloth`、`enable_liger_kernel`、`lora_target`、`flash_attn: fa2` |
| **项目结构**    | 存在 `data/dataset_info.json`、或 `examples/train_lora/`、`examples/deepspeed/ds_z*_config.json`                                                        |

**ms-swift 检测规则**（满足任一即判定为 ms-swift 项目）：

| 检测方式               | 匹配模式                                                                                                                                                          |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **依赖文件**     | `requirements.txt`、`pyproject.toml`、`setup.py` 中包含 `ms-swift`                                                                                        |
| **import 语句**  | `from swift`、`import swift`、`from swift.llm`、`from swift.trainers`                                                                                     |
| **CLI 命令**     | 启动命令包含 `swift sft`、`swift pt`、`swift rlhf`、`megatron sft`、`megatron pt`                                                                       |
| **配置参数**     | YAML/CLI 中包含 ms-swift 特有参数：`tuner_type`、`tuner_backend`、`attn_impl`、`padding_free`、`loss_scale`（非 HF 标准值）、`sequence_parallel_size` |
| **环境变量启动** | 使用 `NPROC_PER_NODE=N swift sft` 方式启动多卡                                                                                                                  |
| **检查点**       | `args.json` 中包含 `swift_version` 字段                                                                                                                       |

**VideoX-Fun 检测规则**（满足任一即判定为 VideoX-Fun 项目）：

| 检测方式              | 匹配模式                                                                                                                                                    |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **依赖文件**    | `requirements.txt`、`pyproject.toml`、`setup.py` 中包含 `videox-fun` 或 `videox_fun`                                                              |
| **import 语句** | `from videox_fun`、`import videox_fun`                                                                                                                  |
| **项目结构**    | 存在 `videox_fun/` 目录（含 `models/`、`data/`、`pipeline/` 子目录）                                                                                |
| **脚本路径**    | 启动命令包含 `scripts/wan2.1_fun/`、`scripts/wan2.2_fun/`、`scripts/cogvideox_fun/`、`scripts/hunyuanvideo/` 等 VideoX-Fun 脚本路径                 |
| **配置文件**    | 存在 `config/wan2.1/wan_civitai.yaml`、`config/wan2.2/` 等 VideoX-Fun 配置                                                                              |
| **训练参数**    | 启动命令/脚本中包含 `--train_mode`（值为 `normal`/`inpaint`/`control_ref`）+ `--video_sample_n_frames`、`--vae_mini_batch`、`--enable_bucket` |
| **Git Remote**  | `.git/config` 中 remote URL 包含 `VideoX-Fun`                                                                                                           |

```bash
# 快速检测命令
FRAMEWORK="generic"

# 检查依赖文件 — LlamaFactory
grep -rls "llamafactory\|llama-factory\|LLaMA-Factory" \
  "${ORIGINAL_PROJECT}/requirements.txt" \
  "${ORIGINAL_PROJECT}/pyproject.toml" \
  "${ORIGINAL_PROJECT}/setup.py" 2>/dev/null && FRAMEWORK="llamafactory"

# 检查配置文件中的 LlamaFactory 特有参数
grep -rls "finetuning_type:\|neat_packing:\|use_unsloth:\|enable_liger_kernel:\|lora_target:" \
  "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/*.yml 2>/dev/null && FRAMEWORK="llamafactory"

# 检查启动命令 — LlamaFactory
echo "${LAUNCH_CMD}" | grep -q "llamafactory-cli" && FRAMEWORK="llamafactory"

# 检查依赖文件 — ms-swift
grep -rls "ms-swift" \
  "${ORIGINAL_PROJECT}/requirements.txt" \
  "${ORIGINAL_PROJECT}/pyproject.toml" \
  "${ORIGINAL_PROJECT}/setup.py" 2>/dev/null && FRAMEWORK="swift"

# 检查配置文件中的 ms-swift 特有参数
grep -rls "tuner_type:\|tuner_backend:\|attn_impl:\|padding_free:\|sequence_parallel_size:" \
  "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/*.yml 2>/dev/null && FRAMEWORK="swift"

# 检查启动命令 — ms-swift
echo "${LAUNCH_CMD}" | grep -qE "swift (sft|pt|rlhf|infer|deploy)|megatron (sft|pt|rlhf)" && FRAMEWORK="swift"

# 检查 import 语句 — ms-swift
grep -rls "from swift\|import swift" \
  "${ORIGINAL_PROJECT}"/*.py 2>/dev/null && FRAMEWORK="swift"

# 检查项目结构 — VideoX-Fun
if [ -d "${ORIGINAL_PROJECT}/videox_fun/models" ] && [ -d "${ORIGINAL_PROJECT}/videox_fun/data" ]; then
  FRAMEWORK="videox_fun"
fi

# 检查依赖文件 — VideoX-Fun
grep -rls "videox-fun\|videox_fun" \
  "${ORIGINAL_PROJECT}/requirements.txt" \
  "${ORIGINAL_PROJECT}/pyproject.toml" \
  "${ORIGINAL_PROJECT}/setup.py" 2>/dev/null && FRAMEWORK="videox_fun"

# 检查启动命令 — VideoX-Fun
echo "${LAUNCH_CMD}" | grep -qE "scripts/(wan2\.[12]|cogvideox|hunyuanvideo|flux|longcatvideo)" && FRAMEWORK="videox_fun"

# 检查配置文件 — VideoX-Fun
[ -f "${ORIGINAL_PROJECT}/config/wan2.1/wan_civitai.yaml" ] || \
  [ -d "${ORIGINAL_PROJECT}/config/wan2.2" ] && FRAMEWORK="videox_fun"

# 检查 Git remote — VideoX-Fun fork
git -C "${ORIGINAL_PROJECT}" remote -v 2>/dev/null | grep -qi "VideoX-Fun" && FRAMEWORK="videox_fun"

# 检查依赖文件 — Flow-Factory
grep -rls "flow-factory\|flow_factory" \
  "${ORIGINAL_PROJECT}/requirements.txt" \
  "${ORIGINAL_PROJECT}/pyproject.toml" \
  "${ORIGINAL_PROJECT}/setup.py" 2>/dev/null && FRAMEWORK="flow_factory"

# 检查 import 语句 — Flow-Factory
grep -rls "from flow_factory\|import flow_factory" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && FRAMEWORK="flow_factory"

# 检查启动命令 — Flow-Factory
echo "${LAUNCH_CMD}" | grep -qE "ff-train|flow-factory-train" && FRAMEWORK="flow_factory"

# 检查配置文件中的 Flow-Factory 特有参数
grep -rls "trainer_type:\|dynamics_type:\|offload_samples_to_cpu:\|sample_group_size:" \
  "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/*.yml "${ORIGINAL_PROJECT}"/examples/**/*.yaml 2>/dev/null && FRAMEWORK="flow_factory"

# 检查项目结构 — Flow-Factory
if [ -d "${ORIGINAL_PROJECT}/src/flow_factory" ]; then
  FRAMEWORK="flow_factory"
fi

# 检查 Git remote — Flow-Factory
git -C "${ORIGINAL_PROJECT}" remote -v 2>/dev/null | grep -qi "Flow-Factory" && FRAMEWORK="flow_factory"

# 检查是否使用 vLLM 推理框架（独立推理或 RLHF 集成）
# vLLM 作为推理框架可与训练框架并存，单独标记
VLLM_DETECTED="no"

# 检查依赖文件 — vLLM
grep -rls "^vllm\|vllm[>=<]" \
  "${ORIGINAL_PROJECT}/requirements.txt" \
  "${ORIGINAL_PROJECT}/pyproject.toml" \
  "${ORIGINAL_PROJECT}/setup.py" 2>/dev/null && VLLM_DETECTED="yes"

# 检查 import 语句 — vLLM
grep -rls "from vllm import\|from vllm\.\|import vllm" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && VLLM_DETECTED="yes"

# 检查启动命令 — vLLM
echo "${LAUNCH_CMD}" | grep -qE "vllm serve|python -m vllm" && VLLM_DETECTED="yes"

# 检查配置中的 vLLM 集成参数（RLHF 场景）
grep -rls "use_vllm.*[Tt]rue\|vllm_mode\|infer_backend.*vllm\|enable_sleep_mode" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/*.yml \
  "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && VLLM_DETECTED="yes"

# 检查 vLLM 特有配置参数
grep -rls "gpu_memory_utilization\|enforce_eager\|enable_prefix_caching\|enable_chunked_prefill\|max_num_seqs\|max_num_batched_tokens\|speculative_config" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && VLLM_DETECTED="yes"

# 如果仅有 vLLM（无训练框架），设为 vllm 框架
if [ "${FRAMEWORK}" = "generic" ] && [ "${VLLM_DETECTED}" = "yes" ]; then
  FRAMEWORK="vllm"
fi

# 检查是否使用 SGLang 推理框架（独立推理或 RLHF 集成）
# SGLang 作为推理框架可与训练框架并存，单独标记
SGLANG_DETECTED="no"

# 检查依赖文件 — SGLang
grep -rls "^sglang\|sglang[>=<]\|sgl-kernel" \
  "${ORIGINAL_PROJECT}/requirements.txt" \
  "${ORIGINAL_PROJECT}/pyproject.toml" \
  "${ORIGINAL_PROJECT}/setup.py" 2>/dev/null && SGLANG_DETECTED="yes"

# 检查 import 语句 — SGLang
grep -rls "from sglang import\|from sglang\.\|import sglang\|from sglang_router" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && SGLANG_DETECTED="yes"

# 检查启动命令 — SGLang
echo "${LAUNCH_CMD}" | grep -qE "sglang\.launch_server|sglang_router\.launch_server" && SGLANG_DETECTED="yes"

# 检查配置中的 SGLang 集成参数（RLHF 场景）
grep -rls "infer_backend.*sglang\|sglang_maxlen\|sglang_mem_fraction\|enable_memory_saver\|update_weights_from_distributed\|update_weights_from_tensor" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/*.yml \
  "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && SGLANG_DETECTED="yes"

# 检查 SGLang 特有配置参数
grep -rls "mem_fraction_static\|schedule_policy\|chunked_prefill_size\|enable_dp_attention\|radix_cache\|piecewise_cuda_graph" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && SGLANG_DETECTED="yes"

# 如果仅有 SGLang（无训练框架且无 vLLM），设为 sglang 框架
if [ "${FRAMEWORK}" = "generic" ] && [ "${SGLANG_DETECTED}" = "yes" ]; then
  FRAMEWORK="sglang"
fi

# 检查是否直接使用 HuggingFace Transformers Trainer（非上层框架封装）
# 仅在未检测到上层框架时检测
if [ "${FRAMEWORK}" = "generic" ]; then
  grep -rls "from transformers import Trainer\|from transformers import TrainingArguments\|from transformers import Seq2Seq" \
    "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && FRAMEWORK="hf_trainer"
  grep -rls "Trainer(model=\|TrainingArguments(" \
    "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && FRAMEWORK="hf_trainer"
fi

echo "检测到训练框架: ${FRAMEWORK}"
echo "检测到 vLLM 推理框架: ${VLLM_DETECTED}"
echo "检测到 SGLang 推理框架: ${SGLANG_DETECTED}"
```

> **框架扩展说明**：当前支持 LlamaFactory、ms-swift、VideoX-Fun、Flow-Factory、HuggingFace Transformers Trainer、vLLM 和 SGLang 专项审计。vLLM/SGLang 作为推理框架可与训练框架并存（如 LlamaFactory+vLLM、ms-swift+SGLang），会同时触发训练框架审计和推理框架审计。未识别框架时走通用 `training-acceleration-audit` 审计流程。HF Trainer 检测优先级最低，仅在其他框架均未匹配时触发。

#### 1.3 多机多卡检测

检查启动命令和 prompt 中的多机标志，并区分两种多机类型：

**类型 A — 可从单节点拉起的多机命令**（无需降级，直接多机 Profiling）：

```
检测标志：
  pdsh             → 分发到多节点执行
  mpirun / mpiexec → MPI launcher 自动编排
  deepspeed --hostfile → DeepSpeed launcher 自动 SSH 到各节点
  torchrun --nnodes>1 且无 --node_rank → elastic/rdzv 模式，单命令拉起
  prompt 中标注 "多机模式: 是（可从单节点拉起）"
```

**类型 B — 需在每个节点手动执行的命令**（需降级为单节点 Profiling）：

```
检测标志：
  torchrun --nnodes>1 --node_rank=X → 用户需在每个节点设置不同 node_rank
  prompt 中标注 "多机模式: 是（需降级为单节点）"
```

**两种类型的处理策略对比**：

| 步骤                | 类型 A（单节点可拉起）                 | 类型 B（需每节点手动执行）         |
| ------------------- | -------------------------------------- | ---------------------------------- |
| **Profiling** | 直接使用多机命令运行，采集真实多机数据 | 降级为单节点模式（`--nnodes=1`） |
| **代码优化**  | 包含通信优化，可直接实施并验证         | 单机优化 + 通信优化作为建议        |
| **验证**      | 多机环境真实验证                       | 仅单节点验证                       |
| **最终报告**  | 所有优化均为真实验证结果               | 区分"已验证"和"待多机验证"         |

**类型 B 降级方法**：

```bash
# 原始多机命令（需每节点执行）
torchrun --nnodes=4 --nproc_per_node=8 --node_rank=0 --master_addr=10.0.0.1 --master_port=29500 train.py

# 降级为单节点 profiling 命令（去除跨节点参数，保留单机参数）
torchrun --nproc_per_node=8 train.py
```

> **建议用户改造为类型 A**：如果用户的命令属于类型 B，在报告中建议改造为单节点可拉起的方式（torchrun elastic / deepspeed --hostfile / pdsh），以便后续获得完整的多机 Profiling 和优化验证。

#### 1.3 创建工作目录

```bash
# 工作目录 = 原始项目路径_optimized
ORIGINAL_PROJECT="<用户提供的项目路径>"
WORK_DIR="${ORIGINAL_PROJECT}_optimized"

mkdir -p "${WORK_DIR}/reports"
echo "工作目录已创建: ${WORK_DIR}"
echo "  ├── project/    # fork 的项目副本（所有修改在此）"
echo "  └── reports/    # 各阶段分析报告"
```

#### 1.3 记录元信息

在 `${WORK_DIR}/reports/` 中创建 `meta.md`，记录：

- 原始项目路径
- 原始启动命令
- 优化目标
- 开始时间
- 各步骤耗时（后续逐步填充）

---

### 第二步：代码静态分析

**目标**：在不运行代码的情况下，分析项目已使用和未使用的加速策略。

#### 2.1 执行通用训练加速审计（所有项目必做）

**无论是否为 LlamaFactory 项目**，都先按照 **`training-acceleration-audit` skill** 的执行指令，对**原始项目**（只读扫描）进行通用全面审计：

1. 扫描项目结构：依赖文件、训练入口、配置文件、启动脚本
2. 逐项检查 9 大类别（A-I）的使用状态：并行策略、训练框架、显存优化、计算优化、通信优化、数据 I/O、训练策略、多模态优化、基础设施
3. 生成带评分的审计报告

> 通用审计覆盖面广（torch.compile、CUDA Graph、Triton 自定义算子、通信优化等），能发现 LlamaFactory YAML 配置无法覆盖的优化机会。

#### 2.2 执行 LlamaFactory 专项审计（仅 LlamaFactory 项目）

**仅当 `FRAMEWORK=llamafactory` 时**，额外按照 **`llamafactory-optimization` skill** 的执行指令进行专项审计：

1. 识别训练模式：`stage`（pt/sft/dpo/kto/...）、`finetuning_type`（full/lora/freeze）、模型系列与规模
2. 逐项检查 10 大类别（A-J）的优化状态：
   - A. PEFT（LoRA/QLoRA/DoRA/rsLoRA/PiSSA/OFT）
   - B. 混合精度（BF16/FP16/pure_bf16/FP8/upcast）
   - C. 激活重计算（gradient_checkpointing/Unsloth GC/梯度累积）
   - D. 分布式显存优化（ZeRO-2/3/Offload/FSDP/FSDP+QLoRA/Megatron/AutoTP）
   - E. 高效优化器（GaLore/BAdam/APOLLO/Adam-mini/Muon）
   - F. 注意力优化（FA2/SDPA/S²-Attn）
   - G. 加速引擎（Unsloth/Liger Kernel/KTransformers）
   - H. 数据处理（neat_packing/streaming/预tokenize/DataLoader）
   - I. 训练策略（NEFTune/RoPE scaling/DFT/ASFT/EAFT/MoD）
   - J. 推理加速（vLLM/SGLang）
3. 基于 LlamaFactory 显存估算表评估当前配置合理性
4. 生成带评分（/58 分）的审计报告，包含场景化配置建议

> **两份报告互补**：LlamaFactory 专项审计聚焦框架内 YAML 可配置的优化项；通用审计覆盖框架外的深层优化（如 torch.compile、自定义 Triton 算子、CUDA Graph、通信拓扑优化、NCCL 调优等）。两者结合才能最大化优化空间。

#### 2.3 执行 ms-swift 专项审计（仅 ms-swift 项目）

**仅当 `FRAMEWORK=swift` 时**，额外按照 **`swift-optimization` skill** 的执行指令进行专项审计：

1. 识别训练模式：`swift sft` / `swift pt` / `swift rlhf --rlhf_type <type>` / `megatron sft`
2. 识别 RLHF 算法（如适用）：`dpo` / `kto` / `cpo` / `simpo` / `orpo` / `grpo` / `ppo` / `gkd`
3. 识别微调方式（`tuner_type`）：`lora` / `full` / `adalora` / `longlora` / `llamapro` / `vera` / `boft` / `bone`
4. 逐项检查 10 大类别（A-J）的优化状态：
   - A. PEFT（LoRA/QLoRA-BnB/QLoRA-HQQ/QLoRA-EETQ/FP8/DoRA/LoRA+/RS-LoRA/PiSSA/LoRA-GA/AdaLoRA/LISA/UnSloth）
   - B. 混合精度（BF16/FP16/FP8 Megatron）
   - C. 激活重计算与梯度优化（gradient_checkpointing/vit_gc/梯度累积/use_logits_to_keep/激活CPU卸载）
   - D. 分布式显存优化（ZeRO-2/3/Offload/ZeRO++/AutoTP/FSDP2/device_map/Megatron TP+PP+CP+EP）
   - E. 高效优化器（GaLore/Q-GaLore）
   - F. 注意力优化（Flash Attention 2/3/SDPA/FlexAttention/序列并行）
   - G. 加速引擎（Liger Kernel/torch.compile/UnSloth）
   - H. 数据处理（packing/padding_free/streaming/lazy_tokenize/DataLoader）
   - I. 训练策略（loss_scale/NEFTune/RoPE scaling/DFT/多模态冻结/分层学习率）
   - J. RLHF/GRPO 优化（vLLM推理加速/异步生成/CPU卸载/sleep_level/动态采样/prefix_caching）
5. 基于 ms-swift 显存估算表评估当前配置合理性
6. 生成带评分（/60 分）的审计报告，包含场景化配置建议

> **两份报告互补**：ms-swift 专项审计聚焦框架内 YAML/CLI 可配置的优化项（ms-swift 使用 `attn_impl` 而非 HF 的 `attn_implementation`，DeepSpeed 使用字符串快捷方式 `zero2`/`zero3` 等）；通用审计覆盖框架外的深层优化。两者结合才能最大化优化空间。

#### 2.4 执行 VideoX-Fun 专项审计（仅 VideoX-Fun 项目）

**仅当 `FRAMEWORK=videox_fun` 时**，额外按照 **`videox-fun-optimization` skill** 的执行指令进行专项审计：

1. 识别模型系列（Wan2.1-Fun/Wan2.2/CogVideoX-Fun/HunyuanVideo/Flux 等）和规模（1.3B/5B/14B）
2. 识别训练模式（`train_mode`）：`normal`（T2V）/ `inpaint`（I2V）/ `control_ref`（控制生成）
3. 识别微调方式：全参（`train.py`）/ LoRA（`train_lora.py`）/ Control / Reward LoRA / Distillation
4. 逐项检查 9 大类别（A-I）的优化状态：
   - A. 混合精度与量化（BF16/FP16/TF32/FP8/8-bit Adam/CAME）
   - B. 激活重计算（标准 GC/分数 GC/梯度累积）
   - C. 模型卸载与低显存（low_vram/vae_mini_batch/multi_stream）
   - D. 分布式显存优化（ZeRO-2/3/CPU Offload/FSDP Full Shard）
   - E. LoRA 配置（PEFT/自定义/rank/alpha/target_name/dropout）
   - F. 注意力优化（FA2/FA3/SageAttention/Variable-Length FA/SDPA/Sparse Linear Attention）
   - G. 数据管道（Bucket 采样/random_hw_adapt/auto_tile_batch_size/预编码数据集/DataLoader workers）
   - H. 训练策略（LR 调度/异常梯度裁剪/EMA/均匀时间步/运动子损失/Reward LoRA）
   - I. 深度优化机会（torch.compile/Liger Kernel/CUDA Graph/QLoRA/VAE 空间 tiling/FP8 训练/ZeRO++/FusedAdam/序列并行训练）
5. 基于 VideoX-Fun 显存估算表评估当前配置合理性（分辨率×帧数×模型规模）
6. 生成带评分（/61 分）的审计报告，包含场景化配置建议

> **两份报告互补**：VideoX-Fun 专项审计聚焦框架内 CLI 参数和脚本可配置的优化项（如 `--low_vram`、`--enable_bucket`、`--auto_tile_batch_size`、FSDP wrap class 等）；通用审计覆盖框架外的深层优化（如 torch.compile、Liger Kernel、CUDA Graph 等）。视频生成模型因分辨率×帧数导致显存需求极高，两层优化结合尤为关键。

#### 2.5 执行 Flow-Factory 专项审计（仅 Flow-Factory 项目）

**仅当 `FRAMEWORK=flow_factory` 时**，额外按照 **`flow-factory-optimization` skill** 的执行指令进行专项审计：

1. 识别 RL 算法（`trainer_type`）：grpo/nft/awm/dpo/dgpo/crd/diffusion-opd
2. 识别模型类型（`model_type`）：wan2_t2v/flux1/sd3-5/ltx2_t2av 等
3. 识别微调方式（`finetune_type`）：full/lora，LoRA rank/alpha/target_modules
4. 逐项检查 10 大类别（A-J）的优化状态：
   - A. 混合精度与数据类型（BF16/FP16/master_weight_dtype/latent_storage_dtype）
   - B. 梯度检查点与显存管理（gradient_checkpointing/offload_samples_to_cpu/ema_device/ref_param_device）
   - C. LoRA 配置（lora_rank/lora_alpha/target_modules/target_components）
   - D. 分布式训练（FSDP2/FSDP/DeepSpeed ZeRO-1/2/3/CPU Offload/多节点）
   - E. 注意力后端（flash_hub/flash_3_hub/sage/xformers/SDPA）
   - F. 优化器与训练策略（learning_rate/adam_params/max_grad_norm/LR scheduler 缺口）
   - G. 数据管道（dataloader_num_workers/sampler/caching/preprocessing）
   - H. 奖励系统优化（async_reward/batch_size/device/dtype/multi-source routing）
   - I. EMA 与调度器（ema_decay/ema_device/dynamics_type/flow_shift）
   - J. 深度优化机会（torch.compile/FusedAdam/8-bit Adam/Liger Kernel/QLoRA/LR scheduler）
5. 基于 Flow-Factory 显存估算表评估当前配置合理性
6. 生成带评分（/49 分）的审计报告，包含场景化配置建议

> **两份报告互补**：Flow-Factory 专项审计聚焦框架内 YAML 可配置的优化项（attn_backend、mixed_precision、offload_samples_to_cpu、FSDP2/DeepSpeed 选择等）和框架当前未内置但可通过代码实现的高收益优化（torch.compile、FusedAdam、QLoRA 等）；通用审计覆盖更广泛的基础设施和通信优化。Flow-Factory 作为 RL 微调框架，奖励系统优化是独特的优化维度。

#### 2.6 执行 HuggingFace Transformers Trainer 专项审计（仅 HF Trainer 项目）

**仅当 `FRAMEWORK=hf_trainer` 时**，额外按照 **`transformers-optimization` skill** 的执行指令进行专项审计：

1. 识别训练任务类型（CausalLM/分类/Seq2Seq）、Trainer 类型、模型规模
2. 识别 PEFT 使用情况（LoRA/QLoRA/无）和量化配置（BitsAndBytes/GPTQ/AWQ/TorchAO/FP8）
3. 逐项检查 10 大类别（A-J）的优化状态：
   - A. 混合精度（bf16/fp16/tf32/full_eval）
   - B. 梯度检查点与显存管理（gradient_checkpointing/torch_empty_cache_steps/auto_find_batch_size）
   - C. PEFT 与量化（LoRA/QLoRA/GPTQ/AWQ/TorchAO/FP8/DoRA）
   - D. 优化器（adamw_torch_fused/adamw_bnb_8bit/4bit/paged/adafactor/galore/apollo/lomo/schedule_free）
   - E. 分布式优化（DDP/FSDP full_shard/hybrid_shard/offload/FSDP2/DeepSpeed ZeRO/parallelism_config TP/CP/SP）
   - F. 注意力优化（sdpa/flash_attention_2/flash_attention_3/flex_attention）
   - G. 计算加速（torch_compile/use_liger_kernel/liger_kernel_config）
   - H. 数据处理（dataloader_num_workers/pin_memory/persistent_workers/prefetch_factor/group_by_length）
   - I. 训练策略（neftune_noise_alpha/gradient_accumulation/label_smoothing/save_only_model）
   - J. 推理优化（Static/Quantized/Offloaded KV Cache/推测解码/torch.compile 推理）
4. 基于 Transformers 显存估算表评估当前配置合理性
5. 生成带评分（/62 分）的审计报告，包含场景化 TrainingArguments 配置建议

> **两份报告互补**：Transformers Trainer 专项审计聚焦 `TrainingArguments` 参数和 `from_pretrained()` 配置可优化的项目（如 `torch_compile`、`use_liger_kernel`、`optim`、`fsdp`、`attn_implementation` 等）；通用审计覆盖框架外的深层优化（如自定义 Triton 算子、CUDA Graph、通信拓扑优化等）。两者结合才能最大化优化空间。

#### 2.6 执行 vLLM 推理优化审计（检测到 vLLM 时）

**当 `VLLM_DETECTED=yes` 时**（无论训练框架是什么），额外按照 **`vllm-optimization` skill** 的执行指令进行推理专项审计：

1. 识别 vLLM 使用模式：`standalone_serve`（独立部署）/ `standalone_offline`（离线批量）/ `rlhf_server`（RL server 模式）/ `rlhf_colocate`（RL colocate 模式）/ `framework_integrated`（框架集成）
2. 识别模型规模、GPU 架构、并行配置
3. 逐项检查 10 大类别（A-J）的优化状态：
   - A. 引擎与编译优化（优化级别 O0-O3/CUDA Graph/torch.compile/V1 引擎）
   - B. 显存管理（gpu_memory_utilization/max_model_len/KV Cache 量化/Swap/Offload）
   - C. 并行策略（TP/PP/DP/EP）
   - D. 量化策略（FP8/AWQ/GPTQ/BitsAndBytes/在线量化/混合量化）
   - E. 调度与批处理（chunked_prefill/max_num_seqs/max_num_batched_tokens/prefix_caching）
   - F. 注意力后端（Flash Attention 2/3/4/FlashInfer/FlashMLA）
   - G. 推测解码（EAGLE/MTP/Draft Model/N-gram）
   - H. RLHF/训练集成（sleep mode/weight transfer/colocate/async RL/动态 LoRA）
   - I. 生产部署（LoRA 服务/多进程 API/Disaggregated Prefill/多模态限制）
   - J. 环境与调优（FastTokens/CPU 核心/Dev Mode/预抢占监控）
4. 基于 vLLM 性能参考表评估当前配置合理性
5. 生成带评分（/52 分）的审计报告，包含场景化配置建议

> **vLLM 审计与训练框架审计互补**：vLLM 审计聚焦推理引擎的性能优化（显存利用率、调度策略、量化、推测解码等），训练框架审计聚焦训练侧优化（LoRA、混合精度、梯度检查点等）。在 RLHF/GRPO 场景中，两者结合覆盖完整的训练+推理优化空间。

#### 2.7 执行 SGLang 推理优化审计（检测到 SGLang 时）

**当 `SGLANG_DETECTED=yes` 时**（无论训练框架是什么），额外按照 **`sglang-optimization` skill** 的执行指令进行推理专项审计：

1. 识别 SGLang 使用模式：`standalone_serve`（独立部署）/ `standalone_offline`（离线批量）/ `rlhf_colocate`（RL colocate 模式）/ `rlhf_server`（RL server 模式）/ `framework_integrated`（框架集成）/ `router_gateway`（SMG 多副本）
2. 识别模型规模、模型架构类型（Dense/MoE/MLA/DSA）、GPU 架构、并行配置
3. 逐项检查 10 大类别（A-J）的优化状态：
   - A. 引擎与计算优化（Piecewise CUDA Graph/CUDA Graph Max BS/torch.compile/Overlap Scheduling/Two-Batch Overlap/DeepGEMM JIT）
   - B. 显存管理（mem_fraction_static/max_total_tokens/context_length/KV Cache 量化/chunked_prefill_size/HiCache）
   - C. 前缀缓存（RadixAttention/驱逐策略/LPM 调度/Chunked Prefix Cache/Page Size）
   - D. 并行策略（TP/DP/EP/PP/DP Attention/SGLang Model Gateway/NCCL NVLS）
   - E. 量化策略（FP8/AWQ/GPTQ/Marlin/BitsAndBytes/ModelOpt/TorchAO/MoE 专用量化）
   - F. 注意力后端（FlashInfer/FA3/FA4/FlashMLA/CutlassMLA/TRTLLM/DSA/混合后端）
   - G. 推测解码（EAGLE-3/EAGLE-2/MTP/STANDALONE/N-gram）
   - H. RLHF/训练集成（Memory Saver/权重更新三策略/暂停继续/确定性推理/R-Fork）
   - I. 生产部署（SGLang Model Gateway/PD Disaggregation/LoRA 服务/结构化输出/HiSparse/监控）
   - J. 环境与调优（schedule_conservativeness/Docker 优化/token usage 监控/CPU 核心数）
4. 基于 SGLang 性能参考表评估当前配置合理性
5. 生成带评分（/54 分）的审计报告，包含场景化配置建议

> **SGLang 审计与训练框架审计互补**：SGLang 审计聚焦推理引擎的性能优化（RadixAttention、DP Attention、调度策略、显存管理等），训练框架审计聚焦训练侧优化。在 RLHF/GRPO 场景中，SGLang 的 Memory Saver（sleep/wake）、三种权重更新策略和确定性推理等特性是 RL 训练效率的关键。

#### 2.8 保存报告

```
# 所有项目都会生成通用审计报告
${WORK_DIR}/reports/v0_training-acceleration-audit-report.md

# LlamaFactory 项目额外生成专项审计报告
${WORK_DIR}/reports/v0_llamafactory-optimization-report.md

# ms-swift 项目额外生成专项审计报告
${WORK_DIR}/reports/v0_swift-optimization-report.md

# VideoX-Fun 项目额外生成专项审计报告
${WORK_DIR}/reports/v0_videox-fun-optimization-report.md

# Flow-Factory 项目额外生成专项审计报告
${WORK_DIR}/reports/v0_flow-factory-optimization-report.md

# HF Trainer 项目额外生成专项审计报告
${WORK_DIR}/reports/v0_transformers-optimization-report.md

# 检测到 vLLM 时额外生成推理优化审计报告
${WORK_DIR}/reports/v0_vllm-optimization-report.md

# 检测到 SGLang 时额外生成推理优化审计报告
${WORK_DIR}/reports/v0_sglang-optimization-report.md
```

#### 2.8 提取关键发现

从审计报告（通用 + 框架专项 + vLLM 推理专项）中合并提取：

- **已使用的加速项**：作为后续优化的基线
- **未使用的高优先级项**：作为优化候选，区分"参数可配置"和"需改代码"
- **综合得分**：作为优化前基准
- **（LlamaFactory）推荐配置模板**：匹配用户场景（模型规模+硬件）的最优配置
- **（ms-swift）推荐配置模板**：匹配用户场景的最优 YAML/CLI 配置（参考 swift-optimization skill 的场景化模板）
- **（VideoX-Fun）推荐配置模板**：匹配用户场景（模型系列+规模+分辨率+帧数+硬件）的最优启动命令（参考 videox-fun-optimization skill 的场景化模板）
- **（Flow-Factory）推荐配置模板**：匹配用户场景（模型类型+RL算法+规模+硬件）的最优 YAML 配置（参考 flow-factory-optimization skill 的场景化模板）
- **（HF Trainer）推荐 TrainingArguments 配置**：匹配用户场景的最优 TrainingArguments 参数和 from_pretrained 加载参数
- **（vLLM）推荐 vLLM 配置**：匹配用户场景的最优 vLLM 引擎参数（gpu_memory_utilization、量化、调度、推测解码、RLHF 集成等）
- **（SGLang）推荐 SGLang 配置**：匹配用户场景的最优 SGLang 引擎参数（mem_fraction_static、RadixAttention、DP Attention、量化、推测解码、RLHF 集成等）
- **（通用）框架外优化机会**：自定义算子、通信调优等需要改代码的高收益项

> 此步骤仅读取原始项目，不做任何修改。

---

### 第三步：Fork 项目

**目标**：创建项目的完整副本，后续所有修改仅在副本中进行。

#### 3.1 复制项目

```bash
cp -r "${ORIGINAL_PROJECT}" "${WORK_DIR}/project"
```

#### 3.2 验证 fork 完整性

```bash
# 验证文件数量一致
echo "原始项目文件数: $(find "${ORIGINAL_PROJECT}" -type f | wc -l)"
echo "Fork 项目文件数: $(find "${WORK_DIR}/project" -type f | wc -l)"
```

#### 3.3 排除不必要的大文件（可选）

如果项目中包含大量数据文件、模型权重等，可以使用符号链接代替拷贝：

```bash
# 对大文件目录使用符号链接（保持只读语义）
# ln -s "${ORIGINAL_PROJECT}/data" "${WORK_DIR}/project/data"
# ln -s "${ORIGINAL_PROJECT}/checkpoints" "${WORK_DIR}/project/checkpoints"
```

> **此刻起，所有代码修改仅在 `${WORK_DIR}/project/` 中进行。**

---

### 第四步：性能 Profiling 基准采集

**目标**：在 fork 项目中插入 profiler 代码，运行短时训练采集性能基准数据。

#### 4.0 多机模式：Profiling 启动命令处理

**如果在第一步检测到多机模式**，根据类型选择不同的 Profiling 策略：

**类型 A（可从单节点拉起）**：直接使用用户的多机启动命令执行 Profiling，无需降级。Profiling 数据包含真实的跨节点通信开销，可完整诊断通信瓶颈。

**类型 B（需每节点手动执行）**：将启动命令降级为单节点模式再执行 Profiling：

1. **去除跨节点参数**：移除 `--nnodes`、`--node_rank`、`--master_addr`、`--master_port` 等
2. **保留单机参数**：保留 `--nproc_per_node`（使用当前节点的 GPU 数量）
3. **调整 DDP 初始化**：确保 fork 代码兼容单节点运行（通常 `torchrun` 会自动处理）

```bash
# 示例：降级后的单节点 Profiling 命令
NGPUS=$(nvidia-smi -L | wc -l)
torchrun --nproc_per_node=${NGPUS} train.py [其他非分布式参数]
```

> **说明**：单节点 Profiling 能够定位大部分性能瓶颈（计算效率、混合精度、算子融合、DataLoader 等）。多机通信瓶颈无法在单节点复现，将在第六步作为优化建议输出。建议用户改造为类型 A 命令以获得完整的多机 Profiling。

**如果是单机模式**，跳过此步，直接进入 4.1。

#### 4.0.1 端口冲突防护（所有涉及 torchrun/分布式训练的场景必做）

**⚠️ 重要**：每次运行 `torchrun` 或其他分布式训练命令前，**必须**执行以下端口管理步骤，防止因残留进程占用端口导致 EADDRINUSE 错误。

**1. 清理残留进程**：

```bash
# 查找并杀死占用默认端口（29500）的残留进程
cleanup_port() {
    local PORT=${1:-29500}
    local PIDS=$(lsof -t -i :${PORT} 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "发现残留进程占用端口 ${PORT}，正在清理: $PIDS"
        echo "$PIDS" | xargs kill -TERM 2>/dev/null
        sleep 2
        # 仍存活的进程强制杀死
        PIDS=$(lsof -t -i :${PORT} 2>/dev/null)
        if [ -n "$PIDS" ]; then
            echo "$PIDS" | xargs kill -KILL 2>/dev/null
        fi
    fi
}
cleanup_port 29500
```

**2. 使用随机端口**：

```bash
# 生成随机可用端口（避免与其他训练进程冲突）
get_free_port() {
    python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}
RANDOM_PORT=$(get_free_port)

# 如果环境变量 MASTER_PORT 已预设，优先使用
MASTER_PORT=${MASTER_PORT:-$RANDOM_PORT}
```

**3. 在所有 torchrun 命令中显式指定端口**：

```bash
# ✅ 正确 — 显式指定 --master_port
torchrun --nproc_per_node=N --master_port=${MASTER_PORT} train.py

# ❌ 错误 — 使用默认端口 29500，容易冲突
torchrun --nproc_per_node=N train.py
```

**4. 训练完成后清理**：

```bash
# 每次训练命令执行完毕后，确保所有子进程退出
cleanup_port ${MASTER_PORT}
```

> **为什么需要这步？** 在调优流程中，会多次运行训练命令（基准 profiling → 优化后验证 → 迭代 profiling），如果上一次运行的 `torchrun` 子进程未完全退出（常见于 OOM、SIGTERM、超时等异常退出），端口会被残留进程占用，导致下一次运行立即失败（`EADDRINUSE: address already in use`），进而导致整个优化流程中断。

#### 4.1 修改 fork 代码以插入 Profiler

按照 **`pytorch-profiler` skill** 的模板 C（长训练任务周期性采样），在 fork 项目的训练入口中插入 profiler 代码：

```python
# 在 fork 项目的训练脚本中添加
from torch.profiler import profile, schedule, tensorboard_trace_handler, ProfilerActivity

profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(skip_first=5, wait=2, warmup=1, active=3, repeat=2),
    on_trace_ready=tensorboard_trace_handler("${WORK_DIR}/reports/v0_profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
)
```

**同时修改训练脚本以满足 profiling 需求：**

- **禁用模型保存**：注释或跳过 `save_checkpoint` / `save_pretrained` / `save_model` 等调用
- **限制运行步数**：设置 `max_steps=10`（或用户指定的步数），确保快速完成
- **保留完整训练逻辑**：不跳过 forward/backward/optimizer step，确保 profiling 数据真实

**⚠️ 框架特定的 Profiler 插入策略：**

**LlamaFactory 项目**（`FRAMEWORK=llamafactory`）：

- **步数限制和禁用保存**：优先通过修改 YAML 配置实现，而非改代码：
  ```yaml
  # 在 fork 的 YAML 配置中添加/修改
  max_steps: 10          # 限制步数（覆盖 num_train_epochs）
  save_steps: 999999     # 实质禁用保存
  save_strategy: "no"    # 显式禁用保存策略（如 Trainer 支持）
  logging_steps: 1       # 每步输出日志，方便观察
  ```
- **Profiler 插桩位置**：LlamaFactory 底层使用 HuggingFace Trainer，Profiler 应插入到 `src/llamafactory/train/trainer_utils.py` 或对应 stage 的 `workflow.py` 中，在 `Trainer` 初始化时通过 `TrainerCallback` 注入 profiler，避免深改训练循环
- **自定义启动脚本处理**：如果用户的启动命令是自定义脚本（如 `python start.sh --config xxx.yaml` 或 `bash train.sh`），需要：
  1. 先读取该脚本，理解其功能（通常是设置环境变量后调用 `llamafactory-cli train` 或等效入口）
  2. 找到实际的 YAML 配置文件路径
  3. 通过修改 YAML 配置实现步数限制和保存禁用
  4. 将 Profiler 插桩到 LlamaFactory 的训练入口（而非自定义脚本本身）

**ms-swift 项目**（`FRAMEWORK=swift`）：

- **步数限制**：通过 YAML/CLI 配置 `max_steps: 10`、`save_steps: 999999`
- **Profiler 插桩**：ms-swift 底层也使用 HuggingFace Trainer，插桩策略同 LlamaFactory

**VideoX-Fun 项目**（`FRAMEWORK=videox_fun`）：

- **步数限制**：修改训练脚本的 CLI 参数 `--max_train_steps 10`、`--checkpointing_steps 999999`
- **Profiler 插桩**：在训练入口脚本中插入

**HF Trainer 项目**（`FRAMEWORK=hf_trainer`）：

- **步数限制**：修改 `TrainingArguments(max_steps=10, save_strategy="no")`
- **Profiler 插桩**：通过 `TrainerCallback` 注入

> **原则**：对于使用 Trainer 体系的框架（LlamaFactory、ms-swift、HF Trainer），优先通过配置/参数控制步数和保存，通过 Callback 机制注入 Profiler，避免直接修改训练循环代码。这样既省工作量又降低出错风险。

#### 4.2 Nsight Systems 采集（可选）

**先检测 nsys 是否可用且版本足够：**

```bash
which nsys && nsys --version
nsys profile --help 2>&1 | grep -q "\-\-pytorch" && echo "pytorch_supported=yes" || echo "pytorch_supported=no"
```

- **如果 nsys 不可用或不支持 `--pytorch`**：按照 **`nsys-install` skill** 的完整流程自动安装或升级 nsys。安装完成后再继续下面的采集步骤。如果安装失败（如无 root 权限、网络不通），则跳过 Nsight Systems 采集，仅依赖 PyTorch Profiler。
- **如果 nsys 可用且支持 `--pytorch`**：按照 **`nsight-systems` skill** 的模板 B（PyTorch DL 脚本完整分析），生成包裹命令：

```bash
nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=autograd-nvtx \
  --cuda-memory-usage=true \
  --python-sampling=true \
  --output="${WORK_DIR}/reports/v0_nsys_report" \
  --force-overwrite=true \
  <修改后的启动命令>
```

> 注意：如果原始启动命令使用 `torchrun`，需按 nsight-systems skill 模板 C 的方式适配多卡场景。

- **如果 nsys 不可用（安装也失败）**：跳过 Nsight Systems 采集，仅依赖 PyTorch Profiler 的数据进行分析。在报告中注明 nsys 不可用。

#### 4.3 执行 Profiling

**⚠️ 运行前必做**：按 4.0.1 节清理残留进程并确定 `MASTER_PORT`。

**必选 — PyTorch Profiler**：运行插入了 profiler 代码的训练脚本

```bash
cd "${WORK_DIR}/project"
# 清理残留进程
cleanup_port ${MASTER_PORT:-29500}
# 使用随机端口运行（如启动命令包含 torchrun）
<修改后的启动命令（已含 profiler 插桩，已限制步数，已添加 --master_port=${MASTER_PORT}）>
# 训练完成后清理
cleanup_port ${MASTER_PORT:-29500}
```

**可选 — Nsight Systems**（仅在 4.2 检测到 nsys 可用时执行）：

```bash
cd "${WORK_DIR}/project"
cleanup_port ${MASTER_PORT:-29500}
nsys profile <...参数...> <启动命令（限制步数，禁用保存，已添加 --master_port=${MASTER_PORT}）>
cleanup_port ${MASTER_PORT:-29500}
```

采集完成后（如 nsys 已执行）：

```bash
nsys stats "${WORK_DIR}/reports/v0_nsys_report.nsys-rep" \
  --report cuda_gpu_kern_sum --report cuda_api_sum \
  --format column
```

#### 4.4 分析结果并输出报告

按照 **`pytorch-profiler` skill** 第三步（解读结果）的报告模板生成：

```
${WORK_DIR}/reports/v0_pytorch-profiler-report.md
```

如果执行了 Nsight Systems 采集，还需按照 **`nsight-systems` skill** 第四步（查看和分析结果）的报告模板生成：

```
${WORK_DIR}/reports/v0_nsight-systems-report.md
```

报告中应包含：

- Top 10 耗时 CUDA Kernel
- Top 10 耗时算子
- GPU 利用率（活跃时间占比）
- 显存峰值使用
- CPU-GPU 同步阻塞点
- NCCL 通信耗时（多卡场景）

#### 4.5 Profiling 失败降级处理

**⚠️ 如果 PyTorch Profiler 运行失败**（如环境问题、依赖缺失、训练脚本启动失败等），按以下策略降级：

1. **尝试修复**：分析错误信息，尝试修复（最多 3 次）
2. **降级为无 Profiler 运行**：如果 Profiler 插桩导致失败，去掉 Profiler 代码，仅运行训练脚本采集基本运行时指标（通过 `nvidia-smi` 采集 GPU 利用率和显存）
3. **仅依赖静态分析**：如果训练脚本完全无法运行，跳过 Profiling，仅依赖第二步的静态审计报告制定优化方案。在瓶颈诊断报告中标注"Profiling 不可用，以下优化基于静态代码分析"
4. **继续后续流程**：无论 Profiling 是否成功，都必须继续第五步及后续步骤，最终生成报告

> **关键**：Profiling 失败不应阻塞整个优化流程。静态审计报告（第二步）已提供充足的优化建议，足以指导第六步的优化实施。

---

### 第五步：环境信息采集与瓶颈诊断

**目标**：结合硬件信息和 profiling 数据，综合诊断训练瓶颈。

#### 5.1 采集硬件资源信息

按照 **`system-resources` skill** 的执行指令，逐步采集：

1. CPU 信息（型号、核心数、NUMA）
2. 内存信息（总量、可用）
3. GPU 信息（型号、显存、利用率、温度、功耗）
4. 磁盘信息（NVMe 可用性、存储类型）
5. 网卡信息（RDMA 状态、带宽）
6. GPU 拓扑（NVLink 互联关系）

输出：

```
${WORK_DIR}/reports/v0_system-resources-report.md
```

**如果是迭代轮次（vN, N≥1）**，还需采集**训练运行时的动态信息**：

```bash
# 在训练运行期间采集 GPU 实时状态（每 2 秒采样一次，采 30 次）
nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw --format=csv -l 2 | head -120
```

#### 5.2 综合瓶颈诊断

**交叉分析**以下数据源，识别瓶颈：

| 数据源                                       | 分析目标                                      |
| -------------------------------------------- | --------------------------------------------- |
| `v0_pytorch-profiler-report.md`            | 算子级热点、显存分配模式                      |
| `v0_nsight-systems-report.md`              | GPU 空闲原因、通信重叠度、kernel launch 开销  |
| `v0_system-resources-report.md`            | 硬件能力上限、资源利用率差距                  |
| `v0_training-acceleration-audit-report.md` | 通用审计：已知未使用的加速项（所有项目）      |
| `v0_llamafactory-optimization-report.md`   | LlamaFactory 专项审计（仅 LlamaFactory 项目） |
| `v0_swift-optimization-report.md`          | ms-swift 专项审计（仅 ms-swift 项目）         |
| `v0_videox-fun-optimization-report.md`     | VideoX-Fun 专项审计（仅 VideoX-Fun 项目）     |
| `v0_flow-factory-optimization-report.md`   | Flow-Factory 专项审计（仅 Flow-Factory 项目） |
| `v0_transformers-optimization-report.md`   | HF Trainer 专项审计（仅 HF Trainer 项目）     |
| `v0_vllm-optimization-report.md`           | vLLM 推理优化审计（检测到 vLLM 时）           |
| `v0_sglang-optimization-report.md`         | SGLang 推理优化审计（检测到 SGLang 时）       |

**瓶颈分类与诊断矩阵：**

| 瓶颈类型                         | 诊断信号                                       | 影响程度 |
| -------------------------------- | ---------------------------------------------- | -------- |
| **显存不足**               | OOM、micro_batch_size 被迫很小、显存占用 > 90% | 高       |
| **GPU 空闲等数据**         | GPU 利用率 < 80%、DataLoader 耗时占比 > 30%    | 高       |
| **CPU-GPU 同步阻塞**       | 大量 cudaStreamSynchronize、.item() 调用       | 高       |
| **未使用混合精度**         | 全 FP32 训练、无 AMP/BF16                      | 高       |
| **未使用 Flash Attention** | 标准 attention 算子占比大、显存 O(N²)         | 高       |
| **Kernel Launch 开销**     | 大量极短 kernel（< 10us）间有间隙              | 中       |
| **通信瓶颈**               | NCCL 占比 > 30%、通信与计算不重叠              | 中       |
| **DataLoader 未优化**      | num_workers=0、无 pin_memory、无 prefetch      | 中       |
| **优化器未融合**           | 使用 PyTorch 原生 Adam 而非 FusedAdam          | 低       |
| **Checkpoint 保存频繁**    | save_steps 过小                                | 低       |

输出：

```
${WORK_DIR}/reports/v0_bottleneck-analysis.md
```

报告格式：

```markdown
# 瓶颈诊断报告 (v0)

## 瓶颈总览
| # | 瓶颈类型 | 严重程度 | 诊断依据 | 推荐优化 |
|---|---------|---------|---------|---------|
| 1 | ... | 高/中/低 | <来自哪份报告的哪个数据> | <对应的优化 skill/方法> |

## 详细分析
### 瓶颈 1：<名称>
- **现象**：<profiler 数据>
- **根因**：<分析>
- **解决方案**：<引用对应 skill>
- **预期收益**：<估算>

## 优化方案（按优先级排序）
1. [高] <方案> — 对应 skill: /xxx — 预期收益: xxx
2. [中] <方案> — 对应 skill: /xxx — 预期收益: xxx
3. [低] <方案>
```

---

### 第六步：实施优化

**目标**：根据瓶颈诊断报告，在 fork 项目上实施代码和配置优化。

#### 6.1 制定优化方案

读取 `v{N}_bottleneck-analysis.md` 中的优化方案列表，按优先级从高到低逐项实施。

#### 6.2 按框架调用对应优化策略

---

**🔹 LlamaFactory 项目优化**（`FRAMEWORK=llamafactory`）：

LlamaFactory 项目采用**两层优化策略**：优先通过 YAML 配置实现（低风险），当 YAML 无法覆盖的高收益优化出现时，允许修改 LlamaFactory 框架代码（在 fork 副本中）。

**第一层：YAML 配置优化（优先执行，零风险）**

| 优先级       | 优化项             | YAML 配置修改                                                       | 预期收益                                     |
| ------------ | ------------------ | ------------------------------------------------------------------- | -------------------------------------------- |
| **P0** | Flash Attention 2  | `flash_attn: fa2`                                                 | Attention 显存 O(N²)→O(N)，训练加速 10~30% |
| **P0** | BF16 混合精度      | `bf16: true`（A100/H100/H20）`<br>`或 `fp16: true`（V100/T4） | 显存减半，计算翻倍                           |
| **P0** | Neat Packing       | `neat_packing: true`                                              | 消除 padding 浪费，吞吐提升 20~50%           |
| **P0** | 梯度检查点         | `gradient_checkpointing: true`                                    | 激活显存降低 ~60%                            |
| **P1** | Unsloth 加速       | `use_unsloth: true`                                               | **训练速度 +70%**，显存 -50%（LoRA）   |
| **P1** | Liger Kernel       | `enable_liger_kernel: true`                                       | 吞吐 +20%，显存 -60%                         |
| **P1** | NEFTune            | `neftune_noise_alpha: 5`                                          | 下游效果提升（SFT 阶段）                     |
| **P1** | QLoRA 4bit         | `quantization_bit: 4`、`quantization_type: nf4`                 | 7B→~6GB                                     |
| **P1** | DataLoader 并行    | `dataloader_num_workers: 4`、`preprocessing_num_workers: 8`     | GPU 等待时间 -50%                            |
| **P2** | DeepSpeed ZeRO-2/3 | `deepspeed: ds_z2/z3_config.json`                                 | 多卡显存分片                                 |
| **P2** | FSDP+QLoRA         | FSDP YAML +`quantization_bit: 4`                                  | 70B@2x24GB                                   |
| **P2** | Unsloth GC         | `use_unsloth_gc: true`                                            | 显存极紧张时                                 |
| **P2** | DoRA/GaLore/FP8 等 | 对应 YAML 参数                                                      | 按需启用                                     |

```bash
# YAML 优化实施方式
CONFIG_FILE=$(find "${WORK_DIR}/project" -name "*.yaml" -o -name "*.yml" | head -1)
# 使用 Edit 工具修改 YAML 配置项
# 如需 DeepSpeed/FSDP，在 fork 目录中创建对应配置文件
```

**第二层：框架代码级优化（高收益项，YAML 无法实现时）**

当通用审计（`v0_training-acceleration-audit-report.md`）或 Profiling 发现以下高收益优化机会，且 LlamaFactory YAML 配置无法覆盖时，**允许在 fork 副本中修改 LlamaFactory 框架代码**：

| 优化项                       | 修改目标                                                                                               | 适用场景                                                   | 预期收益                |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- | ----------------------- |
| **torch.compile**      | 在 fork 项目的训练入口中添加 `model = torch.compile(model)`                                          | PyTorch 2.x+，LlamaFactory 未原生支持 torch.compile 配置时 | 算子融合，吞吐 +10~30%  |
| **自定义 Triton 算子** | 参考 `triton-optimization` skill 的决策树和模板，在 fork 项目中添加高性能 Triton kernel 替换热点算子 | Profiling 发现特定算子成为瓶颈且无成熟库覆盖               | 单算子加速 2~5x         |
| **CUDA Graph**         | 在训练循环中启用 CUDA Graph 捕获                                                                       | 大量小 kernel、launch overhead 高                          | kernel launch 开销 -90% |
| **数据管道深度优化**   | 修改数据预处理逻辑（多线程 tokenize、内存映射等）                                                      | DataLoader 仍是瓶颈、prefetch 不足                         | I/O 吞吐提升            |
| **通信拓扑优化**       | 修改启动脚本中的 NCCL 环境变量、绑核策略                                                               | 多机多卡通信效率低                                         | 通信耗时 -20~40%        |
| **自定义 loss 融合**   | 在 fork 中实现 fused loss function（如 Liger FusedLinearCrossEntropy 的自定义变体）                    | 大词表场景 loss 计算成为热点                               | 显存降低、吞吐提升      |
| **混合精度细粒度控制** | 修改模型代码中特定层的精度策略                                                                         | 训练不稳定需要逐层调精度                                   | 稳定性 + 性能平衡       |
| **优化器参数分组**     | 修改训练代码中的参数分组策略（不同 lr/wd）                                                             | 微调效果需要精细控制                                       | 收敛质量提升            |

**代码修改原则**：

1. **所有修改仅在 fork 副本中**（`${WORK_DIR}/project/`），绝不动原始项目
2. **优先 YAML 配置 → 其次框架代码修改 → 最后自定义代码注入**
3. **最小化侵入**：尽量在训练入口点（如 `src/llamafactory/train/tuner.py`、`workflow.py`）添加 wrapper，避免深改框架内部逻辑
4. **可回退**：每项代码修改记录 diff，验证失败时可独立回退

```python
# 示例：在 fork 的 LlamaFactory 训练入口中注入 torch.compile
# 文件: ${WORK_DIR}/project/src/llamafactory/train/sft/workflow.py
# 在 model = load_model(...) 之后添加：

if torch.cuda.is_available() and hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")
```

```bash
# 示例：在启动脚本中注入 NCCL 通信优化环境变量
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_BUFFSIZE=16777216
export NCCL_P2P_LEVEL=NVL
```

> **核心理念**：LlamaFactory YAML 配置能覆盖 ~80% 的常见优化，但 Profiling 可能揭示框架层面的深层瓶颈（如 kernel launch overhead、通信拓扑不合理、特定算子效率低等），这些需要代码级干预才能解决。两层策略结合，确保不遗漏高收益优化机会。

---

**🔹 ms-swift 项目优化**（`FRAMEWORK=swift`）：

ms-swift 项目采用**两层优化策略**：优先通过 YAML/CLI 配置实现（低风险），当配置无法覆盖的高收益优化出现时，允许修改代码（在 fork 副本中）。

**第一层：YAML/CLI 配置优化（优先执行，零风险）**

ms-swift 支持 YAML 配置文件（`--config xxx.yaml`）和 CLI 参数两种方式，优先修改 YAML 配置文件。

| 优先级       | 优化项               | YAML/CLI 配置修改                                                                                                         | 预期收益                                     |
| ------------ | -------------------- | ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| **P0** | Flash Attention 2    | `attn_impl: flash_attn`（注意：ms-swift 使用 `attn_impl` 而非 HF 的 `attn_implementation`）                         | Attention 显存 O(N²)→O(N)，训练加速 10~30% |
| **P0** | BF16 混合精度        | `torch_dtype: bfloat16`（A100/H100/H20）`<br>`或 `torch_dtype: float16`（V100/T4）                                  | 显存减半，计算翻倍                           |
| **P0** | 序列打包             | `packing: true` 或 `padding_free: true`（二选一，padding_free 零预处理开销更优）                                      | 消除 padding 浪费，吞吐提升 20~50%           |
| **P0** | 梯度检查点           | `gradient_checkpointing: true`（ms-swift 默认已开启，确认未被关闭）                                                     | 激活显存降低 ~60%                            |
| **P1** | UnSloth 加速后端     | `tuner_backend: unsloth`（仅 LoRA 训练）                                                                                | **训练速度 +70%**，显存 -50%           |
| **P1** | Liger Kernel         | `use_liger_kernel: true`（全参训练或 UnSloth 不支持时）                                                                 | 吞吐 +20%，显存 -60%                         |
| **P1** | NEFTune              | `neftune_noise_alpha: 5`                                                                                                | 下游效果提升（SFT 阶段）                     |
| **P1** | QLoRA 4bit           | `quant_method: bnb`、`quant_bits: 4`、`bnb_4bit_quant_type: nf4`、`bnb_4bit_use_double_quant: true`               | 7B→~6GB                                     |
| **P1** | DataLoader 并行      | `dataloader_num_workers: 4`、`dataloader_persistent_workers: true`                                                    | GPU 等待时间 -50%                            |
| **P1** | use_logits_to_keep   | `use_logits_to_keep: true`（RLHF 场景）                                                                                 | 仅计算有 label 位置的 logits，节省显存       |
| **P2** | DeepSpeed ZeRO       | `deepspeed: zero2`（LoRA 多卡）或 `deepspeed: zero3`（全参微调）`<br>`ms-swift 支持字符串快捷方式，无需单独配置文件 | 多卡显存分片                                 |
| **P2** | ZeRO++ 多机优化      | `zero_hpz_partition_size: <每节点GPU数>`                                                                                | 节点内高精度+跨节点量化通信                  |
| **P2** | FSDP2                | `fsdp: fsdp2`                                                                                                           | PyTorch 原生分片                             |
| **P2** | 序列并行             | `sequence_parallel_size: 4~8`（超长序列 >128K）                                                                         | 超长序列训练必用                             |
| **P2** | GRPO+vLLM            | `use_vllm: true`、`vllm_mode: colocate`（GRPO/GKD 场景）                                                              | Rollout 推理加速                             |
| **P2** | DoRA/LoRA+/GaLore 等 | 对应 YAML 参数                                                                                                            | 按需启用                                     |

```bash
# YAML 优化实施方式
CONFIG_FILE=$(find "${WORK_DIR}/project" -name "*.yaml" -o -name "*.yml" | head -1)
# 使用 Edit 工具修改 YAML 配置项
# ms-swift 的 DeepSpeed 使用字符串快捷方式（zero2/zero3/zero2_offload/zero3_offload），无需创建单独配置文件
```

**第二层：框架代码级优化（高收益项，YAML/CLI 无法实现时）**

当通用审计或 Profiling 发现以下高收益优化机会，且 ms-swift YAML/CLI 配置无法覆盖时，**允许在 fork 副本中修改代码**：

| 优化项                       | 修改目标                                                                                               | 适用场景                                     | 预期收益                |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------- | ----------------------- |
| **torch.compile**      | ms-swift 已支持 `torch_compile: true`，如版本不支持则手动注入                                        | PyTorch 2.x+                                 | 算子融合，吞吐 +10~30%  |
| **自定义 Triton 算子** | 参考 `triton-optimization` skill 的决策树和模板，在 fork 项目中添加高性能 Triton kernel 替换热点算子 | Profiling 发现特定算子成为瓶颈且无成熟库覆盖 | 单算子加速 2~5x         |
| **CUDA Graph**         | 在训练循环中启用 CUDA Graph 捕获                                                                       | 大量小 kernel、launch overhead 高            | kernel launch 开销 -90% |
| **NCCL 通信调优**      | 修改启动脚本中的 NCCL 环境变量                                                                         | 多机多卡通信效率低                           | 通信耗时 -20~40%        |
| **多模态冻结策略**     | `freeze_llm`/`freeze_vit`/`freeze_aligner` + 分层学习率 `vit_lr`/`aligner_lr`                | 多模态模型训练                               | 训练效率与效果平衡      |

**代码修改原则**：

1. **所有修改仅在 fork 副本中**（`${WORK_DIR}/project/`），绝不动原始项目
2. **优先 YAML/CLI 配置 → 其次代码修改**
3. **最小化侵入**：ms-swift 的大部分优化通过 YAML 配置即可完成，尽量不改框架代码
4. **可回退**：每项修改记录 diff，验证失败时可独立回退

> **核心理念**：ms-swift 的 YAML/CLI 配置覆盖面极广（包括 DeepSpeed 字符串快捷方式、UnSloth/Liger Kernel 一键开关、序列并行、GRPO+vLLM 等），能覆盖 ~90% 的常见优化。仅在 Profiling 揭示深层瓶颈时才需要代码级干预。

---

**🔹 VideoX-Fun 项目优化**（`FRAMEWORK=videox_fun`）：

VideoX-Fun 项目采用**两层优化策略**：优先通过 CLI 参数和脚本配置实现（低风险），当 CLI 无法覆盖的高收益优化出现时，允许修改框架代码（在 fork 副本中）。

**第一层：CLI 参数/脚本配置优化（优先执行，零风险）**

VideoX-Fun 通过 `accelerate launch` + 训练脚本 CLI 参数控制训练行为。优化主要修改 `.sh` 启动脚本。

| 优先级       | 优化项               | CLI 参数修改                                                                                                                                                   | 预期收益                                  |
| ------------ | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| **P0** | 安装 Flash Attention | `pip install flash-attn`（框架自动检测使用）                                                                                                                 | 注意力计算加速 2-3 倍                     |
| **P0** | BF16 混合精度        | `--mixed_precision="bf16"`（确认已设置）                                                                                                                     | 显存减半，计算翻倍                        |
| **P0** | 梯度检查点           | `--gradient_checkpointing`                                                                                                                                   | 激活显存降低 ~60%                         |
| **P0** | Low VRAM 模式        | `--low_vram`                                                                                                                                                 | VAE/TextEncoder CPU 卸载，节省大量显存    |
| **P0** | TF32                 | `--allow_tf32`                                                                                                                                               | Ampere+ matmul 加速                       |
| **P1** | Bucket 采样          | `--enable_bucket`                                                                                                                                            | 消除 padding 浪费                         |
| **P1** | Auto Tile Batch      | `--auto_tile_batch_size`                                                                                                                                     | 小分辨率样本自动增大 batch，GPU 利用率 ↑ |
| **P1** | 8-bit Adam           | `--use_8bit_adam`                                                                                                                                            | 优化器状态显存减半                        |
| **P1** | DataLoader 并行      | `--dataloader_num_workers 8`                                                                                                                                 | GPU 等待数据时间 ↓                       |
| **P1** | 均匀时间步           | `--uniform_sampling`（多卡）                                                                                                                                 | 避免重复时间步采样                        |
| **P1** | 异常梯度裁剪         | `--abnormal_norm_clip_start 100`                                                                                                                             | 训练稳定性 ↑                             |
| **P2** | DeepSpeed ZeRO-2     | `--use_deepspeed --deepspeed_config_file config/zero_stage2_config.json`                                                                                     | LoRA 多卡显存分片                         |
| **P2** | FSDP Full Shard      | `--use_fsdp --fsdp_sharding_strategy "FULL_SHARD" --fsdp_auto_wrap_policy "TRANSFORMER_BASED_WRAP" --fsdp_transformer_layer_cls_to_wrap "WanAttentionBlock"` | 14B 全参训练推荐（比 ZeRO-3 更稳定）      |
| **P2** | 节点同步             | `--keep_all_node_same_token_length`（多机）                                                                                                                  | 防止 straggler 节点                       |
| **P2** | VAE Mini-Batch       | `--vae_mini_batch 8~16`（高分辨率）                                                                                                                          | VAE 编码显存 ↓                           |
| **P2** | 运动子损失           | `--motion_sub_loss --motion_sub_loss_ratio 0.25`                                                                                                             | 视频时间一致性 ↑                         |

**第二层：框架代码级优化（高收益项，CLI 无法实现时）**

当通用审计或 Profiling 发现以下高收益优化机会，且 CLI 参数无法覆盖时，**允许在 fork 副本中修改 VideoX-Fun 框架代码**：

| 优化项                               | 修改目标                                                                              | 适用场景                              | 预期收益                    |
| ------------------------------------ | ------------------------------------------------------------------------------------- | ------------------------------------- | --------------------------- |
| **torch.compile**              | 训练入口添加 `transformer3d = torch.compile(transformer3d, mode="reduce-overhead")` | PyTorch 2.x+，VideoX-Fun 未原生支持   | 算子融合，吞吐 +10~30%      |
| **Liger Kernel**               | 替换 RMSNorm/SwiGLU 等层为 Triton 融合算子                                            | DiT 模型中 norm/activation 成为热点   | 吞吐 +20%，显存 -60%        |
| **CUDA Graph**                 | 固定输入形状的训练步骤启用 Graph 捕获                                                 | Kernel launch overhead 高（小 batch） | Launch 开销 -90%            |
| **QLoRA (NF4)**                | 基座模型 4-bit 量化 + LoRA                                                            | 14B 模型单卡训练                      | 显存 ~28GB → ~10GB         |
| **Wan VAE 空间 Tiling**        | 参考 CogVideoX VAE 实现空间 tiling                                                    | ≥720P 高分辨率训练                   | VAE 显存 -75%               |
| **预编码 Text Embeddings**     | 预跑 Text Encoder 保存 embeddings，训练时加载                                         | 大规模训练                            | 省 ~5GB 显存 + 加速         |
| **FP8 训练**                   | 使用 TransformerEngine 或 torchao                                                     | H100/H200                             | GEMM 计算翻倍               |
| **ZeRO++ 量化通信**            | DeepSpeed 配置启用 `zero_quantized_weights`/`zero_quantized_gradients`            | 多机训练                              | 通信量 -50~75%              |
| **FusedAdam**                  | 替换原生 AdamW 为 DeepSpeed FusedAdam 或 apex FusedAdam                               | 所有场景                              | Kernel 调用 ↓，速度 +5~10% |
| **DataLoader prefetch_factor** | 添加 `prefetch_factor=2~4`                                                          | DataLoader 是瓶颈                     | CPU→GPU 传输重叠           |
| **NCCL 通信调优**              | 启动脚本添加 NCCL 环境变量                                                            | 多机训练                              | 通信耗时 -20~40%            |

**代码修改原则**：

1. **所有修改仅在 fork 副本中**（`${WORK_DIR}/project/`），绝不动原始项目
2. **优先 CLI 参数 → 其次启动脚本修改 → 最后框架代码注入**
3. **视频生成特殊性**：显存与分辨率×帧数强相关（720P ≈ 480P 的 2.25x，81帧 ≈ 49帧的 1.7x），需综合考虑分辨率/帧数/batch_size 的权衡
4. **FSDP wrap class 因模型而异**：Wan 用 `WanAttentionBlock`，CogVideoX 用 `CogVideoXBlock` 等

```python
# 示例：在 fork 的训练脚本中注入 torch.compile
# 文件: ${WORK_DIR}/project/scripts/wan2.1_fun/train.py
# 在 transformer3d = accelerator.prepare(transformer3d) 之前添加：

if hasattr(torch, 'compile'):
    transformer3d = torch.compile(transformer3d, mode="reduce-overhead")
```

> **核心理念**：VideoX-Fun 的 CLI 参数覆盖了大部分训练控制（混合精度、梯度检查点、low_vram、bucket 采样、分布式策略等），能覆盖 ~70% 的常见优化。但视频生成模型因超高显存需求和复杂的 VAE pipeline，Profiling 常揭示框架层面的深层瓶颈（如 VAE 编码/解码效率、注意力 kernel 选择、Text Encoder 重复计算等），需要代码级干预解决。

---

**🔹 Flow-Factory 项目优化**（`FRAMEWORK=flow_factory`）：

Flow-Factory 项目采用**两层优化策略**：优先通过 YAML 配置实现（低风险），当配置无法覆盖的高收益优化出现时，允许修改框架代码（在 fork 副本中）。

**第一层：YAML 配置优化（优先执行，零风险）**

| 优先级       | 优化项                 | YAML 配置修改                                                                         | 预期收益                                     |
| ------------ | ---------------------- | ------------------------------------------------------------------------------------- | -------------------------------------------- |
| **P0** | Flash Attention        | `model.attn_backend: "flash_hub"`（Ampere+）`<br>`或 `"_flash_3_hub"`（Hopper） | Attention 显存 O(N²)→O(N)，训练加速 20~40% |
| **P0** | BF16 混合精度          | `mixed_precision: "bf16"`                                                           | 显存减半，计算翻倍                           |
| **P0** | 梯度检查点             | `train.enable_gradient_checkpointing: true`                                         | 激活显存降低 ~60%                            |
| **P0** | CPU Sample Offload     | `train.offload_samples_to_cpu: true`                                                | **视频模型必须**。Sample 显存完全卸载  |
| **P1** | Latent FP16 存储       | `train.latent_storage_dtype: "fp16"`                                                | 轨迹存储显存 -50%                            |
| **P1** | EMA 放 CPU             | `train.ema_device: "cpu"`                                                           | 节省等同模型大小的 GPU 显存                  |
| **P1** | 参考模型放 CPU         | `train.ref_param_device: "cpu"`                                                     | DPO/CRD 等双模型算法节省显存                 |
| **P1** | LoRA 替代全参          | `model.finetune_type: "lora"<br>``model.lora_rank: 64~128`                        | 可训练参数 -95%+，显存大幅降低               |
| **P2** | FSDP2/DeepSpeed        | `config_file: config/accelerate_configs/fsdp2.yaml<br>`或 DeepSpeed ZeRO-2/3        | 多卡显存分片                                 |
| **P2** | 异步奖励               | `rewards[*].async_reward: true`                                                     | VLM 奖励计算不阻塞训练                       |
| **P2** | 调整 sample_group_size | `train.sample_group_size: 4~8`                                                      | 平衡 advantage 稳定性与显存                  |

**第二层：框架代码级优化（高收益项，YAML 无法实现时）**

| 优化项                    | 修改目标                                                                           | 适用场景                        | 预期收益                     |
| ------------------------- | ---------------------------------------------------------------------------------- | ------------------------------- | ---------------------------- |
| **torch.compile**   | 在 fork 中对 transformer 模型添加 `torch.compile(model, mode="reduce-overhead")` | PyTorch 2.x+                    | 算子融合，吞吐 +10~30%       |
| **FusedAdam**       | 替换 AdamW 为 `torch.optim.AdamW(fused=True)`                                    | 所有场景                        | Kernel 调用减少，速度 +5~10% |
| **8-bit Adam**      | 使用 `bitsandbytes.optim.AdamW8bit`                                              | 优化器显存瓶颈                  | 优化器显存 -75%              |
| **Liger Kernel**    | 替换 RMSNorm/SwiGLU 等层                                                           | 模型中 norm/activation 成为热点 | 吞吐 +20%，显存 -60%         |
| **QLoRA**           | 4-bit 量化基座模型 + LoRA                                                          | 显存极端受限                    | 模型显存 -75%                |
| **LR Scheduler**    | 添加 cosine/warmup 调度器                                                          | 训练收敛质量提升                | 收敛速度 + 最终效果          |
| **DataLoader 优化** | 添加 `pin_memory=True`、`prefetch_factor=2`、`persistent_workers=True`       | DataLoader 瓶颈                 | 数据加载速度提升             |
| **NCCL 通信调优**   | 启动脚本添加 NCCL 环境变量                                                         | 多机训练                        | 通信耗时 -20~40%             |

**代码修改原则**：

1. **所有修改仅在 fork 副本中**（`${WORK_DIR}/project/`），绝不动原始项目
2. **优先 YAML 配置 → 其次 Accelerate/DeepSpeed 配置 → 最后框架代码修改**
3. **Flow-Factory 使用 Accelerate launcher**：配置修改优先通过 `config_file` 对应的 accelerate/deepspeed 配置文件
4. **注意 RL 特殊性**：Flow-Factory 的训练循环包含 sample（生成）和 optimize（更新）两阶段，显存峰值通常在 sample 阶段

> **核心理念**：Flow-Factory 是 Diffusion/Flow-Matching 模型的 RL 微调框架，YAML 配置覆盖了大部分训练控制（混合精度、梯度检查点、注意力后端、分布式策略、EMA、奖励系统等），能覆盖 ~75% 的常见优化。但框架当前缺少一些高级优化（torch.compile、FusedAdam、8-bit Adam、QLoRA、LR scheduler 等），这些需要代码级干预在 fork 中实现。

HF Trainer 项目采用**两层优化策略**：优先通过 `TrainingArguments` 参数和 `from_pretrained()` 配置实现（低风险），当参数配置无法覆盖的高收益优化出现时，允许修改训练代码（在 fork 副本中）。

**第一层：TrainingArguments 参数优化（优先执行，零风险）**

| 优先级       | 优化项            | 参数修改                                                                                       | 预期收益                               |
| ------------ | ----------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------- |
| **P0** | Flash Attention 2 | `from_pretrained(attn_implementation="flash_attention_2")`                                   | Attention 显存 O(N²)→O(N)，速度 2-4x |
| **P0** | BF16 混合精度     | `TrainingArguments(bf16=True)`（A100/H100/H20）`<br>`或 `fp16=True`（V100/T4）           | 显存减半，计算翻倍                     |
| **P0** | torch.compile     | `TrainingArguments(torch_compile=True, torch_compile_mode="max-autotune")`                   | 吞吐 +20~50%，零代码修改               |
| **P0** | 梯度检查点        | `gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False}`        | 激活显存降低 ~60%                      |
| **P1** | Liger Kernel      | `use_liger_kernel=True`                                                                      | 吞吐 +20%，显存 -60%                   |
| **P1** | TF32              | `tf32=True`                                                                                  | Ampere+ matmul 最高 8x 加速            |
| **P1** | Fused 优化器      | `optim="adamw_torch_fused"`                                                                  | 更快的优化器步骤（融合 CUDA kernel）   |
| **P1** | NEFTune           | `neftune_noise_alpha=5`                                                                      | SFT 效果显著提升                       |
| **P1** | DataLoader 并行   | `dataloader_num_workers=4, dataloader_persistent_workers=True, dataloader_prefetch_factor=2` | GPU 等待时间 -50%                      |
| **P1** | 按长度分组        | `train_sampling_strategy="group_by_length"`                                                  | 减少 padding 浪费 20~50%               |
| **P2** | 8-bit 优化器      | `optim="adamw_bnb_8bit"`                                                                     | 优化器显存 -75%                        |
| **P2** | QLoRA 4-bit       | `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` + PEFT LoRA               | 7B→~6GB                               |
| **P2** | FSDP              | `fsdp="full_shard auto_wrap"` + `fsdp_config={...}`                                        | 多卡显存分片                           |
| **P2** | DeepSpeed ZeRO    | `deepspeed="ds_config.json"`                                                                 | 多卡显存分片                           |
| **P2** | GaLore/LOMO       | `optim="galore_adamw_layerwise"` / `optim="lomo"`                                          | 全参训练 LoRA 级别显存                 |

```python
# TrainingArguments 优化实施方式
# 直接修改训练脚本中 TrainingArguments 的实例化参数
# 如需 FSDP/DeepSpeed，在 fork 目录中创建对应配置文件
```

**第二层：训练代码级优化（高收益项，TrainingArguments 无法实现时）**

当通用审计或 Profiling 发现以下高收益优化机会，且 TrainingArguments 参数无法覆盖时，**允许在 fork 副本中修改训练代码**：

| 优化项                       | 修改目标                                                                                               | 适用场景                                     | 预期收益                |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------- | ----------------------- |
| **自定义 Triton 算子** | 参考 `triton-optimization` skill 的决策树和模板，在 fork 项目中添加高性能 Triton kernel 替换热点算子 | Profiling 发现特定算子成为瓶颈且无成熟库覆盖 | 单算子加速 2~5x         |
| **CUDA Graph**         | 在训练循环中启用 CUDA Graph 捕获                                                                       | 大量小 kernel、launch overhead 高            | kernel launch 开销 -90% |
| **数据管道深度优化**   | 修改数据预处理逻辑（内存映射、streaming、多线程 tokenize）                                             | DataLoader 仍是瓶颈                          | I/O 吞吐提升            |
| **NCCL 通信调优**      | 修改启动脚本中的 NCCL 环境变量                                                                         | 多机多卡通信效率低                           | 通信耗时 -20~40%        |
| **Trainer 子类定制**   | 继承 Trainer 覆写 `compute_loss` / `training_step`                                                 | 需要自定义 loss 融合或特殊训练逻辑           | 显存降低、吞吐提升      |
| **混合精度细粒度控制** | 修改模型代码中特定层的精度策略                                                                         | 训练不稳定需要逐层调精度                     | 稳定性 + 性能平衡       |

**代码修改原则**：

1. **所有修改仅在 fork 副本中**（`${WORK_DIR}/project/`），绝不动原始项目
2. **优先 TrainingArguments 参数 → 其次 from_pretrained 配置 → 最后训练代码修改**
3. **最小化侵入**：Transformers Trainer 的大部分优化通过 TrainingArguments 参数即可完成
4. **可回退**：每项修改记录 diff，验证失败时可独立回退

> **核心理念**：HuggingFace Transformers Trainer 的 `TrainingArguments` 提供了 ~65 个性能相关参数，能覆盖 ~85% 的常见优化场景（混合精度、torch.compile、Liger Kernel、FSDP/DeepSpeed、8-bit 优化器、DataLoader 调优等）。仅在 Profiling 揭示深层瓶颈时才需要代码级干预。

---

**🔹 vLLM 推理优化**（`VLLM_DETECTED=yes`）：

当检测到项目使用 vLLM 推理框架时（无论训练框架是什么），额外执行 vLLM 推理优化。vLLM 优化与训练框架优化**并行进行**，二者不冲突。

**第一层：vLLM 引擎参数优化（优先执行，零风险）**

| 优先级       | 优化项                | 参数/代码修改                                                      | 预期收益                          |
| ------------ | --------------------- | ------------------------------------------------------------------ | --------------------------------- |
| **P0** | 限制 max_model_len    | `max_model_len=<实际需求>` 而非模型上限                          | KV Cache 显存大幅降低             |
| **P0** | KV Cache FP8 量化     | `kv_cache_dtype="fp8"`, `calculate_kv_scales=True`             | KV Cache 显存 -50%（Hopper/Ada+） |
| **P0** | 优化级别              | 确认使用 `-O2` 或 `-O3`                                        | CUDA Graph + torch.compile 全启用 |
| **P0** | 前缀缓存              | `enable_prefix_caching=True`（V1 默认）                          | 多轮对话/RAG TTFT 大幅降低        |
| **P1** | 在线 FP8 量化         | `quantization="fp8_per_block"`（Hopper/Ada）                     | 权重显存 -50%，无需预量化         |
| **P1** | 调度参数调优          | `max_num_seqs`、`max_num_batched_tokens` 根据延迟/吞吐目标调整 | 吞吐或延迟 20-50% 改善            |
| **P1** | DP×TP 并行           | `data_parallel_size × tensor_parallel_size`                     | 多 GPU 最大吞吐                   |
| **P1** | 快速 Tokenizer        | `VLLM_USE_FASTOKENS=1`                                           | 高 QPS 下 tokenization 加速       |
| **P2** | 推测解码              | EAGLE/MTP/N-gram                                                   | 输出 token 生成速度 2-3x          |
| **P2** | Disaggregated Prefill | `kv_transfer_config`                                             | 独立扩 prefill，超低 TTFT         |
| **P2** | GPU 显存利用率        | `gpu_memory_utilization=0.93~0.95`                               | 更多 KV Cache 空间                |

**第二层：RLHF/训练集成优化（当 vLLM 用于 RLHF/GRPO 场景）**

| 优先级       | 优化项            | 配置/代码修改                                                   | 预期收益                             |
| ------------ | ----------------- | --------------------------------------------------------------- | ------------------------------------ |
| **P0** | Sleep Mode        | `enable_sleep_mode=True` + `sleep(level=2)` / `wake_up()` | Colocate 模式必须，显存完整释放/恢复 |
| **P0** | 模式选择          | `vllm_mode="colocate"`（GPU 少）或 `"server"`（GPU 多）     | 资源利用率最大化                     |
| **P1** | Weight Transfer   | `weight_transfer_config={"backend": "nccl"}`                  | 训练后权重高效同步到推理引擎         |
| **P1** | Dev Mode          | `VLLM_SERVER_DEV_MODE=1`                                      | 启用 sleep/wake/weight transfer API  |
| **P1** | 动态 LoRA         | `load_lora_adapter(load_inplace=True)`                        | LoRA RL 免重启更新权重               |
| **P2** | Async RL          | `pause_generation()` / `resume_generation()`                | 异步权重同步，减少阻塞               |
| **P2** | Colocate 显存分配 | `gpu_memory_utilization=0.3~0.5`（colocate）                  | 给训练留足够显存                     |

```python
# 示例：在 fork 的训练代码中优化 vLLM 配置（RLHF 场景）
# 文件: ${WORK_DIR}/project/train_grpo.py

# 优化 vLLM 推理引擎配置
from vllm import LLM

llm = LLM(
    model=model_path,
    tensor_parallel_size=2,
    gpu_memory_utilization=0.4,          # colocate 模式留显存给训练
    max_model_len=4096,                  # 限制为实际需求
    enable_sleep_mode=True,              # RLHF 必须
    kv_cache_dtype="fp8",                # Hopper+ 减少 KV Cache 显存
    enable_prefix_caching=True,          # GRPO 多次采样可复用前缀
    quantization="fp8_per_block",        # 在线量化节省权重显存
)
```

```bash
# 示例：优化 vLLM server 模式的启动命令（RLHF 场景）
VLLM_SERVER_DEV_MODE=1 VLLM_USE_FASTOKENS=1 \
  vllm serve ${MODEL_PATH} \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 4096 \
  --kv-cache-dtype fp8 \
  --enable-sleep-mode \
  --enable-prefix-caching \
  -O2
```

**代码修改原则**：

1. **所有修改仅在 fork 副本中**（`${WORK_DIR}/project/`），绝不动原始项目
2. **优先引擎参数调优 → 其次 RLHF 集成优化 → 最后推测解码/Disagg 等高级特性**
3. **vLLM 优化独立于训练框架优化**：可同时应用训练侧优化（如 LoRA、混合精度）和推理侧优化（如 KV Cache FP8、推测解码）
4. **RLHF 场景注意 GPU 显存分配**：colocate 模式需要精确控制 `gpu_memory_utilization` 以避免训练 OOM

> **核心理念**：现代 RLHF/GRPO 训练流程中，vLLM 推理引擎的效率直接影响整体训练速度（rollout generation 通常占 50-70% 时间）。优化 vLLM 的推理效率（KV Cache 量化、推测解码、前缀缓存、sleep mode 高效切换）可以显著加速 RL 训练的总体 wall-clock time。

---

**🔹 SGLang 推理优化**（`SGLANG_DETECTED=yes`）：

当检测到项目使用 SGLang 推理框架时（无论训练框架是什么），额外执行 SGLang 推理优化。SGLang 优化与训练框架优化**并行进行**，二者不冲突。

**第一层：SGLang 引擎参数优化（优先执行，零风险）**

| 优先级       | 优化项               | 参数/代码修改                                                       | 预期收益                          |
| ------------ | -------------------- | ------------------------------------------------------------------- | --------------------------------- |
| **P0** | 限制 context_length  | `--context-length <实际需求>` 而非模型上限                        | KV Cache 显存大幅降低             |
| **P0** | KV Cache FP8 量化    | `--kv-cache-dtype fp8_e4m3`                                       | KV Cache 显存 -50%（Hopper/Ada+） |
| **P0** | RadixAttention 确认  | 确认未被 `--disable-radix-cache` 关闭                             | 自动前缀缓存，多轮/RAG 加速       |
| **P0** | LPM 调度策略         | `--schedule-policy lpm`（共享前缀多时）                           | 缓存命中率大幅提升                |
| **P1** | DP Attention         | `--enable-dp-attention`（MLA 模型必用）                           | Decode 吞吐 1.9x（DeepSeek）      |
| **P1** | 在线 FP8 量化        | `--quantization fp8`（Hopper/Ada）                                | 权重显存 -50%，无需预量化         |
| **P1** | 调度保守度调优       | 观察 token usage，调整 `--schedule-conservativeness`              | 吞吐或稳定性改善                  |
| **P1** | mem_fraction_static  | 逐步提高至接近 OOM                                                  | 最大化 KV Cache 池                |
| **P1** | SGLang Model Gateway | `sglang_router` + `--router-policy cache_aware`                 | 多副本吞吐 +92%                   |
| **P2** | MoE 优化             | `--ep-size N --moe-a2a-backend deepep --enable-two-batch-overlap` | MoE 吞吐 2x                       |
| **P2** | 推测解码             | EAGLE-3 / MTP / N-gram                                              | 输出 token 生成速度 2-3x          |
| **P2** | HiCache              | `--enable-hierarchical-cache --hicache-ratio 2`                   | 长上下文缓存命中率大幅提升        |

**第二层：RLHF/训练集成优化（当 SGLang 用于 RLHF/GRPO 场景）**

| 优先级       | 优化项            | 配置/代码修改                                                                                 | 预期收益                             |
| ------------ | ----------------- | --------------------------------------------------------------------------------------------- | ------------------------------------ |
| **P0** | Memory Saver      | `--enable-memory-saver` + `release/resume_memory_occupation`                              | Colocate 模式必须，完整释放/恢复显存 |
| **P0** | 权重更新策略      | Colocate:`update_weights_from_tensor<br>`Disaggregated: `update_weights_from_distributed` | 训练后高效同步权重到推理引擎         |
| **P1** | 确定性推理        | `--enable-deterministic-inference`                                                          | On-policy RL 训练对齐                |
| **P1** | CPU 权重备份      | `--enable-weights-cpu-backup`                                                               | 权重恢复加速                         |
| **P1** | 暂停/继续         | `pause_generation` → 更新权重 → `continue_generation`                                   | 异步 RL，减少阻塞                    |
| **P2** | R-Fork            | `--load-format remote_instance`                                                             | 秒级拉起新推理实例                   |
| **P2** | Colocate 显存分配 | `--mem-fraction-static 0.3~0.5`（colocate）                                                 | 给训练留足够显存                     |

```python
# 示例：在 fork 的训练代码中优化 SGLang 配置（RLHF 场景）
import requests

SGLANG_URL = "http://localhost:30000"

# 推理阶段：生成 rollout
responses = requests.post(f"{SGLANG_URL}/generate", json={
    "text": prompts,
    "sampling_params": {"temperature": 0.7, "max_new_tokens": 512}
})

# 训练前：释放推理引擎显存
requests.post(f"{SGLANG_URL}/release_memory_occupation",
              json={"tags": ["kv_cache", "weights"]})

# ... 执行训练更新 ...

# 训练后：更新权重并恢复
requests.post(f"{SGLANG_URL}/update_weights_from_tensor",
              json={"serialized_named_tensors": ..., "flush_cache": True})
requests.post(f"{SGLANG_URL}/resume_memory_occupation")
```

```bash
# 示例：SGLang 推理引擎启动命令优化（RLHF colocate 场景）
python -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --enable-memory-saver \
  --enable-weights-cpu-backup \
  --enable-deterministic-inference \
  --mem-fraction-static 0.4 \
  --kv-cache-dtype fp8_e4m3 \
  --context-length 4096 \
  --port 30000
```

```bash
# 示例：SGLang 推理引擎启动命令优化（生产 server 场景）
python -m sglang_router.launch_server \
  --model-path ${MODEL_PATH} \
  --tp 2 --dp-size 2 \
  --router-policy cache_aware \
  --schedule-policy lpm \
  --mem-fraction-static 0.88 \
  --kv-cache-dtype fp8_e4m3 \
  --enable-metrics \
  --port 30000
```

**代码修改原则**：

1. **所有修改仅在 fork 副本中**（`${WORK_DIR}/project/`），绝不动原始项目
2. **优先引擎参数调优 → 其次 RLHF 集成优化 → 最后推测解码/Disagg 等高级特性**
3. **SGLang 优化独立于训练框架优化**：可同时应用训练侧优化和推理侧优化
4. **RLHF 场景注意 GPU 显存分配**：colocate 模式需要精确控制 `--mem-fraction-static` 以避免训练 OOM

> **核心理念**：SGLang 以 RadixAttention（自动前缀缓存）和 DP Attention 为核心创新，是 DeepSeek 系列模型的推荐推理引擎。在 RLHF/GRPO 场景中，SGLang 的 Memory Saver、三种权重更新策略（磁盘/Tensor/分布式）和确定性推理模式，为 RL 训练提供了灵活高效的推理引擎集成方案。

以下为通用训练项目的优化策略，需要修改 Python 代码和配置文件。

**混合精度（如未启用）**：

- 参考 `deepspeed-optimization` skill 第三步 3.2 节的混合精度配置
- 在 fork 项目中启用 BF16（A100/H100+）或 FP16（V100）
- 代码修改：添加 `torch.amp.autocast` 或 DeepSpeed `bf16.enabled: true`

**Flash Attention（如未启用或可升级）**：

- 参考 `flash-attention` skill 的版本选型（第一步）和框架集成（第四步）
- 根据 GPU 架构选择 FA2/FA3
- 代码修改：安装 `flash-attn`，设置 `attn_implementation="flash_attention_2"` 或直接替换 attention 函数

**DeepSpeed ZeRO（如适用）**：

- 参考 `deepspeed-optimization` skill 的 ZeRO Stage 选型（第一步）和配置模板（第二步）
- 根据模型参数量 + GPU 显存选择 Stage
- 生成 `ds_config.json`，修改启动命令

**数据加载优化（如 DataLoader 是瓶颈）**：

- 设置 `num_workers=4~8`
- 启用 `pin_memory=True`
- 设置 `prefetch_factor=2~4`
- 启用 `persistent_workers=True`

**Liger Kernel（如使用 HuggingFace 模型）**：

- 参考 `training-acceleration-audit` skill D 类的 Liger Kernel 检查项
- 安装 `liger-kernel`，添加 `apply_liger_kernel_to_*` 一行代码

**torch.compile（如 PyTorch >= 2.0）**：

- 添加 `model = torch.compile(model)` 启用图编译

**Triton 自定义算子（Profiling 发现热点算子瓶颈时）**：

- 参考 `triton-optimization` skill 的决策树（第一步）判断是否需要 Triton
- 优先使用 Liger Kernel / FlagGems 等成熟 Triton 生态工具
- 若无成熟替代，参考 `triton-optimization` skill 的融合算子模板（第三步）编写自定义 kernel
- 常见高价值融合：Fused RMSNorm、Fused SwiGLU、Fused Linear+CrossEntropy

**通信优化（多卡/多机场景）**：

- 参考 `deepspeed-optimization` skill 模板 F（ZeRO++）
- 启用 `overlap_comm=true`
- 多机启用 `zero_hpz_partition_size` + 量化通信

**多机多卡专项优化**（仅当第一步检测到多机模式时）：

以下优化项在单节点无法验证，作为**建议项**输出到报告，并在最终报告中标注"需多机环境验证"：

| 优化项                        | 说明                                     | 对应配置                                                    |
| ----------------------------- | ---------------------------------------- | ----------------------------------------------------------- |
| **ZeRO++ 机内通信优化** | 节点内用高精度，跨节点用量化通信降低带宽 | `zero_hpz_partition_size=<gpus_per_node>`                 |
| **量化梯度通信**        | 跨节点梯度用 INT4/INT8 传输              | `zero_quantized_gradients: true`                          |
| **量化权重通信**        | ZeRO-3 跨节点权重用 INT4/INT8 传输       | `zero_quantized_weights: true`                            |
| **通信计算重叠**        | 梯度通信与反向传播重叠                   | `overlap_comm: true`                                      |
| **梯度累积**            | 减少通信频率（N 步累积再同步）           | `gradient_accumulation_steps: N`                          |
| **NCCL 环境变量调优**   | 调整 NCCL 缓冲区、通信算法、超时等       | `NCCL_BUFFSIZE`, `NCCL_ALGO`, `NCCL_SOCKET_IFNAME` 等 |
| **高效通信优化器**      | ZeroOneAdam / OneBitAdam 压缩通信量      | DeepSpeed 通信高效优化器配置                                |

```bash
# NCCL 多机调优环境变量参考
export NCCL_IB_DISABLE=0                   # 启用 InfiniBand（如可用）
export NCCL_SOCKET_IFNAME=eth0             # 指定通信网卡
export NCCL_DEBUG=WARN                     # 调试级别
export NCCL_BUFFSIZE=16777216              # 16MB 缓冲区
export NCCL_P2P_LEVEL=NVL                  # NVLink P2P 级别
```

**其他代码级优化**：

- 移除训练循环中的 `.item()` / `print(cuda_tensor)` 同步点
- 设置 `torch.backends.cudnn.benchmark = True`
- 启用 `optimizer.zero_grad(set_to_none=True)`

#### 6.3 修改启动命令

如果引入了 DeepSpeed 等框架，需要修改启动命令：

```bash
# 原始命令
torchrun --nproc_per_node=4 train.py --config config.yaml

# 优化后命令（示例：引入 DeepSpeed）
deepspeed --num_gpus=4 train.py --config config.yaml --deepspeed ds_config.json
```

#### 6.4 记录所有修改

输出：

```
${WORK_DIR}/reports/v{N}_optimization-changes.md
```

格式：

```markdown
# 优化变更记录 (v{N})

## 变更概览
| # | 优化项 | 修改文件 | 变更类型 | 对应瓶颈 |
|---|--------|---------|---------|---------|
| 1 | 启用 BF16 混合精度 | train.py:L45, ds_config.json | 新增配置 | 未使用混合精度 |
| 2 | 启用 Flash Attention 2 | model.py:L120 | 代码修改 | Attention 显存 O(N²) |
| ... | | | | |

## 详细变更
### 1. 启用 BF16 混合精度
**文件**: train.py
**修改内容**:
\```diff
- output = model(inputs)
+ with torch.amp.autocast('cuda', dtype=torch.bfloat16):
+     output = model(inputs)
\```
**原因**: 原代码使用 FP32，计算量翻倍且显存占用大

## 新的启动命令
\```bash
<优化后的启动命令>
\```
```

---

### 第七步：验证优化效果

**目标**：运行优化后的代码，如遇报错则自动修复，成功后采集新的性能数据。

#### 7.1 执行优化后的代码

**⚠️ 运行前必做**：按第四步 4.0.1 节清理残留进程并确定 `MASTER_PORT`。

```bash
cd "${WORK_DIR}/project"
# 清理残留进程（上一轮 profiling 可能遗留）
cleanup_port ${MASTER_PORT:-29500}
<优化后的启动命令（保持限制步数 + 禁用模型保存 + --master_port=${MASTER_PORT}）>
# 训练完成后清理
cleanup_port ${MASTER_PORT:-29500}
```

#### 7.2 自动修复运行错误

如果运行报错，按以下流程处理：

```
报错！
├── 1. 读取完整错误信息（traceback）
├── 2. 分析错误类型：
│   ├── ImportError → 安装缺失依赖（pip install xxx）
│   ├── CUDA OOM → 减小 micro_batch_size 或调整 ZeRO Stage
│   ├── EADDRINUSE（端口占用） → cleanup_port 清理残留进程 + 使用新的随机端口重试
│   ├── Shape mismatch → 检查混合精度/Flash Attention 的输入格式
│   ├── Config error → 修正 DeepSpeed/配置文件参数
│   └── 其他 → 根据具体错误修复代码
├── 3. 修复代码（仅在 fork 目录中）
├── 4. 重新执行
├── 5. 如连续 3 次修复仍失败 → 回退该优化项，标记为"不可用"
└── 6. 如所有优化项都验证失败 → 回退到纯配置优化（YAML/参数级），跳过代码级修改
```

**⚠️ 验证阶段 Turn 保护**：如果在验证阶段累计修复尝试超过 5 次仍无法正常运行，立即停止修复，回退所有导致问题的优化项，仅保留已验证成功的优化，直接进入第八步。不要在错误修复循环中消耗过多 Turn。

#### 7.3 采集新的性能数据

运行成功后，按照第四步和第五步的方法重新采集：

1. **PyTorch Profiler**：运行带 profiler 的版本
2. **Nsight Systems**：用 nsys 包裹运行
3. **System Resources**：采集运行时 GPU 状态

输出（版本号递增）：

```
${WORK_DIR}/reports/v{N}_pytorch-profiler-report.md
${WORK_DIR}/reports/v{N}_nsight-systems-report.md
${WORK_DIR}/reports/v{N}_system-resources-report.md
```

---

### 第八步：迭代优化

**目标**：对比优化前后的性能数据，判断是否继续优化。

#### 8.1 性能对比

从 `v{N}` 和 `v{N-1}` 的 profiler 报告中提取关键指标对比：

```markdown
## 性能对比：v{N-1} → v{N}

| 指标 | v{N-1} | v{N} | 变化 |
|------|--------|------|------|
| 每步耗时 (ms) | ... | ... | -X% |
| 吞吐 (samples/s) | ... | ... | +X% |
| GPU 显存峰值 (GB) | ... | ... | -X GB |
| GPU 利用率 (%) | ... | ... | +X pp |
| Top1 Kernel 耗时 (ms) | ... | ... | -X% |
| NCCL 通信占比 (%) | ... | ... | -X pp |
```

#### 8.2 判断收敛

满足以下**任一条件**则认为收敛，**立即进入第九步生成最终报告**：

| 收敛条件                 | 说明                                         |
| ------------------------ | -------------------------------------------- |
| 吞吐提升 < 5%            | 本轮相对上轮改善不大                         |
| 连续 2 轮无显著改善      | 优化空间已基本耗尽                           |
| 达到最大迭代轮数（5 轮） | **硬性上限，防止无限循环和 Turn 耗尽** |
| 用户主动要求停止         | 人工介入                                     |

**⚠️ 重要**：达到最大迭代轮数（5 轮）后必须**无条件进入第九步**，不得以"还有优化空间"为由继续迭代。保证最终报告的生成优先级高于额外的优化收益。

#### 8.3 未收敛 → 继续优化

如果仍有优化空间**且未达到最大迭代轮数**：

1. 回到**第五步**，基于新的 profiler 数据重新诊断瓶颈
2. 版本号递增：v1 → v2 → v3 → v4 → v5
3. 每轮聚焦于**当前最大瓶颈**，避免重复优化已改善的项目

---

### 第九步：输出最终优化报告

**目标**：生成完整的优化前后对比文档，包含所有优化项说明和性能提升数据。

**⚠️ 本步骤是整个流程的最终交付物，必须执行，不得跳过。** 即使前面某些步骤被跳过或失败，也必须基于已有数据生成报告。如果 Profiling 数据不可用，使用静态审计数据；如果优化验证失败，在报告中说明哪些优化已验证、哪些未能验证。

输出文件：

```
${WORK_DIR}/reports/final_optimization-summary.md
```

报告格式：

```markdown
# GPU 训练智能调优 — 最终优化报告

## 项目信息
| 项目 | 值 |
|------|-----|
| 原始项目路径 | <路径>（未修改） |
| 优化后项目路径 | <WORK_DIR>/project/ |
| 原始启动命令 | `<命令>` |
| 优化后启动命令 | `<命令>` |
| 优化迭代轮数 | N |
| 优化总耗时 | X 分钟 |

## 性能提升总结

| 指标 | 优化前 (v0) | 优化后 (v{N}) | 提升 |
|------|------------|-------------|------|
| 每步耗时 | X ms | Y ms | -Z% |
| 训练吞吐 | X samples/s | Y samples/s | **+Z%** |
| GPU 显存峰值 | X GB | Y GB | -Z GB |
| GPU 利用率 | X% | Y% | +Z pp |
| 预估训练总时间 | X 小时 | Y 小时 | **节省 Z 小时** |

## 使用的优化项

| # | 优化项 | 引入轮次 | 对应 Skill | 效果 |
|---|--------|---------|-----------|------|
| 1 | BF16 混合精度 | v1 | /deepspeed-optimization | 显存 -40%, 吞吐 +30% |
| 2 | Flash Attention 2 | v1 | /flash-attention | Attention 显存 O(N²)→O(N) |
| 3 | DeepSpeed ZeRO Stage 2 | v1 | /deepspeed-optimization | 优化器显存分片 |
| 4 | DataLoader 多进程加载 | v1 | - | GPU 等待时间 -50% |
| ... | | | | |

## 解决的瓶颈点

| # | 瓶颈 | 诊断来源 | 解决方案 | 验证结果 |
|---|------|---------|---------|---------|
| 1 | GPU 利用率仅 45% | v0_nsight-systems-report | DataLoader num_workers=8 | GPU 利用率提升至 85% |
| 2 | 显存不足，micro_batch=1 | v0_pytorch-profiler-report | 启用 BF16 + ZeRO-2 | micro_batch 可增至 4 |
| ... | | | | |

## 关键代码变更

<对比优化前后的核心代码差异，使用 diff 格式>

## 多机通信优化建议（如为多机模式）

> 以下优化基于单节点 Profiling 数据推断，需在多机环境中实际验证。

| # | 建议 | 预期收益 | 验证状态 |
|---|------|---------|---------|
| 1 | <建议> | <收益> | ⏳ 待多机验证 |

## 多机部署说明（如为多机模式）

1. 将 `<WORK_DIR>/project/` 目录同步到所有节点的相同路径
2. 确保所有节点安装相同的 Python 依赖版本
3. 使用以下命令在每个节点上启动（或通过 hostfile 统一启动）

## 优化后的启动命令

```bash
# 说明：使用 DeepSpeed launcher，4 卡 ZeRO-2 + BF16 混合精度
deepspeed --num_gpus=4 \
  ${WORK_DIR}/project/train.py \
  --config ${WORK_DIR}/project/config.yaml \
  --deepspeed ${WORK_DIR}/project/ds_config.json
```

**启动命令参数说明：**

| 参数                           | 说明                            |
| ------------------------------ | ------------------------------- |
| `--num_gpus=4`               | 使用 4 张 GPU                   |
| `--deepspeed ds_config.json` | DeepSpeed 配置（ZeRO-2 + BF16） |
| ...                            | ...                             |

## 所有报告索引

| 版本  | 报告                  | 路径                                                                   |
| ----- | --------------------- | ---------------------------------------------------------------------- |
| v0    | 通用训练加速审计      | reports/v0_training-acceleration-audit-report.md                       |
| v0    | LlamaFactory 专项审计 | reports/v0_llamafactory-optimization-report.md（仅 LlamaFactory 项目） |
| v0    | ms-swift 专项审计     | reports/v0_swift-optimization-report.md（仅 ms-swift 项目）            |
| v0    | VideoX-Fun 专项审计   | reports/v0_videox-fun-optimization-report.md（仅 VideoX-Fun 项目）     |
| v0    | Flow-Factory 专项审计 | reports/v0_flow-factory-optimization-report.md（仅 Flow-Factory 项目） |
| v0    | HF Trainer 专项审计   | reports/v0_transformers-optimization-report.md（仅 HF Trainer 项目）   |
| v0    | vLLM 推理优化审计     | reports/v0_vllm-optimization-report.md（检测到 vLLM 时）               |
| v0    | SGLang 推理优化审计   | reports/v0_sglang-optimization-report.md（检测到 SGLang 时）           |
| v0    | PyTorch Profiler      | reports/v0_pytorch-profiler-report.md                                  |
| v0    | Nsight Systems        | reports/v0_nsight-systems-report.md                                    |
| v0    | 系统资源              | reports/v0_system-resources-report.md                                  |
| v0    | 瓶颈诊断              | reports/v0_bottleneck-analysis.md                                      |
| v1    | 优化变更              | reports/v1_optimization-changes.md                                     |
| v1    | PyTorch Profiler      | reports/v1_pytorch-profiler-report.md                                  |
| ...   | ...                   | ...                                                                    |
| final | 优化总结              | reports/final_optimization-summary.md                                  |

```

---

### 第十步：最终完整性检查

**目标**：确认原始项目未被修改，所有变更仅在 fork 目录中。

#### 10.1 验证原始项目完整性

```bash
# 对比原始项目和 fork 前的状态
# 方法 1：如果原始项目是 git 仓库
cd "${ORIGINAL_PROJECT}"
git status  # 应无任何变更

# 方法 2：对比文件校验和
diff <(find "${ORIGINAL_PROJECT}" -type f -exec md5sum {} \; | sort) \
     <(cat "${WORK_DIR}/reports/original_checksums.txt" | sort)
```

> 在第三步 fork 时，应先保存原始项目的文件校验和：
>
> ```bash
> find "${ORIGINAL_PROJECT}" -type f -exec md5sum {} \; > "${WORK_DIR}/reports/original_checksums.txt"
> ```

#### 10.2 向用户交付结果

最终输出给用户的信息：

```
✅ GPU 训练智能调优完成！

📂 优化后的项目: ${WORK_DIR}/project/
📊 优化报告:     ${WORK_DIR}/reports/final_optimization-summary.md
🚀 启动命令:     <优化后的启动命令>

⚡ 性能提升: 吞吐 +X%, 显存 -Y GB, 预估训练时间 -Z%

📝 原始项目未做任何修改: ${ORIGINAL_PROJECT}
```

---

## 附录：工作目录结构总览

```
<项目路径>_optimized/
├── project/                                        # fork 的项目副本
│   ├── train.py                                    # 优化后的训练脚本
│   ├── ds_config.json                              # DeepSpeed 配置（如新增）
│   ├── ...                                         # 其他项目文件
│   └── (data/ → 符号链接到原始数据)                  # 大文件可用符号链接
└── reports/
    ├── meta.md                                      # 元信息（项目路径、命令、时间等）
    ├── original_checksums.txt                        # 原始项目文件校验和
    │
    ├── v0_training-acceleration-audit-report.md      # [静态] 通用代码审计报告
    ├── v0_llamafactory-optimization-report.md        # [静态] LlamaFactory 专项审计（框架特定）
    ├── v0_videox-fun-optimization-report.md           # [静态] VideoX-Fun 专项审计（框架特定）
    ├── v0_flow-factory-optimization-report.md         # [静态] Flow-Factory 专项审计（框架特定）
    ├── v0_transformers-optimization-report.md          # [静态] HF Trainer 专项审计（框架特定）
    ├── v0_vllm-optimization-report.md                   # [静态] vLLM 推理优化审计（检测到 vLLM 时）
    ├── v0_sglang-optimization-report.md                 # [静态] SGLang 推理优化审计（检测到 SGLang 时）
    ├── v0_pytorch-profiler-report.md                 # [v0] 算子级性能基准
    ├── v0_nsight-systems-report.md                   # [v0] 系统级性能基准
    ├── v0_system-resources-report.md                 # [v0] 硬件资源信息
    ├── v0_bottleneck-analysis.md                     # [v0] 瓶颈诊断
    │
    ├── v1_optimization-changes.md                    # [v1] 优化变更记录
    ├── v1_pytorch-profiler-report.md                 # [v1] 优化后性能
    ├── v1_nsight-systems-report.md                   # [v1] 优化后系统级
    ├── v1_system-resources-report.md                 # [v1] 运行时资源
    ├── v1_bottleneck-analysis.md                     # [v1] 残留瓶颈
    │
    ├── v2_optimization-changes.md                    # [v2] 第二轮优化（如需要）
    ├── ...
    │
    └── final_optimization-summary.md                 # 最终优化总结报告
```

## 附录：Skill 调用关系图

```
第一步: 确认输入 + 框架检测 + vLLM/SGLang 检测
  │
第二步: /training-acceleration-audit ──→ v0_audit-report（所有项目）
  │
  ├── FRAMEWORK=llamafactory ?
  │     │
  │   第二步(额外): /llamafactory-optimization ──→ v0_llamafactory-report
  │
  ├── FRAMEWORK=swift ?
  │     │
  │   第二步(额外): /swift-optimization ──→ v0_swift-report
  │
  ├── FRAMEWORK=videox_fun ?
  │     │
  │   第二步(额外): /videox-fun-optimization ──→ v0_videox-fun-report
  │
  ├── FRAMEWORK=flow_factory ?
  │     │
  │   第二步(额外): /flow-factory-optimization ──→ v0_flow-factory-report
  │
  ├── FRAMEWORK=hf_trainer ?
  │     │
  │   第二步(额外): /transformers-optimization ──→ v0_transformers-report
  │
  ├── VLLM_DETECTED=yes ?（可与任何训练框架并存）
  │     │
  │   第二步(额外): /vllm-optimization ──→ v0_vllm-report
  │
  ├── SGLANG_DETECTED=yes ?（可与任何训练框架并存）
  │     │
  │   第二步(额外): /sglang-optimization ──→ v0_sglang-report
  │
第三步: Fork 项目
  │
第四步: /pytorch-profiler + /nsight-systems ──→ v0_profiler-reports
  │
第五步: /system-resources ──→ v0_resources + v0_bottleneck ◄── 综合分析（合并所有审计报告）
  │
第六步: 实施优化
  │   ├── FRAMEWORK=llamafactory ?
  │   │     ├── 第一层: YAML 配置优化（flash_attn/unsloth/liger/packing/...）
  │   │     └── 第二层: 框架代码修改（torch.compile/Triton（参考 /triton-optimization）/CUDA Graph/通信调优/...）
  │   │
  │   ├── FRAMEWORK=videox_fun ?
  │   │     ├── 第一层: CLI 参数优化（low_vram/bucket/8bit_adam/FSDP/...）
  │   │     └── 第二层: 框架代码修改（torch.compile/Liger Kernel/QLoRA/VAE tiling/FusedAdam/...）
  │   │
  │   ├── FRAMEWORK=flow_factory ?
  │   │     ├── 第一层: YAML 配置优化（attn_backend/mixed_precision/offload/gradient_checkpointing/FSDP2/DeepSpeed/...）
  │   │     └── 第二层: 框架代码修改（torch.compile/FusedAdam/8-bit Adam/Liger Kernel/QLoRA/LR scheduler/...）
  │   │
  │   ├── FRAMEWORK=generic ?
  │   │     └── /flash-attention + /deepspeed-optimization + /triton-optimization(按需) + 代码优化 ──→ v1_changes
  │   │
  │   ├── VLLM_DETECTED=yes ?（与训练框架优化并行）
  │   │     ├── 第一层: vLLM 引擎参数优化（max_model_len/kv_cache_dtype/quantization/scheduling/...）
  │   │     └── 第二层: RLHF 集成优化（sleep_mode/weight_transfer/colocate/async_rl/...）
  │   │
  │   └── SGLANG_DETECTED=yes ?（与训练框架优化并行）
  │         ├── 第一层: SGLang 引擎参数优化（context_length/kv_cache_dtype/dp_attention/schedule_policy/...）
  │         └── 第二层: RLHF 集成优化（memory_saver/weight_update/deterministic/r_fork/...）
  │
第七步: 执行 → 修复 → 重新 profiling ──→ v1_profiler-reports
  │
第八步: 收敛判断
  │     │
  │   未收敛 → 回到第五步（v2, v3...）
  │     │
  │   收敛 ↓
  │
第九步: final_optimization-summary
  │
第十步: 完整性检查 → 交付
```

## 附录：快速启动模板

对于常见场景的简化启动方式：

```
用户: 请帮我优化这个训练项目
      项目路径: /workspace/my_llm_training
      启动命令: torchrun --nproc_per_node=4 train.py --config config.yaml
      模型: LLaMA-7B
```

→ 触发 `/gpu-training-optimizer` skill，自动执行完整 10 步流程。
