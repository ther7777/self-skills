---
name: gpu-static-analysis
description: 纯静态代码分析，无需 GPU/运行环境，扫描训练项目并输出优化建议报告
---

> **依赖说明**：本 skill 调用 `gpu-training-optimizer/_internals/` 下的子 skill（如 `training-acceleration-audit`、`llamafactory-optimization` 等），需同时安装 `gpu-training-optimizer` skill。

## 描述
对 GPU 训练项目进行纯静态代码分析，无需 GPU 或运行时环境。自动检测训练框架，执行通用训练加速审计和框架专项审计，生成包含优化建议、具体代码/配置变更示例和预估影响的完整优化报告。

## 触发条件
当用户要求对训练项目进行纯静态分析、不需要 Profiling 的代码审计时触发。

## 核心约束

- **⚠️ 不运行任何代码**：不执行训练脚本、不做 Profiling、不需要 GPU
- **⚠️ 不修改原始项目**：所有操作为只读扫描
- **⚠️ 不 Fork 项目**：无需创建副本（因为不修改代码）
- **⚠️ 必须生成最终报告**：`static-analysis-report.md` 是用户最终交付物，缺失则整个分析流程等于无效

## 输出目录

所有报告输出到 `<项目路径>_static_analysis/` 目录：

```
<项目路径>_static_analysis/
├── static-analysis-report.md                    # 最终综合优化报告（主交付物）
├── v0_training-acceleration-audit-report.md      # 通用加速审计
├── v0_<framework>-optimization-report.md         # 框架专项审计（如适用）
├── v0_vllm-optimization-report.md                # vLLM 推理优化审计（如适用）
└── v0_sglang-optimization-report.md              # SGLang 推理优化审计（如适用）
```

## 执行指令

你是 GPU 训练优化专家。按以下 4 个步骤完成纯静态代码分析。

---

### 第一步：确认输入与框架检测

#### 1.1 收集信息

从用户的 prompt 中提取以下信息。**不要向用户反问**，直接从 prompt 和项目代码中获取：

| 信息 | 必填 | 获取方式 | 默认策略 |
|------|------|----------|---------|
| **项目路径** | 是 | 从 prompt 提取 | — |
| **启动命令** | 是 | 从 prompt 提取 | — |
| **模型参数量** | 推荐 | 从项目代码/配置推断（搜索 config.json、model_name_or_path 等） | 如无法推断则标注"未知" |
| **优化目标** | 可选 | 从 prompt 提取 | 默认"综合优化（吞吐优先）" |

> **启动命令的作用**：启动命令不会被实际执行，仅用于：(1) 辅助框架检测（如 `swift sft`、`llamafactory-cli`、`deepspeed --hostfile`）；(2) 判断并行策略和多机模式（如 `--nproc_per_node`、`--nnodes`）；(3) 提取训练入口脚本路径和参数；(4) 生成更有针对性的优化建议。

#### 1.2 训练框架检测

扫描项目文件，识别是否使用特定训练框架。框架检测影响后续审计策略的选择。

**LlamaFactory 检测规则**（满足任一即判定为 LlamaFactory 项目）：

| 检测方式 | 匹配模式 |
|---------|---------|
| **依赖文件** | `requirements.txt`、`pyproject.toml`、`setup.py` 中包含 `llamafactory` 或 `llama-factory` |
| **import 语句** | `from llamafactory`、`import llamafactory` |
| **CLI 命令** | 项目脚本中包含 `llamafactory-cli`、`llamafactory.cli` |
| **配置参数** | YAML/JSON 中包含 LlamaFactory 特有参数：`finetuning_type`、`neat_packing`、`use_unsloth`、`enable_liger_kernel`、`lora_target`、`flash_attn: fa2` |
| **项目结构** | 存在 `data/dataset_info.json`、或 `examples/train_lora/`、`examples/deepspeed/ds_z*_config.json` |

**ms-swift 检测规则**（满足任一即判定为 ms-swift 项目）：

| 检测方式 | 匹配模式 |
|---------|---------|
| **依赖文件** | `requirements.txt`、`pyproject.toml`、`setup.py` 中包含 `ms-swift` |
| **import 语句** | `from swift`、`import swift`、`from swift.llm`、`from swift.trainers` |
| **CLI 命令** | 项目脚本中包含 `swift sft`、`swift pt`、`swift rlhf`、`megatron sft`、`megatron pt` |
| **配置参数** | YAML/CLI 中包含 ms-swift 特有参数：`tuner_type`、`tuner_backend`、`attn_impl`、`padding_free`、`loss_scale`、`sequence_parallel_size` |
| **检查点** | `args.json` 中包含 `swift_version` 字段 |

**VideoX-Fun 检测规则**（满足任一即判定为 VideoX-Fun 项目）：

| 检测方式 | 匹配模式 |
|---------|---------|
| **依赖文件** | `requirements.txt`、`pyproject.toml`、`setup.py` 中包含 `videox-fun` 或 `videox_fun` |
| **import 语句** | `from videox_fun`、`import videox_fun` |
| **项目结构** | 存在 `videox_fun/` 目录（含 `models/`、`data/`、`pipeline/` 子目录） |
| **脚本路径** | 脚本中包含 `scripts/wan2.1_fun/`、`scripts/wan2.2_fun/`、`scripts/cogvideox_fun/`、`scripts/hunyuanvideo/` |
| **配置文件** | 存在 `config/wan2.1/wan_civitai.yaml`、`config/wan2.2/` 等 |
| **Git Remote** | `.git/config` 中 remote URL 包含 `VideoX-Fun` |

**Flow-Factory 检测规则**（满足任一即判定为 Flow-Factory 项目）：

| 检测方式 | 匹配模式 |
|---------|---------|
| **依赖文件** | `requirements.txt`、`pyproject.toml`、`setup.py` 中包含 `flow-factory` 或 `flow_factory` |
| **import 语句** | `from flow_factory`、`import flow_factory` |
| **CLI 命令** | 启动命令包含 `ff-train`、`flow-factory-train` |
| **配置参数** | YAML 中包含 `trainer_type:`（值为 grpo/nft/awm/dpo/dgpo/crd/diffusion-opd）、`dynamics_type:`、`offload_samples_to_cpu:`、`sample_group_size:` |
| **项目结构** | 存在 `src/flow_factory/` 目录 |
| **Git Remote** | `.git/config` 中 remote URL 包含 `Flow-Factory` |

**HuggingFace Transformers Trainer 检测规则**（仅在上述框架均未匹配时检测）：

| 检测方式 | 匹配模式 |
|---------|---------|
| **import 语句** | `from transformers import Trainer`、`from transformers import TrainingArguments`、`from transformers import Seq2Seq` |
| **代码模式** | `Trainer(model=`、`TrainingArguments(` |

**vLLM 推理框架检测规则**（可与训练框架并存，单独标记）：

| 检测方式 | 匹配模式 |
|---------|---------|
| **依赖文件** | `requirements.txt`、`pyproject.toml`、`setup.py` 中包含 `vllm` |
| **import 语句** | `from vllm import`、`from vllm.`、`import vllm` |
| **CLI 命令** | 启动命令包含 `vllm serve`、`python -m vllm` |
| **RLHF 集成** | 代码/配置中包含 `use_vllm=True`、`vllm_mode`、`infer_backend: vllm`、`enable_sleep_mode` |
| **vLLM 特有参数** | `gpu_memory_utilization`、`enforce_eager`、`enable_prefix_caching`、`max_num_seqs`、`speculative_config` |

**SGLang 推理框架检测规则**（可与训练框架并存，单独标记）：

| 检测方式 | 匹配模式 |
|---------|---------|
| **依赖文件** | `requirements.txt`、`pyproject.toml`、`setup.py` 中包含 `sglang` 或 `sgl-kernel` |
| **import 语句** | `from sglang import`、`from sglang.`、`import sglang`、`from sglang_router` |
| **CLI 命令** | 启动命令包含 `sglang.launch_server`、`sglang_router.launch_server` |
| **RLHF 集成** | 代码/配置中包含 `infer_backend: sglang`、`sglang_maxlen`、`sglang_mem_fraction`、`enable_memory_saver`、`update_weights_from_distributed` |
| **SGLang 特有参数** | `mem_fraction_static`、`schedule_policy`、`chunked_prefill_size`、`enable_dp_attention`、`radix_cache`、`piecewise_cuda_graph` |

```bash
# 快速检测命令
ORIGINAL_PROJECT="<用户提供的项目路径>"
LAUNCH_CMD="<用户提供的启动命令>"
FRAMEWORK="generic"

# 检查启动命令 — LlamaFactory
echo "${LAUNCH_CMD}" | grep -q "llamafactory-cli" && FRAMEWORK="llamafactory"

# 检查依赖文件 — LlamaFactory
grep -rls "llamafactory\|llama-factory\|LLaMA-Factory" \
  "${ORIGINAL_PROJECT}/requirements.txt" \
  "${ORIGINAL_PROJECT}/pyproject.toml" \
  "${ORIGINAL_PROJECT}/setup.py" 2>/dev/null && FRAMEWORK="llamafactory"

# 检查配置文件中的 LlamaFactory 特有参数
grep -rls "finetuning_type:\|neat_packing:\|use_unsloth:\|enable_liger_kernel:\|lora_target:" \
  "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/*.yml 2>/dev/null && FRAMEWORK="llamafactory"

# 检查依赖文件 — ms-swift
grep -rls "ms-swift" \
  "${ORIGINAL_PROJECT}/requirements.txt" \
  "${ORIGINAL_PROJECT}/pyproject.toml" \
  "${ORIGINAL_PROJECT}/setup.py" 2>/dev/null && FRAMEWORK="swift"

# 检查启动命令 — ms-swift
echo "${LAUNCH_CMD}" | grep -qE "swift (sft|pt|rlhf|infer|deploy)|megatron (sft|pt|rlhf)" && FRAMEWORK="swift"

# 检查配置文件中的 ms-swift 特有参数
grep -rls "tuner_type:\|tuner_backend:\|attn_impl:\|padding_free:\|sequence_parallel_size:" \
  "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/*.yml 2>/dev/null && FRAMEWORK="swift"

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

# 检查是否使用 vLLM 推理框架（可与训练框架并存）
VLLM_DETECTED="no"

grep -rls "^vllm\|vllm[>=<]" \
  "${ORIGINAL_PROJECT}/requirements.txt" \
  "${ORIGINAL_PROJECT}/pyproject.toml" \
  "${ORIGINAL_PROJECT}/setup.py" 2>/dev/null && VLLM_DETECTED="yes"

grep -rls "from vllm import\|from vllm\.\|import vllm" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && VLLM_DETECTED="yes"

echo "${LAUNCH_CMD}" | grep -qE "vllm serve|python -m vllm" && VLLM_DETECTED="yes"

grep -rls "use_vllm.*[Tt]rue\|vllm_mode\|infer_backend.*vllm\|enable_sleep_mode" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/*.yml \
  "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && VLLM_DETECTED="yes"

# 如果仅有 vLLM（无训练框架），设为 vllm 框架
if [ "${FRAMEWORK}" = "generic" ] && [ "${VLLM_DETECTED}" = "yes" ]; then
  FRAMEWORK="vllm"
fi

# 检查是否使用 SGLang 推理框架（可与训练框架并存）
SGLANG_DETECTED="no"

grep -rls "^sglang\|sglang[>=<]\|sgl-kernel" \
  "${ORIGINAL_PROJECT}/requirements.txt" \
  "${ORIGINAL_PROJECT}/pyproject.toml" \
  "${ORIGINAL_PROJECT}/setup.py" 2>/dev/null && SGLANG_DETECTED="yes"

grep -rls "from sglang import\|from sglang\.\|import sglang\|from sglang_router" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && SGLANG_DETECTED="yes"

echo "${LAUNCH_CMD}" | grep -qE "sglang\.launch_server|sglang_router\.launch_server" && SGLANG_DETECTED="yes"

grep -rls "infer_backend.*sglang\|sglang_maxlen\|sglang_mem_fraction\|enable_memory_saver\|update_weights_from_distributed\|update_weights_from_tensor" \
  "${ORIGINAL_PROJECT}"/*.py "${ORIGINAL_PROJECT}"/*.yaml "${ORIGINAL_PROJECT}"/*.yml \
  "${ORIGINAL_PROJECT}"/**/*.py 2>/dev/null && SGLANG_DETECTED="yes"

# 如果仅有 SGLang（无训练框架且无 vLLM），设为 sglang 框架
if [ "${FRAMEWORK}" = "generic" ] && [ "${SGLANG_DETECTED}" = "yes" ]; then
  FRAMEWORK="sglang"
fi

# 检查是否直接使用 HuggingFace Transformers Trainer（仅在未检测到上层框架时）
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

> **框架检测优先级**：LlamaFactory > ms-swift > VideoX-Fun > Flow-Factory > vLLM（独立）> SGLang（独立）> HF Trainer > 通用。vLLM/SGLang 可与任何训练框架并存（通过独立标记）。

#### 1.3 创建输出目录

```bash
OUTPUT_DIR="${ORIGINAL_PROJECT}_static_analysis"
mkdir -p "${OUTPUT_DIR}"
echo "输出目录: ${OUTPUT_DIR}"
```

---

### 第二步：执行通用训练加速审计

**无论检测到哪种框架**，都按照 **`training-acceleration-audit` skill** 的完整执行指令，对原始项目（只读扫描）进行全面审计：

1. 扫描项目结构：依赖文件、训练入口、配置文件、启动脚本
2. 逐项检查 9 大类别（A-I）的使用状态：
   - A. 并行策略（DDP/FSDP/ZeRO/TP/PP/SP/EP）
   - B. 训练框架（DeepSpeed/Megatron/Accelerate/ColossalAI/FSDP/veScale）
   - C. 显存优化（混合精度/FP8/激活重计算/CPU卸载/梯度累积）
   - D. 计算优化（Flash Attention/torch.compile/Triton/CUDA Graph/Liger Kernel/算子融合）
   - E. 通信优化（GDRDMA/通信计算重叠/梯度压缩/NCCL调优）
   - F. 数据与 I/O 优化（多进程加载/预取/Pin Memory/小文件合并/Packing/流式数据集）
   - G. 训练策略（LR调度/LoRA/QLoRA/层冻结/对齐训练Loss优化）
   - H. 多模态专项（视觉编码器冻结/动态分辨率/视觉Token裁剪）
   - I. 基础设施（存储同城/高性能存储/Checkpoint频率/日志频率）
3. 生成带评分的审计报告

**保存为**：`${OUTPUT_DIR}/v0_training-acceleration-audit-report.md`

---

### 第三步：执行框架专项审计（如适用）

根据第一步检测到的框架，执行对应的专项审计 skill：

**LlamaFactory 项目** (`FRAMEWORK=llamafactory`)：
- 按 `/llamafactory-optimization` skill 的完整执行指令审计 10 大类（A-J，/58 分）
- 保存为 `${OUTPUT_DIR}/v0_llamafactory-optimization-report.md`

**ms-swift 项目** (`FRAMEWORK=swift`)：
- 按 `/swift-optimization` skill 的完整执行指令审计 10 大类（A-J，/60 分）
- 保存为 `${OUTPUT_DIR}/v0_swift-optimization-report.md`

**VideoX-Fun 项目** (`FRAMEWORK=videox_fun`)：
- 按 `/videox-fun-optimization` skill 的完整执行指令审计 9 大类（A-I，/61 分）
- 保存为 `${OUTPUT_DIR}/v0_videox-fun-optimization-report.md`

**Flow-Factory 项目** (`FRAMEWORK=flow_factory`)：
- 按 `/flow-factory-optimization` skill 的完整执行指令审计 10 大类（A-J，/49 分）
- 保存为 `${OUTPUT_DIR}/v0_flow-factory-optimization-report.md`

**HuggingFace Transformers Trainer 项目** (`FRAMEWORK=hf_trainer`)：
- 按 `/transformers-optimization` skill 的完整执行指令审计 10 大类（A-J，/62 分）
- 保存为 `${OUTPUT_DIR}/v0_transformers-optimization-report.md`

**通用项目** (`FRAMEWORK=generic`)：
- 无额外专项审计，仅依赖第二步的通用审计结果

**vLLM 推理优化** (`VLLM_DETECTED=yes`，可与任何训练框架并存)：
- 按 `/vllm-optimization` skill 的完整执行指令审计 10 大类（A-J，/52 分）
- 保存为 `${OUTPUT_DIR}/v0_vllm-optimization-report.md`

**SGLang 推理优化** (`SGLANG_DETECTED=yes`，可与任何训练框架并存)：
- 按 `/sglang-optimization` skill 的完整执行指令审计 10 大类（A-J，/54 分）
- 保存为 `${OUTPUT_DIR}/v0_sglang-optimization-report.md`

> **多份报告互补**：通用审计覆盖面广（torch.compile、CUDA Graph、Triton 自定义算子、通信优化等），框架专项审计聚焦该框架内可配置的优化项。vLLM/SGLang 审计聚焦推理引擎的性能优化，在 RLHF/GRPO 场景中与训练框架审计互补。所有报告结合才能最大化优化空间。

---

### 第四步：生成综合优化报告

基于第二步和第三步的审计结果，生成最终的综合优化报告。这是用户的**最终交付物**，必须生成。

**输出文件**：`${OUTPUT_DIR}/static-analysis-report.md`

**报告格式**：

```markdown
# GPU 训练项目静态分析 — 优化建议报告

## 项目概览

| 项目 | 值 |
|------|-----|
| 项目路径 | <路径> |
| 启动命令 | `<命令>` |
| 检测到的训练框架 | <框架名称> |
| 模型类型 | <LLM / MLLM / 视频生成 / 未知> |
| 模型规模 | <推断的参数量或"未知"> |
| 优化目标 | <用户指定或默认> |
| 分析时间 | <完成时间> |

## 审计评分总结

### 通用训练加速审计（9 大类）

| # | 类别 | 得分 | 满分 | 关键发现 |
|---|------|------|------|----------|
| A | 并行策略 | X | 7 | ... |
| B | 训练框架 | X | 6 | ... |
| C | 显存优化 | X | 5 | ... |
| D | 计算优化 | X | 6 | ... |
| E | 通信优化 | X | 4 | ... |
| F | 数据与 I/O 优化 | X | 10 | ... |
| G | 训练策略 | X | 6 | ... |
| H | 多模态专项 | X | 3 或 N/A | ... |
| I | 基础设施 | X | 5 | ... |
| | **总分** | **X** | **Y** | |

### <框架名>专项审计（如适用）

| # | 类别 | 得分 | 满分 | 关键发现 |
|---|------|------|------|----------|
| ... | | | | |
| | **总分** | **X** | **Y** | |

## 优化建议（按优先级排序）

### 🔴 高优先级（预估提升 > 30%）

#### 1. <优化项名称>

- **当前状态**: <未使用/部分使用，附证据>
- **推荐操作**: <具体操作说明>
- **预估影响**: <吞吐提升/显存节省的预估>
- **实施难度**: 低/中/高
- **具体变更**:

（根据框架提供对应的配置/代码变更示例）

对于 LlamaFactory 项目提供 YAML 配置：
```yaml
# LlamaFactory YAML 配置示例
```

对于 ms-swift 项目提供 CLI/YAML 配置：
```yaml
# ms-swift 配置示例
```

对于 HF Trainer 项目提供 TrainingArguments：
```python
# TrainingArguments 配置示例
```

对于通用项目提供 Python 代码：
```python
# 代码变更示例
```

#### 2. ...

### 🟡 中优先级（预估提升 10-30%）

（同上格式）

### 🟢 低优先级（预估提升 < 10%）

（同上格式）

## 推荐配置模板

根据项目特征和检测到的框架，提供完整的场景化配置模板。

### 场景 1：<描述（如 "7B 模型 LoRA 微调 on 8xA100"）>

```yaml
# 或 python/json，取决于框架
# 完整可用的配置模板
```

### 场景 2：<描述>

```yaml
# 完整可用的配置模板
```

## Flash Attention 配置建议

根据项目实际情况，参考 `/flash-attention` skill 的选型指南：

- **推荐版本**: <FA2/FA3/FA4，根据模型架构和 GPU 推断>
- **安装方式**: <pip 命令>
- **集成方式**: <具体代码或配置>

## DeepSpeed / 分布式配置建议

根据项目的模型规模和训练方式，参考 `/deepspeed-optimization` skill 的配置建议：

- **推荐 ZeRO Stage**: <Stage 0/1/2/3，附决策理由>
- **是否需要 Offload**: <是/否，附理由>
- **推荐配置**:

```json
{
  // 完整的 ds_config.json 模板
}
```

## 所有报告索引

| 报告 | 路径 | 说明 |
|------|------|------|
| 综合优化建议报告 | static-analysis-report.md | 本报告（主交付物） |
| 通用训练加速审计 | v0_training-acceleration-audit-report.md | 9 大类通用审计 |
| <框架>专项审计 | v0_<framework>-optimization-report.md | 框架专项审计 |
| vLLM 推理优化审计 | v0_vllm-optimization-report.md | vLLM 推理专项审计（检测到 vLLM 时） |
| SGLang 推理优化审计 | v0_sglang-optimization-report.md | SGLang 推理专项审计（检测到 SGLang 时） |
```

#### 报告生成要求

1. **每条建议都必须包含具体的代码/配置变更示例**，不要只给抽象建议
2. **优先级排序严格按照 `training-acceleration-audit` skill 第四步的规则**
3. **预估影响要具体**：给出百分比范围（如"吞吐提升 20-30%"），而非模糊描述
4. **配置模板必须完整可用**：用户可以直接复制使用，不需要额外修改
5. **Flash Attention 和 DeepSpeed 建议要结合项目实际**：根据模型架构、参数量、GPU 类型（如可推断）给出针对性建议
6. **如果某些信息无法从静态分析中获取**（如 GPU 型号、实际显存），明确标注"需根据实际硬件调整"
