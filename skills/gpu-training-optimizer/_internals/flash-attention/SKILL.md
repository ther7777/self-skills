---
name: flash-attention
description: Flash Attention 版本选型与最佳配置
user-invocable: false
---

## 描述
指导用户为其 GPU 训练/推理工作负载选择并配置最优的 Flash Attention 版本，涵盖版本选型、安装、API 使用、HuggingFace/框架集成、推理 KV Cache 优化等，目标是"开启到最佳"。

## 触发条件
当用户需要启用或优化 Flash Attention 时触发。适用场景包括：新项目选型 Flash Attention 版本、升级到更高效的 FA 版本、排查 FA 安装/兼容性问题、配置推理 KV Cache、在 HuggingFace/DeepSpeed/Megatron 中集成 FA 等。

## 执行指令

你是 Flash Attention 配置优化专家。根据用户的 GPU 硬件、使用场景和框架生态，指导用户启用最佳的 Flash Attention 配置。

---

### 第一步：确定 GPU 硬件与版本选型

Flash Attention 有多个版本，**核心选型依据是 GPU 架构**：

| 版本 | GPU 架构要求 | 数据类型 | 特点 | 推荐场景 |
|------|------------|---------|------|---------|
| **Flash Attention 2** | Ampere (A100/A10/A30) 及以上 | FP16 / BF16 | 最成熟稳定，生态最广 | **通用首选**，绝大多数训练和推理场景 |
| **Flash Attention 3** | Hopper (H100/H200) | FP16 / BF16 / **FP8** | 利用 H100 异步特性 (WGMMA + TMA + pingpong)，FP8 支持 | H100 集群追求极致性能 |
| **Flash Attention 4** | Hopper (H100) + Blackwell (B200) | FP16 / BF16 / FP8 / FP4 | 基于 CuTeDSL (CUDA C++)，支持 FP4，自动化 kernel 生成 | 最新硬件上的前沿探索 |

**选型决策树：**

```
你的 GPU 是什么架构？
├── Ampere (A100, A10, A30, RTX 30xx/40xx)  → Flash Attention 2
├── Hopper (H100, H200)
│   ├── 需要 FP8 训练？  → Flash Attention 3
│   └── 不需要 FP8      → Flash Attention 2 或 3 均可
└── Blackwell (B200, GB200)
    └── Flash Attention 4（实验性）
```

> **注意**：Volta (V100) 及更早架构**不支持** Flash Attention。V100 用户应使用 `xformers.ops.memory_efficient_attention` 或 PyTorch 原生 SDPA。

---

### 第二步：安装

#### 2.1 Flash Attention 2（推荐）

**方式 1：pip 安装（推荐）**
```bash
pip install flash-attn --no-build-isolation
```

**方式 2：从源码编译（需要自定义 CUDA 架构时）**
```bash
# 指定 GPU 架构加速编译（例如 A100 = sm_80, H100 = sm_90）
FLASH_ATTENTION_FORCE_BUILD=TRUE \
TORCH_CUDA_ARCH_LIST="8.0" \
pip install flash-attn --no-build-isolation
```

**依赖要求：**
- PyTorch >= 2.0
- CUDA >= 11.6（推荐 11.8+）
- Linux（不支持 Windows/macOS）

**常见安装问题：**

| 问题 | 解决方案 |
|------|---------|
| 编译时间过长 (>30min) | 设置 `MAX_JOBS=4` 限制并行编译数，或指定 `TORCH_CUDA_ARCH_LIST` 仅编译目标架构 |
| CUDA 版本不兼容 | 确认 `nvcc --version` 和 `torch.version.cuda` 一致 |
| 内存不足 (OOM during build) | 设置 `MAX_JOBS=2` 减少并行编译 |
| `No module named 'flash_attn'` | 确认安装时未报错，且 Python 环境一致 |

#### 2.2 Flash Attention 3（H100 专用）

```bash
# FA3 在 Hopper 子目录中
cd hopper
python setup.py install
```

#### 2.3 PyTorch 原生 SDPA（零安装方案）

PyTorch >= 2.0 内置了 `scaled_dot_product_attention`，自动选择 Flash Attention / Memory-Efficient / Math 后端：

```python
import torch.nn.functional as F

# PyTorch 自动选择最优后端（包括 Flash Attention）
output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
```

> SDPA 是零依赖方案，但功能不如 `flash-attn` 包完整（如不支持 ALiBi、sliding window、paged KV cache 等高级特性）。

---

### 第三步：API 使用与最佳配置

#### 3.1 核心 API 速查

| API | 用途 | 输入格式 |
|-----|------|---------|
| `flash_attn_func` | 通用 attention | `q, k, v` 各自独立，shape `(batch, seqlen, nheads, headdim)` |
| `flash_attn_qkvpacked_func` | QKV 打包（MHA） | `qkv` shape `(batch, seqlen, 3, nheads, headdim)` |
| `flash_attn_kvpacked_func` | KV 打包（MQA/GQA） | `q` + `kv` shape `(batch, seqlen, 2, nheads_kv, headdim)` |
| `flash_attn_with_kvcache` | 推理 KV Cache | 支持 paged attention、RoPE on-the-fly |
| `flash_attn_varlen_func` | 变长序列 | 通过 `cu_seqlens` 指定每个样本长度，避免 padding |

#### 3.2 训练最佳配置

```python
from flash_attn import flash_attn_func

# 标准训练 attention（推荐配置）
output = flash_attn_func(
    q, k, v,                    # (batch, seqlen, nheads, headdim)
    causal=True,                # 自回归模型必须开启
    softmax_scale=None,         # 默认 1/sqrt(headdim)，通常不需要改
    deterministic=False,        # True 可保证反向传播确定性，但会慢 ~5-10%
)
```

**支持的 head_dim**：FA2 支持最大 256（任意值到 256），推荐使用 64、128、256。

#### 3.3 高级功能启用

| 功能 | 参数 | 说明 |
|------|------|------|
| **因果注意力** | `causal=True` | GPT 类自回归模型必须开启 |
| **滑动窗口注意力** | `window_size=(left, right)` | 如 Mistral: `window_size=(4096, 0)` 表示左看 4096 token |
| **ALiBi 位置编码** | `alibi_slopes` | shape `(nheads,)` 或 `(batch, nheads)`，替代 RoPE 时使用 |
| **Softcapping** | `softcap=50.0` | Gemma-2 / Grok 使用，对 attention logits 做 `tanh(x/softcap)*softcap` |
| **确定性反向传播** | `deterministic=True` | 牺牲 ~5-10% 性能换取可复现梯度，调试时有用 |

#### 3.4 变长序列（避免 padding 浪费）

对于 batch 内序列长度不一的场景，使用 `flash_attn_varlen_func` 避免 padding 浪费：

```python
from flash_attn import flash_attn_varlen_func

# cu_seqlens: 累积序列长度，如 3 个序列长度为 [100, 200, 150]
# cu_seqlens = [0, 100, 300, 450]
output = flash_attn_varlen_func(
    q, k, v,                    # (total_seqlen, nheads, headdim)
    cu_seqlens_q=cu_seqlens_q,  # (batch+1,) 累积长度
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    causal=True,
)
```

> 对于数据打包 (packing) 场景，`flash_attn_varlen_func` 是必需的——它能让多个短序列拼接后仍然正确计算各自的 attention，不会跨序列"看到"其他样本。

#### 3.5 推理 KV Cache 优化

```python
from flash_attn import flash_attn_with_kvcache

output = flash_attn_with_kvcache(
    q,                          # (batch, 1, nheads, headdim) — 当前 token
    k_cache,                    # (batch, max_seqlen, nheads_kv, headdim)
    v_cache,                    # (batch, max_seqlen, nheads_kv, headdim)
    k_new=k_new,                # 新 token 的 key
    v_new=v_new,                # 新 token 的 value
    cache_seqlens=cache_seqlens,  # 每个样本当前 cache 长度
    causal=True,
    # Paged KV Cache（大规模推理推荐）
    # block_table=block_table,  # (batch, max_num_blocks)，PagedAttention block 映射
    # RoPE on-the-fly（可选）
    # rotary_cos=rotary_cos,
    # rotary_sin=rotary_sin,
    # rotary_interleaved=False,
)
```

**Paged KV Cache** 特别适合大规模推理服务（如 vLLM）：
- KV cache 按固定大小 block 分配，支持非连续内存
- 减少显存碎片，提高 batch 并发量
- 通过 `block_table` 指定每个序列的 block 映射

---

### 第四步：框架集成

#### 4.1 HuggingFace Transformers（推荐，最简单）

```python
from transformers import AutoModelForCausalLM

# 方式 1：一行启用 Flash Attention 2
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B",
    attn_implementation="flash_attention_2",  # 关键参数
    torch_dtype=torch.bfloat16,               # FA 要求 FP16/BF16
    device_map="auto",
)
```

> **注意**：`attn_implementation` 支持的值：
> - `"flash_attention_2"` — 使用 flash-attn 包
> - `"sdpa"` — 使用 PyTorch 原生 SDPA
> - `"eager"` — 原始 Python 实现（最慢，仅用于调试）

**HuggingFace 集成注意事项：**
- 需要 `pip install flash-attn`
- 模型必须以 FP16 或 BF16 加载（`torch_dtype=torch.float16` 或 `torch.bfloat16`）
- 大多数主流模型已内置支持：LLaMA、Qwen、Mistral、Gemma、Phi、ChatGLM、InternLM 等

#### 4.2 PyTorch 原生 SDPA

```python
import torch
import torch.nn.functional as F

# 自动选择最优后端
with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(
        query, key, value,
        is_causal=True,
    )
```

#### 4.3 DeepSpeed 集成

在 DeepSpeed 配置中启用：
```json
{
  "replace_with_kernel_inject": true
}
```

或在代码中使用 `deepspeed.ops.transformer.inference` 模块时，DeepSpeed 会自动检测并使用 Flash Attention。

#### 4.4 Megatron-LM 集成

Megatron-LM 已原生支持 Flash Attention：
```python
# 在 Megatron 参数中添加
--use-flash-attn
```

---

### 第五步：性能优化 Checklist

按优先级逐项检查，确保 Flash Attention 发挥最佳性能：

| # | 检查项 | 操作 | 优先级 |
|---|--------|------|--------|
| 1 | **确认 FA 已生效** | 检查日志或用 `torch.profiler` 确认 kernel 名称包含 `flash` | 高 |
| 2 | **使用 BF16 而非 FP32** | FA 仅支持 FP16/BF16，FP32 会 fallback 到慢路径 | 高 |
| 3 | **开启 causal mask** | 自回归模型设置 `causal=True`，比传统 mask 矩阵更高效 | 高 |
| 4 | **head_dim 对齐** | head_dim 为 64/128/256 时性能最优，非标准值可能降速 | 中 |
| 5 | **使用 varlen 避免 padding** | 变长序列用 `flash_attn_varlen_func` 或数据打包 | 中 |
| 6 | **启用滑动窗口** | 长序列场景（如 Mistral）用 `window_size` 限制注意力范围 | 中 |
| 7 | **推理启用 KV Cache** | 使用 `flash_attn_with_kvcache` 避免重复计算 | 中 |
| 8 | **Paged KV Cache** | 大规模推理服务启用 paged attention 减少显存碎片 | 中 |
| 9 | **H100 考虑 FA3** | H100 上 FA3 的 FP8 支持可进一步提升吞吐 | 低 |
| 10 | **确定性模式按需开** | `deterministic=True` 仅在需要可复现性时开启 | 低 |

---

### 第六步：验证 Flash Attention 是否生效

#### 方法 1：检查 HuggingFace 模型日志

加载模型时会打印 attention 实现：
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
)
# 检查模型 config
print(model.config._attn_implementation)  # 应输出 "flash_attention_2"
```

#### 方法 2：用 PyTorch Profiler 验证

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    output = model(input_ids)

# 查找 flash 相关 kernel
for event in prof.key_averages():
    if "flash" in event.key.lower():
        print(f"{event.key}: {event.cuda_time_total}us")
```

如果看到类似 `void flash_fwd_kernel` 或 `flash::` 开头的 CUDA kernel，说明 Flash Attention 已生效。

#### 方法 3：对比显存占用

```python
# 不用 FA（标准 attention）
torch.cuda.reset_peak_memory_stats()
output_standard = standard_attention(q, k, v)
mem_standard = torch.cuda.max_memory_allocated()

# 用 FA
torch.cuda.reset_peak_memory_stats()
output_fa = flash_attn_func(q, k, v, causal=True)
mem_fa = torch.cuda.max_memory_allocated()

print(f"Standard: {mem_standard/1e9:.2f} GB, Flash: {mem_fa/1e9:.2f} GB")
# FA 的显存应显著低于 standard（尤其长序列时）
```

---

### 第七步：常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `No module named 'flash_attn'` | 未安装或环境不一致 | `pip install flash-attn --no-build-isolation` |
| 安装报错 `CUDA extension` | CUDA 版本不匹配 | 确认 `nvcc --version` 与 PyTorch CUDA 版本一致 |
| `RuntimeError: FlashAttention only supports Ampere GPUs or newer` | GPU 架构过旧 | V100 等不支持 FA，改用 `xformers` 或 SDPA |
| HF 模型加载后仍用 eager attention | `torch_dtype` 未设置为 FP16/BF16 | 添加 `torch_dtype=torch.bfloat16` |
| `CUDA out of memory` 反而更严重 | 非 FA 问题，其他算子占用 | 用 profiler 定位真正的显存大户 |
| FA 比预期慢 | head_dim 非标准值或 batch 太小 | 确保 head_dim 为 64/128/256，增大 batch |
| `deterministic` 模式结果不同 | 非确定性反向传播（默认行为） | 设置 `deterministic=True` |

---

### 附录：Flash Attention 版本详细对比

| 特性 | FA2 | FA3 | FA4 |
|------|-----|-----|-----|
| **GPU 架构** | Ampere+ | Hopper | Hopper + Blackwell |
| **数据类型** | FP16, BF16 | FP16, BF16, FP8 | FP16, BF16, FP8, FP4 |
| **实现语言** | CUDA | CUDA (cute) | CuTeDSL (CUDA C++) |
| **Causal** | ✅ | ✅ | ✅ |
| **滑动窗口** | ✅ | ✅ | ✅ |
| **ALiBi** | ✅ | ✅ | ❓ |
| **MQA/GQA** | ✅ | ✅ | ✅ |
| **Paged KV Cache** | ✅ | ✅ | ✅ |
| **Softcapping** | ✅ | ✅ | ✅ |
| **torch.compile** | ✅ | ❓ | ❓ |
| **Varlen** | ✅ | ✅ | ✅ |
| **确定性反向传播** | ✅ | ✅ | ❓ |
| **Head dim 范围** | 任意值到 256 | 64-256 | 64-256 |
| **生态成熟度** | ⭐⭐⭐ | ⭐⭐ | ⭐ |

### 附录：推理专项优化 — PagedAttention、FlashDecoding 与 Continuous Batching

Flash Attention 在推理场景中有不同于训练的优化策略和用法。

#### PagedAttention（分页注意力）

传统 KV Cache 为每个序列预分配连续显存（按 max_seq_len），实际使用率通常只有 50-60%，导致大量碎片浪费。PagedAttention 将 KV Cache 分为固定大小的"页"（block），按需分配：

```python
# flash_attn_with_kvcache 支持 paged KV cache（FA2 2.5.7+）
from flash_attn import flash_attn_with_kvcache

output = flash_attn_with_kvcache(
    q,                    # [batch, 1, heads, head_dim] — decode 阶段单 token
    k_cache,              # [num_blocks, block_size, kv_heads, head_dim]
    v_cache,              # [num_blocks, block_size, kv_heads, head_dim]
    cache_seqlens=seq_lens,  # [batch] — 每个序列当前长度
    block_table=block_table, # [batch, max_blocks_per_seq] — 页表
    causal=True,
)
```

**关键收益**：
- 显存利用率从 ~60% 提升到 > 95%（消除碎片）
- 支持更大并发（同等显存下可服务更多请求）
- 序列间可共享相同的 KV Cache 页（如 system prompt）

**注意**：PagedAttention 是 vLLM 的核心技术，通过 `flash_attn_with_kvcache` 的 `block_table` 参数在 Flash Attention 中原生支持。

#### FlashDecoding / Split-KV

Decode 阶段（逐 token 生成）中，Q 只有 1 个 token 但 KV 可以非常长。标准 Flash Attention 的并行度不足：

- **标准 FA decode**：按 batch 和 head 并行 → batch=1 时 GPU 利用率低
- **FlashDecoding**：额外按 KV 序列长度维度切分并行（Split-KV），多个 thread block 处理同一个 head 的不同 KV 区间，最后 reduce

```python
# flash_attn_with_kvcache 自动启用 Split-KV（当序列足够长时）
output = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=seq_lens,
    causal=True,
    # num_splits 自动选择，或可手动设置
)
```

**适用场景**：长序列 decode（seq_len > 2048），尤其是小 batch（batch_size=1-4）。

#### GQA/MQA 对推理 KV Cache 的影响

GQA（Grouped Query Attention）和 MQA（Multi-Query Attention）通过减少 KV head 数显著降低 KV Cache 大小：

| Attention 类型 | KV head 数 | KV Cache 大小（相对 MHA） | 代表模型 |
|---------------|-----------|------------------------|---------|
| MHA | = Q head 数 | 1x | GPT-3, LLaMA-1 |
| GQA | Q head 数 / group | 1/group（如 1/4, 1/8） | LLaMA-2/3, Qwen-2, Gemma |
| MQA | 1 | 1/num_heads | Falcon, PaLM |

Flash Attention 对 GQA/MQA 的支持：
```python
# GQA：q_heads=32, kv_heads=8 → 自动 broadcast
# 无需手动 repeat_kv
output = flash_attn_func(
    q,  # [batch, seq, 32, head_dim]
    k,  # [batch, seq, 8, head_dim]  — FA 自动处理 head 数不一致
    v,  # [batch, seq, 8, head_dim]
    causal=True,
)
```

#### Continuous Batching 中的 Flash Attention

在 vLLM/SGLang 等推理服务中，不同请求的序列长度不同。`flash_attn_varlen_func` 是处理变长 batch 的关键：

```python
from flash_attn import flash_attn_varlen_func

# 多个不同长度的序列拼接成一个大 tensor
# cu_seqlens 标记每个序列的起止位置（cumulative sequence lengths）
output = flash_attn_varlen_func(
    q_unpadded,          # [total_tokens, heads, head_dim]
    k_unpadded,          # [total_tokens, kv_heads, head_dim]
    v_unpadded,          # [total_tokens, kv_heads, head_dim]
    cu_seqlens_q=cu_seqlens_q,  # [batch+1]，如 [0, 128, 350, 512]
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_q_len,
    max_seqlen_k=max_k_len,
    causal=True,
)
```

**优势**：零 padding 浪费。在 Continuous Batching 场景中，不同请求可以在任意时间点加入/退出 batch，通过动态更新 `cu_seqlens` 实现高效的迭代级调度。

### 附录：Flash Attention vs 原生 Attention 性能参考

对于序列长度 2048、head_dim=128 的典型 LLM 场景：

| 指标 | 原生 Attention | Flash Attention 2 |
|------|---------------|-------------------|
| 前向速度 | 1x | ~2-4x |
| 反向速度 | 1x | ~2-3x |
| 显存占用 | O(N²) | O(N)（不显式存储 attention 矩阵） |
| 序列长度 8K | OOM 风险高 | 正常运行 |
| 序列长度 32K+ | 基本不可行 | 可行（配合滑动窗口更优） |

### 附录：与其他工具的配合

| 工具 | 配合方式 |
|------|---------|
| **Liger Kernel** | FA 处理 attention，Liger 处理 RMSNorm/RoPE/SwiGLU/CrossEntropy，两者互补不冲突 |
| **torch.compile** | FA2 支持 `torch.compile`，可获得额外的图级优化 |
| **FSDP / DeepSpeed** | FA 与分布式训练框架完全兼容 |
| **混合精度 (AMP)** | FA 原生 FP16/BF16，与 AMP 自然配合 |
| **Activation Checkpointing** | FA 已自带高效的反向重计算，通常不需要对 attention 层额外做 checkpointing |
