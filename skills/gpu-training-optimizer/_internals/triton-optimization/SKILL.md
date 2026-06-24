---
name: triton-optimization
description: Triton 算子性能优化与热点算子融合加速
---

# Skill: Triton 算子性能优化

## 描述
指导用户使用 OpenAI Triton 编写高性能自定义 GPU 算子，替换训练/推理中的热点算子瓶颈。涵盖 Triton 编程模型、常见融合算子模板（Fused Softmax、LayerNorm/RMSNorm、SwiGLU/GeGLU、RoPE、CrossEntropy、Fused Linear+Loss）、auto-tuning 策略、内存优化技巧、PyTorch 集成、生态工具（Liger Kernel/Unsloth/FlagGems）选型以及性能调试方法论。

## 触发条件
当以下任一条件满足时触发：
1. **Profiling 发现热点算子瓶颈**：PyTorch Profiler 或 Nsight Systems 显示某个��子（如 LayerNorm、Softmax、Activation、RoPE、CrossEntropy 等）占比显著（>10% 总耗时），且无现成优化库可用
2. **Kernel Launch Overhead 高**：大量小 kernel 导致 GPU 利用率低，需要通过算子融合减少 kernel 数量
3. **显存瓶颈由中间结果引起**：多个相邻算子的中间 tensor 占用大量显存，fusion 可消除中间分配
4. **用户显式要求 Triton 优化**：用户指定使用 Triton 编写自定义 kernel
5. **现有 Triton 算子需要调优**：项目中已有 Triton kernel 但性能不佳

## 执行指令

你是 Triton GPU 算子优化专家。根据 Profiling 数据和项目特点，判断是否需要编写自定义 Triton kernel，并提供高性能实现方案。

---

### 第一步：判断是否需要 Triton 自定义算子

在编写 Triton kernel 前，**先评估是否有更简单的替代方案**。Triton 自定义算子是代码级优化的最后手段。

#### 1.1 决策树

```
热点算子被 Profiling 识别
├── 该算子是否有成熟库实现？
│   ├── Attention 热点 → 使用 Flash Attention（参考 /flash-attention skill）
│   ├── RMSNorm/RoPE/SwiGLU/CrossEntropy 热点 → 使用 Liger Kernel（一行代码）
│   ├── LayerNorm/Dropout/Softmax 热点 → 检查 torch.compile 是否已自动融合
│   └── 无现成库
│       ├── torch.compile 能否自动融合？
│       │   ├── 能 → 启用 torch.compile（零代码修改）
│       │   └── 不能（动态 shape / 自定义逻辑 / compile 性能不佳）
│       │       └── ✅ 编写 Triton 自定义算子
│       └── 是多个 element-wise 操作的组合？
│           ├── 是 → ✅ Triton 融合 kernel 收益显著
│           └── 否（纯 GEMM 等）→ cuBLAS/CUTLASS 通常更优，Triton 不一定更快
└── 是否为多个相邻算子的组合？（如 Linear + Activation + Norm）
    └── ✅ Triton 融合 kernel 可消除中间 tensor 显存分配
```

#### 1.2 Triton 适用场景 vs 不适用场景

| 适用场景（Triton 优势大） | 不适用场景（其他方案更优） |
|--------------------------|-------------------------|
| Element-wise 融合（多个逐元素操作合并为一个 kernel） | 纯矩阵乘法（cuBLAS/CUTLASS 更优） |
| Reduction 融合（Softmax、LayerNorm、RMSNorm 等） | 已有高度优化的 CUDA 库（cuDNN、NCCL） |
| Fused Linear + Activation + Norm | 非计算密集型操作（如纯内存拷贝） |
| Fused Linear + Loss（消除大 logits tensor） | 简单操作 torch.compile 已能自动融合 |
| 自定义 Attention 变体（标准 FA 不覆盖的 pattern） | I/O 密集型瓶颈（应优化数据管道而非算子） |
| Quantized GEMM（INT8/FP8 矩阵乘） | 通信瓶颈（应优化 NCCL 而非计算） |

#### 1.3 性能收益预估

| 优化类型 | 典型加速比 | 典型显存节省 | 示例 |
|---------|-----------|-------------|------|
| 多个 element-wise 融合 | 2-5x | 消除中间 tensor | Activation + Dropout 融合 |
| Reduction kernel 优化 | 1.5-3x | 中间结果不写回 HBM | Fused Softmax、Fused LayerNorm |
| Fused Linear + Loss | 2-4x | **50-80%**（消除大 logits tensor） | Fused Linear CrossEntropy |
| Fused Norm + Activation | 2-3x | 减少 kernel launch | Fused RMSNorm + SwiGLU |
| 自定义量化 GEMM | 2-4x（vs FP16） | 模型大小减半+ | INT8/FP8 Matmul |

---

### 第二步：Triton 编程基础

#### 2.1 核心概念

Triton 是一个**基于 block（tile）的 GPU 编程模型**，使用 Python 语法编写 GPU kernel，由编译器自动处理 shared memory、warp 调度等底层细节。

| 概念 | 说明 |
|------|------|
| **Program** | 一个 Triton kernel 实例，对应 CUDA 中的一个 block |
| **Block（Tile）** | kernel 一次处理的数据块大小（如 BLOCK_SIZE=1024 个元素） |
| **program_id** | 类似 CUDA blockIdx，标识当前 block 在 grid 中的位置 |
| **tl.load / tl.store** | 从 HBM 读写数据，支持 mask 处理边界 |
| **tl.arange** | 生成 block 内的偏移量序列 |
| **@triton.jit** | 标记函数为 Triton kernel |
| **@triton.autotune** | 自动搜索最优 block/warp/pipeline 配置 |
| **tl.constexpr** | 编译期常量，用于 block size 等参数 |

#### 2.2 基础模板：Vector Add

```python
import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 当前 block 处理的数据范围
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # mask 处理越界（最后一个 block 可能不满）
    mask = offsets < n_elements

    # 从 HBM 读取 → 计算 → 写回 HBM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    output = torch.empty_like(x)
    n_elements = output.numel()
    # grid lambda：根据 meta 参数（如 BLOCK_SIZE）动态计算 grid
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output
```

**关键模式**：
1. `tl.program_id(axis=0)` 获取 block ID
2. `tl.arange(0, BLOCK_SIZE)` 生成 block 内偏移
3. `mask = offsets < n_elements` 处理边界
4. `tl.load(..., mask=mask)` 带 mask 的安全加载
5. `kernel[grid](...)` 启动 kernel，grid 可以是 lambda

#### 2.3 内存访问优化要点

| 要点 | 说明 | 实践建议 |
|------|------|---------|
| **合并访问 (Coalescing)** | 同一 warp 内的线程访问连续内存地址 | 确保最内层维度是连续的，使用行主序 |
| **BLOCK_SIZE 对齐** | 块大小应为 2 的幂且 >= 64 | 常用值：64, 128, 256, 512, 1024 |
| **减少 HBM 读写** | 算子融合的核心价值在于减少中间结果写回 HBM | 多个操作合并在同一个 kernel 内完成 |
| **向量化加载** | Triton 编译器自动向量化 tl.load | 保持数据对齐有助于生成更优的向量化代码 |
| **Eviction Policy** | 控制 L2 cache 驱逐策略 | 对不会复用的数据用 `eviction_policy='evict_first'` |

---

### 第三步：常见融合算子模板

以下是训练场景中最常见的 Triton 融合算子模板，按优化收益排序。

#### 3.1 Fused Softmax（融合 Softmax）

**适用场景**：Attention 中 softmax 占比高，且无法使用 Flash Attention（如自定义 attention 变体）。

**核心优化**：将 max reduction、exp、sum reduction、normalize 四步融合为一个 kernel，中间结果仅保留在 SRAM（寄存器/shared memory）中，避免多次 HBM 读写。

```python
@triton.jit
def fused_softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_cols

    # 1. Load entire row to SRAM
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
    # 2. Fused: max → subtract → exp → sum → divide
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)  # 数值稳定
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    # 3. Write back
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)
```

**性能要点**：
- 整行数据在 SRAM 中完成全部计算，只读写 HBM 各一次（原生实现需 4-5 次）
- 适合 `n_cols` 能放入一个 block 的场景（如 head_dim <= 8192）
- 超大 `n_cols` 需要分块 + 在线 softmax 算法（参考 Flash Attention）

#### 3.2 Fused RMSNorm（融合 RMS 归一化）

**适用场景**：LLaMA、Qwen、Mistral 等主流 LLM 使用 RMSNorm 替代 LayerNorm。

```python
@triton.jit
def fused_rms_norm_kernel(
    output_ptr, input_ptr, weight_ptr,
    stride, n_cols, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start = input_ptr + row_idx * stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load row and weight
    row = tl.load(row_start + col_offsets, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # RMSNorm: x * weight / sqrt(mean(x^2) + eps)
    row_sq = row * row
    mean_sq = tl.sum(row_sq, axis=0) / n_cols
    rrms = 1.0 / tl.sqrt(mean_sq + eps)
    normed = row * rrms * weight

    # Write back (can cast to fp16/bf16 here)
    output_row_start = output_ptr + row_idx * stride
    tl.store(output_row_start + col_offsets, normed, mask=mask)
```

**融合扩展**：RMSNorm + Residual Add（前一层的 residual 与 norm 融合，减少一次 HBM 读写）

```python
# Fused: output = RMSNorm(x + residual) * weight
# 同时写回 normed output 和 x + residual（给后续 residual 用）
row = tl.load(input_ptr + offsets, mask=mask).to(tl.float32)
residual = tl.load(residual_ptr + offsets, mask=mask).to(tl.float32)
x = row + residual
# ... RMSNorm 计算 ...
tl.store(output_ptr + offsets, normed, mask=mask)
tl.store(residual_ptr + offsets, x, mask=mask)  # in-place 更新 residual
```

#### 3.3 Fused SwiGLU / GeGLU（融合门控激活）

**适用场景**：LLaMA/Qwen 等使用 SwiGLU (SiLU Gate)、Gemma 使用 GeGLU (GELU Gate)。

```python
@triton.jit
def fused_swiglu_kernel(
    output_ptr, gate_ptr, up_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    gate = tl.load(gate_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(up_ptr + offsets, mask=mask).to(tl.float32)

    # SwiGLU: silu(gate) * up = gate * sigmoid(gate) * up
    sigmoid_gate = tl.sigmoid(gate)
    result = gate * sigmoid_gate * up

    tl.store(output_ptr + offsets, result, mask=mask)
```

**进阶融合**：SwiGLU + RMSNorm（在 MLP block 内将 norm 和 activation 融合）

#### 3.4 Fused RoPE（融合旋转位置编码）

**适用场景**：几乎所有主流 LLM 都使用 RoPE 位置编码。

```python
@triton.jit
def fused_rope_kernel(
    q_ptr, k_ptr,              # Q, K tensors
    cos_ptr, sin_ptr,          # 预计算的 cos/sin 表
    output_q_ptr, output_k_ptr,
    seq_len, head_dim,
    stride_seq, stride_head,
    BLOCK_SIZE: tl.constexpr,
):
    pid_seq = tl.program_id(0)     # 序列位置
    pid_head = tl.program_id(1)    # head 索引

    half_dim = head_dim // 2
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_dim

    # 加载 Q 的前半和后半
    base = pid_seq * stride_seq + pid_head * stride_head
    q_first = tl.load(q_ptr + base + offsets, mask=mask).to(tl.float32)
    q_second = tl.load(q_ptr + base + half_dim + offsets, mask=mask).to(tl.float32)

    # 加载 cos/sin
    cos = tl.load(cos_ptr + pid_seq * half_dim + offsets, mask=mask).to(tl.float32)
    sin = tl.load(sin_ptr + pid_seq * half_dim + offsets, mask=mask).to(tl.float32)

    # RoPE: [x1, x2] → [x1*cos - x2*sin, x1*sin + x2*cos]
    out_first = q_first * cos - q_second * sin
    out_second = q_first * sin + q_second * cos

    tl.store(output_q_ptr + base + offsets, out_first, mask=mask)
    tl.store(output_q_ptr + base + half_dim + offsets, out_second, mask=mask)

    # 对 K 做同样的操作（可融合在同一 kernel 中）
    # ...
```

**优化要点**：Q 和 K 的 RoPE 融合在同一 kernel 中，减少一半的 kernel launch 开销。

#### 3.5 Fused CrossEntropy（融合交叉熵损失）

**适用场景**：LLM 训练中 CrossEntropy 需要处理 `(batch*seq, vocab_size)` 的大 logits tensor，vocab_size 通常 32K-128K+，显存占用巨大。

**核心优化**：Online Softmax + Log + NLL 融合，避免实例化完整的 softmax 中间结果。

```python
@triton.jit
def fused_cross_entropy_kernel(
    loss_ptr, logits_ptr, labels_ptr,
    n_cols,   # vocab_size
    logits_row_stride,
    ignore_index,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    label = tl.load(labels_ptr + row_idx)

    # Skip ignored labels
    if label == ignore_index:
        tl.store(loss_ptr + row_idx, 0.0)
        return

    logits_start = logits_ptr + row_idx * logits_row_stride

    # Online softmax: 分块计算 max 和 sum(exp)
    m = -float('inf')  # running max
    d = 0.0            # running sum(exp)
    target_logit = 0.0

    for block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols
        logits = tl.load(logits_start + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)

        # 提取 target logit
        is_target = col_offsets == label
        target_logit += tl.sum(tl.where(is_target, logits, 0.0))

        # Online max and sum update
        block_max = tl.max(logits, axis=0)
        new_m = tl.maximum(m, block_max)
        d = d * tl.exp(m - new_m) + tl.sum(tl.exp(logits - new_m), axis=0)
        m = new_m

    # Loss = log(sum(exp)) + max - target_logit
    loss = tl.log(d) + m - target_logit
    tl.store(loss_ptr + row_idx, loss)
```

**性能收益**：
- 不再实例化 `(batch*seq, vocab_size)` 的 softmax 中间 tensor → 显存大幅降低
- 对于 vocab_size=128K，单个 logits tensor 在 BF16 下占 256KB/token，在长序列 batch 中轻松消耗数 GB
- Liger Kernel 的 `LigerCrossEntropyLoss` 就是基于此原理

#### 3.6 Fused Linear + CrossEntropy（最高价值融合）

**适用场景**：LLM 的最后一层 `lm_head (linear)` + `cross_entropy_loss`。这是**单一最高价值**的 Triton 融合优化。

**核心原理**：将 `logits = linear(hidden, weight)` 和 `loss = cross_entropy(logits, labels)` 融合，按 vocab_size 分块计算，每个 block 计算一部分 logits 并立即用于 online softmax，**永远不实例化完整的 logits tensor**。

```python
# 伪代码框架（完整实现参考 Liger Kernel 的 LigerFusedLinearCrossEntropyLoss）
@triton.jit
def fused_linear_cross_entropy_kernel(
    loss_ptr, hidden_ptr, weight_ptr, labels_ptr,
    hidden_dim, vocab_size,
    BLOCK_V: tl.constexpr,  # vocab 分块大小
    BLOCK_H: tl.constexpr,  # hidden 分块大小
):
    row_idx = tl.program_id(0)
    label = tl.load(labels_ptr + row_idx)

    m = -float('inf')
    d = 0.0
    target_logit = 0.0

    # 分块遍历 vocab 维度
    for v_start in range(0, vocab_size, BLOCK_V):
        # 计算 hidden[row] @ weight[v_start:v_start+BLOCK_V].T
        # 得到部分 logits（仅 BLOCK_V 个值，不需要完整 vocab_size）
        partial_logits = tl.zeros([BLOCK_V], dtype=tl.float32)
        for h_start in range(0, hidden_dim, BLOCK_H):
            # 分块矩阵乘（在 SRAM 中）
            h = tl.load(hidden_ptr + row_idx * hidden_dim + h_start + tl.arange(0, BLOCK_H))
            w = tl.load(weight_ptr + (v_start + tl.arange(0, BLOCK_V)[:, None]) * hidden_dim
                       + h_start + tl.arange(0, BLOCK_H)[None, :])
            partial_logits += tl.sum(w * h[None, :], axis=1)

        # Online softmax update（与 3.5 相同）
        # ...

    loss = tl.log(d) + m - target_logit
    tl.store(loss_ptr + row_idx, loss)
```

**显存收益分析**：

| 模型 | vocab_size | seq_len | batch | 标准方式 logits 显存 | Fused 方式 logits 显存 |
|------|-----------|---------|-------|---------------------|----------------------|
| LLaMA-3-8B | 128,256 | 2048 | 4 | **~2 GB** (BF16) | **0 GB**（永不实例化） |
| Qwen-2-7B | 151,936 | 4096 | 2 | **~2.3 GB** | **0 GB** |
| LLaMA-3-70B | 128,256 | 8192 | 1 | **~2 GB** | **0 GB** |

> **推荐**：在大多数场景下，直接使用 Liger Kernel 的 `LigerFusedLinearCrossEntropyLoss` 而非手写。仅当 Liger 不支持特定 loss 变体时才需自行实现。

---

### 第四步：Auto-Tuning 策略

Triton 的 `@triton.autotune` 装饰器可自动搜索最优的 kernel 配置参数。

#### 4.1 基本用法

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],  # 触发重新搜索的参数（shape 变化时重新 tune）
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # ... kernel body
    pass
```

#### 4.2 配置参数含义

| 参数 | 说明 | 调优建议 |
|------|------|---------|
| `BLOCK_*` | Tile 大小 | 必须是 2 的幂；增大 → 更多数据复用但寄存器压力增大 |
| `num_warps` | 每个 block 使用的 warp 数 | 2/4/8；计算密集型用 8，内存密集型用 4 |
| `num_stages` | 软件流水线阶段数 | 2-5；增大可隐藏内存延迟但增加寄存器使用 |
| `key` | 触发重新 tune 的参数 | 通常设为影响 kernel 行为的 shape 参数 |

#### 4.3 常见配置搜索空间

**Element-wise / Reduction kernel：**
```python
configs=[
    triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
    triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
]
```

**矩阵乘（GEMM）kernel：**
```python
configs=[
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
]
```

#### 4.4 L2 Cache 优化（GEMM 场景）

矩阵乘中，Tile 的遍历顺序影响 L2 cache 命中率。**Grouped ordering** 比简单的行优先遍历显著提升缓存命中：

```python
# 在 GEMM kernel 中，使用 grouped ordering 提升 L2 cache 利用率
pid = tl.program_id(axis=0)
num_pid_m = tl.cdiv(M, BLOCK_M)
num_pid_n = tl.cdiv(N, BLOCK_N)
num_pid_in_group = GROUP_M * num_pid_n
group_id = pid // num_pid_in_group
first_pid_m = group_id * GROUP_M
group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

**原理**：将输出矩阵 C 的 tile 按 `GROUP_M` 行为一组遍历，同组内的 tile 共享 A 矩阵的相同行，提高 L2 cache 中 A 数据块的复用率。

---

### 第五步：PyTorch 集成

#### 5.1 Wrapper 函数模式（最常用）

```python
import torch
import triton
import triton.language as tl

class FusedRMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6):
        # 保存反向传播需要的数据
        output = torch.empty_like(x)
        n_cols = x.shape[-1]
        n_rows = x.numel() // n_cols

        # 选择 BLOCK_SIZE（需 >= n_cols 且为 2 的幂）
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        fused_rms_norm_kernel[(n_rows,)](
            output, x, weight,
            x.stride(-2), n_cols, eps,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        # 反向传播 kernel（同样用 Triton 实现）
        grad_x, grad_weight = fused_rms_norm_backward(
            grad_output, x, weight, ctx.eps, ctx.BLOCK_SIZE
        )
        return grad_x, grad_weight, None

# 用户接口
def fused_rms_norm(x, weight, eps=1e-6):
    return FusedRMSNorm.apply(x, weight, eps)
```

#### 5.2 替换 PyTorch 原生模块

```python
import torch.nn as nn

class TritonRMSNorm(nn.Module):
    """Drop-in 替换 PyTorch RMSNorm"""
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        return fused_rms_norm(x, self.weight, self.eps)

# 在模型中替换
def patch_model_with_triton_rmsnorm(model):
    """将模型中所有 RMSNorm 替换为 Triton 版本"""
    for name, module in model.named_modules():
        if isinstance(module, (nn.RMSNorm,)) or type(module).__name__ == 'RMSNorm':
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            triton_norm = TritonRMSNorm(module.weight.shape[0], getattr(module, 'eps', 1e-6))
            triton_norm.weight = module.weight  # 共享权重
            setattr(parent, child_name, triton_norm)
    return model
```

#### 5.3 与 torch.compile 配合

Triton kernel 天然与 `torch.compile` 兼容（Triton 本身就是 torch.compile/Inductor 的后端）：

```python
model = MyModel()
model = patch_model_with_triton_rmsnorm(model)
model = torch.compile(model)  # Triton 自定义 kernel 不会被 compile 重写
```

> **注意**：`torch.compile` 不会重新编译已有的 `@triton.jit` kernel，两者互补。compile 优化模型其他部分的图级融合，Triton 优化你手写的热点 kernel。

---

### 第六步：生态工具选型

在手写 Triton kernel 前，**优先考虑成熟的 Triton 生态工具**：

| 工具 | 覆盖范围 | 接入成本 | 推荐场景 |
|------|---------|---------|---------|
| **[Liger Kernel](https://github.com/linkedin/Liger-Kernel)** | RMSNorm, RoPE, SwiGLU, GeGLU, CrossEntropy, FusedLinearCrossEntropy, FusedLinear+DPO/ORPO/SimPO/CPO/KTO Loss, KLDiv, JSD | **一行代码** `apply_liger_kernel_to_llama()` | **首选**。HuggingFace 模型训练，吞吐 +20%，显存 -60% |
| **[Unsloth](https://github.com/unslothai/unsloth)** | 全套 LLM 训练 Triton kernel（Attention, MLP, Norm, Embedding, Loss） | `tuner_backend: unsloth`（框架集成） | LoRA 微调场景，速度 +70%，显存 -50% |
| **[FlagGems](https://github.com/FlagOpen/FlagGems)** | 300+ 通用 Triton 算子（替换 PyTorch ATen 算子） | `import flag_gems; flag_gems.enable()` | 通用算子加速，支持 AMD ROCm |
| **[Triton-Puzzles](https://github.com/srush/Triton-Puzzles)** | 学习资源 | — | Triton 编程学习 |

#### 6.1 Liger Kernel 快速接入

```python
# 方式 1：自动 Monkey-Patch（推荐）
from liger_kernel.transformers import AutoLigerKernelForCausalLM
model = AutoLigerKernelForCausalLM.from_pretrained("meta-llama/Llama-3-8B", ...)

# 方式 2：按模型 Patch
from liger_kernel.transformers import apply_liger_kernel_to_llama
apply_liger_kernel_to_llama()  # 必须在 model = AutoModelForCausalLM.from_pretrained() 之前调用

# 方式 3：HuggingFace Trainer 一行开启
training_args = TrainingArguments(use_liger_kernel=True, ...)

# 方式 4：单独使用 Liger 的 Triton 算子
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
from liger_kernel.ops.rope import LigerRopeFunction
from liger_kernel.ops.swiglu import LigerSiLUMulFunction
from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
```

**Liger Kernel 支持的模型**：LLaMA, Qwen2, Mistral, Gemma, Phi3, GLM, InternVL, Mixtral, Mllama, DeepSeek 等 20+ 主流模型。

#### 6.2 FlagGems 通用算子加速

```python
import flag_gems
flag_gems.enable()  # 自动替换 PyTorch 算子为 Triton 实现

# 之后正常写 PyTorch 代码，底层自动使用 Triton kernel
output = torch.softmax(x, dim=-1)  # 实际调用 FlagGems 的 Triton softmax
output = torch.layer_norm(x, normalized_shape)  # Triton layer_norm
```

#### 6.3 选型决策

```
需要优化的算子是什么？
├── RMSNorm/RoPE/SwiGLU/CrossEntropy/FusedLinear+Loss
│   └── 使用 Liger Kernel（一行代码，生产验证）
├── LoRA 微调整体加速
│   └── 使用 Unsloth（LlamaFactory/ms-swift 集成）
├── 通用 PyTorch 算子加速（softmax/layernorm/gelu/dropout 等）
│   └── 使用 FlagGems
├── 以上工具不覆盖的自定义算子
│   └── 手写 Triton kernel（参考本 skill 第三步模板）
└── 以上工具不覆盖 + 性能要求极致 + 有 CUDA 经验
    └── 考虑 CUDA C++ / CUTLASS（超出本 skill 范围）
```

---

### 第七步：性能调试与验证

#### 7.1 Benchmarking

```python
import triton

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(10, 18)],  # 1K to 128K
        line_arg='provider',
        line_vals=['triton', 'torch'],
        line_names=['Triton', 'PyTorch'],
        styles=[('blue', '-'), ('green', '-')],
        ylabel='GB/s',
        plot_name='rmsnorm-performance',
        args={'M': 4096},
    )
)
def benchmark(M, N, provider):
    x = torch.randn(M, N, device='cuda', dtype=torch.float16)
    weight = torch.randn(N, device='cuda', dtype=torch.float16)

    if provider == 'triton':
        ms = triton.testing.do_bench(lambda: fused_rms_norm(x, weight))
    else:
        rms_norm = torch.nn.RMSNorm(N).cuda().half()
        ms = triton.testing.do_bench(lambda: rms_norm(x))

    gbps = lambda ms: 2 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
    return gbps(ms)

benchmark.run(print_data=True, save_path='./bench/')
```

#### 7.2 正确性验证

```python
def verify_triton_kernel(triton_fn, torch_fn, *args, atol=1e-2, rtol=1e-2):
    """验证 Triton kernel 与 PyTorch 参考实现的数值一致性"""
    # 使用相同输入
    triton_out = triton_fn(*args)
    torch_out = torch_fn(*args)

    # 数值对比
    if torch.allclose(triton_out, torch_out, atol=atol, rtol=rtol):
        print(f"PASS: max diff = {(triton_out - torch_out).abs().max().item():.6f}")
    else:
        diff = (triton_out - torch_out).abs()
        print(f"FAIL: max diff = {diff.max().item():.6f}, "
              f"mean diff = {diff.mean().item():.6f}")
        # 定位差异位置
        idx = diff.argmax()
        print(f"  Worst element: triton={triton_out.flatten()[idx]:.6f}, "
              f"torch={torch_out.flatten()[idx]:.6f}")

    # 梯度对比（训练场景必须验证）
    # ... 类似对比反向传播梯度
```

**验证 Checklist**：

| # | 检查项 | 方法 |
|---|--------|------|
| 1 | 前向数值精度 | `torch.allclose(triton_out, torch_out, atol=1e-2, rtol=1e-2)` |
| 2 | 反向梯度精度 | 对比 `grad_input`, `grad_weight` |
| 3 | 边界条件 | 测试 `n_elements` 不被 `BLOCK_SIZE` 整除的情况 |
| 4 | 数据类型 | 分别测试 FP16、BF16、FP32 |
| 5 | 大尺寸输入 | 测试实际训练中的 shape（如 seq_len=8192, hidden=4096） |
| 6 | 训练稳定性 | 在完整训练中运行 100+ step 确认 loss 曲线正常 |

#### 7.3 常见性能问题排查

| 问题 | 可能原因 | 排查方法 | 解决方案 |
|------|---------|---------|---------|
| Triton kernel 比 PyTorch 慢 | BLOCK_SIZE 不合理 | `triton.autotune` 搜索 | 调整 config 搜索空间 |
| 显存未减少 | 中间 tensor 仍然被分配 | `torch.cuda.memory_snapshot()` | 检查 wrapper 是否保存了不必要的 tensor |
| 编译时间过长 | 首次 JIT 编译 | 正常现象（后续有缓存） | `TRITON_CACHE_DIR` 持久化缓存 |
| 精度下降 | FP16 累加精度不足 | 对比 FP32 参考实现 | reduction 使用 `.to(tl.float32)` 累加 |
| 无法 autotune | config 导致 OOM | 减小 BLOCK_SIZE 搜索上限 | 移除过大的 config |
| kernel launch overhead | 输入太小不值得 kernel | 添加 fallback 逻辑 | 小输入走 PyTorch，大输入走 Triton |

---

### 第八步：实战整合流程

当 `/gpu-training-optimizer` 在 Step 6（代码级优化）中决定使用 Triton 自定义算子时，按以下流程执行：

#### 8.1 从 Profiling 到 Triton Kernel

```
1. PyTorch Profiler 识别 Top-K 热点算子（如 aten::layer_norm 占 15%）
2. 检查是否有成熟替代（Liger Kernel / FlagGems / torch.compile）
3. 若无替代 → 确认算子类型：
   ├── Element-wise → 参考模板 3.3 (SwiGLU) 或 3.4 (RoPE)
   ├── Reduction → 参考模板 3.1 (Softmax) 或 3.2 (RMSNorm)
   ├── Linear + Something → 参考模板 3.6 (Fused Linear + Loss)
   └── 其他 → 基于 2.2 基础模板开发
4. 编写 Triton kernel + torch.autograd.Function wrapper
5. 数值验证（第七步 7.2）
6. 性能 benchmark（第七步 7.1）
7. 在 fork 项目中集成（Monkey-Patch 或 Module 替换）
8. 完整训练验证（运行 100+ step，对比 loss 曲线）
```

#### 8.2 在 Fork 项目中集成

```python
# 文件: ${WORK_DIR}/project/triton_kernels/__init__.py
# 新建 triton_kernels 目录存放自定义 kernel

# 文件: ${WORK_DIR}/project/triton_kernels/fused_rmsnorm.py
# 存放 Triton kernel 实现

# 文件: ${WORK_DIR}/project/patch_model.py
# 集成脚本，在训练入口处调用
from triton_kernels.fused_rmsnorm import TritonRMSNorm, patch_model_with_triton_rmsnorm

# 在训练脚本的模型初始化之后、训练之前插入：
model = patch_model_with_triton_rmsnorm(model)
```

#### 8.3 修改记录格式

每个 Triton 优化都应在 `v{N}_optimization-changes.md` 中记录：

```markdown
### X. 自定义 Triton Fused RMSNorm

**对应瓶颈**: PyTorch Profiler 显示 `aten::_native_rms_norm` 占总耗时 12%

**修改文件**:
- 新增: `triton_kernels/fused_rmsnorm.py` — Triton kernel 实现
- 修改: `train.py:L85` — 添加 `patch_model_with_triton_rmsnorm(model)` 调用

**验证结果**:
- 数值精度: max diff = 1.2e-4 (FP16), PASS
- 性能: RMSNorm 算子耗时 3.2ms → 1.1ms (-66%)
- 显存: 峰值显存 42.1GB → 41.3GB (-0.8GB)

**收益**: 该算子占总耗时从 12% 降至 4.5%
```

---

### 附录：Triton API 速查

| API | 用途 | 常用场景 |
|-----|------|---------|
| `tl.load(ptr, mask, other)` | 从 HBM 加载数据 | 所有 kernel |
| `tl.store(ptr, value, mask)` | 写回 HBM | 所有 kernel |
| `tl.arange(start, end)` | 生成偏移序列 | block 内索引 |
| `tl.program_id(axis)` | 获取 block ID | grid 定位 |
| `tl.max(x, axis)` | 规约求最大值 | Softmax、Norm |
| `tl.sum(x, axis)` | 规约求和 | Norm、Loss |
| `tl.exp(x)` | 指数函数 | Softmax |
| `tl.log(x)` | 对数函数 | CrossEntropy |
| `tl.sigmoid(x)` | Sigmoid | SiLU/SwiGLU |
| `tl.where(cond, x, y)` | 条件选择 | Mask 操作 |
| `tl.dot(a, b)` | Tile 矩阵乘 | GEMM kernel |
| `tl.atomic_add(ptr, val)` | 原子加 | 梯度累加 |
| `tl.cdiv(a, b)` | 向上取整除 | Grid 计算 |
| `tl.constexpr` | 编译期常量 | Block size |
| `triton.next_power_of_2(n)` | 下一个 2 的幂 | BLOCK_SIZE 选择 |

### 附录：访存优化 (Memory Access Optimization)

GPU 性能优化中，访存模式的优化常常比计算优化收益更大。许多 kernel 慢的根因不是"算不动"，而是"访存烂"。

#### A. Coalesced Access（合并访存）

GPU 中同一 warp 内的线程应该访问**连续的内存地址**，这样硬件可以把多次访问合并为一次事务。

```python
@triton.jit
def good_access_kernel(x_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    # ✅ 好：连续线程访问连续地址
    offsets = pid * N + tl.arange(0, N)
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x)

@triton.jit
def bad_access_kernel(x_ptr, out_ptr, N: tl.constexpr, stride: tl.constexpr):
    pid = tl.program_id(0)
    # ❌ 差：跨步访问，每个线程隔 stride 个元素，无法合并
    offsets = pid * N + tl.arange(0, N) * stride
    x = tl.load(x_ptr + offsets)
    tl.store(out_ptr + offsets, x)
```

**Triton 中的实践**：
- `tl.arange` 天然产生连续偏移，直接用即可保证 coalesced
- 当处理多维张量时，确保最内层维度（stride=1）由 `tl.arange` 遍历
- 如果必须做转置或非连续访问，先 load 到 block 内再处理

#### B. 向量化加载

GPU 硬件支持单条指令加载 128 bit 数据（如 `float4` = 4 个 float32、`half8` = 8 个 float16）。Triton 编译器在 BLOCK_SIZE 为 2 的幂且访存连续时通常能自动向量化，但需要确保：

- **BLOCK_SIZE 足够大**：至少 128（让每个线程处理足够多的元素）
- **数据对齐**：指针地址是元素大小的倍数（通常 PyTorch tensor 自动满足）
- **连续访存**：如上述 coalesced access 所述

#### C. Shared Memory 与 Bank Conflict

Triton 对 shared memory 的使用是隐式的（编译器自动决定哪些数据放 shared memory），但理解 bank conflict 有助于选择更好的 BLOCK_SIZE：

- Shared memory 分为 32 个 bank（每 4 字节一个 bank）
- 同一 warp 内多个线程访问同一 bank 的不同地址 → **bank conflict** → 串行化
- **实践建议**：
  - BLOCK_SIZE 选 32 的倍数（而非 24、48 等非对齐值）
  - 在 GEMM kernel 中，如果 BLOCK_K 是 32 的倍数，编译器更容易避免 bank conflict
  - 如果 ncu 显示大量 bank conflict，尝试调整 BLOCK 维度或增加 padding

#### D. 软件流水线 (Software Pipelining / Double Buffering)

`num_stages` 参数控制 Triton 编译器的**软件流水线深度**——在计算当前 tile 的同时，提前异步加载下一个 tile：

```python
triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4)
#                                                               ^^^^^^^^^^^
# num_stages=4 表示最多同时有 4 个阶段的数据在 pipeline 中
# = 当前计算 tile_k + 预取 tile_{k+1} + 预取 tile_{k+2} + 预取 tile_{k+3}
```

| num_stages | 效果 | 适用场景 |
|------------|------|---------|
| 1 | 无流水线（load 完再 compute） | 极简 kernel、调试 |
| 2 | 双缓冲（经典 double buffering） | 大多数 memory-bound kernel |
| 3-5 | 深度流水线（隐藏更多延迟） | A100+，register 充裕时 |
| > 5 | 寄存器压力过大，通常适得其反 | 极少使用 |

**与 L2 Cache 的关系**：num_stages 增大意味着同时在 shared memory 中缓存更多 tile，这会增加 register 和 shared memory 使用量。需要在**延迟隐藏**和**occupancy**之间找平衡。

### 附录：Occupancy 与资源压力调优

Occupancy（占用率）指 SM 上实际驻留的 warp 数与理论最大 warp 数之比。它不是越高越好，但太低通常意味着性能问题。

#### Occupancy 的影响因素

| 资源 | 对 Occupancy 的影响 | Triton 中的控制方式 |
|------|-------------------|-------------------|
| **Register 使用量** | 每线程 register 越多 → 每 SM 可驻留的 block 越少 → occupancy 下降 | 减小 BLOCK_SIZE、减少临时变量、降低 num_stages |
| **Shared Memory 使用量** | 每 block shared memory 越大 → 同 SM 可驻留的 block 越少 | 减小 BLOCK 维度（尤其 GEMM 的 BLOCK_M/N/K） |
| **Block Size (num_warps)** | 每 block 线程越多 → 单 block 消耗更多资源 | 调整 num_warps（2/4/8） |

#### Register Spill 问题

当编译器分配的寄存器超过硬件上限，超出部分会 **spill 到 local memory**（实际走 global memory + cache），性能大幅下降：

- **检测**：用 ncu 看 `spill loads` 和 `spill stores` 指标，或 Triton 编译时的 register 使用报告
- **缓解策略**：
  - 减小 BLOCK_SIZE / BLOCK 维度
  - 减小 num_stages（减少 pipeline 缓冲区占用的寄存器）
  - 简化 kernel 内的临时计算（少 unroll、减少中间变量）
  - 将 reduction 路径拆分为多个 kernel 而非在单 kernel 中全部完成

#### 实用经验

| kernel 类型 | 推荐 num_warps | 推荐 BLOCK_SIZE | occupancy 目标 |
|------------|---------------|----------------|---------------|
| Element-wise | 4 | 1024-4096 | > 50% |
| Reduction (Norm/Softmax) | 4-8 | 与 hidden_dim 对齐 | > 30%（可接受低些，因为是计算密集） |
| GEMM | 4-8 | BLOCK_M/N: 64-256, BLOCK_K: 32-64 | > 40% |
| Fused Linear+Loss | 4-8 | BLOCK_M: 64-128, BLOCK_V: vocab 分块 | > 30% |

**Memory-bound kernel 对 occupancy 更敏感**：更多活跃 warp 可以在等待内存返回时切换到其他 warp 执行，从而隐藏内存延迟。Compute-bound kernel 对 occupancy 不太敏感，因为计算本身就能让 pipeline 保持忙碌。

### 附录：Warp 级优化注意事项

GPU 的真实执行单位是 **warp**（32 个线程）。虽然 Triton 抽象了大部分底层细节，但理解 warp 行为有助于写出更高效的 kernel。

#### Warp Divergence

同一 warp 内的线程必须执行相同指令。如果分支导致 warp 内线程走不同路径，这些路径会被**串行执行**：

```python
@triton.jit
def kernel(x_ptr, out_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * N + tl.arange(0, N)
    x = tl.load(x_ptr + offsets, mask=offsets < total_size, other=0.0)

    # ⚠️ tl.where 比 Python if/else 更友好于 warp
    # tl.where 本质上是 predication（两个分支都算，根据条件选结果），无 divergence
    result = tl.where(x > 0, x, x * 0.01)  # LeakyReLU

    tl.store(out_ptr + offsets, result)
```

**最佳实践**：
- 用 `tl.where` 替代分支——predication 模式下无 divergence
- 让同一 block 内的线程尽量走相同路径
- Mask 操作（`tl.load(..., mask=...)` ）本身不产生 divergence（Triton 编译器做了特殊处理）

#### Warp-level Primitives

Triton 通过 `tl.reduce` 和 `tl.scan` 暴露了底层的 warp shuffle 操作：

```python
# tl.sum 内部使用 warp shuffle 实现高效的 block-level reduction
# 比显式的 shared memory reduction 更快（避免 shared memory 读写）
row_sum = tl.sum(x, axis=-1)  # 编译为 warp shuffle + cross-warp reduce
```

- `tl.reduce`（sum/max/min）→ 编译为 warp shuffle + cross-warp sync
- `tl.scan`（cumsum/cumprod）→ 编译为 warp shuffle 的 prefix scan
- 这些 primitive 比手动写 shared memory reduction 通常更快

### 附录：Loop Unrolling 与 ILP

#### 循环展开的作用

Triton 编译器会自动对 `for` 循环做展开优化以提高**指令级并行（ILP）**。但展开程度可以通过代码结构影响：

```python
@triton.jit
def matmul_kernel(..., BLOCK_K: tl.constexpr):
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # 这个循环会被编译器展开——因为 BLOCK_K 是 constexpr
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + ...)
        b = tl.load(b_ptr + ...)
        acc = tl.dot(a, b, acc)
```

**展开的收益**：
- 减少循环控制开销（branch 指令）
- 让编译器看到更多独立指令，提高 ILP
- 使软件流水线（num_stages）有更多阶段可以 overlap

**展开的风险**：
- 增加寄存器使用量 → 可能导致 register spill
- 代码体积膨胀 → 指令 cache 压力增大

**实践建议**：
- `tl.constexpr` 声明的循环上界会触发完全展开——对短循环（< 32 次迭代）有效
- 对长循环（如 GEMM 的 K 维度），依赖 num_stages 做软件流水线而非完全展开
- 如果 ncu 显示 register spill，考虑增大 BLOCK_K 减少循环次数，或减小其他 BLOCK 维度释放寄存器空间

### 附录：与其他工具的配合

| 工具 | 配合方式 |
|------|---------|
| **Flash Attention** | FA 处理 Attention，Triton 处理 Norm/Activation/Loss/RoPE，两者互补 |
| **Liger Kernel** | Liger 本身基于 Triton，用 Liger 覆盖常见算子，仅 Liger 不覆盖的才手写 |
| **torch.compile** | compile 不重写 `@triton.jit` kernel，两者可同时使用 |
| **DeepSpeed / FSDP** | 分布式框架与 Triton 算子完全兼容（算子在单卡上执行，分布式管理权重分片） |
| **混合精度 (AMP)** | Triton kernel 内部精度自行控制，不受 AMP autocast 影响；建议 reduction 路径用 FP32 |
| **CUDA Graph** | Triton kernel 可被 CUDA Graph 捕获，但首次编译不能在 graph 内 |
| **PyTorch Profiler** | 自定义 Triton kernel 会以 `triton_*` 名称出现在 profiler 中 |
| **Nsight Systems** | Triton kernel 可在 timeline 中以 CUDA kernel 形式可见 |

### 附录：安装与环境要求

```bash
# Triton 随 PyTorch 2.0+ 自动安装
pip install torch  # Triton 已包含在内

# 单独安装（特定版本）
pip install triton

# 验证安装
python -c "import triton; print(triton.__version__)"
```

| 要求 | 最低版本 | 推荐版本 |
|------|---------|---------|
| PyTorch | 2.0 | 2.4+ |
| Triton | 2.0 | 3.0+（随 PyTorch 自动安装） |
| CUDA | 11.6 | 12.1+ |
| GPU | Volta (V100)+ | Ampere (A100)+ |
| Python | 3.8 | 3.10+ |

> **注意**：Triton 的 CUDA backend 支持 V100 及以上 GPU。与 Flash Attention（仅 Ampere+）不同，Triton 可在 V100 上运行，但 Ampere+ 的 tensor core 指令可能带来更好的性能。Triton 也支持 AMD ROCm 后端。
