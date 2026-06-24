---
name: pytorch-profiler
description: PyTorch Profiler 性能分析，提供算子级瓶颈定位
user-invocable: false
---

## 描述
使用 PyTorch Profiler 对训练或推理代码进行性能分析，定位算子耗时、显存瓶颈、GPU 利用率问题，并给出优化建议。

## 触发条件
当用户需要对 PyTorch 训练/推理代码进行性能 profiling、定位瓶颈、分析 GPU 利用率、排查显存问题时触发。或当用户要求查看某段代码的算子耗时、CUDA kernel 时间分布等。

## 执行指令

你是 PyTorch 性能分析专家。被调用时，根据用户场景选择合适的 profiling 方式，插入 profiler 代码，运行分析，并解读结果给出优化建议。

---

### 第一步：确定分析场景

根据用户需求判断属于哪种场景：

| 场景 | 特征 | 推荐方式 |
|------|------|----------|
| **快速算子耗时分析** | 想知道哪些算子最耗时 | 基础 `profile` context manager |
| **GPU kernel 分析** | 想看 CUDA kernel 粒度耗时 | 启用 `ProfilerActivity.CUDA` |
| **显存分析** | OOM 或想了解显存分布 | 启用 `profile_memory=True` |
| **长训练任务采样** | 训练循环中周期性采集 | 使用 `schedule` + `on_trace_ready` |
| **可视化 Trace** | 想在 Chrome/Perfetto 中查看时间线 | `export_chrome_trace` |
| **定位代码位置** | 想知道耗时算子来自哪行代码 | 启用 `with_stack=True` |

---

### 第二步：插入 Profiler 代码

根据场景提供对应代码模板：

#### 模板 A：快速算子耗时分析（最常用）

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

# 包裹要分析的代码段
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
) as prof:
    with record_function("training_step"):  # 用户自定义标签
        # ... 被分析的代码 ...
        output = model(inputs)
        loss = criterion(output, targets)
        loss.backward()

# 打印按 CUDA 总时间排序的 Top 20 算子
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# 也可以按 CPU 时间排序
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# 按算子输入 shape 分组（需 record_shapes=True）
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=20))
```

**关键参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `activities` | list | 采集范围：`ProfilerActivity.CPU`（CPU 算子）、`ProfilerActivity.CUDA`（GPU kernel）、`ProfilerActivity.XPU` |
| `record_shapes` | bool | 记录算子输入 tensor 的 shape，帮助定位不同 shape 的性能差异 |
| `with_flops` | bool | 估算矩阵乘法和 2D 卷积的 FLOPS |
| `with_stack` | bool | 记录调用栈（文件名+行号），开销较大，按需启用 |
| `with_modules` | bool | 记录模块层级（如 `ModelA.LayerB`），目前仅支持 TorchScript |
| `profile_memory` | bool | 跟踪 tensor 显存分配/释放 |

**输出表格关键列解读：**

| 列名 | 含义 |
|------|------|
| `Self CPU` | 算子本身 CPU 耗时（不含子算子） |
| `CPU total` | 算子 CPU 总耗时（含子算子） |
| `Self CUDA` | 算子本身 GPU kernel 耗时 |
| `CUDA total` | 算子 GPU 总耗时 |
| `CPU Mem` / `Self CPU Mem` | 内存分配量（需 `profile_memory=True`） |
| `# of Calls` | 调用次数 |

---

#### 模板 B：显存分析

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
) as prof:
    output = model(inputs)
    loss = criterion(output, targets)
    loss.backward()

# 按显存分配量排序
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=20))
# 或按总显存（含子算子）排序
# print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=20))

# 导出显存时间线（HTML 格式可直接在浏览器查看）
# prof.export_memory_timeline("memory_timeline.html", device="cuda:0")
```

注意：`export_memory_timeline` 已标记为 deprecated，新版推荐使用：
```python
torch.cuda.memory._record_memory_history()
# ... 运行代码 ...
torch.cuda.memory._export_memory_snapshot("memory_snapshot.pickle")
```

---

#### 模板 C：长训练任务周期性采样（推荐用于实际训练）

```python
from torch.profiler import profile, schedule, tensorboard_trace_handler

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(
        skip_first=10,   # 跳过前 10 步（避免初始化干扰）
        wait=5,          # 每个周期等待 5 步（profiler 不活跃）
        warmup=1,        # 预热 1 步（结果丢弃，消除 profiling 开销偏差）
        active=3,        # 实际采集 3 步
        repeat=2,        # 重复 2 个周期后停止（0 = 无限）
    ),
    on_trace_ready=tensorboard_trace_handler("./profiler_logs"),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    for step, batch in enumerate(train_loader):
        if step >= 10 + (5 + 1 + 3) * 2:  # 超过采集范围可提前退出
            break
        train_step(batch)
        prof.step()  # 必须：通知 profiler 当前 step 结束
```

**`schedule` 参数详解：**

| 参数 | 说明 |
|------|------|
| `skip_first` | 跳过最初 N 步（默认 0），用于跳过数据加载/模型初始化 |
| `wait` | 每个周期的空闲步数，profiler 不采集 |
| `warmup` | 预热步数，开始追踪但结果丢弃（消除 profiling 启动开销） |
| `active` | 实际采集步数 |
| `repeat` | 重复周期数（0 = 持续到结束） |

**执行时序**（以上述配置为例）：
```
步骤: 0-9(skip) | 10-14(wait) | 15(warmup) | 16-18(active→trace_ready) | 19-23(wait) | 24(warmup) | 25-27(active→trace_ready)
```

---

#### 模板 D：导出 Chrome Trace（用 Perfetto 可视化）

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    model(inputs)

prof.export_chrome_trace("trace.json")
# 用浏览器打开 https://ui.perfetto.dev/ 或 chrome://tracing 加载 trace.json
```

---

#### 模板 E：定位调用栈来源

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    with_stack=True,
) as prof:
    model(inputs)

# group_by_stack_n=5 表示按调用栈前 5 帧分组
print(prof.key_averages(group_by_stack_n=5).table(
    sort_by="self_cuda_time_total", row_limit=10
))

# 导出火焰图格式的栈信息
# prof.export_stacks("profiler_stacks.txt", metric="self_cuda_time_total")
```

---

#### 模板 F：使用 record_function 标记自定义代码段

```python
from torch.profiler import record_function

def train_step(batch):
    with record_function("data_preprocess"):
        inputs, labels = preprocess(batch)

    with record_function("forward"):
        outputs = model(inputs)

    with record_function("loss_compute"):
        loss = criterion(outputs, labels)

    with record_function("backward"):
        loss.backward()

    with record_function("optimizer_step"):
        optimizer.step()
        optimizer.zero_grad()
```

`record_function` 会在 profiler 输出和 trace 中显示为独立的标记区域，帮助快速定位各阶段耗时。

---

#### 模板 G：动态开关采集（高级用法）

```python
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
) as prof:
    code_to_profile_0()

    # 临时关闭 CUDA 采集
    prof.toggle_collection_dynamic(False, [ProfilerActivity.CUDA])
    code_to_skip()

    # 重新开启 CUDA 采集
    prof.toggle_collection_dynamic(True, [ProfilerActivity.CUDA])
    code_to_profile_1()
```

---

### 第三步：解读结果并给出优化建议

分析 profiler 输出时，关注以下模式：

#### 3.1 常见瓶颈及对策

| 瓶颈模式 | 诊断信号 | 优化建议 |
|----------|---------|---------|
| **DataLoader 过慢** | `enumerate(DataLoader)` 占比大，GPU idle 时间长 | 设置 `num_workers=4~8`，`pin_memory=True`，`persistent_workers=True` |
| **CPU-GPU 同步阻塞** | CPU 时间远大于 CUDA 时间，出现 `cudaStreamSynchronize` | 避免 `tensor.item()`、`print(cuda_tensor)`、`.cpu()` 调用 |
| **显存碎片化 / OOM** | 显存分配量大，`aten::empty` 频繁 | 启用 gradient checkpointing，减小 batch size，预分配显存 |
| **小 kernel 启动开销** | 大量小 CUDA kernel，每个耗时 < 10us | 使用 `torch.compile` 做算子融合，或启用 CUDA Graph |
| **未使用 Tensor Core** | matmul 算子 FLOPS 低于理论峰值 | 确认输入维度为 8 的倍数，启用 AMP (BF16/FP16) |
| **卷积未选最优算法** | conv 算子耗时异常 | 设置 `torch.backends.cudnn.benchmark = True` |
| **通信等待** | `nccl:all_reduce` 等通信算子占比大 | 启用通信计算重叠 (`overlap_comm`)，检查负载均衡 |
| **逐元素操作瓶颈** | 大量 `aten::add`、`aten::mul` 等 pointwise 操作 | 用 `torch.compile` 自动融合，或 channels_last 内存格式 |

#### 3.2 关键指标判断标准

| 指标 | 健康值 | 说明 |
|------|--------|------|
| GPU 利用率 | > 80% | < 50% 说明存在严重瓶颈 |
| Tensor Core 使用率 | > 50%（有 TC 的 GPU） | 0% 说明未启用混合精度或维度不对齐 |
| DataLoader 耗时占比 | < 10% | > 30% 说明数据管线是瓶颈 |
| 通信耗时占比（多卡） | < 20% | > 40% 需优化通信或检查负载均衡 |
| CPU-GPU 同步次数 | 尽量少 | 每个 step 不应超过 2-3 次必要同步 |

#### 3.3 瓶颈类型判定

根据 profiler 输出指标，将瓶颈归类到以下类型，以指导后续优化方向：

| 瓶颈类型 | 诊断条件 | 典型 profiler 特征 | 优化方向 |
|----------|---------|-------------------|---------|
| **Compute-bound** | GPU 利用率高、吞吐远低于理论峰值 | CUDA kernel 持续执行无 gap，但 Self CUDA 时间长；matmul/conv 等计算算子占比 > 70% | 混合精度（BF16/FP16）、Tensor Core 对齐（维度为 8 的倍数）、torch.compile |
| **Memory-bound** | 大量 element-wise 操作、显存分配频繁 | `aten::add`、`aten::mul`、`aten::copy_` 等 pointwise 算子占比高；`aten::empty`/`aten::to` 频繁出现 | 算子融合（Liger Kernel/torch.compile）、减少中间张量、Flash Attention |
| **Launch-bound** | 大量极短 kernel | 平均 Self CUDA < 10μs 的 kernel 数量 > 50%；CPU total 远大于 CUDA total | CUDA Graph、torch.compile、persistent kernel、减少 Python 调用 |
| **Data I/O-bound** | DataLoader 占比高 | `enumerate(DataLoader)` 或数据预处理算子在 CPU 时间排行靠前；GPU 周期性 idle | num_workers、pin_memory、prefetch_factor、数据格式优化 |
| **Communication-bound** | 通信算子占比高（多卡场景） | `nccl:all_reduce`、`nccl:all_gather` 占 CUDA 时间 > 30%；通信与计算未重叠 | overlap_comm、梯度压缩、ZeRO++、更高效并行策略 |
| **Sync-bound** | CPU-GPU 频繁同步 | `cudaStreamSynchronize`、`cudaDeviceSynchronize` 在 CPU 排行靠前；CPU wait 时间长 | 去掉 `.item()`、`print(cuda_tensor)`、延迟同步到 step 结束 |

---

### 第四步：输出分析报告

按以下格式向用户呈现结果：

```markdown
# PyTorch 性能分析报告

## 环境信息
- PyTorch 版本：X.Y.Z
- GPU：<型号> x <数量>
- 分析范围：<代码段描述>

## Top 10 耗时算子

| 排名 | 算子 | Self CUDA | CUDA Total | 调用次数 | 备注 |
|------|------|-----------|------------|---------|------|
| 1 | ... | ... | ... | ... | ... |

## GPU 利用率概览
- GPU 活跃时间占比：X%
- Tensor Core 使用情况：是/否
- 主要空闲原因：<DataLoader/CPU同步/通信等待>

## 显存分析（如启用）
- 峰值显存占用：X GB
- 主要显存消耗算子：...
- 是否存在碎片化：...

## 瓶颈诊断
1. **[严重程度]** <瓶颈描述> — 影响：<说明>
2. ...

## 优化建议（按优先级）
1. **[高]** <建议> — 预期收益：<说明>
   ```python
   # 代码示例
   ```
2. **[中]** <建议> — 预期收益：<说明>
3. **[低]** <建议> — 预期收益：<说明>
```

---

### 附录：常见 sort_by 排序键

| 排序键 | 用途 |
|--------|------|
| `cpu_time_total` | CPU 总耗时 |
| `self_cpu_time_total` | CPU 自身耗时（不含子算子） |
| `cuda_time_total` | CUDA 总耗时 |
| `self_cuda_time_total` | CUDA 自身耗时 |
| `cpu_memory_usage` | CPU 内存分配总量 |
| `self_cpu_memory_usage` | CPU 内存自身分配量 |
| `cuda_memory_usage` | GPU 显存分配总量 |
| `self_cuda_memory_usage` | GPU 显存自身分配量 |

### 附录：可视化工具

| 工具 | 用途 | 使用方式 |
|------|------|----------|
| **Perfetto** | 查看 trace 时间线（推荐） | 导出 `trace.json`，在 https://ui.perfetto.dev/ 打开 |
| **Chrome Tracing** | 查看 trace 时间线 | 导出 `trace.json`，在 `chrome://tracing` 打开 |
| **TensorBoard** | 综合视图（已 deprecated 但仍可用） | `pip install torch_tb_profiler` + `tensorboard --logdir=./log` |
| **HTA (Holistic Trace Analysis)** | 分布式训练 trace 分析 | TensorBoard Profiler 的替代品 |

### 附录：PyTorch 性能调优速查清单

以下是从 PyTorch 官方性能调优指南中提取的关键优化项，可结合 profiler 结果逐项检查：

| 优化项 | 代码/配置 | 适用场景 |
|--------|----------|---------|
| 异步数据加载 | `DataLoader(num_workers=4~8, pin_memory=True)` | 所有训练任务 |
| 推理/验证时关闭梯度 | `with torch.no_grad():` | 验证/推理 |
| Conv+BN 去掉 bias | `nn.Conv2d(..., bias=False)` | Conv 后接 BatchNorm |
| 高效梯度清零 | `optimizer.zero_grad(set_to_none=True)` | 所有训练任务 |
| torch.compile 算子融合 | `model = torch.compile(model)` | PyTorch 2.x |
| channels_last 内存格式 | `model.to(memory_format=torch.channels_last)` | CV 模型 |
| Gradient Checkpointing | `torch.utils.checkpoint.checkpoint(fn, ...)` | 显存受限 |
| 关闭调试 API | 禁用 `detect_anomaly`、`emit_nvtx` 等 | 正式训练 |
| Tensor Core 精度控制 | `torch.set_float32_matmul_precision('high')` | 有 Tensor Core 的 GPU |
| CUDA Graph | `torch.compile(model, mode="reduce-overhead")` | 固定形状输入 |
| cuDNN 自动调优 | `torch.backends.cudnn.benchmark = True` | 卷积网络 |
| 避免 CPU-GPU 同步 | 不用 `.item()`、`print(cuda_tensor)` | 训练循环内 |
| 直接在 GPU 创建 tensor | `torch.rand(size, device='cuda')` | 减少 H2D 拷贝 |
| 混合精度训练 | `torch.amp.autocast('cuda', dtype=torch.bfloat16)` | 所有 GPU 训练 |
| 变长输入预分配显存 | 先用最大长度跑一遍 forward+backward | NLP/语音模型 |
| DDP 梯度累积时跳过 AllReduce | `with model.no_sync():` | 梯度累积 N-1 步 |
| NUMA 绑定（CPU 训练） | `numactl --cpunodebind=N --membind=N python train.py` | 多 socket CPU |
| 高效内存分配器 | `LD_PRELOAD=jemalloc.so` 或 `tcmalloc.so` | CPU 训练 |
| Tensor Core 维度对齐 | 确保 hidden_dim/vocab_size/head_dim 为 8 的倍数 | 所有使用 matmul 的模型 |
| 避免不必要的 .contiguous() | 只在 kernel 要求连续内存时调用 | 减少无意义的显存拷贝 |
| 使用 inplace 操作 | `F.relu(x, inplace=True)`、`tensor.add_(...)` | 减少中间张量分配 |
| 预分配 buffer 复用 | `torch.empty(..., out=buffer)` | 重复运算场景 |
| Memory-bound 检测 | 看 pointwise 算子占比，DRAM 吞吐 vs 峰值带宽比 | 判断是否值得做算子融合 |
| H2D/计算 overlap | 多 CUDA stream + `non_blocking=True` + prefetch | 数据传输与计算并行 |

### 附录：何时需要 Nsight Compute (ncu)

PyTorch Profiler 擅长 **算子级分析**（哪个算子慢、调用了多少次），但无法深入 kernel 内部。当你通过 profiler 定位到具体的热点 kernel 后，需要用 **Nsight Compute (ncu)** 分析 kernel 内部性能：

| 需要 ncu 的场景 | ncu 能给你的信息 |
|----------------|-----------------|
| 热点 kernel 优化方向不明确 | Roofline 模型判断 compute-bound vs memory-bound |
| matmul 性能未达预期 | SM 利用率、Tensor Core 命中率、pipeline stall |
| 自定义 Triton/CUDA kernel 调优 | achieved occupancy、register 使用量、shared memory bank conflict |
| 怀疑访存效率低 | L2 hit rate、DRAM throughput、global load efficiency |
| warp 级性能问题 | warp stall reason 分布、branch efficiency |

**使用流程**：PyTorch Profiler 定位热点 → nsys 看系统级时间线 → ncu 分析具体 kernel → 优化 → 验证。详见 `/nsight-systems` skill 的 ncu 附录。
