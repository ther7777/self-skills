---
name: nsight-systems
description: NVIDIA Nsight Systems 系统级 GPU Profiling
user-invocable: false
---

## 描述
使用 NVIDIA Nsight Systems (`nsys`) 对 GPU 训练/推理工作负载进行系统级性能分析。Nsight Systems 提供 CPU + GPU 的全局时间线视图，能捕获 CUDA API 调用、GPU kernel 执行、NVTX 标注、内存操作、NCCL 通信等，是定位 GPU 利用率瓶颈、CPU-GPU 交互问题和分布式通信开销的首选工具。

## 触发条件
当用户需要对 CUDA/GPU 工作负载进行系统级 profiling（而非单纯的算子级分析）时触发。适用场景包括：需要查看 CPU-GPU 时间线、定位 GPU idle 原因、分析 NCCL 通信开销、排查多卡训练瓶颈、对比 DeepSpeed/FSDP 并行效率等。

## 执行指令

你是 NVIDIA Nsight Systems 性能分析专家。根据用户的场景选择合适的 `nsys` 命令和选项，指导用户采集和分析性能数据。

---

### 第一步：确定分析场景

| 场景 | 推荐命令模板 | 说明 |
|------|------------|------|
| **单 GPU 训练快速分析** | 模板 A | 最常用，采集 CUDA + NVTX |
| **PyTorch DL 脚本完整分析** | 模板 B | 包含 CUDA/cuDNN/cuBLAS/NVTX + Python 采样 |
| **多卡分布式训练 (torchrun)** | 模板 C | 选择性 profile 指定 rank |
| **DeepSpeed 分布式训练** | 模板 D | 通过 wrapper 脚本适配 DeepSpeed launcher |
| **仅采集指定代码段** | 模板 E | 使用 NVTX 或 cudaProfilerApi 控制采集范围 |
| **长训练任务采样** | 模板 F | 延迟启动 + 限时采集 |
| **事后统计分析** | 模板 G | `nsys stats` 对已有报告做统计汇总 |

---

### 第二步：选择并生成 nsys 命令

#### 模板 A：单 GPU 快速分析（最常用）

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output=report_output \
  --force-overwrite=true \
  python train.py [args]
```

#### 模板 B：PyTorch DL 脚本完整分析（推荐用于深度学习）

```bash
nsys profile \
  --trace=cuda,cudnn,cublas,osrt,nvtx \
  --pytorch=autograd-nvtx \
  --cudabacktrace=all \
  --python-backtrace=cuda \
  --python-sampling=true \
  --cuda-memory-usage=true \
  --output=dl_profile \
  --force-overwrite=true \
  python train.py [args]
```

**PyTorch 专用选项 `--pytorch`：**

| 值 | 说明 |
|----|------|
| `autograd-nvtx` | 自动为 PyTorch autograd 操作添加 NVTX 标注 |
| `autograd-shapes-nvtx` | 同上，额外记录 tensor shape |
| `functions-trace` | 追踪 PyTorch 函数调用，提供详细的网络结构和执行信息 |
| `none` | 禁用 |

可组合使用：`--pytorch=functions-trace,autograd-nvtx`

#### 模板 C：多卡 torchrun 选择性 Profile

创建 `run_nsys.py`：
```python
import subprocess, sys, os

local_rank = int(os.environ["LOCAL_RANK"])
args_string = " ".join(sys.argv[1:])

if local_rank == 0:  # 只 profile rank 0
    command = f"nsys profile -t cuda,nvtx,cudnn -o rank0_profile python {args_string}"
else:
    command = f"python {args_string}"

subprocess.run(command, shell=True)
```

启动：
```bash
torchrun --nnodes=1 --nproc-per-node=8 run_nsys.py train_script.py [args]
```

#### 模板 D：DeepSpeed 分布式 Profile

创建 `nsys_profile.sh`：
```bash
#!/bin/bash
nsys profile \
  -t cuda,mpi,nvtx,cudnn \
  -o deepspeed_profile.%p \
  python "$@"
```

启动（注意 `--no_python`）：
```bash
deepspeed --no_python [deepspeed args] ./nsys_profile.sh train_script.py [args]
```

`%p` 会被替换为进程 PID，避免多进程输出冲突。

#### 模板 E：仅采集指定代码段（NVTX 控制）

**方式 1：NVTX 标注控制（推荐）**

在 Python 代码中插入 NVTX 标记：
```python
import torch.cuda.nvtx as nvtx

# 或使用 nvtx 包: pip install nvtx
import nvtx

for step in range(num_steps):
    if step == target_step:
        nvtx.range_push("profile_region")

    train_step(batch)

    if step == target_step:
        nvtx.range_pop()
```

启动采集：
```bash
nsys profile \
  --capture-range=nvtx \
  --nvtx-capture=profile_region \
  --capture-range-end=stop \
  --trace=cuda,nvtx \
  python train.py
```

**方式 2：cudaProfilerApi 控制**

在代码中插入：
```python
torch.cuda.cudart().cudaProfilerStart()
# ... 要 profile 的代码 ...
torch.cuda.cudart().cudaProfilerStop()
```

启动采集：
```bash
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --trace=cuda,nvtx \
  python train.py
```

#### 模板 F：长训练任务 — 延迟启动 + 限时采集

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --delay=120 \
  --duration=30 \
  --kill=none \
  --output=long_run_profile \
  python train.py [args]
```

- `--delay=120`：启动后等待 120 秒再开始采集（跳过 warmup）
- `--duration=30`：采集 30 秒后停止
- `--kill=none`：采集结束后不终止应用，让训练继续

#### 模板 G：事后统计分析 (`nsys stats`)

```bash
# 显示默认统计（CUDA API、Kernel、Memory）
nsys stats report.nsys-rep

# 仅显示 CUDA GPU kernel 汇总
nsys stats --report cuda_gpu_kern_sum report.nsys-rep

# 仅显示 CUDA GPU trace（每个 kernel 的详细执行记录）
nsys stats --report cuda_gpu_trace report.nsys-rep

# 仅显示 CUDA API 调用汇总
nsys stats --report cuda_api_sum report.nsys-rep

# 多份报告，多种格式
nsys stats \
  --report cuda_gpu_kern_sum \
  --report cuda_api_sum \
  --format csv,column \
  --output report_kernels.csv,- \
  report.nsys-rep
```

也可以在 profile 时直接生成统计：
```bash
nsys profile --stats=true python train.py
```

---

### 第三步：核心 CLI 选项速查

#### `--trace` / `-t`（指定采集哪些 API 的 trace）

| 值 | 说明 | 适用场景 |
|----|------|---------|
| `cuda` | CUDA Runtime & Driver API | **必选**，所有 GPU 工作负载 |
| `nvtx` | NVIDIA Tools Extension 标注 | **推荐**，用于代码段标记 |
| `osrt` | OS Runtime（pthread、文件 I/O 等） | 分析 DataLoader / I/O 瓶颈 |
| `cudnn` | cuDNN API | DL 训练（conv、RNN 等） |
| `cublas` | cuBLAS API | 矩阵运算（线性层） |
| `mpi` | MPI API | 多机分布式训练 |
| `opengl` | OpenGL API | 图形应用 |
| `vulkan` | Vulkan API | 图形应用 |
| `nccl` | NCCL 通信（需插件） | 多卡集合通信分析 |

多选用逗号分隔：`--trace=cuda,nvtx,cudnn,cublas,osrt`

#### 常用控制选项

| 选项 | 说明 | 常用值 |
|------|------|--------|
| `--output` / `-o` | 输出文件名（不含扩展名） | `my_profile`，`%p` 替换 PID，`%h` 替换 hostname |
| `--duration` / `-d` | 采集时长（秒） | `30` |
| `--delay` / `-y` | 延迟启动（秒） | `60` |
| `--capture-range` / `-c` | 采集触发方式 | `none`（立即）, `cudaProfilerApi`, `nvtx`, `hotkey` |
| `--capture-range-end` | 采集结束行为 | `stop-shutdown`, `stop`, `repeat[:N]` |
| `--kill` | 采集结束后是否终止应用 | `sigterm`（默认）, `none` |
| `--force-overwrite` / `-f` | 覆盖已有文件 | `true` |
| `--sample` / `-s` | CPU 采样范围 | `process-tree`（默认）, `system-wide`, `none` |
| `--backtrace` / `-b` | 回溯方式 | `auto`, `dwarf`, `fp`, `lbr`, `none` |
| `--export` | 额外导出格式 | `sqlite`, `jsonlines`, `arrow`, `none` |

#### CUDA 专用选项

| 选项 | 说明 |
|------|------|
| `--cuda-memory-usage=true` | 跟踪 CUDA kernel 的 GPU 显存使用（有开销） |
| `--cudabacktrace=all` | 为 CUDA API 调用采集调用栈（开销大） |
| `--cuda-graph-trace=graph` | CUDA Graph 整体追踪（推荐，低开销） |
| `--cuda-trace-scope=process-tree` | 追踪目标进程及子进程的 CUDA 活动 |

#### Python / PyTorch 专用选项

| 选项 | 说明 |
|------|------|
| `--python-sampling=true` | 采集 Python 调用栈采样 |
| `--python-backtrace=cuda` | 在 CUDA API 调用时采集 Python 调用栈 |
| `--pytorch=autograd-nvtx` | 自动 NVTX 标注 PyTorch autograd 操作 |
| `--pytorch=functions-trace` | 追踪 PyTorch 函数，提供网络结构执行详情 |

#### GPU 指标采集选项

| 选项 | 说明 |
|------|------|
| `--gpu-metrics-devices=all` | 采集所有 GPU 指标 |
| `--gpu-metrics-frequency=10000` | GPU 指标采样频率（Hz），默认 10KHz |
| `--nic-metrics=true` | 采集 NIC/HCA 网络指标 |

---

### 第四步：查看和分析结果

#### 4.1 可视化工具

| 工具 | 用途 | 操作方式 |
|------|------|---------|
| **Nsight Systems GUI (`nsys-ui`)** | 全功能时间线 + 统计分析（推荐） | `nsys-ui report.nsys-rep` |
| **CLI Stats** | 终端内快速查看统计 | `nsys stats report.nsys-rep` |
| **nsys analyze** | 专家系统自动诊断 | `nsys analyze report.nsys-rep` |
| **导出 SQLite** | 用 Python/SQL 自定义分析 | `nsys export --type=sqlite report.nsys-rep` |

#### 4.2 时间线视图关键看点

在 Nsight Systems GUI 中，关注以下模式：

| 看什么 | 在哪看 | 说明 |
|--------|-------|------|
| **GPU idle 间隙** | CUDA HW 行 | 间隙 = GPU 等待 CPU 提交工作 |
| **CUDA API 调用耗时** | CUDA API 行 | 长条 = 同步阻塞（如 cudaMemcpy、cudaStreamSync） |
| **NVTX 标记区间** | NVTX 行 | 对应用户标记的代码段（forward/backward/optimizer） |
| **CPU 线程活动** | CPU 行 | DataLoader 工作线程是否繁忙 |
| **NCCL 通信** | NCCL 行 | AllReduce/AllGather 的耗时和重叠情况 |
| **Kernel 时长分布** | GPU Kernel 行 | 大量短 kernel = launch overhead，少量长 kernel = 计算密集 |

#### 4.3 常见瓶颈模式及对策

| 瓶颈模式 | 时间线表现 | 优化建议 |
|----------|-----------|---------|
| **GPU 空闲等数据** | GPU 行大段空白，CPU DataLoader 线程忙 | 增加 `num_workers`，启用 `pin_memory`，数据预加载 |
| **CPU-GPU 同步阻塞** | CUDA API 行有长条 `cudaStreamSynchronize` / `cudaDeviceSynchronize` | 减少 `.item()`、`print(cuda_tensor)` 等同步点 |
| **Kernel Launch 开销大** | 大量极短 kernel 间有间隙 | 使用 `torch.compile`、CUDA Graph、算子融合 |
| **通信不重叠** | NCCL AllReduce 和计算串行排列 | 启用通信计算重叠（FSDP `backward_prefetch`、DeepSpeed `overlap_comm`） |
| **通信耗时过长** | NCCL 操作占总时间 > 30% | 检查 GDRDMA 是否启用，网络拓扑是否合理，负载是否均衡 |
| **显存拷贝瓶颈** | 长条 `cudaMemcpy` (H2D/D2H) | 使用 `pin_memory`，减少 CPU-GPU 数据传输 |
| **Python GIL 开销** | Python 采样显示大量解释器等待 | 减少 Python 侧逻辑，使用 `torch.compile` 减少 dispatch 开销 |
| **cuDNN 算法选择慢** | 首次 conv 调用特别长 | 设置 `torch.backends.cudnn.benchmark = True` |

---

### 第五步：输出分析报告

```markdown
# Nsight Systems 性能分析报告

## 采集环境
- nsys 版本：X.Y.Z
- GPU：<型号> x <数量>
- 采集命令：`nsys profile ...`
- 采集时长：X 秒

## 总体 GPU 利用率
- GPU 活跃时间占比：X%
- 主要空闲原因：<DataLoader / CPU-GPU 同步 / 通信等待>

## Top 10 CUDA Kernel（按总耗时）
| 排名 | Kernel 名称 | 总耗时 | 调用次数 | 平均耗时 |
|------|------------|--------|---------|---------|
| 1 | ... | ... | ... | ... |

## CUDA API 耗时分布
| API | 总耗时 | 调用次数 | 占比 |
|-----|--------|---------|------|
| cudaLaunchKernel | ... | ... | ... |
| cudaMemcpy | ... | ... | ... |

## 通信分析（多卡场景）
- NCCL AllReduce 总耗时：X ms
- 通信与计算重叠度：X%
- 各 rank 耗时是否均衡：是/否

## 瓶颈诊断
1. **[严重程度]** <瓶颈描述>
2. ...

## 优化建议（按优先级）
1. **[高]** <建议> — 预期收益：<说明>
2. **[中]** <建议>
3. **[低]** <建议>
```

---

### 附录：nsys vs torch.profiler 对比

| 维度 | Nsight Systems (`nsys`) | PyTorch Profiler (`torch.profiler`) |
|------|------------------------|--------------------------------------|
| **视角** | 系统级：CPU + GPU + 通信全局时间线 | 框架级：PyTorch 算子粒度 |
| **CUDA Kernel** | 精确到 GPU kernel，含 SM 占用率 | 聚合到 PyTorch 算子 |
| **NCCL 通信** | 直接可见通信 kernel 和重叠 | 需额外配置 |
| **CPU 采样** | 全系统 CPU 采样 + OS 调度 | 仅 PyTorch 线程 |
| **Python 栈** | `--python-sampling` 支持 | `with_stack=True` |
| **侵入性** | 无需改代码（命令行包裹即可） | 需在代码中插入 profiler 上下文 |
| **结果大小** | 较大（MB~GB 级） | 较小 |
| **可视化** | nsys-ui GUI（功能丰富） | Perfetto / Chrome Tracing |
| **推荐用途** | 定位系统级瓶颈（GPU idle 原因、通信开销） | 定位算子级瓶颈（哪个 op 最慢） |

**最佳实践**：先用 `torch.profiler` 做算子级快速分析；如需深入分析 GPU 利用率、通信重叠、CPU-GPU 交互，再用 `nsys`。

### 附录：从 nsys 时间线判断瓶颈类型

nsys 时间线是判断系统瓶颈的最直观工具。以下是各瓶颈类型在时间线中的典型特征：

| 瓶颈类型 | 时间线特征 | 具体表现 | 下一步 |
|----------|-----------|---------|--------|
| **Compute-bound** | GPU kernel 行密集无 gap，持续高利用率 | Kernel 连续执行，单个 kernel 耗时长（> 1ms），SM 利用率高 | 用 ncu 看是否命中 Tensor Core、能否用混合精度 |
| **Memory-bound** | GPU 活跃但大量短 kernel，频繁 memcpy | element-wise kernel 密集，cudaMemcpy 占比高 | 算子融合减少中间张量写回，Flash Attention |
| **Launch-bound** | GPU kernel 间有规律性小 gap | Kernel launch API (cudaLaunchKernel) 密集，kernel 间 gap > kernel 自身耗时 | CUDA Graph、torch.compile、减少 Python 层调度 |
| **Data I/O-bound** | GPU 周期性长 idle，与 DataLoader/H2D 交替 | 大段 GPU idle 后跟着密集计算，idle 期间 CPU 在做数据加载 | num_workers、pin_memory、预取、数据格式优化 |
| **Communication-bound** | NCCL kernel 占比大，与计算不重叠 | AllReduce/AllGather kernel 长条与 compute kernel 串行排列而非并行 | overlap_comm、梯度压缩、更高效并行策略 |
| **Sync-bound** | CPU 长时间 wait，GPU 短暂忙后 idle | cudaStreamSynchronize / cudaDeviceSynchronize 的 CPU 时间条很长 | 去掉 `.item()`、减少同步点 |

**判断步骤**：
1. 打开 nsys-ui 或 Perfetto，看 GPU 行的整体 pattern
2. GPU 几乎满载 → Compute 或 Memory bound → 用 ncu 区分
3. GPU 有规律 gap → 看 gap 期间 CPU 在做什么 → 对应 Data I/O / Launch / Sync bound
4. 多卡场景看 NCCL 行 → 通信 kernel 与 compute 是否重叠

### 附录：Nsight Compute (ncu) — Kernel 级深度分析

nsys 告诉你 **"哪里慢"**，ncu 告诉你 **"为什么慢"**。当通过 nsys 定位到具体的热点 kernel 后，用 ncu 深入分析 kernel 内部性能。

#### 何时用 ncu

| 场景 | 用 nsys 还是 ncu？ |
|------|-------------------|
| 找到整体瓶颈在哪 | nsys（全局时间线） |
| 分析具体 kernel 为什么慢 | ncu（kernel 内部指标） |
| 判断 compute-bound vs memory-bound | ncu（roofline model） |
| 看 GPU occupancy / register / shared memory | ncu |
| 看 warp stall reason | ncu |
| 优化自定义 Triton/CUDA kernel | ncu |

#### 常用 ncu 命令

```bash
# 模板 A：分析单个 kernel（交互式，适合调试）
ncu --set full \
    --target-processes all \
    --kernel-name "regex:.*my_kernel.*" \
    --launch-count 5 \
    python my_script.py

# 模板 B：生成 Roofline 模型（判断 compute vs memory bound 的利器）
ncu --set roofline \
    --kernel-name "regex:.*matmul.*" \
    --launch-count 3 \
    -o roofline_report \
    python my_script.py

# 模板 C：对 PyTorch 训练采样特定 step 的 kernel
ncu --set full \
    --target-processes all \
    --launch-skip 1000 --launch-count 50 \
    -o training_kernels \
    python train.py

# 模板 D：对比两次运行（优化前后对比）
ncu --set full -o before.ncu-rep python train_v0.py
ncu --set full -o after.ncu-rep python train_v1.py
# 在 ncu-ui 中对比两个 .ncu-rep 文件
```

#### 关键 ncu 指标解读

| 指标 | 含义 | 健康值 | 偏离说明 |
|------|------|--------|---------|
| **SM Efficiency** | SM 活跃时间占比 | > 80% | 低 → block 数不够或 kernel 太短 |
| **Achieved Occupancy** | 实际驻留的 warp 数 / 理论最大 warp 数 | > 50% | 低 → register 或 shared memory 使用过多 |
| **DRAM Throughput** | 全局显存带宽利用率 | 接近峰值 → memory-bound | 远低于峰值 → 可能有 cache 命中或 compute-bound |
| **L2 Hit Rate** | L2 cache 命中率 | > 80% | 低 → 随机访存或 working set 太大 |
| **Compute (SM) Throughput** | 计算单元利用率 | 接近峰值 → compute-bound | 低且 DRAM 高 → memory-bound |
| **Warp Stall Reasons** | warp 无法发射的原因分布 | 无主导 stall 原因 | `stall_memory` 高 → 访存瓶颈；`stall_not_selected` 高 → occupancy OK 只是调度；`stall_barrier` 高 → 同步过多 |
| **Register File Usage** | 每线程寄存器数 | < 128 | 过高导致 occupancy 下降和 spill |
| **Shared Memory Usage** | 每 block shared memory 大小 | 平衡使用 | 过高导致 occupancy 下降 |
| **Global Load/Store Efficiency** | 有效数据量 / 实际传输量 | > 80% | 低 → 非 coalesced 访问或 misaligned |

#### Roofline 模型解读

Roofline 是 ncu 最实用的功能之一。它将 kernel 性能画在"算力天花板"和"带宽天花板"组成的图上：

```
          Compute Ceiling (FLOPS peak)
         ___________________________
        /
       /   ← Kernel 在这个区域 = memory-bound
      /       优化方向：减少访存、提高 cache 命中
     /
    /          ← Kernel 在这个区域 = compute-bound
   /              优化方向：混合精度、Tensor Core、算法优化
  /
 / ← Memory Bandwidth Ceiling
```

- **Kernel 在带宽线下方**：memory-bound，优化访存（coalescing、tiling、cache）
- **Kernel 在算力线下方**：compute-bound，优化计算（混合精度、Tensor Core）
- **Kernel 远离两条线**：可能有其他瓶颈（latency、divergence、spill）

### 附录：多卡场景仅对 rank 0 采集 GPU 指标

```bash
#!/bin/bash
# 用于 mpirun / srun 的 wrapper
if [ "$OMPI_COMM_WORLD_LOCAL_RANK" -eq 0 ]; then
    nsys profile --nic-metrics=true --gpu-metrics-devices=all "$@"
else
    nsys profile "$@"
fi
```

### 附录：容器内使用 nsys

```bash
# Docker 需要额外权限
docker run --cap-add=SYS_ADMIN --cap-add=SYS_PTRACE ...

# 或在 Kubernetes 中添加 securityContext
# securityContext:
#   capabilities:
#     add: ["SYS_ADMIN", "SYS_PTRACE"]
```
