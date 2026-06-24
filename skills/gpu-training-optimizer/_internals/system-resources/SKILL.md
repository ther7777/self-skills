---
name: system-resources
description: 机器资源信息采集（CPU/内存/GPU/磁盘/网卡）
user-invocable: false
---

## 描述
采集当前机器的 CPU、内存、GPU 显存、磁盘、网卡及 RDMA、GPU 拓扑等真实运行资源信息，并生成结构化报告，为代码优化和性能调优（如 batch size 选择、ZeRO Stage 选型、CPU Offload 可行性评估、多机通信策略等）提供数据支撑。

## 触发条件
当用户需要了解当前机器的硬件资源状况时触发。适用场景包括：训练前评估硬件条件、排查 OOM 问题时查看显存/内存使用、选择并行策略前了解 GPU 拓扑和网卡带宽、评估 CPU Offload / NVMe Offload 可行性、性能调优时需要了解真实硬件瓶颈等。

## 执行指令

你是系统资源分析专家。按以下步骤逐项采集当前机器的硬件资源信息，然后生成汇总报告并给出优化建议。

> **执行原则**：每个步骤的命令都应实际运行，采集真实数据。命令不可用时跳过并标注"不可用"。

---

### 第一步：采集 CPU 信息

```bash
lscpu
```

**关注指标：**

| 指标 | 含义 | 优化关联 |
|------|------|---------|
| Model name | CPU 型号 | 判断算力档次 |
| Socket(s) | 物理 CPU 数 | 多 socket 需注意 NUMA 绑核 |
| Core(s) per socket | 每 socket 核心数 | 影响 DataLoader `num_workers` 上限 |
| Thread(s) per core | 超线程倍数 | 逻辑核 = socket × core × thread |
| CPU(s) | 总逻辑核数 | DataLoader / CPU Offload 的并行能力 |
| NUMA node(s) | NUMA 节点数 | CPU Offload 时应绑定到 GPU 所在 NUMA 节点 |
| Flags (avx/avx2/avx512) | 向量指令集 | 影响 CPU 侧计算效率（如 DeepSpeedCPUAdam） |

如需查看 NUMA 与 CPU 核心的对应关系：
```bash
lscpu | grep "NUMA node"
```

---

### 第二步：采集内存信息

```bash
free -h
```

**关注指标：**

| 指标 | 含义 | 优化关联 |
|------|------|---------|
| total | 物理内存总量 | CPU Offload 可用空间上限 |
| used | 已使用内存 | 当前占用情况 |
| free | 空闲内存 | 可立即分配的内存 |
| available | 实际可用内存（含可回收缓存） | **最重要**：真实可用于 Offload 的空间 |
| Swap total | 交换分区大小 | 0 表示无 swap（GPU 训练推荐无 swap） |

**CPU Offload 可行性速算**（FP16 + Adam 优化器）：
- 优化器状态每参数 12 bytes（FP32 权重 + 动量 + 方差）
- 10B 模型 offload 优化器 ≈ 120 GB CPU 内存
- 70B 模型 offload 优化器 ≈ 840 GB CPU 内存
- **规则**：available 内存 > 模型参数量 × 12 bytes 才可安全启用 offload_optimizer

如需更详细的内存信息：
```bash
cat /proc/meminfo | head -20
```

---

### 第三步：采集 GPU 信息

#### 3.1 基础信息总览

```bash
nvidia-smi
```

#### 3.2 结构化数据采集（便于分析）

```bash
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.max.sm,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits
```

**关注指标：**

| 指标 | 含义 | 优化关联 |
|------|------|---------|
| name | GPU 型号 | 决定 Flash Attention 版本、FP8/BF16 支持 |
| memory.total | 显存总量 (MiB) | 决定 micro_batch_size、ZeRO Stage |
| memory.used | 已用显存 | 排查显存泄漏、判断是否有其他进程占用 |
| memory.free | 可用显存 | 实际可用于训练的显存 |
| utilization.gpu | GPU 计算利用率 (%) | 训练中 < 80% 说明有瓶颈（数据加载/通信/CPU 同步） |
| utilization.memory | 显存带宽利用率 (%) | 高值说明显存带宽受限 |
| temperature.gpu | GPU 温度 (°C) | > 80°C 需关注散热，可能降频 |
| power.draw / power.limit | 实际功耗 / 功耗上限 (W) | 接近上限说明计算密集 |
| pcie.link.gen / width | PCIe 代数和宽度 | Gen4 x16 = 32 GB/s，影响 CPU↔GPU 传输速度 |

#### 3.3 查看 GPU 上运行的进程

```bash
nvidia-smi --query-compute-apps=pid,name,gpu_uuid,used_memory --format=csv
```

> 如果显存被占用但不知道被谁占了，此命令可定位进程。

#### 3.4 GPU 架构判断

| GPU 型号 | 架构 | 支持特性 |
|---------|------|---------|
| V100 | Volta | FP16，不支持 Flash Attention |
| A100 / A10 / A30 | Ampere | BF16、Flash Attention 2、Tensor Core |
| H100 / H200 / H20 | Hopper | BF16、FP8、Flash Attention 2/3、高带宽 NVLink |
| B200 / GB200 | Blackwell | FP4、Flash Attention 4 |

---

### 第四步：采集磁盘信息

#### 4.1 文件系统使用情况

```bash
df -h | grep -v "tmpfs\|overlay\|shm" || df -h
```

> 过滤 tmpfs/overlay 等虚拟文件系统，聚焦实际存储。如过滤后无结果则显示全部。

#### 4.2 块设备与 NVMe 识别

```bash
lsblk -d -o NAME,SIZE,TYPE,ROTA,TRAN,MODEL 2>/dev/null || lsblk
```

**关注指标：**

| 指标 | 含义 | 优化关联 |
|------|------|---------|
| 数据目录挂载点容量与使用率 | 数据存储空间 | 训练数据和 checkpoint 是否放得下 |
| NVMe 设备是否存在 | 高速本地存储 | ZeRO-Infinity NVMe Offload 的前提 |
| ROTA=0 | 非旋转介质（SSD/NVMe） | 数据 I/O 速度 |
| 存储类型（ceph/nfs/local） | 存储层级 | 本地 SSD > Ceph > NFS，训练数据应尽量放本地 |

**NVMe Offload 可行性判断：**
- `lsblk` 中出现 `nvme` 设备 → 具备 NVMe Offload 条件
- 检查 NVMe 是否已挂载且有可用空间
- ZeRO-Infinity 推荐 NVMe 可用空间 > 模型参数量 × 16 bytes

---

### 第五步：采集网卡与 RDMA 信息

#### 5.1 网卡列表与状态

```bash
ip -br link show | grep -v "lo\|docker\|veth\|br-"
```

> `-br` 为简洁输出模式，过滤容器虚拟网卡。

#### 5.2 物理网卡速率（需逐一检查关键网卡）

```bash
# 查看 bond 接口或主要物理网卡的速率
for iface in $(ip -br link show | awk '/bond|eth0|ens|eno/{print $1}' | head -10); do
    echo "=== $iface ==="
    ethtool "$iface" 2>/dev/null | grep -E "Speed|Link detected|Settings for" || echo "  (无法获取)"
done
```

#### 5.3 RDMA/InfiniBand 状态（如可用）

```bash
# 检查 RDMA 设备
ibstat 2>/dev/null | head -60 || echo "ibstat 不可用，跳过 RDMA 检查"
```

```bash
# 查看 RDMA 设备数量和状态摘要
ibstatus 2>/dev/null | grep -E "Infiniband|rate|state" | head -30 || echo "ibstatus 不可用"
```

**关注指标：**

| 指标 | 含义 | 优化关联 |
|------|------|---------|
| 网卡速率 | 单口带宽 | 25/100/200 Gbps，影响多机通信吞吐 |
| Bond 配置 | 链路聚合 | 多口聚合可提升总带宽 |
| RDMA 状态 (Active/LinkUp) | GPUDirect RDMA 可用性 | 多机训练必须确认 RDMA 可用 |
| RDMA 速率 | RDMA 带宽 | 200 Gb/s (HDR) / 400 Gb/s (NDR) |
| RDMA 设备数量 | 可用 RDMA 通道数 | 影响 NCCL 通信并行度 |

---

### 第六步：采集 GPU 拓扑信息

```bash
nvidia-smi topo -m
```

**关注指标：**

| 连接类型 | 含义 | 带宽 |
|---------|------|------|
| NV# (如 NV18) | NVLink 直连，#表示 link 数 | 每 link 50 GB/s (NVLink4)，NV18 ≈ 900 GB/s |
| PIX | 同一 PCIe switch 下 | ~32 GB/s (PCIe Gen4 x16) |
| PHB | 经过 PCIe Host Bridge | ~32 GB/s |
| NODE | 同一 NUMA 节点，不同 PCIe switch | 较低 |
| SYS | 跨 NUMA 节点 | 最低，需经 CPU 互联 |

**拓扑对优化的影响：**
- **全 NVLink 互联** → 张量并行（TP）效率高，优先在 NVLink GPU 组内做 TP
- **无 NVLink** → 避免 TP，优先用数据并行 + ZeRO
- **GPU-NIC 亲和性** → NCCL 通信应绑定到 GPU 对应的 RDMA 网卡（NUMA 节点一致）

---

### 第七步：生成资源摘要报告

将以上采集结果整合为结构化报告：

```markdown
# 机器资源摘要报告

## 硬件概览
| 项目 | 值 |
|------|-----|
| CPU | <型号> × <socket 数>，<总核心数> 核 / <总逻辑核数> 线程 |
| 内存 | 总量 <X> GB，可用 <Y> GB |
| GPU | <型号> × <数量>，单卡显存 <X> GB，总显存 <Y> GB |
| 存储 | 本地 NVMe: <有/无> (<容量>)，数据目录: <挂载点> (<可用空间>) |
| 网卡 | <速率> × <数量>，RDMA: <可用/不可用> (<速率>) |
| GPU 互联 | <NVLink/PCIe>，拓扑: <描述> |
| CUDA | 驱动 <版本>，CUDA <版本> |

## GPU 状态详情
| GPU # | 型号 | 显存总量 | 已用 | 可用 | 利用率 | 温度 | 功耗 |
|-------|------|---------|------|------|--------|------|------|
| 0 | ... | ... | ... | ... | ... | ... | ... |

## 存储详情
| 挂载点 | 容量 | 已用 | 可用 | 类型 |
|--------|------|------|------|------|
| ... | ... | ... | ... | NVMe/SSD/Ceph/NFS |

## 网卡详情
| 接口 | 速率 | 状态 | 类型 |
|------|------|------|------|
| ... | ... | UP/DOWN | Ethernet/RDMA |

## 资源评估与优化建议

### 显存预算
- 单卡可用显存：<X> GB
- 4 卡总可用显存：<Y> GB
- **建议**：<根据显存推荐 ZeRO Stage 和 micro_batch_size 范围>

### CPU Offload 可行性
- 可用 CPU 内存：<X> GB
- 可支撑 offload 的最大模型参数量：约 <Y>B（按 12 bytes/参数估算）
- **建议**：<是否推荐 CPU Offload，适合多大模型>

### NVMe Offload 可行性
- NVMe 可用空间：<X> TB
- **建议**：<是否具备 ZeRO-Infinity 条件>

### 多机通信
- RDMA 状态：<可用/不可用>
- 网卡带宽：<X> Gb/s
- **建议**：<是否推荐 GPUDirect RDMA，通信优化策略>

### 并行策略
- GPU 互联：<NVLink 描述>
- **建议**：<TP/PP/DP 策略建议>

### 数据 I/O
- 数据存储位置：<本地/远程>
- 存储类型：<NVMe/SSD/Ceph/NFS>
- **建议**：<数据加载优化建议，是否需要预拷贝到本地>

### CPU 绑核
- NUMA 节点数：<N>
- **建议**：<DeepSpeed --bind_cores_to_rank / numactl 建议>
```

---

### 附录：资源信息与优化决策对照表

| 硬件指标 | 阈值 / 条件 | 推荐优化操作 |
|---------|------------|------------|
| **GPU 显存** < 40 GB | 单卡 V100-32G / A10-24G | ZeRO Stage 2+, gradient checkpointing, micro_batch=1 |
| **GPU 显存** 40-80 GB | 单卡 A100-40G/80G | ZeRO Stage 2 起步，按模型大小升级 |
| **GPU 显存** > 80 GB | 单卡 H100-80G / H20-96G | ZeRO Stage 1-2 可训练较大模型 |
| **CPU 内存** > 模型参数×12B | 内存充足 | 可安全启用 offload_optimizer 到 CPU |
| **CPU 内存** < 模型参数×12B | 内存不足 | 不宜启用 CPU Offload，或考虑 NVMe Offload |
| **NVMe 可用** | lsblk 中有 nvme 设备 | 可启用 ZeRO-Infinity NVMe Offload |
| **无 NVMe** | 仅 HDD/远程存储 | 不可用 NVMe Offload，依赖 GPU+CPU 内存 |
| **RDMA 可用** | ibstat 显示 Active | 多机训练启用 GPUDirect RDMA，设置 NCCL 环境变量 |
| **RDMA 不可用** | ibstat 不存在或 Down | 多机通信走 TCP/IP，带宽受限，考虑 ZeRO++ 量化通信 |
| **NVLink 互联** | nvidia-smi topo 显示 NV# | 优先在 NVLink 组内做张量并行（TP） |
| **仅 PCIe 互联** | topo 显示 PIX/PHB/SYS | 避免 TP，优先数据并行 + ZeRO |
| **GPU 利用率** < 50% | nvidia-smi utilization.gpu | 瓶颈可能在数据加载或 CPU-GPU 同步，用 profiler 进一步分析 |
| **GPU 温度** > 80°C | temperature.gpu > 80 | 可能触发降频，检查散热/功耗限制 |
| **数据在远程存储** (NFS/Ceph) | df 显示远程挂载 | 预拷贝训练数据到本地 NVMe/SSD |
| **数据在本地 NVMe** | df 显示本地 nvme | I/O 通常不是瓶颈 |
| **NUMA ≥ 2** | lscpu NUMA nodes ≥ 2 | CPU Offload 时建议绑核（`--bind_cores_to_rank` 或 `numactl`） |
| **CPU 核心数** > 32 | 多核服务器 | DataLoader `num_workers` 建议 4-8 per GPU |

### 附录：快速一键采集脚本

如需一次性采集所有信息，运行以下脚本：

```bash
echo "========== CPU =========="
lscpu | grep -E "Model name|Socket|Core|Thread|CPU\(s\)|NUMA|Flags" | head -15

echo -e "\n========== 内存 =========="
free -h

echo -e "\n========== GPU =========="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,power.draw,power.limit --format=csv 2>/dev/null || echo "nvidia-smi 不可用"

echo -e "\n========== GPU 拓扑 =========="
nvidia-smi topo -m 2>/dev/null || echo "nvidia-smi topo 不可用"

echo -e "\n========== 磁盘 =========="
lsblk -d -o NAME,SIZE,TYPE,ROTA,TRAN,MODEL 2>/dev/null || lsblk
echo "---"
df -h | grep -vE "tmpfs|overlay|shm|udev|proc" || df -h

echo -e "\n========== 网卡 =========="
ip -br link show | grep -vE "lo|docker|veth|br-" | head -20

echo -e "\n========== RDMA =========="
ibstatus 2>/dev/null | grep -E "Infiniband|rate|state" | head -20 || echo "RDMA 不可用"

echo -e "\n========== GPU 进程 =========="
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv 2>/dev/null || echo "无 GPU 进程"
```

### 附录：常用 nvidia-smi 查询命令

```bash
# 持续监控 GPU（每 1 秒刷新）
nvidia-smi -l 1

# 仅看显存使用
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# 查看 NVLink 状态和带宽
nvidia-smi nvlink --status

# 查看 GPU 时钟频率（是否被降频）
nvidia-smi --query-gpu=index,clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory --format=csv

# 查看 ECC 错误（排查硬件问题）
nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv

# 查看 PCIe 带宽（rx/tx 吞吐）
nvidia-smi dmon -s t -c 5
```
