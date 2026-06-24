---
name: nsys-install
description: Nsight Systems 自动检测、安装与升级
---

# Skill: Nsight Systems 安装与升级

## 描述
检测当前环境的 Nsight Systems (`nsys`) 安装状态，如果缺失或版本过旧（不支持 `--pytorch` 等关键特性），则自动下载安装对应版本的 Nsight Systems 独立包。支持 x86_64 和 aarch64 架构，覆盖 Ampere/Hopper/Blackwell GPU。

## 触发条件
当以下任一条件满足时触发：
- 运行 `nsys` 命令提示找不到
- 当前 nsys 版本不支持 `--pytorch` 自动标注（低于 2025.1）
- 用户明确要求安装或升级 Nsight Systems
- GPU 训练优化器流程中第四步检测到 nsys 不可用或版本过低

## 执行指令

你是 NVIDIA 工具链安装专家。按以下步骤检测环境并安装合适版本的 Nsight Systems。

---

### 第一步：环境检测

运行以下命令采集环境信息，**不要向用户提问**，直接从系统获取：

```bash
echo "=== Architecture ==="
uname -m

echo "=== OS ==="
cat /etc/os-release 2>/dev/null | head -5

echo "=== glibc ==="
ldd --version 2>&1 | head -1

echo "=== GPU ==="
nvidia-smi --query-gpu=gpu_name,driver_version,compute_cap --format=csv,noheader 2>/dev/null | head -1

echo "=== CUDA Toolkit ==="
cat /usr/local/cuda/version.json 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['cuda']['version'])" 2>/dev/null || nvcc --version 2>/dev/null | grep release || echo "not found"

echo "=== Current nsys ==="
which nsys 2>/dev/null && nsys --version 2>/dev/null || echo "nsys not found"

echo "=== nsys --pytorch support ==="
nsys profile --help 2>&1 | grep -q "\-\-pytorch" && echo "supported" || echo "not supported"

echo "=== Disk space (/opt) ==="
df -h /opt 2>/dev/null | tail -1
```

---

### 第二步：判断是否需要安装

根据第一步的结果判断：

| 情况 | 行动 |
|------|------|
| nsys 未安装 | → 需要安装，继续第三步 |
| nsys 已安装但**不支持 `--pytorch`**（版本 < 2025.1） | → 需要升级，继续第三步 |
| nsys 已安装且支持 `--pytorch`（版本 >= 2025.1） | → 无需操作，输出当前版本信息后结束 |

如果无需安装，直接输出：
```
✅ Nsight Systems 已是最新版本，支持 --pytorch 自动标注
当前版本: <版本号>
路径: <nsys路径>
```

---

### 第三步：选择安装版本

根据 **GPU 架构**和**驱动版本**选择合适的 Nsight Systems 版本：

#### 版本兼容矩阵

| Nsight Systems 版本 | 最低驱动版本 | 支持 GPU 架构 | `--pytorch` | 下载文件名 |
|---------------------|-------------|--------------|-------------|-----------|
| **2026.1.1** | ≥ 525.60 | Ampere / Hopper / Blackwell | ✅ | `NsightSystems-linux-public-2026.1.1.204-3717666.run` |
| **2025.2.1** | ≥ 525.60 | Ampere / Hopper / Blackwell | ✅ | `NsightSystems-linux-public-2025.2.1.130-3554082.run` |
| **2025.1.1** | ≥ 525.60 | Ampere / Hopper | ✅ | `NsightSystems-linux-public-2025.1.1.131-3503307.run` |

#### 选择策略

```
if driver_version >= 525.60:
    if gpu_arch in (Blackwell):   → 2026.1.1（Blackwell 需要最新版本）
    elif gpu_arch in (Hopper):    → 2026.1.1（推荐最新，兼容性最好）
    elif gpu_arch in (Ampere):    → 2025.2.1 或 2026.1.1
else:
    → 提示用户升级驱动后再安装
```

> **注意**：如果驱动版本低于 525.60，nsys 的 GPU trace 功能会受限。建议先升级驱动。

#### GPU 架构识别

| Compute Capability | 架构 | 代表型号 |
|-------------------|------|---------|
| 8.0, 8.6, 8.9 | Ampere | A100, A10, A30, L40 |
| 9.0 | Hopper | H100, H200, H20, H800 |
| 10.0, 12.0 | Blackwell | B100, B200, GB200 |

---

### 第四步：下载安装包

#### 4.0 配置外网代理

下载 NVIDIA 安装包需要访问外网。**在下载之前，先配置代理确保网络连通**。

#### 4.1 构造下载 URL

下载地址模式：
```
https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/<YEAR>_<MINOR>/<FILENAME>.run
```

各版本对应 URL：

| 版本 | URL |
|------|-----|
| 2026.1.1 | `https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_1/NsightSystems-linux-public-2026.1.1.204-3717666.run` |
| 2025.2.1 | `https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/NsightSystems-linux-public-2025.2.1.130-3554082.run` |
| 2025.1.1 | `https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_1/NsightSystems-linux-public-2025.1.1.131-3503307.run` |

#### 4.2 选择下载目录

```bash
# 优先使用有空间的目录
DOWNLOAD_DIR="/tmp"

# 检查磁盘空间（安装包约 800MB，安装后约 2GB）
df -h "${DOWNLOAD_DIR}" | tail -1
```

#### 4.3 下载

```bash
NSYS_VERSION="2026.1.1"           # 根据第三步选择结果填入
NSYS_FILENAME="NsightSystems-linux-public-2026.1.1.204-3717666.run"  # 对应文件名
NSYS_URL="https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_1/${NSYS_FILENAME}"

cd "${DOWNLOAD_DIR}"

# 下载（使用 wget，如有代理先配置）
wget -q --show-progress -O "${NSYS_FILENAME}" "${NSYS_URL}"

# 如果 wget 失败（需要认证或网络问题），尝试 curl
if [ $? -ne 0 ]; then
    echo "wget 下载失败，尝试 curl..."
    curl -fL -o "${NSYS_FILENAME}" "${NSYS_URL}"
fi

# 如果仍然失败（NVIDIA 需要登录），提示用户手动下载
if [ ! -f "${NSYS_FILENAME}" ] || [ $(stat -c%s "${NSYS_FILENAME}" 2>/dev/null || echo 0) -lt 1000000 ]; then
    echo "⚠️ 自动下载失败（NVIDIA 可能要求登录）"
    echo "请手动下载: ${NSYS_URL}"
    echo "下载后放到 ${DOWNLOAD_DIR}/ 目录，然后重新运行此 skill"
    # 退出安装流程
fi
```

---

### 第五步：安装

#### 5.1 静默安装

```bash
cd "${DOWNLOAD_DIR}"

# 赋予执行权限
chmod +x "${NSYS_FILENAME}"

# 静默安装（--quiet 无 TUI，--accept 接受 EULA）
bash "${NSYS_FILENAME}" --quiet --accept
```

默认安装到：`/opt/nvidia/nsight-systems/<VERSION>/`

#### 5.2 验证安装目录

```bash
# 查看安装目录
INSTALL_DIR="/opt/nvidia/nsight-systems/${NSYS_VERSION}"
ls -la "${INSTALL_DIR}/bin/"
ls -la "${INSTALL_DIR}/target-linux-x64/" 2>/dev/null || ls -la "${INSTALL_DIR}/target-linux-sbsa-armv8/" 2>/dev/null
```

---

### 第六步：配置 PATH

#### 6.1 创建符号链接

```bash
INSTALL_DIR="/opt/nvidia/nsight-systems/${NSYS_VERSION}"

# 创建 nsys 符号链接（覆盖旧版本的链接）
ln -sf "${INSTALL_DIR}/bin/nsys" /usr/local/bin/nsys

# 如果有 nsys-ui 也创建链接
if [ -f "${INSTALL_DIR}/bin/nsys-ui" ]; then
    ln -sf "${INSTALL_DIR}/bin/nsys-ui" /usr/local/bin/nsys-ui
fi
```

#### 6.2 处理与 CUDA Toolkit 自带 nsys 的冲突

CUDA Toolkit 会在 `/usr/local/cuda/bin/nsys` 放一个 wrapper 脚本指向旧版本。确保新版本优先：

```bash
# 检查 PATH 中 nsys 的优先级
which -a nsys 2>/dev/null

# 如果 /usr/local/cuda/bin/nsys 排在前面，需要确保 /usr/local/bin 在 PATH 中更靠前
# 或者直接覆盖 CUDA 的 nsys wrapper
if [ -f /usr/local/cuda/bin/nsys ]; then
    # 备份旧的 wrapper
    mv /usr/local/cuda/bin/nsys /usr/local/cuda/bin/nsys.bak.$(date +%Y%m%d)
    # 创建新的符号链接
    ln -sf "${INSTALL_DIR}/bin/nsys" /usr/local/cuda/bin/nsys
fi
```

---

### 第七步：验证安装

```bash
echo "=== 版本验证 ==="
nsys --version

echo "=== --pytorch 支持验证 ==="
nsys profile --help 2>&1 | grep -q "\-\-pytorch" && echo "✅ --pytorch 支持可用" || echo "❌ --pytorch 不支持"

echo "=== 环境检查 ==="
nsys status -e 2>&1 | head -15

echo "=== 安装路径 ==="
which nsys
readlink -f $(which nsys) 2>/dev/null
```

预期输出：
```
=== 版本验证 ===
NVIDIA Nsight Systems version 2026.1.1.xxx-xxxxxxxxxxxxx
=== --pytorch 支持验证 ===
✅ --pytorch 支持可用
```

---

### 第八步：清理

```bash
# 删除下载的安装包（约 800MB）
rm -f "${DOWNLOAD_DIR}/${NSYS_FILENAME}"
echo "安装包已清理"
```

---

### 输出格式

安装完成后输出以下摘要：

```
============================================================
  Nsight Systems 安装完成
============================================================
  版本:        <新版本号>
  安装目录:    /opt/nvidia/nsight-systems/<VERSION>/
  二进制:      /usr/local/bin/nsys
  旧版本:      <旧版本号或"无">
  --pytorch:   ✅ 支持
  磁盘占用:    ~2 GB
============================================================

可通过以下命令验证:
  nsys --version
  nsys profile --help | grep pytorch
```

---

## 附录：常见问题

### Q: 下载时报 403 / 需要登录
NVIDIA 的下载链接有时需要登录 developer.nvidia.com。解决方案：
1. 在浏览器中登录 https://developer.nvidia.com/nsight-systems/get-started
2. 手动下载对应版本的 `.run` 文件
3. 上传到服务器的 `/tmp/` 目录
4. 重新运行此 skill

### Q: 安装报 glibc 版本过低
Nsight Systems 2025+ 需要 glibc >= 2.17。可通过 `ldd --version` 检查。如果 glibc 过低，考虑使用容器环境。

### Q: 安装后 nsys 仍然是旧版本
检查 PATH 优先级：`which -a nsys`。CUDA Toolkit 自带的 `/usr/local/cuda/bin/nsys` 可能覆盖了新安装的版本。按第六步处理冲突。

### Q: nsys profile 报 "permission denied" 或 "perf_event_open failed"
```bash
# 临时解决
echo 1 | sudo tee /proc/sys/kernel/perf_event_paranoid

# 永久解决（需要 root）
echo 'kernel.perf_event_paranoid=1' | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

### Q: 多版本共存
独立安装的 nsys 在 `/opt/nvidia/nsight-systems/<VERSION>/`，不同版本互不干扰。通过修改符号链接切换版本：
```bash
ln -sf /opt/nvidia/nsight-systems/<目标版本>/bin/nsys /usr/local/bin/nsys
```
