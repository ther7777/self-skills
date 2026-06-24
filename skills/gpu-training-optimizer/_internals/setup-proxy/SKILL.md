---
name: setup-proxy
description: 外网代理配置，支持受限网络环境中访问 GitHub/PyPI 等外网资源
---

# Skill: 设置外网代理

## 描述
在受限网络环境（企业内网、隔离集群等）中自动检测和配置 HTTP/HTTPS 代理，使终端能够访问 GitHub、PyPI 等外网资源。

## 触发条件
当用户需要访问外网资源（如 git clone GitHub 仓库、pip install 外网包、curl 外网地址）但因网络受限而失败时触发，或用户主动要求设置代理 / 配置外网访问时触发。

## 执行指令

### 第零步：检查当前网络状态

先测试外网是否已可达，避免不必要的代理配置：

```bash
curl -sI --max-time 5 https://github.com 2>&1 | head -3
```

- **已可达**（返回 `HTTP/1.1 200` 或 `HTTP/2 200`）：告知用户当前网络已可访问外网，无需配置代理，终止流程。
- **不可达**（超时、连接失败、403/407）：继续第一步。

### 第一步：检测可用代理

按优先级依次尝试以下来源，找到可用的代理地址：

**来源 1 — 平台预置环境变量（如企业平台自动注入）：**

```bash
# 常见平台代理变量名，按顺序检查
for var in ENV_VENUS_PROXY HTTP_PROXY HTTPS_PROXY http_proxy https_proxy all_proxy ALL_PROXY; do
  val="$(printenv "$var" 2>/dev/null)"
  if [ -n "$val" ]; then
    echo "FOUND: $var=$val"
    break
  fi
done
```

**来源 2 — 标准环境变量：**
```bash
printenv HTTP_PROXY || printenv http_proxy
```

**来源 3 — 用户手动提供：**
如果以上均未找到，询问用户提供代理地址（格式：`http://user:password@host:port` 或 `http://host:port`）。

### 第二步：设置代理环境变量

**重要**：使用 `$(printenv <VAR>)` 而非直接 `$<VAR>` 进行赋值，避免密码中特殊字符导致变量展开异常。

```bash
PROXY_URL="<检测到的代理地址>"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
```

**no_proxy 配置**：根据当前网络环境设置内网域名白名单，避免内网请求绕远走代理：

```bash
# 通用内网域名白名单（可按需追加）
export no_proxy="localhost,127.0.0.1,.local,.internal"
export NO_PROXY="$no_proxy"
```

> 如果检测到特定平台环境（如设置了 `ENV_VENUS_PROXY`），自动追加该平台的内网域名到 `no_proxy`。

### 第三步：验证代理连通性

```bash
curl -sI --max-time 10 https://github.com 2>&1 | head -5
```

- 看到 `HTTP/1.1 200` 或 `HTTP/2 200`：代理设置成功。
- 超时或连接失败：检查代理地址是否正确、代理服务是否可达。

### 第四步：向用户报告结果

**成功时**：
```
代理已设置成功，当前终端可访问外网。

常见用法：
- git clone https://github.com/xxx/yyy.git
- pip install <package>
- curl https://<外网地址>

注意：代理仅对当前终端会话生效。新开终端需重新设置。
如需持久化，可将以下内容添加到 ~/.bashrc：

  PROXY_URL="<代理地址>"
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export no_proxy="localhost,127.0.0.1,.local,.internal"
```

**失败时**：
- 检查代理地址是否正确
- 尝试 `curl -sv --proxy "$PROXY_URL" --max-time 10 https://github.com` 查看详细错误
- 确认当前环境是否允许使用该代理（部分环境可能有白名单限制）

### 第五步：配置 Git 代理（按需）

如果用户需要使用 git 访问外网仓库，还需配置 git 代理：

```bash
git config --global http.proxy "$PROXY_URL"
git config --global https.proxy "$PROXY_URL"
```

取消 git 代理：
```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

## 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 代理地址含特殊字符导致变量展开异常 | 密码中含有 `@`、`#` 等字符 | 使用 `$(printenv <VAR>)` 方式赋值 |
| 代理连通但 HTTPS 握手失败 | 代理不支持 CONNECT 隧道 | 确认代理类型，尝试 HTTP 目标验证 |
| 内网服务变慢或不可达 | 内网请求也走了代理 | 将内网域名加入 `no_proxy` |
| git clone 外网仓库超时 | git 未使用代理 | 执行 `git config --global http.proxy` |
| 当前环境无需代理 | 外网已可达 | 跳过代理配置，直接使用 |
