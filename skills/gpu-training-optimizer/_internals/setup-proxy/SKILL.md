---
name: setup-proxy
description: 腾讯内网代理配置，支持访问 GitHub/PyPI 等外网资源
user-invocable: false
---

## 描述
在腾讯内网环境（Venus 平台等）中配置 HTTP/HTTPS 代理，使终端能够访问 GitHub、PyPI 等外网资源。

## 触发条件
当用户需要访问外网资源（如 git clone GitHub 仓库、pip install 外网包、curl 外网地址）但因网络受限而失败时触发，或者当用户主动要求设置代理 / 配置外网访问时触发。

## 执行指令

### 第一步：检测代理环境变量

检查 `ENV_VENUS_PROXY` 是否已设置：

```bash
printenv ENV_VENUS_PROXY
```

- **已设置**：继续第二步。
- **未设置**：告知用户当前环境没有预配置的代理地址，请用户手动提供代理地址（格式：`http://user:password@host:port`），然后继续第二步。

### 第二步：设置代理环境变量

**重要**：使用 `$(printenv ENV_VENUS_PROXY)` 而非直接 `$ENV_VENUS_PROXY` 进行赋值，避免密码中特殊字符导致变量展开异常。

执行以下命令：

```bash
PROXY_URL="$(printenv ENV_VENUS_PROXY)"
export http_proxy="$PROXY_URL"
export https_proxy="$PROXY_URL"
export HTTP_PROXY="$PROXY_URL"
export HTTPS_PROXY="$PROXY_URL"
export no_proxy="localhost,.woa.com,.oa.com,.tencent.com,.tencentcos.cn,.myqcloud.com"
export NO_PROXY="$no_proxy"
```

注意：`no_proxy` 列表包含腾讯内网常见域名后缀，访问这些域名时不走代理，避免内网请求绕远。

### 第三步：验证代理连通性

用一个轻量请求验证代理是否可用：

```bash
curl -sI --max-time 10 https://github.com 2>&1 | head -5
```

- 看到 `HTTP/1.1 200` 或 `HTTP/2 200`：代理设置成功。
- 超时或连接失败：检查代理地址是否正确、代理服务是否可达。

### 第四步：向用户报告结果

报告代理设置状态，并给出后续使用提示：

**成功时**：
```
代理已设置成功，当前终端可访问外网。

常见用法：
- git clone https://github.com/xxx/yyy.git
- pip install <package>
- curl https://外网地址

注意：代理仅对当前终端会话生效。新开终端需重新设置。
如需持久化，可将以下内容添加到 ~/.bashrc：

  PROXY_URL="$(printenv ENV_VENUS_PROXY)"
  export http_proxy="$PROXY_URL"
  export https_proxy="$PROXY_URL"
  export HTTP_PROXY="$PROXY_URL"
  export HTTPS_PROXY="$PROXY_URL"
  export no_proxy="localhost,.woa.com,.oa.com,.tencent.com,.tencentcos.cn,.myqcloud.com"
  export NO_PROXY="$no_proxy"
```

**失败时**：
- 检查 `ENV_VENUS_PROXY` 值是否正确
- 尝试 `curl -sv --proxy "$PROXY_URL" --max-time 10 https://github.com` 查看详细错误
- 确认当前环境是否允许使用该代理（部分集群可能有不同代理配置）

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
| `$ENV_VENUS_PROXY` 直接赋值后变量为空 | 密码中含有特殊字符（如 `@`），shell 展开时被截断 | 使用 `$(printenv ENV_VENUS_PROXY)` 赋值 |
| 代理连通但 HTTPS 握手失败 | 代理不支持 CONNECT 隧道 | 确认代理类型，尝试 HTTP 目标验证 |
| 内网服务变慢 | 内网请求也走了代理 | 检查 `no_proxy` 是否包含目标域名 |
| git clone 超时 | git 未使用代理 | 执行 `git config --global http.proxy` 设置 git 代理 |
