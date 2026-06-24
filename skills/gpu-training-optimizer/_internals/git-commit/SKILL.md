---
name: git-commit
description: 提交代码到远程仓库
user-invocable: false
---

# Skill: 提交代码到远程仓库

## 描述
将代码变更提交到远程 Git 仓库（GitHub、GitLab 等）。自动完成分支创建、暂存、提交、推送的完整流程。

## 触发条件
当用户说"提交代码"、"推送代码"、"push 代码"、"commit 并 push"等意图时触发。

## 规则（必须严格遵守）

### 规则 1：禁止直接提交 master/main 分支
- **绝对不允许**直接在 master 或 main 分支上 commit 或 push
- 如果当前在 master/main 分支，必须先创建并切换到 develop 分支
- 如果用户强制要求提交到 master，明确拒绝并解释原因

### 规则 2：分支命名规范
- 分支格式：`develop_<summary>`
- `<summary>` 根据本次变更内容自动总结，要求：
  - 使用英文小写字母、数字和连字符（`-`）
  - 长度**不超过 20 个字符**
  - 准确反映变更特征（如 `add-lf-swift-skill`、`fix-flash-attn`、`update-readme`）
- 如果远程已存在同名分支且与本次变更相关，复用该分支继续提交

### 规则 3：提交信息规范
- 使用 Conventional Commits 格式：`<type>: <description>`
- type 取值：`feat`（新功能）、`fix`（修复）、`docs`（文档）、`refactor`（重构）、`chore`（杂务）、`perf`（性能）、`test`（测试）
- description 用简洁英文描述变更内容，1-2 句话

## 执行指令

### 第一步：检查当前状态

并行执行以下命令：

```bash
git status -u
git diff --stat
git log --oneline -5
git branch --show-current
```

- 如果没有任何变更（无 modified、无 untracked），告知用户"没有需要提交的变更"并终止
- 记录当前分支名称，后续步骤需要用到

### 第二步：分析变更内容

1. 查看变更详情：
```bash
git diff
```

2. 如果有新文件，查看新文件列表确认内容

3. 根据变更内容：
   - 总结变更特征，生成不超过 20 字符的英文 summary（小写+连字符）
   - 确定 commit type（feat/fix/docs/refactor/chore/perf/test）
   - 草拟 commit message

### 第三步：分支管理

1. **检查当前分支**：
```bash
git branch --show-current
```

2. **如果当前在 master 或 main 分支**（必须切换）：
   - 生成目标分支名：`develop_<summary>`
   - 检查远程是否已有该分支：
     ```bash
     git ls-remote --heads origin develop_<summary>
     ```
   - 如果远程不存在，从当前分支创建并切换：
     ```bash
     git checkout -b develop_<summary>
     ```
   - 如果远程已存在，拉取并切换：
     ```bash
     git fetch origin develop_<summary>
     git checkout -b develop_<summary> origin/develop_<summary>
     ```

3. **如果当前已在 `develop_*` 分支**：
   - 直接使用当前分支，无需切换

4. **如果当前在其他非保护分支**：
   - 告知用户当前分支名称，询问是否继续在该分支提交或创建新的 develop 分支

### 第四步：暂存文件

1. **排除敏感文件**：以下文件绝不暂存
   - `.env`、`credentials.json`、`*token*`、`*secret*`
   - 任何包含凭据或密钥的配置文件

2. **按文件暂存**：
   - 对修改的已跟踪文件和新文件，按文件名逐个 `git add`
   - **不使用** `git add -A` 或 `git add .`

```bash
git add <file1> <file2> ...
```

3. 暂存后确认：
```bash
git status
```

### 第五步：提交

使用 HEREDOC 格式提交，确保消息格式正确：

```bash
git commit -m "$(cat <<'EOF'
<type>: <concise description>

<optional body: what changed and why>
EOF
)"
```

### 第六步：推送到远程

```bash
git push -u origin <branch-name> 2>&1
```

- 如果推送失败（认证问题），提示用户检查凭据配置或手动执行 push
- 如果推送失败（冲突），先 `git pull --rebase origin <branch>` 再重试

### 第七步：输出结果

推送成功后，输出以下信息：

```
- commit: <hash> — <commit message>
- branch: <branch-name>
- 变更: <文件数> 个文件，+<additions>/-<deletions> 行
- 远程: <remote-url>
```

如果是新分支的首次推送，提示用户可以在远程仓库平台创建 Pull Request / Merge Request。

## 注意事项

- 如果遇到网络问题（如内网环境需要代理访问外网仓库），请先执行 `setup-proxy` 配置代理
- 不要修改用户的 git 全局配置（`git config --global`）
- 如果用户明确指定了分支名，使用用户指定的名称（但仍需验证不是 master/main）
