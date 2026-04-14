# self-skills

这是一个用于沉淀个人工作流的 Claude Code / Codex Skills 仓库。

这些技能模板来自可复用的实战流程总结，目标是：

- 减少重复劳动；
- 降低遗漏关键步骤的概率；
- 让 Claude / Codex 在不同任务中保持一致、稳定的执行节奏。

---

## 目录结构

所有 skill 统一放在 `skills/<skill-name>/` 目录下，每个 skill 的核心入口为 `SKILL.md`。

```
skills/
├── copilot-rule/              # 自己原创
├── tex-resume-validator/      # 自己原创
├── resume-optimizer/          # 自己原创
├── ssh-remote-access/         # 自己原创
├── web-access/                # 下载自社区
├── paper-analyze/             # OrbitOS 生态（来源见下）
├── paper-search/              # OrbitOS 生态（来源见下）
├── extract-paper-images/      # OrbitOS 生态（来源见下）
├── conf-papers/               # OrbitOS 生态（来源见下）
├── start-my-day/              # 下载自 MarsWang42/OrbitOS
├── ai-research-advisor-framework/  # 来源待确认
├── huashu-nuwa/               # 下载自 alchaincyf/nuwa-skill
└── pua/                       # 下载自 tanweai/pua（含所有人格子 skill）
```

---

## 1. 自己原创 / 个人适配的技能

这些技能完全由我个人编写或针对本地环境（如 3090 服务器）深度定制，无外部上游仓库。

| Skill | 说明 |
|-------|------|
| `copilot-rule` | 对话连续性规则。每个阶段结束后主动给出下一步选项，避免任务完成后对话突然中断。 |
| `tex-resume-validator` | LaTeX 简历编辑闭环。任何 `.tex` 改动后，按固定流程执行编译、查错、预览 PDF、清理临时文件并汇报结果。 |
| `resume-optimizer` | 算法/AI 工程岗位（校招/实习）简历优化 SOP。将原始笔记、逐字稿或项目描述提炼为可量化、可追问、可面试展开的简历要点。 |
| `ssh-remote-access` | 远程 SSH 访问流程。在修改远程代码或运行实验前，先建立并验证到 Linux 服务器的连接与操作链路。 |
| `connect-3090` | 连接本地 3090 GPU 服务器的快捷 skill，包含路径跳转、环境检查等私有配置。**（尚未同步到本仓库）** |

---

## 2. 从社区下载的技能（来源已确认）

以下技能来自 GitHub 或公开社区，已备份到本仓库。如需更新，请直接访问原始项目地址：

### 2.1 联网与浏览器自动化

| Skill | 原始项目地址 | 说明 |
|-------|-------------|------|
| `web-access` | https://github.com/eze-is/web-access | 一泽Eze 开发的完整联网 skill。支持搜索、网页抓取、CDP 浏览器自动化、社交媒体内容抓取等。 |

### 2.2 论文阅读与研究工作流（OrbitOS 官方）

| Skill | 原始项目地址 | 说明 |
|-------|-------------|------|
| `start-my-day` | https://github.com/MarsWang42/OrbitOS | MarsWang42 的 OrbitOS 系统中的「启动一天」skill，生成今日论文/资讯推荐笔记。 |

### 2.3 人格化 / 高能动性引擎

| Skill | 原始项目地址 | 说明 |
|-------|-------------|------|
| `pua` | https://github.com/tanweai/pua | OpenPUA 官方仓库。包含核心 PUA 模式、P7/P8/P9/P10 职级模式、`mama`（妈妈唠叨）、`yes`（夸夸模式）、`pro`（自进化/KPI）、`shot`（浓缩版）、`pua-loop`（自动迭代）等全部子 skill 及命令。 |
| `huashu-nuwa` | https://github.com/alchaincyf/nuwa-skill | 花叔（alchaincyf）的「女娲 · Skill 造人术」。将名人/角色的思维方式蒸馏为可运行的 Claude Code skill，内含 Jobs、Musk、Feynman、Taleb 等多个示例 perspective。 |

---

## 3. 论文相关技能（OrbitOS 生态 / 来源待确认）

以下 4 个 skill 内部大量引用 OrbitOS 的 vault 路径与命名规范（如 `$OBSIDIAN_VAULT_PATH/20_Research/Papers/...`），但**在 MarsWang42/OrbitOS 官方仓库中未找到同名的独立 skill 目录**。它们可能是：

1. 其他开发者基于 OrbitOS 风格二次创作的社区 skill；
2. 我个人根据 OrbitOS 规范本地化适配后的版本；
3. 来自 Lobehub 等 skill 市场但已被下架或未索引的 skill。

无论来源如何，它们都与 `start-my-day` 共享同一套 Obsidian vault 结构，属于同一工作流生态。

| Skill | 说明 | 备注 |
|-------|------|------|
| `paper-analyze` | 深度分析单篇论文，生成图文并茂的详细笔记和评估。 | 基于 OrbitOS 风格 |
| `paper-search` | 在已整理的论文笔记库中按关键词、作者、领域搜索。 | 基于 OrbitOS 风格 |
| `extract-paper-images` | 从论文中提取图片，优先从 arXiv 源码包获取真正的论文图。 | 基于 OrbitOS 风格 |
| `conf-papers` | 顶会论文搜索推荐（CVPR/ICCV/ECCV/ICLR/AAAI/NeurIPS/ICML）。 | 基于 OrbitOS 风格 |

> **如果你有这些 skill 的原始 GitHub/市场地址，欢迎补充到本 README。**

---

## 4. 其他来源待确认的技能

| Skill | 说明 | 备注 |
|-------|------|------|
| `ai-research-advisor-framework` | AI 研究顾问：自动判断用户问题是「方向选择」还是「实验设计」，并行调度 Ilya/芒格/塔勒布/Karpathy/费曼/马斯克 等子 agent 分析后整合输出。 | 暂未找到公开上游仓库，可能来自中文社区分享或个人整理。 |

---

## 更新维护建议

- **自己原创的技能**：直接在本仓库修改，无需担心上游冲突。
- **来源已确认的社区 skill**：建议定期 `git pull` 原始仓库或关注 release，再将改动同步回本仓库备份。
- **OrbitOS 生态 / 来源待确认 skill**：如后续找到原始出处，请更新本 README 并将链接补入对应表格。

---

## License

- 自己原创 skill：按个人使用自由分发。
- 下载自社区的 skill：遵循各自原始仓库的 License（如 `web-access` 为 MIT，`pua` 为 MIT，`nuwa-skill` 按原仓库协议）。
