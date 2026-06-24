# self-skills

个人 AI 编程助手技能（Claude Code、Codex、Cursor 等兼容），工作流中积累复用，自己日常使用。

---

## 目录

```
skills/<skill-name>/SKILL.md
```

```
skills/
├── gpu-training-optimizer/         # 含 17 个 _internals/ 子技能
├── gpu-static-analysis/
├── paper-analyze/
├── paper-search/
├── extract-paper-images/
├── conf-papers/
├── start-my-day/
├── ai-research-advisor-framework/
├── multi-agent-council/
├── research-decision-gate/
├── external-ai-brief/
├── copilot-rule/
├── resume-optimizer/
├── tex-resume-validator/
├── ssh-remote-access/
├── setup-proxy/
├── git-commit/
├── web-access/
├── huashu-nuwa/
└── pua/
```

---

## GPU 训练优化

| 技能 | 来源 | 说明 |
|-------|------|------|
| `gpu-training-optimizer` | — | 端到端 GPU 训练智能调优：静态分析 → 硬件采集 → 性能剖析 → 瓶颈诊断 → 代码优化 → 验证迭代。支持 LlamaFactory、ms-swift、VideoX-Fun、Flow-Factory、HF Trainer、vLLM、SGLang 等框架。内含 17 个内部子技能。 |
| `gpu-static-analysis` | — | GPU 训练项目纯静态代码分析，无需 GPU。依赖 `gpu-training-optimizer` 内的子技能。 |

## 论文与科研

| 技能 | 来源 | 说明 |
|-------|------|------|
| `paper-analyze` | — | 深度分析单篇论文，生成详细笔记与图表。 |
| `paper-search` | — | 在已有论文笔记中按关键词、作者、领域搜索。 |
| `extract-paper-images` | — | 从论文中提取图片，优先使用 arXiv 源码包。 |
| `conf-papers` | — | 顶会论文搜索推荐（CVPR、ICCV、ECCV、ICLR、AAAI、NeurIPS、ICML）。 |
| `start-my-day` | [MarsWang42/OrbitOS](https://github.com/MarsWang42/OrbitOS) | 生成每日论文与资讯推荐笔记。 |
| `ai-research-advisor-framework` | — | 将研究问题路由到多视角子代理（Ilya、芒格、塔勒布、Karpathy、费曼、马斯克），汇总输出。 |

## Agent 工作流

| 技能 | 来源 | 说明 |
|-------|------|------|
| `copilot-rule` | — | 对话连续性规则：每完成一项任务后主动给出下一步选项。 |
| `multi-agent-council` | — | 多 Agent 协作议会，受 OpenRouter Fusion 启发。多个子代理独立回答同一问题，主代理对比合成。 |
| `research-decision-gate` | — | 研究决策检查点：GO / SMALL BET / REFRAME / STOP。检查泛化性、创新边界、评估可信度、ROI。 |
| `external-ai-brief` | — | 生成自包含决策上下文，供无法访问本地文件的外部 AI 使用。 |

## 简历与求职

| 技能 | 来源 | 说明 |
|-------|------|------|
| `resume-optimizer` | — | AI/算法岗（校招/实习）简历优化 SOP，将原始笔记转化为可量化的面试要点。 |
| `tex-resume-validator` | — | LaTeX 简历编辑闭环：编译 → 查错 → 预览 PDF → 清理临时文件。 |

## 开发工具

| 技能 | 来源 | 说明 |
|-------|------|------|
| `ssh-remote-access` | — | 远程 SSH 连接建立与验证，修改代码或运行实验前的准备流程。 |
| `setup-proxy` | — | 自动检测并配置 HTTP/HTTPS 代理，适配受限网络环境。 |
| `git-commit` | — | 结构化 Git 提交流程：分支管理 → 暂存 → 提交 → 推送，遵循 Conventional Commits 规范。 |

## 其他

| 技能 | 来源 | 说明 |
|-------|------|------|
| `web-access` | [eze-is/web-access](https://github.com/eze-is/web-access) | 联网搜索、网页抓取、CDP 浏览器自动化、社交媒体内容抓取。 |
| `huashu-nuwa` | [alchaincyf/nuwa-skill](https://github.com/alchaincyf/nuwa-skill) | 女娲造人：将名人/角色的思维方式蒸馏为可运行的 AI 技能（Jobs、Musk、Feynman、Taleb 等）。 |
| `pua` | [tanweai/pua](https://github.com/tanweai/pua) | 高能动性模式，含 P7-P10 职级模式及多种子模式。MIT 协议。 |

---

## 说明

本仓库为个人使用维护。未标注来源的技能为自写或自行适配。来自社区的技能备份于此，更新请以原仓库为准。
