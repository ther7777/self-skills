# self-skills

这是一个用于沉淀个人工作流的 Codex Skills 仓库。

这些技能模板来自可复用的实战流程总结，目标是：

- 减少重复劳动；
- 降低遗漏关键步骤的概率；
- 让 Codex 在不同任务中保持一致、稳定的执行节奏。

## 已收录技能

- `skills/copilot-rule`：对话连续性规则。每个阶段结束后主动给出下一步选项，避免任务完成后对话突然中断。
- `skills/tex-resume-validator`：LaTeX 简历编辑闭环。任何 `.tex` 改动后，按固定流程执行编译、查错、预览 PDF、清理临时文件并汇报结果。
- `skills/resume-optimizer`：算法/AI 工程岗位（校招/实习）简历优化 SOP。将原始笔记、逐字稿或项目描述提炼为可量化、可追问、可面试展开的简历要点。
- `skills/ssh-remote-access`：远程 SSH 访问流程。在修改远程代码或运行实验前，先建立并验证到 Linux 服务器的连接与操作链路。

## 使用说明

- 本仓库为“自用流程沉淀”，不保证适配所有团队或项目，请按需裁剪。
- 目录结构遵循 `skills/<skill-name>/SKILL.md`。