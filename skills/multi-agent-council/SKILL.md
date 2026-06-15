---
name: multi-agent-council
description: 多 Agent 协作议会 - 主 agent 派遣多个 sub-agent 对同一问题独立给出完整答案，主 agent 担任裁判+合成器，输出对比分析后的最终答案。灵感来自 OpenRouter Fusion。适用于需要多视角、深度调研、避免单模型偏见的复杂问题。触发词：多模型协作、模型议会、多视角分析、llm council、fusion、议会讨论、几个 AI 一起讨论。
description_zh: 多 Agent 协作议会
description_en: Multi-Agent Council
disable: false
agent_created: true
---

# multi-agent-council

## When to use

当用户的提问满足以下任一条件时，激活本 skill：

- 明确要求"多模型协作/模型议会/council/fusion/多视角/几个 AI 一起讨论"
- 问题复杂度高，单一视角容易遗漏重要维度或陷入盲区
- 用户希望对比不同分析路径并保留分歧
- 需要做深度调研、文献综述、技术选型对比、方案权衡
- 任何"先独立思考再综合"性质的复杂任务

**不适合的场景：** 简单问答、单步操作、已经有明确答案的事实查询。

## 核心设计原则

**每个 sub-agent 独立完成完整答卷，不做分工。**

- ❌ **错误做法：** 把问题拆成 N 个子任务分给不同 sub-agent（"你查事实，你做分析，你给方案"）
  - 后果：答卷残缺 → 无法交叉验证 → 误差累积
- ✅ **正确做法：** 同一个问题完整交给 N 个 sub-agent，每个独立走"理解→调研→推理→结论"全流程
  - 主 agent 拿到 N 份**可比较的完整答卷**，能做真正的共识/分歧对比

## 参考依据：Fusion 博客的关键数据

**博客原文：** OpenRouter 的 Fusion 实验验证了三件事：

1. **合成步骤本身贡献最大收益。** Opus 4.8 + Opus 4.8（同一模型两份答卷，自融合）从 58.8% 升到 65.5%（**+6.7pp**）。博客明确说："This suggests that a meaningful chunk of Fusion's lift comes from the synthesis step itself, not just from combining different model architectures." 
2. **模型多样性有增量贡献。** Opus 4.8 + GPT-5.5（跨模型）得分 67.6%，比自融合的 65.5% 再高 **+2.1pp**。博客说："We believe this demonstrates the benefits of model diversity... Bringing multiple different perspectives to complex problems yields superior results."
3. **自融合已经拿到大部分收益。** +6.7pp 的自融合提升占跨模型总提升（+8.8pp）的 **76%**。说明多跑几次 + 对比合成比换模型更重要。

**博客中提及的模型：** Claude Fable 5、Claude Opus 4.8、GPT-5.5、Gemini 3.1 Pro、Gemini 3 Flash、Kimi K2.6、DeepSeek V4 Pro。

**博客未提及的概念：** subagent_type 差异、工具集差异、不同训练数据/RLHF/对齐——这些都不是博客中提到的多样性来源。所有 panel 模型统一使用 web_search + web_fetch。

## Steps

### 步骤 0：选择模型（激活后立即执行）

激活本 skill 后，**第一时间询问用户**：选择参与议会的 sub-agent 模型。

使用 `AskUserQuestion` 工具，提供预设方案和自定义选项：

**问题 1：每个 sub-agent 用什么模型？**

默认配置（推荐，一键确认）：

| 议员 | model | 来源 |
|---|---|---|
| 议员 A | `kimi-k2.6` | Moonshot |
| 议员 B | `deepseek-v4-pro` | DeepSeek |
| 议员 C | `MiniMax-M3` | MiniMax |

> 组合灵感来自 Fusion 平价面板（Kimi K2.6 + DeepSeek V4 Pro + Gemini 3 Flash 得分 64.7%，接近 Fable 5 的 65.3%，成本减半）。适配当前环境用 MiniMax M3 替代 Gemini Flash。
>
> `model` 字段支持三类取值：(1) 内置/自定义模型别名（取决于当前 CodeBuddy 版本和 `~/.codebuddy/models.json` 配置，例如 `gpt-5.1-codex` / `gemini-3.1-pro` 等）；(2) `"inherit"` — 继承主会话当前模型；(3) 省略不写 — 回退到默认 sub-agent 模型（`CODEBUDDY_CODE_SUBAGENT_MODEL` 环境变量或 agent 配置 `models[0]`）。

备选方案：

| 预设 | 描述 |
|---|---|
| 默认（推荐） | Kimi K2.6 + DeepSeek V4 Pro + MiniMax M3 |
| 当前模型 ×3 | 3 个 sub-agent 都用 current model（model 传 `"inherit"` 或省略不写），纯靠多次采样 + 合成获取收益 |
| 自定义 | 手动输入每个 sub-agent 的 model 标识符（内置别名 / `inherit` / 在 `~/.codebuddy/models.json` 中自定义） |

**问题 2：参与议会的成员数量？**
- 2 个（最少，已有合成收益）
- 3 个（推荐）
- 4 个（更多视角，但成本翻倍）

用户选择后，立即进入阶段 1。

---

### 阶段 1：分派任务（必须并行）

**在同一条消息中**发起对应数量的 `Agent` 工具调用。

**关键约束：每个 sub-agent 收到的 prompt 必须包含完整的原始问题 + 完整的工作要求，不做任务拆分。**

**subagent_type 选择：**
- `general-purpose` 通用性最好，推荐作为默认
- `Explore` 偏搜索/调研，适合信息密集型问题
- 所有 sub-agent 统一使用 `general-purpose` 即可，博客未涉及 subagent_type 差异

**示例 prompt 模板（原样发给每个 sub-agent）：**

```
请独立、完整地回答以下问题。

【问题】
<用户原始问题，逐字保留>

【要求】
- 独立完成"理解问题 → 调研/查证 → 推理 → 得出结论"全流程
- 可以自由使用 WebSearch / WebFetch / Read / Bash 等工具
- 工具调用次数不限，直到你对自己的答案有信心
- 不需要参考其他 agent 的输出

【输出格式】
## 我的结论
（明确的最终答案或立场）

## 论证过程
（3-5 条具体证据/数据/案例支撑）

## 不确定的地方
（如实标注，包括可能的错误）
```

### 阶段 2：对比分析（主 agent 担任 Judge）

收集到所有答卷后，整理对比：

```markdown
### 共识
- [所有答卷都同意的结论 1] — 强信号
- [多数同意的结论 2] — 中等信号

### 分歧
| 争议点 | A 立场 | B 立场 | C 立场 |
|---|---|---|---|

### 独特洞见
- **X 提到**：独有的角度/证据

### 盲区
- [所有答卷都遗漏的维度]

### 事实核验
- [如果有答卷之间存在事实矛盾，交叉验证]
```

### 阶段 3：综合输出（主 agent 担任 Synthesizer）

基于对比分析撰写最终答案：

1. 优先呈现共识（高置信度）
2. 明确标注分歧（不回避、不平均化）
3. 核验事实矛盾（交叉验证后再下结论）
4. 补充遗漏维度（发现盲区纳入答案）
5. 保留独特洞见（不让其丢失）
6. 不要全量展示原始答卷（用户只需要综合后的答案）

## Pitfalls

- ❌ **把问题拆给 sub-agent 分工**——这是流水线思维不是议会思维。后果：答卷残缺、误差累积
- ❌ **串行调用 sub-agent**：必须在同一条消息发起多个 Agent 调用
- ❌ **主 agent 偷懒**：直接转述某个 sub-agent 的输出，没有真正综合
- ❌ **平均化所有观点**：为了"客观"把所有意见中和，丢失立场和洞见
- ❌ **绕过对比直接出结果**：跳过阶段 2 的分析直接到阶段 3
- ❌ **限制 sub-agent 工具调用次数**：不限制，让 sub-agent 自主调研

## Verification

- [ ] 步骤 0：是否询问了用户选择模型？
- [ ] 阶段 1：≥2 个并行 Agent 调用？model 参数是否按用户选择填入？
- [ ] 每个 sub-agent 收到了完整的原始问题？
- [ ] 每个 sub-agent 给出了完整答卷（理解→调研→结论）？
- [ ] 最终答案引用了具体证据？
- [ ] 分歧被明确标注而非回避？
- [ ] 对事实矛盾做了交叉验证？

## Notes

- 不依赖外部 API，全程 Agent 环境内完成
- 合成步骤是主要收益来源（参考 Fusion 自融合的 +6.7pp），模型多样性是增量（+2.1pp）。即使 sub-agent 全跑同一模型，多次采样 + 对比合成仍然有价值。
- 主 agent 既是 Judge 也是 Synthesizer
- 灵感来自 OpenRouter Fusion 博客 https://openrouter.ai/blog/announcements/fusion-beats-frontier/
