---
name: research-decision-gate
description: "Use before research or engineering-research decisions: deciding whether to continue a direction, run an experiment, change route, interpret results, write a paper claim, or invest more work when there is risk of local optimization, self-drawn targets, weak evaluation, LLM self-evaluation loops, system-specific bug fixes disguised as innovation, or poor ROI. Produces a concise decision gate covering generality, innovation boundary, evaluation credibility, ROI and stop conditions, paper narrative fit, next action, and what not to do."
---

# Research Decision Gate

Use this as a pre-decision gate, not as a brainstorming mode. Be skeptical, concise, and willing to stop work.

## Procedure

1. Start with one verdict: `GO`, `SMALL BET`, `REFRAME`, or `STOP`.
2. Test generality: decide whether the issue remains after removing project-specific bugs, prompt artifacts, data quirks, implementation defects, and accidental constraints.
3. Set the innovation boundary: name the strongest defensible claim and the claims that would be overreach.
4. Audit evaluation: check gold source, baselines, negative examples, leakage, judge independence, metric gaming, and whether the result would convince someone outside this project.
5. Check ROI: every proposed experiment must change a real decision; define the cheapest decisive test and the stop condition before running it.
6. Check narrative fit: decide whether the result supports the paper or product thesis, or only improves a local subsystem.
7. Give one next action and one explicit non-action.

## Output Format

**一句话判断**: `GO / SMALL BET / REFRAME / STOP` + one sentence.

**通用性判断**: Is this a problem for a class of RAG or AI systems, or mainly our implementation bug?

**创新边界**: What claim is defensible? What claim is forbidden?

**评测风险**: What could make the evaluation circular, unfair, leaky, or self-drawn?

**ROI/止损**: What is the cheapest decisive test? What result stops this line?

**论文叙事位置**: Which paper claim, section, or ablation would this support?

**下一步动作**: One concrete action to do now.

**不该做什么**: One tempting action to avoid.

## Red Flags

- The claim disappears after fixing one project-specific bug.
- The baseline is weaker because it lacks equivalent evidence, prompts, tools, retrieval budget, or tuning effort.
- Negative examples are produced by the same method being evaluated without independent audit.
- LLM labels are used as both gold and judge.
- More experiments only improve a local metric without changing the paper claim.
- The work can only be explained as "our system had this issue", not "this class of systems has this failure mode".

## Decision Bias

Prefer `REFRAME` when the phenomenon is real but the current claim is too broad.

Prefer `SMALL BET` when evaluation is weak but upside is high and cost is low.

Prefer `STOP` when the next experiment cannot change the decision.
