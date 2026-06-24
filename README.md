# self-skills

My personal collection of reusable skills for AI coding agents (Claude Code, Codex, Cursor, and compatible tools). Built from daily workflows to reduce repetition and keep execution consistent across sessions.

---

## Directory

```
skills/<skill-name>/SKILL.md
```

```
skills/
├── gpu-training-optimizer/         # 17 sub-skills in _internals/
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

## GPU Training

| Skill | Source | Description |
|-------|--------|-------------|
| `gpu-training-optimizer` | — | End-to-end GPU training optimization: static analysis → hardware profiling → bottleneck diagnosis → code optimization → validation. Supports LlamaFactory, ms-swift, VideoX-Fun, Flow-Factory, HF Trainer, vLLM, SGLang. Contains 17 internal sub-skills. |
| `gpu-static-analysis` | — | Read-only static analysis for GPU training projects (no GPU required). Depends on sub-skills bundled with `gpu-training-optimizer`. |

## Research & Papers

| Skill | Source | Description |
|-------|--------|-------------|
| `paper-analyze` | — | Deep-dive analysis of a single paper with detailed notes and figures. |
| `paper-search` | — | Keyword, author, and field-based search across an organized paper notes library. |
| `extract-paper-images` | — | Extracts paper figures, preferring arXiv source packages when available. |
| `conf-papers` | — | Conference paper search and recommendations (CVPR, ICCV, ECCL, ICLR, AAAI, NeurIPS, ICML). |
| `start-my-day` | [MarsWang42/OrbitOS](https://github.com/MarsWang42/OrbitOS) | Generates a daily paper and news recommendation note. |
| `ai-research-advisor-framework` | — | Routes research questions to perspective agents (Ilya, Munger, Taleb, Karpathy, Feynman, Musk) and synthesizes their outputs. |

## Agent Workflow

| Skill | Source | Description |
|-------|--------|-------------|
| `copilot-rule` | — | Enforces conversation continuity — agent must suggest next steps after every completed task. |
| `multi-agent-council` | — | Multi-agent council inspired by OpenRouter Fusion. Sub-agents independently answer the same question; the lead synthesizes results. |
| `research-decision-gate` | — | Decision checkpoint for research work: GO / SMALL BET / REFRAME / STOP. Checks generality, innovation boundary, evaluation quality, and ROI. |
| `external-ai-brief` | — | Generates a self-contained decision context for consulting external AI that cannot access local files. |

## Resume & Career

| Skill | Source | Description |
|-------|--------|-------------|
| `resume-optimizer` | — | SOP for AI/algorithm engineering resumes (campus & internship). Converts raw notes into quantifiable, interview-ready bullet points. |
| `tex-resume-validator` | — | Full LaTeX resume edit loop: compile, lint, preview PDF, clean up after every `.tex` change. |

## Dev Tools

| Skill | Source | Description |
|-------|--------|-------------|
| `ssh-remote-access` | — | Reliably establishes SSH connections to Linux servers before remote code edits or experiments. |
| `setup-proxy` | — | Auto-detects and configures HTTP/HTTPS proxy for restricted network environments. |
| `git-commit` | — | Structured git commit workflow: branch → stage → commit → push, with Conventional Commits and safety rules. |

## Extras

| Skill | Source | Description |
|-------|--------|-------------|
| `web-access` | [eze-is/web-access](https://github.com/eze-is/web-access) | Full web access: search, scraping, CDP browser automation, social media capture. |
| `huashu-nuwa` | [alchaincyf/nuwa-skill](https://github.com/alchaincyf/nuwa-skill) | Distills notable thinkers' mental models into runnable AI skills (Jobs, Musk, Feynman, Taleb, etc.). |
| `pua` | [tanweai/pua](https://github.com/tanweai/pua) | High-agency mode with tiered corporate pressure dynamics (P7-P10), self-evolution, and more. MIT licensed. |

---

## Note

This is a personal skill collection maintained for my own use. Skills without a linked source are written or adapted by me. Community-sourced skills are mirrored here as backups — check their upstream repos for the latest versions.
