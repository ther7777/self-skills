# self-skills

Personal collection of reusable skills for AI coding agents (Claude Code, Codex, Cursor, and compatible tools).

Each skill encodes a repeatable workflow — from GPU training optimization to paper analysis — to reduce manual repetition and ensure consistent execution across sessions.

---

## Directory

All skills live under `skills/<skill-name>/`, with `SKILL.md` as the entry point.

```
skills/
├── copilot-rule/                # Original
├── tex-resume-validator/        # Original
├── resume-optimizer/            # Original
├── ssh-remote-access/           # Original
├── external-ai-brief/           # Original
├── research-decision-gate/      # Original
├── multi-agent-council/         # Original
├── gpu-training-optimizer/      # Original (17 sub-skills in _internals/)
├── gpu-static-analysis/         # Original (depends on gpu-training-optimizer)
├── web-access/                  # Community
├── huashu-nuwa/                 # Community
├── pua/                         # Community
├── start-my-day/                # Community (OrbitOS ecosystem)
├── paper-analyze/               # OrbitOS ecosystem
├── paper-search/                # OrbitOS ecosystem
├── extract-paper-images/        # OrbitOS ecosystem
├── conf-papers/                 # OrbitOS ecosystem
└── ai-research-advisor-framework/  # Source unconfirmed
```

---

## Original Skills

Skills created or significantly adapted for personal workflows.

| Skill | Description |
|-------|-------------|
| `copilot-rule` | Enforces conversation continuity — the agent must suggest next steps after every completed task, preventing silent drop-offs. |
| `tex-resume-validator` | Full LaTeX resume edit loop: compile, lint, preview PDF, and clean up artifacts after any `.tex` change. |
| `resume-optimizer` | SOP for polishing AI/algorithm engineering resumes (campus & internship). Converts raw notes into quantifiable, interview-ready bullet points. |
| `ssh-remote-access` | Reliably establishes and verifies SSH connections to Linux servers before editing remote code or running experiments. |
| `connect-3090` | Quick-connect shortcut for a local 3090 GPU server. Contains private paths and environment checks. *(Not synced to this repo.)* |
| `external-ai-brief` | Generates a self-contained decision context for consulting external AI that cannot access local files. |
| `research-decision-gate` | Research decision checkpoint: GO / SMALL BET / REFRAME / STOP. Audits generality, innovation boundary, evaluation credibility, and ROI before committing more work. |
| `multi-agent-council` | Multi-agent council inspired by OpenRouter Fusion. Multiple sub-agents independently answer the same question; the lead agent synthesizes consensus, flags disagreements, and delivers the final result. |
| `gpu-training-optimizer` | End-to-end GPU training optimization: static analysis → hardware profiling → bottleneck diagnosis → code optimization → validation. Supports LlamaFactory, ms-swift, VideoX-Fun, Flow-Factory, HF Trainer, vLLM, and SGLang. Includes 17 sub-skills under `_internals/`. |
| `gpu-static-analysis` | Read-only static analysis for GPU training projects (no GPU required). Produces an optimization report with concrete configuration and code suggestions. Requires `gpu-training-optimizer/_internals/` sub-skills to be installed. |

---

## Community Skills

Skills sourced from public repositories, mirrored here as backups. Check upstream projects for the latest versions.

### Browser & Network

| Skill | Source | Description |
|-------|--------|-------------|
| `web-access` | [eze-is/web-access](https://github.com/eze-is/web-access) | Full web access skill: search, scraping, CDP browser automation, social media capture. |

### Agent Personality & Agency

| Skill | Source | Description |
|-------|--------|-------------|
| `pua` | [tanweai/pua](https://github.com/tanweai/pua) | High-agency mode with corporate pressure dynamics. Includes tiered modes (P7-P10), `mama` nag mode, `pro` self-evolution, and more. MIT licensed. |
| `huashu-nuwa` | [alchaincyf/nuwa-skill](https://github.com/alchaincyf/nuwa-skill) | Distills the thinking patterns of notable figures into runnable AI skills. Ships with perspectives from Jobs, Musk, Feynman, Taleb, and others. |

---

## OrbitOS Ecosystem Skills

These skills follow the [OrbitOS](https://github.com/MarsWang42/OrbitOS) workflow conventions and share its Obsidian vault structure. The paper-related skills (`paper-analyze`, `paper-search`, `extract-paper-images`, `conf-papers`) were not found as standalone directories in the official OrbitOS repository and may be community adaptations or local customizations.

| Skill | Description |
|-------|-------------|
| `start-my-day` | Generates a daily paper and news recommendation note. From the official OrbitOS distribution. |
| `paper-analyze` | Deep analysis of a single paper with detailed notes and figures. |
| `paper-search` | Searches the organized paper notes library by keyword, author, or field. |
| `extract-paper-images` | Extracts images from papers, preferring arXiv source packages over PDF snapshots. |
| `conf-papers` | Top-tier conference paper search (CVPR, ICCV, ECCV, ICLR, AAAI, NeurIPS, ICML). |

---

## Source Unconfirmed

| Skill | Description |
|-------|-------------|
| `ai-research-advisor-framework` | Routes research questions to the right perspective agents (Ilya, Munger, Taleb, Karpathy, Feynman, Musk), then synthesizes their outputs. Origin unknown — likely a community share or personal adaptation. |

---

## Maintenance

- **Original skills**: Edit directly in this repository.
- **Community skills**: Pull from upstream periodically for fixes and features, then sync back to this backup.
- **OrbitOS / unconfirmed skills**: If you recognize the original source of any uncredited skill, contributions to update this README are welcome.

---

## License

- Original skills in this repository are free for personal use.
- Community skills follow their respective upstream licenses (e.g., `web-access` is MIT, `pua` is MIT).
