---
name: resume-optimizer
description: Business-oriented SOP for optimizing algorithm/AI engineering resumes targeting top-tier tech company campus recruiting or internship positions. Use whenever a user wants to write, rewrite, review, or polish a resume entry — whether it's a project, a paper, or a competition. Also trigger when the user provides raw project notes, interview reflections, paper drafts, competition summaries, or draft bullet points and wants to turn them into compelling resume content. Trigger proactively when the user shares a .tex resume file and asks for feedback, edits, or review.
---

# Resume Optimizer — Algo/AI Engineering (Campus & Internship)

This skill encodes proven SOP for writing resume entries for **algorithm / AI engineering** roles at tier-1 tech companies. It covers three entry types with distinct formats: **projects**, **research papers**, and **competitions**.

The underlying principle: a resume entry should be a **structured answer to the unspoken interview question** — "What problem did you tackle, how did you solve it, and what specifically did you contribute?" Every entry should make the reviewer feel *you* were the key person, not just a bystander.

---

## Entry Types and When to Use Each

| Type | Use when | Key accent |
|------|----------|------------|
| **Project** | Internship/job engineering deliverables | Business problem → your solution → impact metrics |
| **Paper** | Academic or industry research, submitted/published | Technical contribution → your specific modules → venue + results |
| **Competition** | Hackathons, shared tasks (Kaggle, etc.) | Task framing → your method → ranking |

---

## SOP 1: Project Entries

### Format

```
\item \textbf{[Deidentified Project Name]}: [One-sentence context: business scenario, scale, core challenges]
  \begin{itemize}
    \item \textbf{[Module/Contribution Area]}: [Solution approach] + [Why this specific approach/highlights] + [Key metrics/results]
    \item \textbf{[Module/Contribution Area]}: ...
  \end{itemize}
```

### Rules

**Background Line (1st line)**:
- One sentence covering: Business context + Scale/Volume (e.g., 10M+ users/100B+ data) + Core challenges (max 2).
- Use "Designed and implemented..." instead of "Responsible for...", showing ownership.
- Deidentify project names (e.g., "Province X Medical Insurance System" instead of "Harbin Medical Insurance").

**Bullets (2-3 items)**:
- Each bullet = Solution highlight (architecture/strategy/tech used + *why*) + Quantifiable metrics.
- Avoid listing technical keywords; explain how the solution solved a problem the previous version couldn't.
- Numbers drive credibility: Accuracy/Recall from X% to Y%, Latency reduced by Z%, etc.

**Anti-patterns (Avoid)**:
- ❌ "Participated in building..." — Too weak; use "Architected", "Lead development", or "Core developer".
- ❌ Detail dumping: Listing every minor tech stack without priority.
- ❌ Result-less bullets: Bullets without numbers cannot be validated.

### Example (Deidentified)

```
\item \textbf{High-Concurrency Intent Recognition System}: Core developer for an intelligent assistant covering 300+ business categories, handling 10M+ daily requests with challenges in ASR errors and colloquial intent boundaries.
  \begin{itemize}
    \item \textbf{Cascaded LLM-SLM Architecture}: Designed a cascaded routing architecture (BERT for speed + 35B LLM for fallback); optimized SFT data strategy; overall accuracy 94%, inference latency reduced by 65%.
    \item \textbf{Multi-turn Intent Tracking}: Implemented sliding-window state tracking for context retention; integrated BPM flow constraints with LLM execution, achieving 96% task completion rate.
  \end{itemize}
```

---

## SOP 2: Research Paper Entries

### Format

```
\item \textbf{[Full Paper Title]} \hfill {\small \textit{[Conference/Journal] ([CCF Rank] [Status])}}\\
  {\small \textit{[Author 1]*, [Author 2]* (*Equal contribution)} \quad [Context: Internship/Lab project, relationship]}
  \begin{itemize}
    \item {\small [Problem definition: Targeted XX bottleneck to propose XX framework (core idea)].}
    \item {\small \textbf{Core Contributions}: [Designed Module A (Methodology)] + [Implemented System B (Engineering)]; [Key Results].}
  \end{itemize}
```

### Rules

**Header Line**:
- Use full title, venue abbreviation + CCF Rank (A/B/C/Top) + Status (Submitted/Under Review/Published).
- CCF ranking is critical for quick screening by reviewers.

**Author Line**:
- List yourself first, mark equal contributions with *, specify the relationship (e.g., "Co-first author with mentor").
- Stating the context (Internship vs. Lab) highlights different skill sets.

**Bullets (using \small font)**:

Bullet 1 = **Problem + Core Idea** (1-2 sentences)
- Explicitly state: What specific gap/bottleneck was addressed?
- Explicitly state: What is the core methodological novelty?

Bullet 2 = **Core Contributions** (Crucial part)
- Distinguish between two dimensions:
  - **Methodological Design**: Modules you designed to solve specific sub-problems.
  - **Engineering/Deployment**: Distributed training, optimization, or scale (e.g., "Using Ray/vLLM for 100B model distillation").
- End with key experimental results (vs. SOTA metrics).

### Example (Deidentified)

```
\item \textbf{FRAMEWORK: Method for Controllable LLM Reasoning} \hfill {\small \textit{EMNLP 2026 Under Review (CCF A)}}\\
  {\small \textit{Applicant*, Mentor* (*Equal contribution)} \quad Internship project at Tier-1 Lab}
  \begin{itemize}
    \item {\small Addressed LLM hallucinations in structured reasoning by proposing a dual-track decoupling framework for semantic and fine-grained constraint control.}
    \item {\small \textbf{Core Contributions}: Designed a 4-stage linguistic Chain-of-Thought (CoT) for task解耦; implemented a process-level reward model (PRM) for DPO/GRPO training on 8*H800 cluster. Achieved SOTA results on GSM8K/WMT, exceeding GPT-4o by +4.2 points.}
  \end{itemize}
```

---

## SOP 3: Competition Entries

### Format

```
\item \textbf{[Competition Name] -- [Task Description]} \hfill \textbf{[Rank: Champion / Top-X]}\\
  {\small \textit{[Role: Core Contributor, Tech Report/Paper Pending]}}\\
  {\small [Challenge context]; [Core solution novelty]; [Metrics + Lead over 2nd place]}
```

### Rules

**Header Line**:
- Right-align and bold the rank; this is the primary signal.
- Use the formal competition name and a brief task description.

**Role Line**:
- Specify your role: Lead developer? Core contributor? Did you write the tech report?

**Content (Concise, 2-3 sentences)**:
1. Task difficulty (Set the context).
2. Winning logic (Why did your method win? High-level strategy).
3. Leading margins (e.g., "Outperformed 2nd place by 7pp").

### Example (Deidentified)

```
\item \textbf{Global Shared Task 2026 -- Topic Classification} \hfill \textbf{Grand Champion}\\
  {\small \textit{Core Contributor, Tech Report submitted to NeurIPS Competition Track}}\\
  {\small Addressed low inter-annotator agreement (40.6%) and ambiguous class boundaries; designed a CAMSR-COT framework using Gate-routing for confidence-based error correction. Achieved F1=0.89, leading the runner-up by 7 percentage points.}
```

---

## Overall Structure Advice

Recommend organizing sections by **output type** rather than by affiliation:

```
Education
Research Papers    ← All papers, regardless of lab/internship
Work Experience    ← Projects only
Competitions       ← Separate section
Skills & Others    ← Coding, Open source, etc.
```

---

## General Writing Principles

**Ownership over Participation**
- ❌ `The system achieved...`
- ✅ `Designed and deployed XX architecture, improving...`

**Quantify Everything**
An entry without numbers is just an opinion. Every core bullet must have at least one metric.

**One-Page Rule**
If space is limited, sacrifice adjectives ("innovative", "advanced") and minor tech details before sacrificing metrics or core modules.

---

## Execution Checklist for Copilot

1. **Classify entry type**: Project, Paper, or Competition?
2. **Deidentify names**: Ensure no internal project/company names are exposed.
3. **Draft using SOP format**: Follow the LaTeX boilerplate provided.
4. **Validation**:
   - [ ] Is there at least one metric?
   - [ ] Is the owner-role clear?
   - [ ] Is the venue/rank prominent?
   - [ ] Is the language professional?
