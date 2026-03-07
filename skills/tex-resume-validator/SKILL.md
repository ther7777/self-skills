---
name: tex-resume-validator
description: Specialized skill for compiling, validating, and iteratively refining LaTeX resumes. Use this skill whenever a .tex file is modified (content or layout), a LaTeX resume needs to be compiled, or PDF output needs to be verified. Automatically triggers after ANY edit to a .tex file in a resume workspace — enforce this even when the user only asks for a "small change".
---

# TeX Resume Validator — Compile, Validate, Iterate

This skill governs the full lifecycle of LaTeX resume editing: from source modification through PDF validation to clean workspace management.

## Core Workflow

Every `.tex` edit follows this fixed sequence:

```
Edit .tex → xelatex compile → Check log → Inspect PDF → Clean up → Report to user
```

Never skip a step. Compile immediately after every edit, even minor ones.

## Step 1 — Compile

Run xelatex in nonstopmode so it doesn't hang on errors:

```powershell
xelatex -interaction=nonstopmode <filename>.tex
```

Capture the exit code:
```powershell
if ($LASTEXITCODE -ne 0) { # enter error recovery flow }
```

## Step 2 — Validate Log Output

After compilation, check for:

| Signal | What to look for |
|--------|-----------------|
| **Fatal errors** | Lines starting with `!` — must fix before proceeding |
| **Missing packages** | `LaTeX Error: File '*.sty' not found` |
| **Font errors** | `Font ... not found`, missing `.ttf`/`.otf` files |
| **Overfull hboxes** | `Overfull \hbox` — warns of content overflowing margins |
| **Success** | `Output written on <file>.pdf (N pages)` |

## Step 3 — Validate PDF Layout

After a successful compile, mentally verify:
- Content does not overflow page margins
- Photo placeholder / framebox is correctly positioned
- Section spacing looks consistent
- No orphaned lines or widowed headings across pages

If the user hasn't provided feedback on layout yet, proactively ask.

## Error Recovery Logic

### Attempt < 3 — Auto-fix

Analyze the error type and fix directly without asking the user:

| Error Type | Fix Strategy |
|-----------|-------------|
| Missing `\usepackage` | Add the required package to preamble |
| Undefined control sequence | Check for typos in command names |
| Missing `}` or `{` | Scan nearby lines for unmatched braces |
| `\textbf` inside `\textbf` | Simplify nested bold commands |
| Font not found | Fall back to a known-available font or comment out font declaration |

Re-compile after each fix. Repeat up to 3 attempts.

### Attempt ≥ 3 — Escalate to User

If 3 consecutive compile attempts all fail, stop auto-fixing and report:
1. The exact error message from the log
2. The root cause diagnosis (missing file, package conflict, etc.)
3. Two or more alternative approaches for the user to choose from

## Step 4 — Clean Up

After a successful compile, remove intermediate files to keep the workspace clean:

```powershell
Remove-Item -Path "<basename>.aux","<basename>.log","<basename>.out" -ErrorAction SilentlyContinue
```

Only clean up after a **successful** compile. Retain `.log` on failure — it's needed for debugging.

## Reporting Format

After completing the compile-validate-cleanup cycle, report to the user in this format:

> **Compile result**: Success / Failed (attempt N)
> **Pages**: N
> **Warnings**: [list any significant overfull hboxes or font warnings, or "None"]
> **Cleaned up**: .aux .log .out

Then invoke `ask_questions` to proceed (per copilot-rule).
