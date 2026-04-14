# Experiment Design Framework

Use this reference when the user's question is about how to validate an idea, design an experiment, set up baselines, or debug an algorithmic improvement.

## Core Principle

Before giving advice, internally route through three lenses: Karpathy (engineering realism), Feynman (scientific validation), Musk (first principles / removal). Output the synthesis in a structured format.

---

## Lens 1: Karpathy — Engineering Realism & Minimum Viable Experiment

**Key question**: What's the fastest, most engineering-grounded validation path?

**Checklist**:
- **Minimum viable experiment**: Can you get a preliminary result in 1-2 days?
- **Data check**: The first step is never touching model code — it's thoroughly checking the data
- **Benchmark selection**:
  - Don't let average performance hide tail problems
  - Sort dataset by loss descending — "you are guaranteed to find something surprising and useful"
- **Code / implementation**: Can it be quickly reproduced? Is there an open-source baseline?
- **March of Nines**: How will this experiment perform on the hardest 5% of cases?

**Warning signals**:
- Experiment design is too complex with many prerequisites → simplify first
- Only tested average case, not tail behavior → missing engineering realism
- Data unchecked, model trained directly → classic mistake

---

## Lens 2: Feynman — Scientific Validation & Anti-Self-Deception

**Key question**: What core assumption is this experiment actually testing? Am I fooling myself?

**Checklist**:
- **Core hypothesis**:
  - What's the null hypothesis?
  - If the result is negative, which core belief of mine would be falsified?
- **Cargo cult detection**:
  - Does this experiment have scientific form without scientific substance?
  - If you strip away all implementation complexity, does the core logic still hold?
- **Naming ≠ understanding**:
  - Can you explain what this experiment does without jargon, in one sentence?
  - If someone else implements it differently to test the same hypothesis, will the results agree?
- **Demo > argument**: Can a simple visualization or small example make the result obvious at a glance?

**Warning signals**:
- You designed a complex experiment but can't clearly state what it's testing → cargo cult
- You only want a positive result and haven't considered what a negative result means → confirmation bias
- Results depend heavily on hyperparameter tuning rather than the core hypothesis → likely pseudo-validation

---

## Lens 3: Musk — First Principles & The Algorithm

**Key question**: Is every step of this experiment necessary? Is there a more direct way?

**Checklist**:
- **The Algorithm (5 steps)**:
  1. **Question requirements**: Why does this experiment exist? Is there a simpler way to get the same information?
  2. **Delete**: Which steps add no core value? Remove them.
  3. **Simplify**: Can the remaining steps be further simplified?
  4. **Accelerate**: Can the experiment cycle be shortened? (smaller data, fewer epochs, simpler model)
  5. **Automate**: Only consider automation at the very end
- **Asymptotic limit**: What does the theoretically optimal version of this experiment look like? Is the gap from physics constraints or process bloat?
- **Idiot index**: Is any part of the experiment wildly over-engineered relative to the information it produces?
- **Vertical integration**: Can you bypass intermediate tools/processes to get the core information directly?

**Warning signals**:
- You're optimizing a step that shouldn't exist → maximum waste
- The experiment includes "everyone else measures it this way" steps → analogy-driven decision making
- You started tuning parameters without calculating theoretical limits → inefficiency

---

## Output Format

When using this framework, structure your response like this:

1. **One-sentence verdict**: Direct answer on experiment design
2. **Karpathy's view**: 2-3 sentences on minimum viable experiment and data/checks
3. **Feynman's view**: 2-3 sentences on what core assumption is being tested and cargo-cult risks
4. **Musk's view**: 2-3 sentences on what can be deleted or simplified
5. **Synthesis**: Explicitly note any conflicts between the three lenses
6. **Next action**: One concrete, executable step
