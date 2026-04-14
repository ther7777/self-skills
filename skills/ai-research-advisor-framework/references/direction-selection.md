# Direction Selection Framework

Use this reference when the user's question is about choosing between research directions, evaluating whether an approach is worth pursuing, or feeling纠结 about algorithmic improvements.

## Core Principle

Before giving advice, internally route through three lenses: Ilya (taste), Munger (inversion), Taleb (convexity/risk). Output the synthesis in a structured format.

---

## Lens 1: Ilya — Research Taste & Compression

**Key question**: Is this direction elegant? Is it leading to better understanding/compression?

**Checklist**:
- Is it doing better compression, or just benchmark hacking?
- Is it an improvement or a transformation? ("Just scale" era is over.)
- Does it fit "research aesthetics" — simple, elegant, biologically inspired?
- Is data the bottleneck for this direction?
- Will this be the main road or a side road in 5-10 years?

**Warning signals**:
- Needs many hacks and special cases to work → "ugly research" red flag
- Only improves a benchmark score without better compression/understanding → likely a side road
- Optimizing pre-training when pre-training as we know it is ending → wrong timing

---

## Lens 2: Munger — Inversion & Cognitive Biases

**Key question**: Under what conditions will this direction definitely waste my time?

**Checklist**:
- **Inversion**: Don't ask "why is this good?" Ask "what would guarantee failure?"
- **Three baskets**: Yes (confident), No (confidently skip), Too Hard (not enough info). Most things are Too Hard.
- **Bias detection**:
  - Social proof: "Everyone is doing it" ≠ "worth doing"
  - Overconfidence: Am I overestimating my unique insight?
  - Deprivation super-reaction: Am I driven by FOMO?
  - Lollapalooza: Are multiple biases reinforcing each other?
- **Incentive structure**: Is my choice influenced by external pressure (advisor preference, reviewer taste, hot topics)?

**Warning signals**:
- You're doing it because others are → FOMO-driven research
- You can't argue the opponent's side better than they can → not qualified to hold this opinion
- It has to be complicated to work → probably doesn't

---

## Lens 3: Taleb — Convexity & Tail Risk

**Key question**: Does this strategy have convexity? What's the worst case?

**Checklist**:
- **Convexity test**:
  - If it fails, is the cost limited and known?
  - If it succeeds, is the payoff unknown but potentially large?
  - Can you gain information through small cheap experiments?
- **Ergodicity test**: If you repeat this 100 times, will you be wiped out in one of them?
- **Tail risk**: What's the worst case? (3 months wasted? 1 year? Entire PhD?)
- **Skin in the game**: Do the people promoting this direction bear real costs if they're wrong?

**Warning signals**:
- All-in on an unvalidated big direction → betting against ergodicity
- Downside is catastrophic (e.g. entire PhD), upside is just a normal paper → negative convexity
- Experts all say "this is safe" → beware consensus narratives

---

## Output Format

When using this framework, structure your response like this:

1. **One-sentence verdict**: Direct answer (pursue / skip / small bet / need more info)
2. **Ilya's view**: 2-3 sentences on elegance, compression, and long-term ceiling
3. **Munger's view**: 2-3 sentences on failure modes and cognitive biases
4. **Taleb's view**: 2-3 sentences on convexity and tail risk
5. **Synthesis**: Explicitly note any conflicts between the three lenses
6. **Next action**: One concrete, executable step
