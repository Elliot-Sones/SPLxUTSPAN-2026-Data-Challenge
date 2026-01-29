<python>
  Use uv for everything: uv run, uv pip, uv venv.
</python>

<test-results>
  After running any algorithm as a test, you need to add:
    - What we ran with EXACT precision
    - Results with exact precision
  The standard is that if I want to reproduce this test, I should be able to read the description and reproduce to exact precision
  This should include, data we used, model, config and anything else that is worth noting.
</test-results>


<submission>
When running on test.csv file to get target for submission, the   output should be a file name + number (1,2,3,4,5) in the submission folder. Also add the submission details (exactly what model and how it was trained with other details so i can reproduce), the submission Why don't you results and pointing to the submission file in the folder
</submission>


<principles>
  <style>No emojis. No em dashes - use hyphens or colons instead.</style>

  <epistemology>
    Assumptions are the enemy. Never guess numerical values - benchmark instead of estimating.
    When uncertain, measure. Say "this needs to be measured" rather than inventing statistics.
  </epistemology>

  <scaling>
    Validate at small scale before scaling up. Run a sub-minute version first to verify the
    full pipeline works. When scaling, only the scale parameter should change.
  </scaling>

  <interaction>
    Clarify unclear requests, then proceed autonomously. Only ask for help when scripts timeout
    (>2min), sudo is needed, or genuine blockers arise.
  </interaction>

  <ground-truth-clarification>
    For non-trivial tasks, reach ground truth understanding before coding. Simple tasks execute
    immediately. Complex tasks (refactors, new features, ambiguous requirements) require
    clarification first: research codebase, ask targeted questions, confirm understanding,
    persist the plan, then execute autonomously.
  </ground-truth-clarification>

  <spec-driven-development>
    When starting a new project, after compaction, or when SPEC.md is missing/stale and
    substantial work is requested: invoke /spec skill to interview the user. The spec persists
    across compactions and prevents context loss. Update SPEC.md as the project evolves.
    If stuck or losing track of goals, re-read SPEC.md or re-interview.
  </spec-driven-development>

  <first-principles-reimplementation>
    Building from scratch can beat adapting legacy code when implementations are in wrong
    languages, carry historical baggage, or need architectural rewrites. Understand domain
    at spec level, choose optimal stack, implement incrementally with human verification.
  </first-principles-reimplementation>

  <constraint-persistence>
    When user defines constraints ("never X", "always Y", "from now on"), immediately persist
    to project's local CLAUDE.md. Acknowledge, write, confirm.
  </constraint-persistence>
</principles>

<machines>
    Remote access- GPU via vast.ai (Any gpu)
</machines>

<skill-spec>
  Prevent wrong assumptions and context loss across compactions by building a comprehensive,
  persistent specification through structured interviewing.

  <when-to-use>
    - Starting a new project (no SPEC.md exists)
    - After compaction when resuming substantial work
    - User explicitly invokes /spec
    - Requirements are ambiguous and substantial work is requested
    - User says "interview me" or asks to clarify architecture
  </when-to-use>

  <process>
    1. Read existing SPEC.md and project CLAUDE.md if they exist
    2. Interview using AskUserQuestion tool:
       - Architecture: data structures, concrete shapes, performance constraints, tradeoffs, boundaries
       - Scope: what's OUT of scope, success criteria, edge cases, dependencies
       - Implementation: patterns/anti-patterns, preserve vs rewrite, scaling, testing
       - Risks: what needs measurement, technical uncertainties, failure modes
    3. Question style:
       - Ask non-obvious questions - skip anything derivable from code
       - Use concrete examples: "If batch_size=1024, does each element represent X or Y?"
       - Challenge assumptions: "You said X, but that conflicts with Y - which takes priority?"
       - Offer adversarial interpretations to surface hidden requirements
    4. Continue until user signals completion, summarize and ask for corrections each round
    5. Write to ./SPEC.md with: Objective, Success Criteria, Architecture (data structures,
       boundaries), Constraints (non-negotiable, tradeoffs, out of scope), Implementation
       Strategy (optimization params, patterns, anti-patterns), Open Questions, Reference Examples
  </process>

  <rules>
    - Never guess numerical values - mark as "TBD: needs benchmarking"
    - Spec can be large (1000+ lines) - comprehensiveness beats brevity
    - Update spec incrementally as project evolves
    - After writing spec, remind user to review and correct
  </rules>
</skill-spec>

<skill-sub-agent-delegation>
  Delegate complex tasks to sub-agents for parallel autonomous work.

  <permissions>
    - NEVER spawn without explicit permission
    - ASK first: "I've identified [TASK] for sub-agent delegation. Should I spawn one?"
    - Explain WHY before requesting
  </permissions>

  <when-to-delegate>
    - GPU kernel optimization with iterative benchmarking
    - Numerical correctness verification across test cases
    - Performance profiling and analysis
    - Parallel investigation of independent code paths
    - Long-running validation suites
  </when-to-delegate>

  <patterns>
    - Parallel: Optimize independent kernels simultaneously (attention to A, MLP to B)
    - Correctness First: Make tests pass before performance
    - Incremental: Iterate until target speedup or report blockers
  </patterns>

  <kernel-optimization-template>
    Optimize [OPERATION] in [FILE].
    Context: [current impl], [bottleneck source], [target HW: 3090/H100], [use case: train/inference]
    Requirements:
    1. Implement with Triton/CUDA
    2. Verify: torch.allclose(atol=1e-5, rtol=1e-5), gradients match autograd
    3. Benchmark: warmup=10, bench=100, report min/max/mean/std us
    4. Scales: (1,128), (8,512), (32,2048)
    Report: correctness status, perf table (scale, baseline_us, opt_us, speedup), memory
  </kernel-optimization-template>

  <workflow>Setup -> Develop -> Verify -> Benchmark -> Report</workflow>

  <requirements>
    - Report measured numbers, never estimates
    - Include methodology (warmup, iterations, sync)
    - Flag regressions immediately
  </requirements>
</skill-sub-agent-delegation>

<skill-paper-implementation>
  Implement research papers from arxiv.

  <input>Require arxiv link (not PDF). Fetch LaTeX: https://arxiv.org/e-print/XXXX.XXXXX (.tar.gz with .tex files)</input>

  <phases>
    1. Discovery: Parse LaTeX for architecture, algorithms, hyperparameters, loss functions, datasets.
       Search existing GitHub implementations. Note ambiguities.
       Gather: Scope (train/inference/finetune)? Scale (model size, compute)? Baseline codebase?
       Priority (accuracy/speed/memory)? Validation method?
    2. Verification: If repo exists, audit against LaTeX source. Check architecture, hyperparameters,
       training procedure. Identify discrepancies.
    3. Refinement: Present findings, ask questions, iterate until execution steps are perfectly clear.
    4. Implementatio7n: Build/modify code. Write correctness tests. Profile performance.
    5. Optimization (optional): Profile with nsight-systems/torch.profiler. Write custom CUDA/Triton
       kernels. Benchmark with measurements.
  </phases>

  <on-arxiv-link>
    1. Parse LaTeX source
    2. Search existing implementations
    3. Present structured questions
    4. Wait for answers, refine, then proceed autonomously
  </on-arxiv-link>
</skill-paper-implementation>

</claude-instructions>