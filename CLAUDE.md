# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Execution Planning

Before starting a nontrivial task, read the whole request and form a complete plan before touching code. As part of that plan, identify every step that would need approval, elevated permissions, network access, is destructive, long-running, irreversible, or is a repository operation (push, merge, rebase, branch deletion, etc.).

For each one: check whether it's actually necessary, and prefer a simpler approach that avoids it when that doesn't compromise the work — see "Working efficiently in this repo" below for concrete examples already found this way in this repo (direct pixi binaries, no `pixi run`; real pytest tests instead of ad hoc snippets). Separate genuinely risky/consequential actions from ones that merely happen to trigger a permission prompt. Actions already pre-authorized elsewhere in this file (e.g. commit/push after a round, below) don't need to be re-asked each time — this review is about *new or unreviewed* risk. For a step that is genuinely risky or consequential and not already pre-authorized: stop before implementing it, explain what will be done and why, state what permission is needed, describe alternatives considered, and wait for approval.

If the plan changes mid-task in a way that introduces a new risky or consequential action, stop and repeat this review before continuing.

## Design Principles

Prefer simple, explicit, composable code over clever or hidden abstraction.

**Explicit pipelines.** Prefer a visible sequence of transformations over a wrapper that hides several conceptual stages:

```python
phase = hilbert_phase(...)
epochs = identify_cycles_from_phase(phase, ...)
samples, cycles, details = fit_pca_phase_coordinates(X, epochs=epochs)
```

over a single function that estimates phase, identifies cycles, and fits coordinates internally. This mirrors the package's own phase-estimation / cycle-identification / coordinate-estimation / diagnostics split — don't collapse it back together.

**Single responsibility.** Each function performs one clearly defined conceptual transformation. Avoid functions that combine several of: phase estimation, cycle identification, coordinate estimation, diagnostics, report formatting.

**Shared representations.** Use well-defined shared structures (e.g. `CycleEpochs`) as the interface between stages, and validate their invariants early (at construction) so downstream code can rely on them without re-checking.

**No package-level convenience wrappers for tests/demos.** Don't add a wrapper to the package just to shorten a test, example, or notebook. Put convenience helpers in the test file, demo script, or notebook that needs them.

**Clean development architecture.** On development branches, prefer clear architecture over preserving historical APIs. Don't add compatibility wrappers without a concrete, current need; handle merge/release compatibility deliberately when that work actually comes up, not preemptively.

## Environment

Dependencies are managed with **pixi**, resolved into `.pixi/envs/default` (shared across worktrees at the repo root: `../../../.pixi/envs/default` relative to a checkout under `.claude/worktrees/`). The system/bare `python`/`python3` have no project deps installed.

```bash
# Run full test suite (excludes slow/MCMC tests)
<repo-root>/.pixi/envs/default/bin/pytest tests/ -q -m "not slow"

# Run everything including the slow Bayesian smoke test
<repo-root>/.pixi/envs/default/bin/pytest tests/ -q
```

## Working efficiently in this repo

**Commit and push after each round.** When doing a round of fixes (e.g. resolving review feedback), commit and push to the current branch's remote as soon as the round is done, without waiting for separate confirmation. Still: one commit per round (don't amend), descriptive message, no force-push.

**Invoke pixi directly, never through `pixi run`.** Call `.pixi/envs/default/bin/python` / `.../bin/pytest` directly — they're already-resolved binaries. `pixi run pytest ...` and the direct-path form run the same thing, but `pixi run` is a different leading command and prompts every time even though the underlying command is already trusted; the direct-path form also lets a permission allowlist entry actually match it (see below).

**A pytest allowlist entry, if you have one, is personal, not committed.** `.claude/settings.json` is shared/committed, but a `Bash(.../pytest *)` entry needs the pixi env's *absolute* path, which is machine-specific (different clone location, different user) — committing one would silently stop matching, or worse, leak someone's local path, for everyone else. Add it to your own `.claude/settings.local.json` (already gitignored) instead if you want it, matching the exact absolute path from `<repo-root>/.pixi/envs/default/bin/pytest` on your machine.

**Never prefix that call with `cd ...`, `VAR=value`, or chain it after another command.** The allowlist match is against the *entire* command string as a prefix, so `cd /path && PYTHONPATH=... .../bin/pytest tests/` does not match `Bash(.../bin/pytest *)` even though the pytest invocation itself is identical to an allowlisted one — it prompts every time purely because of the leading `cd`/env-var. Fix: issue `cd <dir>` as its own separate command first (bare `cd` is always auto-allowed and the cwd persists across subsequent commands), then run the pytest/python command completely bare, nothing before the binary path. `PYTHONPATH` is also unnecessary once cwd is correct — `phase_coordinates` resolves on its own from the local checkout.

**Prefer real pytest tests over ad hoc `python -c` snippets.** A one-off `python -c "..."` is unique text each time and can never be pre-approved, even when the check is trivial (shape asserts, import checks, printing a value) — every single one prompts for approval. Write the check as an actual test (or extend an existing one) in `tests/` and run it through the pre-approved pytest binary instead. For genuine one-off numeric exploration that doesn't belong in the test suite, write it to a script under `docs/debug/scripts/` with the Write tool (no prompt) and execute it once via the direct pixi python binary — this turns N prompts into 1 and leaves a reviewable artifact. Prefer Read/Grep over spinning up Python at all for structural questions ("does this function exist", "what's this signature").
