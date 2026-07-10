# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

**Invoke pixi directly, never through `pixi run`.** Call `.pixi/envs/default/bin/python` / `.../bin/pytest` directly — they're already-resolved binaries. `pixi run pytest ...` and the direct-path form run the same thing, but only the direct-path form matches the project's `Bash(.../pytest *)` allowlist entry in `.claude/settings.json` at the repo root; `pixi run` is a different leading command and prompts every time even though the underlying command is already trusted.

**Prefer real pytest tests over ad hoc `python -c` snippets.** A one-off `python -c "..."` is unique text each time and can never be pre-approved, even when the check is trivial (shape asserts, import checks, printing a value) — every single one prompts for approval. Write the check as an actual test (or extend an existing one) in `tests/` and run it through the pre-approved pytest binary instead. For genuine one-off numeric exploration that doesn't belong in the test suite, write it to a script under `docs/debug/scripts/` with the Write tool (no prompt) and execute it once via the direct pixi python binary — this turns N prompts into 1 and leaves a reviewable artifact. Prefer Read/Grep over spinning up Python at all for structural questions ("does this function exist", "what's this signature").
