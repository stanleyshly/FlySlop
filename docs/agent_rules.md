# Rules for every FlySlop implementation subagent

Repo: /Users/stanley/Desktop/FlySlop. The plan is in PLAN.md. The frozen contracts are in docs/architecture.md, in the section "Contracts (P0, frozen)". Read both before you write code. Implement against the contracts. Do not change them. If a contract is unworkable, say why in your report.

1. **RAM.** The user's hard rule is that everything stays under a configurable cap.
   - Use `backend/memory_budget.py`: `check_fits`, `start_watchdog`, `worker_budget`, and the env var `FLYSLOP_MAX_RAM_GB`.
   - Several agents run at the same time, so every heavy command YOU run must use your stated per-agent budget. Prefix the command with `FLYSLOP_MAX_RAM_GB=<budget>`.
   - Anything the user will run later must read the cap. Default to 4 GB and size worker counts from it.
2. **Context (strict, the user pays for it).** Keep your own context small.
   - Hard limit: hand off at ~80k tokens. Final report must be under 40 lines.
   - Read files with `grep -n` and `sed -n` ranges only. Send ALL command output to a file and read only `tail -20`.
   - Run the full test suite once, at the end, with `2>&1 | tail -5`.
   - Use grep/head/sed -n. Don't cat big files. Send long logs to files under the scratchpad and tail them.
   - If you think you are past ~80k tokens, stop at a clean point and hand off (see rule 7).
3. **No long training.** Smoke tests must each finish in under ~2 minutes. The user runs full training offline.
4. **Git.** The repo has no commits. Never run git commit, reset, checkout, stash or worktree. Just edit files.
5. **File ownership.** Edit ONLY the files your prompt lists as yours.
   - Other agents are editing other files at the same time.
   - If you need a change in a file you don't own, don't make it. Describe the exact change in your report.
   - For `pyproject.toml` and `uv.lock`, don't edit them. Put the dependency you need in your report.
6. **Tests.**
   - Add unittest tests under tests/ using your own file names.
   - Run your tests and `uv run python -m unittest discover -s tests`. Other agents' in-progress work may cause failures. Report those failures, but don't fix files you don't own.
   - Use `uv run --extra training` for torch.
7. **Final report.** Keep it concise and in this order:
   - files changed
   - commands run
   - measured numbers, including peak RSS of heavy commands
   - test results
   - remaining risks
   - requested changes to files you don't own
   - if you stopped early, a precise "next steps" handoff
8. **Code style.** Match the style of the surrounding code. Keep the code deterministic where the plan requires it (hashes, seeds).
