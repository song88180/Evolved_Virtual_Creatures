# Project Instructions

When running Python scripts, tests, or checks for this project, always use the conda environment named `mujoco`.

Use:

```bash
conda run -n mujoco --no-capture-output python -m pytest
```

# Codex Git Workflow

When modifying tracked files in this project:

1. Before editing, inspect `git status` and avoid overwriting user changes.
2. After editing, do not commit automatically.
3. At the end of the response, include:
   - Proposed commit message for `git commit -m`.
   - Ask whether to (1) commit, (2) keep changes uncommitted, (3) commit to a new branch, or (4) revert Codex's changes.
4. If the user approves the commit:
   - Run `git add` only for files changed for this task.
   - Run `git commit -m "<proposed message>"`.
5. If the user says to keep changes without committing:
   - Leave the working tree as-is.
6. If the user rejects the changes:
   - Revert only Codex's changes from this task.
   - Do not revert unrelated user changes.
7. Never push unless the user explicitly asks.
