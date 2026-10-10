# Development Loop — ZIP → Local → GitHub PR → Kaggle

## One-time empty repository bootstrap

1. Clone `https://github.com/vastolordede/MachineLearningRandomForest.git`.
2. On its unborn `main`, create an initial empty commit: `git commit --allow-empty -m "chore: bootstrap main branch"`.
3. Push `main`; create/push `dev` from `main`.
4. From `dev`, create `feature/project-scaffold` and apply the first patch.

## Every feature patch

1. `git switch dev` / `git pull --ff-only origin dev` / `git status` (must be clean).
2. `git switch -c feature/<cluster>`.
3. Place ZIP under `work/`, run `python work\\apply_patch.py <ZIP> --dry-run`.
4. Review `CREATE/REPLACE`, then `python work\\apply_patch.py <ZIP>`.
5. Run precise tests specified by the patch. Do not commit failed work.
6. `git status`, `git diff --check`, stage only relevant paths, `git diff --cached --check`, `git diff --cached`.
7. Commit, `git log --oneline -1`, push to `origin`, open PR targeting `dev`.
8. Only mark matching checklist IDs complete after PR is merged and acceptance conditions met.

## ML-specific safety

- No dataset/model artifact in Git; use versioned dataset/artifact storage.
- Freeze source commit + dataset revision + preprocessing version + seed for each experiment.
- `/kaggle/working/` is ephemeral before notebook outputs are saved; artifacts require a persistent backup.
- Data split and input schema are provisional until Thảo/Sang/Đăng agree to D12/D13.
- No imputer/encoder fitted on all rows before the split.
