# Contributing

Shared policy for the Venti / DISP-CAL / VLM repositories (kept in
`00_tools/standards/CONTRIBUTING.md`; repo-specific notes go below the line).

## Branches

| Branch | Purpose |
|---|---|
| `main` | mirrors upstream (`opera-adt/*`) or, for repos without an upstream, the reviewed state. **Nobody commits here directly.** |
| `feature/<topic>` | one concern: a feature, a fix, a refactor, a cleanup, or a chore. It becomes one PR. |
| `ops` (SAS repos) | `main` plus the feature branches an operational image actually pins. |

Work happens on the `mgovorcin/*` forks. PRs to `opera-adt/*` are opened only at
milestone boundaries (gamma 0.3, v0.5 CalVal, ...).

## Pull requests

- **One concern per PR.** Never mix a feature with a fix, a refactor with a
  cleanup, or either with a dependency bump. If the work is already mixed, split
  it (`git-commit-pr` skill).
- **Small.** Readable in one sitting. A reviewer should be able to say yes or no
  without untangling anything.
- **Conventional commits**: `feat:`, `fix:`, `refactor:`, `docs:`, `test:`,
  `chore:`, `perf:`. The subject says what changed; the body says why.
- **Tests in the same commit** as the behaviour they cover (`unit-tests` skill).
  "Tests to follow" is not accepted.
- **Value changes are explicit.** If a PR can change what a run produces, it
  either keeps the golden green or regenerates it on purpose with the diff
  explained (`workflow-regression` skill). Science changes land behind an
  `algorithm_parameters.yaml` flag whose default reproduces the previous result.
- **Claude Code** may author PRs under these rules; the repo owner reviews and
  merges. Claude never pushes to `main`.

## Before you push

```bash
pixi run lint      # pre-commit: ruff, black, mypy, nbstripout, SPDX headers
pixi run test
pixi run golden    # if the repo has a golden dataset
```

## Licensing

Every `.py` file carries an SPDX header matching the repository license (the
`spdx-header` pre-commit hook enforces it). Code copied or adapted from another
repository keeps that repository's notice; see its `NOTICE` file.

---
