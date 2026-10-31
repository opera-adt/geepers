<!-- One concern per PR: a feature, a fix, a refactor, a cleanup, or a chore. Not two. -->

## What

<!-- One or two sentences. Link the issue / plan task: Closes #NN, implements T## -->

## Why

<!-- The problem, with numbers if there are any. -->

## Checklist

- [ ] Tests for the new or changed behaviour are in this PR
- [ ] `pixi run lint` and `pixi run test` pass locally
- [ ] Golden: unaffected / regenerated on purpose (diff explained below)
- [ ] Value-changing: behind a flag that defaults to the previous behaviour, or the changelog explains the new default
- [ ] Docs updated (algorithm parameters, CLI, product spec) where behaviour is visible to users

## Golden / e2e impact

<!-- "none", or: which layers changed, median/max difference, link to the e2e report -->
