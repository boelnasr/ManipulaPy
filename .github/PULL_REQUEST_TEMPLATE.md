<!--
Thanks for contributing to ManipulaPy! Please read CONTRIBUTING.md and fill in the
sections below. Keep each PR focused on a single change.
-->

## Summary

<!-- What does this PR do, and why? Link any related issue, e.g. "Closes #123". -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor / maintenance
- [ ] CI / build

## Checklist

- [ ] Code follows PEP8 (`black` + `isort` + `flake8` pass).
- [ ] Tests added or updated for the change (RED→GREEN for bug fixes; see CONTRIBUTING.md).
- [ ] Regression tests live in `tests/test_v132_regressions.py` where appropriate.
- [ ] Full suite passes: `python -m pytest tests/ -v -m "not (cuda or vision or simulation)"`.
- [ ] Docstrings / docs updated; Sphinx builds clean if docs changed.
- [ ] Commit messages clearly describe the change.

## AI usage disclosure

<!-- Per CONTRIBUTING.md, if you used AI assistance, note it here. All submitted code
must be human-understood and tested. -->

## Additional notes

<!-- Anything reviewers should know: trade-offs, follow-ups, screenshots, etc. -->
