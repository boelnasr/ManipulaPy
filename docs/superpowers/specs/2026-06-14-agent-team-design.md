# Agent Team Design — 2026-06-14

A set of 13 user-global subagents (`~/.claude/agents/`), each a thin persona grounded
in skills already installed (gstack, superpowers, claude-mem). User-global scope means
they are available in every project, including the ROS2 workspaces.

## Design constraints
- **Mixed advisor/doer split** (chosen by user): strategy/review roles are read-only
  advisors; engineering/asset roles are doers.
- **Subagents cannot spawn subagents.** Advisors return verdicts; the user (top level)
  dispatches doers to act on them. No self-running management hierarchy.
- Each agent earns its place — no decorative roles (skipped: product-manager,
  community-manager, scrum-master, data-engineer as redundant for a solo library).

## Roster

| Agent | Mode | Primary skills | Job |
|-------|------|----------------|-----|
| ceo | advisor | gstack:plan-ceo-review, office-hours | Strategic/product verdicts |
| cto | advisor | gstack:plan-eng-review, health, superpowers:systematic-debugging | Architecture/tech-strategy verdicts |
| marketing-manager | advisor | gstack:brand, design, banner-design | Positioning, copy, launch plans |
| repo-researcher | advisor | claude-mem:mem-search + code search | Codebase + past-work research |
| security-reviewer | advisor | security-review, gstack:cso | Secrets/unsafe-pattern scan |
| python-specialist | doer | superpowers:test-driven-development, code-review, simplify | Primary feature/bugfix coding |
| physics-validator | doer | superpowers:test-driven-development, systematic-debugging | Robotics-math correctness |
| qa-engineer | doer | verify, gstack:qa-only, verification-before-completion | Suite + notebook execution / regressions |
| devops-engineer | doer | gstack:ship, land-and-deploy, review | CI/env/branch/ship mechanics |
| release-manager | doer | claude-mem:version-bump, gstack:ship, document-release | Semver, CHANGELOG, tag, release |
| documentation-writer | doer | gstack:document-generate, document-release, make-pdf | Docs, docstrings, release notes |
| figures-specialist | doer | diagram + _build_nb* figure pipeline | Scientific matplotlib/TikZ figures |
| frontend-uiux-engineer | doer | ui-ux-pro-max, ui-styling, design-review | Web demos, docs-site UI/UX |

## Cross-cutting rules baked into doer prompts
- Commit policy (devops-engineer, release-manager): no Claude/AI/claude-flow mentions,
  no internal-planning-doc references in commits/PRs/release notes.
- Verification before "done"; report failures faithfully with real output.
- Advisors translate concerns into concrete changes and hand back; they do not edit.
