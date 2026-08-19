# Initialize an Existing Repository

## Purpose

Use this process when `.forge/` is added to a repository containing source code, configuration, documentation, or an existing product. The goal is to create a trustworthy baseline of what exists before proposing changes.

## Read first

Read [`00-overview.md`](00-overview.md), then inspect the repository without modifying source code. Classify findings as **observed** (directly supported by files or executed commands), **inferred** (reasonable interpretation), or **desired** (requested future behavior).

## Procedure

1. **Inventory the repository.** Identify languages, frameworks, package managers, entry points, deployment files, tests, generated files, and important directories.
2. **Determine runnable state.** Locate documented commands, install dependencies when permitted, run the safest available checks, and record failures verbatim. Do not assume a failing or missing command is a green gate.
3. **Recover product intent.** Read existing documentation and ask only the questions needed to distinguish intended behavior from current behavior.
4. **Draft product overview.** Populate [`../requirements/product-overview.md`](../requirements/product-overview.md), separating current state, desired outcomes, stakeholders, constraints, and known gaps.
5. **Extract feature requirements.** Create or update files under [`../requirements/features/`](../requirements/features/). Use stable IDs and preserve observed behavior when it is part of the current contract.
6. **Review requirements.** Spawn a fresh-context reviewer to identify ambiguity, unsupported claims, conflicts, and missing acceptance criteria. Resolve blockers before blueprinting.
7. **Map the technical system.** Populate [`../blueprints/technical-context.md`](../blueprints/technical-context.md) from repository evidence. Add container, component, and feature blueprints with links to real paths or symbols where useful.
8. **Review blueprints.** Spawn a fresh-context reviewer to compare documented architecture with repository evidence and the requirements. Record discrepancies rather than silently correcting them.
9. **Configure policy.** Replace the `TODO` values in [`00-overview.md`](00-overview.md) with verified commands, branch expectations, manual gates, tools, and parallelism rules.
10. **Recommend the first initiative.** If the repository is broken, prioritize restoring a runnable baseline. Otherwise recommend the smallest valuable change that exercises the documented workflow.

## Required completion evidence

- Repository inventory and runnable-state evidence are recorded.
- Observed, inferred, and desired behavior are distinguishable in durable documents.
- Requirements and blueprints have independent review findings.
- Project policy contains verified commands where available and named owners for unknowns.
- The first initiative addresses the highest-value baseline gap or delivers a small vertical slice.

## Handoff

Point the orchestrator to [`02-initiative.md`](02-initiative.md). If the repository cannot be safely understood or validated, mark initialization `blocked` and recommend a discovery or recovery initiative rather than guessing.
