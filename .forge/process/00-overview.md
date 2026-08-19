# Software Forge Process Overview

## Purpose

This directory contains the instructions that agents use to initialize a repository, plan an initiative, execute work orders, validate changes, and close out the resulting branch. The workflow is documentation-driven: agents must read the durable requirements and blueprints before proposing or implementing changes, and must leave an auditable record of decisions and evidence.

The installed workflow begins at [`../_index.md`](../_index.md). This file defines the shared vocabulary, lifecycle, defaults, and project-specific policy that all other process documents inherit.

## Artifact boundaries

- [`../requirements/`](../requirements/) is the durable product definition: what the system should do, for whom, and why.
- [`../blueprints/`](../blueprints/) is the durable technical definition: how the system is organized and how behavior is realized.
- [`../initiatives/`](../initiatives/) contains bounded work. An initiative may update requirements or blueprints only when its work orders explicitly authorize those updates.

Temporary planning notes, agent conversations, and scratchpads are inputs to the workflow, not durable sources of truth. When a decision becomes relevant to future work, record it in the appropriate durable document or initiative log.

## Shared lifecycle

### Initialization

Use [`01-init-new.md`](01-init-new.md) for a new or empty repository and [`01-init-existing.md`](01-init-existing.md) for a repository with existing code or artifacts. Both paths produce a product overview, requirements/blueprints baseline, project policy, independent validation findings, and a recommended first initiative.

### Initiative execution

Use [`02-initiative.md`](02-initiative.md) to turn a requested change into a scoped initiative and ordered work orders. Validate the proposed design with [`03-design-review.md`](03-design-review.md), implement each work order with [`04-work-order.md`](04-work-order.md), validate implementation with [`05-implementation-review.md`](05-implementation-review.md), and complete the final reconciliation with [`06-closeout-review.md`](06-closeout-review.md).

## Status vocabulary

Use these statuses unless the project policy defines a stricter compatible vocabulary:

| Status        | Meaning                                                                        |
| ------------- | ------------------------------------------------------------------------------ |
| `draft`       | The artifact is being authored and is not ready for execution.                 |
| `in-review`   | A reviewer or human is evaluating the artifact.                                |
| `approved`    | Required approval for the current gate has passed.                             |
| `in-progress` | Work is actively being performed.                                              |
| `blocked`     | Safe progress requires a decision, fix, external action, or manual gate.       |
| `complete`    | The artifact's scoped work and required validation are complete.               |
| `closed`      | The initiative has completed closeout and its follow-up state is discoverable. |

Do not mark an artifact `complete` merely because an agent stopped working. Record the evidence and gate outcome that support the status transition.

## Default roles

- **Orchestrator:** owns the initiative state, dispatches agents, enforces gates, and maintains the execution log.
- **Architect/planner:** collaborates on intent, constraints, design, and work-order decomposition.
- **Implementer:** executes one assigned work order and records the resulting changes.
- **Reviewer:** evaluates an artifact or implementation with fresh context and reports findings without silently changing the reviewed work.
- **Human approver:** resolves decisions and approvals required by the project policy.

One agent may perform more than one role only when the project policy permits it. A reviewer must not approve its own implementation work.

## Default gate policy

The default is human approval after design review. Human approval is not required after every implementation review unless the project policy says so; implementation reviews still run according to the configured cadence, which defaults to one review per work order. Closeout is mandatory.

An automated bypass is valid only when it is documented in this policy section or in an initiative-specific decision that is stricter than the policy. Every bypass must identify the policy basis, approver or standing authority, and validation evidence.

## Project-specific policy

Initialization must replace the placeholders in this section with facts for the host repository. Until that happens, use the defaults above and treat unknown commands or gates as unknown—not as permission to skip validation.

### Repository and branch

- **Repository type:** Python 3.10 CLI for virtual-production 3D Gaussian Splatting; Pixi/setuptools package with pytest, Ruff, and mypy dev tools (observed in [`pyproject.toml`](../../pyproject.toml:1) and [`pixi.toml`](../../pixi.toml:1)).
- **Default branch:** `master` (observed from `.git/HEAD` and `.git/config`; origin is `johnwildauer/gsforge`).
- **Initiative branch policy:** Create a dedicated development branch per initiative. Closeout merges that branch back into `master`, or into a specifically named release branch when that future policy is adopted. Branch naming format is not yet specified and must be chosen during initiative planning.
- **Commit/pull-request policy:** Human review is required at the configured Forge gates. Exact hosting protection rules and required status checks are unknown; do not infer them as passing. Closeout must include human approval before merge.

### Commands

- **Install/setup:** `pixi install` (documented in [`README.md`](../../README.md:68); lockfile present). Outcome during initialization: not executed because this session exposed no command-execution tool; treat as unverified.
- **Format:** `pixi run format` (declared in [`pixi.toml`](../../pixi.toml:57)). Outcome: not executed; unverified.
- **Lint/static analysis:** `pixi run lint` and, if separately configured, `pixi run mypy`; Ruff task is declared, no mypy task is declared. Outcome: not executed; unverified.
- **Unit/integration tests:** `pixi run test` (declared in [`pixi.toml`](../../pixi.toml:55); equivalent `pytest`). Outcome: not executed; test files were inspected, but no pass/fail claim is made.
- **Build/package:** `python -m build` is conventional but not declared; no repository build task was observed. Outcome: unknown/unverified.
- **Run/manual verification:** `pixi run gsforge --help` or `gsforge --help`, then a manual workstation smoke test covering ingest, COLMAP/GLOMAP, training, preview/checkpoint creation, PLY inspection, and export. Outcome: not executed; external tools and GPU are manual gates.

### Gates and intervention

- **Design approval:** Human must remain in the loop and iterate during every design-phase prompt of an initiative unless explicitly commanded otherwise.
- **Implementation review cadence:** One independent review per work order by default.
- **Manual/external gates:** Human approval after all work orders complete and before closeout/merge; real FFmpeg and COLMAP smoke test; CUDA/PyTorch/gsplat training smoke test; inspection of preview/checkpoint/final PLY/export artifacts; maintainer confirmation of unresolved product/implementation discrepancies.
- **Automation bypasses:** None standing. A specific explicit human instruction may waive the design-loop or another gate only for that initiative, with the waiver and evidence recorded in its execution log.
- **Network or destructive-action restrictions:** Network access is limited to documented dependency/setup needs. Do not delete or overwrite user media, models, checkpoints, or exports without explicit authorization. Use isolated temporary project directories for smoke tests.

### Tools and parallelism

- **Required tools/MCP servers:** Repository file inspection and Markdown editing; no project runtime tool is mandated by the repository. A command-execution capability is required to run validation, but was unavailable in this initialization session.
- **Optional tools/MCP servers:** Clangd is not relevant to this Python repository. Browser or other MCP tools are not required unless an initiative introduces an external UI or service.
- **Default parallelism:** `sequential work orders`
- **Permitted parallelism:** Parallel documentation inspection is safe. Implementation work orders remain sequential unless they have disjoint ownership and an explicit merge/reconciliation plan; all branches merge through the initiative closeout gate.

### Failure and recovery

When a review or command fails, the orchestrator must record the exact finding, affected artifact, attempted remediation, evidence, decision-maker, and next state in the initiative execution log. A failed gate blocks completion until it is fixed, explicitly waived under this policy, or tabled as follow-up work.

## Authoring rules shared by all processes

1. Read the relevant process file, initiative artifacts, requirements, and blueprints before acting.
2. Prefer evidence from the repository over assumptions. Label observed, inferred, desired, and provisional information where relevant.
3. Keep scope explicit. Every work order must state what is in scope and out of scope.
4. Preserve traceability from work order to requirements/blueprints and from implementation back to the work order.
5. Make the smallest durable documentation update that accurately reflects the resulting system.
6. Append execution events; do not rewrite history to hide failed attempts or discarded decisions.
7. Stop and mark the work `blocked` when required information or validation is unavailable.
