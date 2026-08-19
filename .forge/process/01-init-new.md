# Initialize a New Repository

## Purpose

Use this process when the host repository is empty or has no reliable product/codebase baseline. The goal is not to generate a large speculative architecture; it is to establish enough shared context to start a first initiative safely.

## Read first

Read [`00-overview.md`](00-overview.md), then inspect the host repository and any user-provided notes. Treat technical choices as provisional until implementation validates them.

## Procedure

1. **Establish context.** Confirm the repository root, available tools, constraints, target platform, and what “empty” means. Record unknowns rather than inventing facts.
2. **Elicit product intent.** Collaborate with the user on the problem, target users, desired outcome, MVP boundary, non-goals, and success measures.
3. **Draft product overview.** Populate [`../requirements/product-overview.md`](../requirements/product-overview.md), clearly marking provisional decisions.
4. **Define initial features.** Create one file per cohesive feature under [`../requirements/features/`](../requirements/features/) using the feature template. Use testable requirement and acceptance-criterion IDs.
5. **Review requirements.** Spawn a fresh-context reviewer. Resolve blockers and record warnings before continuing.
6. **Choose a technical baseline.** Based on requirements and user constraints, document runtime, technology, dependency, and validation assumptions in [`../blueprints/technical-context.md`](../blueprints/technical-context.md).
7. **Create blueprints.** Add only the container, component, and feature blueprints needed to make the MVP implementable. Mark uncertain architecture as provisional.
8. **Review blueprints.** Spawn a fresh-context reviewer to compare the blueprints with requirements and the stated technical constraints. Resolve blockers.
9. **Configure policy.** Replace the `TODO` values in the project-specific policy section of [`00-overview.md`](00-overview.md), including setup, build, test, run, gate, tool, and parallelism rules.
10. **Recommend the first initiative.** Describe the smallest initiative that can produce a validated vertical slice or establish the minimum runnable foundation. Do not begin implementation in this initialization step.

## Required completion evidence

- Product overview exists and has an explicit MVP boundary.
- Each initial feature has testable requirements or is explicitly deferred.
- Technical context and necessary blueprints exist, with provisional choices labeled.
- Requirements and blueprint review findings are recorded and resolved or accepted.
- Project-specific policy is populated or its unknowns are explicitly assigned.
- A first initiative is recommended with rationale, scope, and expected validation.

## Handoff

Tell the user or orchestrator that initialization is complete and point to [`02-initiative.md`](02-initiative.md). If a required decision remains unresolved, mark initialization `blocked` and record the decision needed instead of starting implementation.
